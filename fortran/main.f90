! Copyright 2023 FortAI-Hub contributors.
! Released under the MIT License. See LICENSE file for full license information.

program main
    use, intrinsic :: iso_c_binding
    use mod_essentials
    use mod_real_precision
    use mod_trie_tokenizer, only : load_tokenizer => load_trie_tokenizer, rwkv_tokenizer => trie_tokenizer
    use mod_rwkv_lm, load_model => load_rwkv_lm_model
    use mod_generation
    use mod_prompt_utils
    implicit none

    character(len=128) :: tokenizer_filename, model_filename
    type(rwkv_lm_type) :: model
    type(rwkv_tokenizer) :: tokenizer
    type(generation_options) :: gen_opts

    intrinsic signal
    integer, parameter :: sigint = 2 ! SIGINT interrupt signal (Ctrl-C) in Unix-based systems
    logical :: in_generation, stop_generation_requested

    call signal(sigint, handle_interrupt_signal)

    call get_arguments(tokenizer_filename, model_filename)
    call load_files(tokenizer_filename, model_filename, tokenizer, model)
    call precompute_layer_norm_embeddings(model)
    call warm_model(model)
    call run_chat_example(model, tokenizer, gen_opts)

contains

    subroutine get_arguments(tokenizer_filename, model_filename)
        character(len=128), intent(out) :: tokenizer_filename, model_filename
        integer :: ierr

        call get_command_argument(1, tokenizer_filename, status=ierr)
        if (ierr /= 0) then
            write (stdout,*) 'Usage: Please provide tokenizer_filename as an argument.'
            stop 1
        end if

        call get_command_argument(2, model_filename, status=ierr)
        if (ierr /= 0) then
            write (stdout,*) 'Usage: Please provide model_filename as an argument.'
            stop 1
        end if

        model_filename = trim(model_filename)
        tokenizer_filename = trim(tokenizer_filename)
    end subroutine

    subroutine load_files(tokenizer_filename, model_filename, tokenizer, model)
        character(len=128), intent(in) :: tokenizer_filename, model_filename
        type(rwkv_tokenizer), intent(out) :: tokenizer
        type(rwkv_lm_type), intent(out) :: model

        integer(8) :: count_rate, t1, t2

        call system_clock(count_rate=count_rate)

        write(stderr,'(a)') "> Loading tokenizer " // tokenizer_filename // "..."
        call system_clock(count=t1)
        tokenizer = load_tokenizer(tokenizer_filename)
        call system_clock(count=t2)
        write(stderr,'(a)') "    done. (" // real_to_str(real(t2-t1)/count_rate) // "s)"

        write(stderr,'(a)') "> Loading model " // model_filename // "..."
        call system_clock(count=t1)
        model = load_model(model_filename)
        call system_clock(count=t2)
        write(stderr,'(a)') "    done. (" // real_to_str(real(t2-t1)/count_rate) // "s)"

        call display_model_info(model)
    end subroutine

    subroutine display_model_info(m)
        type(rwkv_lm_type), intent(in) :: m

        write(stderr,'(a)') "> Model parameters:"
        write(stderr,'(a, i0)') "    d_model: ", m%d_model
        write(stderr,'(a, i0)') "    vocab_size: ", m%vocab_size
        write(stderr,'(a, i0)') "    n_layers: ", m%n_layers
    end subroutine

    subroutine precompute_layer_norm_embeddings(model)
        use mod_functions, only : layer_norm_2d
        type(rwkv_lm_type), intent(inout) :: model

        integer(8) :: count_rate, t1, t2

        call system_clock(count_rate=count_rate)

        write(stderr,'(a)') "> Precomputing of the layer normalization for the embeddings..."
        call system_clock(count=t1)
        model%emb = layer_norm_2d(model%emb, model%ln_emb%g, model%ln_emb%b, model%ln_emb%eps)
        model%precomputed_ln_emb = .true.
        call system_clock(count=t2)
        write(stderr,'(a)') "    done. (" // real_to_str(real(t2-t1)/count_rate) // "s)"
    end subroutine

    subroutine warm_model(model)
        use mod_state, only: state_type
        type(rwkv_lm_type), intent(in) :: model

        type(state_type) :: state
        integer :: inputs(13)
        real(sp), allocatable :: logits(:)

        integer(8) :: count_rate, t1, t2

        call system_clock(count_rate=count_rate)

        state = model%init_state()

        ! A simple sentence to warm up the model
        inputs = [53648, 59, 300, 47284, 57192, 4811, 32438, 4833, 22590, 39043, 261, 53671, 59] ! Token ids from rwkv_vocab_v20230424.csv

        write(stderr,'(a,i0,a)') "> Model warm up with ", size(inputs), " tokens..."
        call system_clock(count=t1)
        logits = model%forward_batch(inputs, state)
        call system_clock(count=t2)
        write(stderr,'(a)') "    done. (" // real_to_str(real(t2-t1)/count_rate) // "s)"
    end subroutine

    subroutine run_chat_example(model, tokenizer, gen_opts)
        use mod_state, only: state_type

        type(rwkv_lm_type), intent(in) :: model
        type(rwkv_tokenizer), intent(in) :: tokenizer
        type(generation_options), intent(in) :: gen_opts

        character(len=10000) :: input
        character(:), allocatable :: prompt
        integer :: iostat
        type(state_type) :: state

        state = model%init_state()

        do
            read(stdin, '(a)', iostat=iostat) input
            if (iostat /= 0) exit

            prompt = generate_prompt(instruction=replace_slash_n(trim(adjustl(input))))

            state = model%init_state() ! reset the state at every input

            call process_and_generate_text(model, tokenizer, gen_opts, state, prompt)
        end do
    end subroutine

    subroutine process_and_generate_text(model, tokenizer, opts, state, prompt)
        use mod_state, only: state_type

        type(rwkv_lm_type), intent(in) :: model
        type(rwkv_tokenizer), intent(in) :: tokenizer
        type(generation_options), intent(in) :: opts
        type(state_type), intent(inout) :: state
        character(:), allocatable, intent(in) :: prompt

        integer, allocatable :: inputs(:)
        real(sp), allocatable :: logits(:)

        integer :: i

        integer(8) :: count_rate, t1, t2

        call system_clock(count_rate=count_rate)

        write(stderr,'(a)') "> Tokenize prompt..."
        call system_clock(count=t1)
        inputs = tokenizer%encode(prompt)
        call system_clock(count=t2)
        write(stderr,'(a)') "    done. (" // real_to_str(real(t2-t1)/count_rate) // "s)"

        write(stderr,'(a)') "> Tokens:"
        call write_i0_arr_1d(stderr, inputs)

        write(stderr,'(a,i0,a)') "> Preprocessing ", size(inputs), " tokens..."
        call system_clock(count=t1)
        logits = model%forward_batch(inputs, state)
        call system_clock(count=t2)
        write(stderr,'(a)') "    done. (" // real_to_str(real(t2-t1)/count_rate) // "s)"

        write(stderr,'(a)') "> Generating..."
        call system_clock(count=t1)
        call generate_text(model, tokenizer, state, logits, opts)
        call system_clock(count=t2)
        write(stdout, '(a)') ''
        write(stderr,'(a)') "    done. (" // real_to_str(real(t2-t1)/count_rate) // "s)"
    end subroutine

    subroutine generate_text(model, tokenizer, state, input_logits, opts)
        use mod_state, only: state_type

        type(rwkv_lm_type), intent(in) :: model
        type(rwkv_tokenizer), intent(in) :: tokenizer
        type(state_type), intent(inout) :: state
        real(sp), intent(in) :: input_logits(:)
        type(generation_options), intent(in) :: opts

        real(sp) :: occurrence(size(model%proj, 1)) ! corresponding to the number of logits
        real(sp), allocatable :: logits(:)
        integer, allocatable :: sampled_indices(:)
        integer :: i, token_id
        integer(c_int) :: token_len
        character(:), allocatable :: token
        logical :: end_of_generation

        occurrence = 0
        in_generation = .true.
        stop_generation_requested = .false.

        logits = input_logits

        do i = 1, opts%max_token_limit
            if (stop_generation_requested) then
                stop_generation_requested = .false.
                write(stderr,'(a)') "> Stop generation due to user request."
                exit ! forced end of generation
            end if

            token_id = generate_next_token(logits, occurrence, opts, end_of_generation)

            if (end_of_generation) then
                exit
            end if

            token = tokenizer%decode([token_id])

            if (i > 1 .or. .not. is_whitespace(token)) then
                call print_token(token)
            end if

            logits = model%forward(token_id, state)
        end do

        in_generation = .false.
    end subroutine

    subroutine print_token(token)
        implicit none
        character(len=*), intent(in) :: token
        write(stdout, '(a)', advance='no') token
    end subroutine

    pure logical function is_whitespace(string)
        implicit none
        character(len=*), intent(in) :: string
        integer :: i

        is_whitespace = .true.

        do i = 1, len_trim(string)
            if (string(i:i) /= ' ' .and. string(i:i) /= char(9) .and. string(i:i) /= char(10) .and. string(i:i) /= char(13)) then
                is_whitespace = .false.
                exit
            end if
        end do
    end function

    function replace_slash_n(s) result(res)
        character(len=*), intent(in) :: s
        character(len=len(s)) :: res, temp
        integer :: pos

        res = s
        pos = index(res, "\n")

        do while (pos /= 0)
            if (pos == len_trim(res)) then
                temp = res(1:pos-1) // achar(10)
            else
                temp = res(1:pos-1) // achar(10) // res(pos+2:)
            end if
            res = temp
            pos = index(res, "\n")
        end do
    end function

    subroutine handle_interrupt_signal(signal_number)
        integer, intent(in) :: signal_number
        if (in_generation) then
            stop_generation_requested = .true.
        else
            stop "Exiting due to user interrupt"
        end if
    end subroutine

end program
