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

    integer, parameter :: sigint = 2 ! SIGINT interrupt signal (Ctrl-C) in Unix-based systems
    integer, parameter :: num_omp_threads = 8 ! found optimal on Apple M1 Max, 10-core CPU.

    character(len=128) :: tokenizer_filename, model_filename
    type(rwkv_lm_type) :: model
    type(rwkv_tokenizer) :: tokenizer
    intrinsic signal
    type(generation_options) :: gen_opts

    call omp_set_num_threads(num_omp_threads)

    call signal(sigint, handle_interrupt_signal)

    call get_arguments(tokenizer_filename, model_filename)
    call load_files(tokenizer_filename, model_filename, tokenizer, model)
    call precompute_layer_norm_embeddings(model)
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
    end subroutine get_arguments

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
    end subroutine load_files

    subroutine display_model_info(m)
        type(rwkv_lm_type), intent(in) :: m

        write(stderr,'(a)') "> Model parameters:"
        write(stderr,'(a, i0)') "    d_model: ", m%d_model
        write(stderr,'(a, i0)') "    vocab_size: ", m%vocab_size
        write(stderr,'(a, i0)') "    n_layers: ", m%n_layers
    end subroutine display_model_info

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
    end subroutine precompute_layer_norm_embeddings

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
    end subroutine run_chat_example

    subroutine process_and_generate_text(model, tokenizer, opts, state, prompt)
        use mod_state, only: state_type

        type(rwkv_lm_type), intent(in) :: model
        type(rwkv_tokenizer), intent(in) :: tokenizer
        type(generation_options), intent(in) :: opts
        type(state_type), intent(inout) :: state
        character(:), allocatable, intent(in) :: prompt

        integer, allocatable :: inputs(:)
        real(sp), allocatable :: logits(:)
        procedure(generated_token_handler), pointer :: print_generated_token

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

        print_generated_token => print_token

        write(stderr,'(a)') "> Generating..."
        call system_clock(count=t1)
        call generate_text(model, tokenizer, state, logits, opts, print_generated_token)
        call system_clock(count=t2)
        write(stdout, '(a)') ''
        write(stderr,'(a)') "    done. (" // real_to_str(real(t2-t1)/count_rate) // "s)"
    end subroutine process_and_generate_text

    subroutine print_token(token)
        implicit none
        character(len=*), intent(in) :: token
        WRITE(stdout, '(a)', advance='no') token
    end subroutine print_token

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
    end function replace_slash_n

    subroutine handle_interrupt_signal(signal_number)
        integer, intent(in) :: signal_number
        if (in_generation) then
            stop_generation_requested = .true.
        else
            stop "Exiting due to user interrupt"
        end if
    end subroutine handle_interrupt_signal

end program main
