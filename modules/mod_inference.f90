! Copyright 2023 FortAI-Hub contributors.
! Released under the MIT License. See LICENSE file for full license information.

module mod_inference
    use iso_fortran_env, only: error_unit, output_unit
    use mod_essentials, only : write_i0_arr_1d
    use mod_functions, only : layer_norm_2d
    use mod_generation, only: generation_options, generate_next_token
    use mod_prompt_utils, only: generate_prompt
    use mod_readline, only: readline
    use mod_real_precision, only: sp
    use mod_rwkv_lm, only: rwkv_lm_type, load_rwkv_lm_model
    use mod_stats, only: sample_once_from_multinomial
    use mod_timer, only: timer
    use mod_trie_tokenizer, only: trie_tokenizer, load_trie_tokenizer

    implicit none

    private
    public :: inference_options, run_inference

    type :: inference_options
        character(:), allocatable :: tokenizer_filename
        character(:), allocatable :: model_filename
        character(:), allocatable :: draft_model_filename
        type(generation_options) :: generation
        integer :: speculative_sampling_lookahead = 7
    end type

    type :: inference
        type(inference_options) :: options
        type(trie_tokenizer) :: tokenizer
        type(rwkv_lm_type) :: model
        type(rwkv_lm_type), allocatable :: draft_model
        logical :: speculative_sampling_enabled
    contains
        procedure :: run => inference_run
        procedure, private :: generate_text => inference_generate_text
        procedure, private :: generate_text_with_speculative_sampling => inference_generate_text_with_speculative_sampling
        procedure, private :: sample_token => inference_sample_token
        procedure, private :: load_tokenizer => inference_load_tokenizer
        procedure, private :: print_token => inference_print_token
        procedure, private :: print_tokens => inference_print_tokens
    end type

    interface inference
        module procedure :: inference_constructor
    end interface

    ! Variables for signals handling
    integer, parameter :: unix_sigint = 2, unix_sigquit = 3
    integer :: default_sigint_handler, default_sigquit_handler
    logical :: signal_received

contains

    subroutine run_inference(options)
        type(inference_options), intent(in) :: options
        type(inference) :: p
        p = inference(options)
        call p%run()
    end subroutine

    type(inference) function inference_constructor(options) result(self)
        type(inference_options), intent(in) :: options
        character(*), parameter :: warmup_text = &
            'Question: A simple sentence to warm up the model' &
            // new_line('') // new_line('') // 'Response:'
        integer, allocatable :: warmup_inputs(:)

        self%options = options
        self%speculative_sampling_enabled = allocated(options%draft_model_filename)

        call self%load_tokenizer()
        warmup_inputs = self%tokenizer%encode(warmup_text)

        self%model = load_model('model', options%model_filename, warmup_inputs)
        if (self%speculative_sampling_enabled) then
            self%draft_model = load_model('draft-model', options%draft_model_filename, warmup_inputs)
        end if
    end function

    subroutine inference_load_tokenizer(self)
        class(inference), intent(inout) :: self
        type(timer) :: t
        t = timer('Loading tokenizer ' // self%options%tokenizer_filename)
        self%tokenizer = load_trie_tokenizer(self%options%tokenizer_filename)
        call t%done()
    end subroutine

    subroutine inference_run(self)
        use iso_fortran_env, only: real64
        class(inference), intent(inout) :: self

        character(:), allocatable :: instruction
        character(:), allocatable :: prompt
        logical :: is_eof
        real(sp), dimension(:,:,:), allocatable :: state, draft_state  ! d_model, n_components, n_layers
        integer, allocatable :: prompt_tokens(:)
        real(sp) :: logits(self%model%vocab_size)
        real(sp), allocatable :: draft_logits(:)
        integer :: input_count = 0
        integer :: tokens_count = 0
        type(timer) :: t
        real(real64) :: elapsed_time

        call reset_state()

        do
            call read_user_input(instruction, is_eof)
            if (is_eof) exit
            if (len(instruction) == 0) cycle

            if (input_count > 0) then
                call reset_state()
            end if

            prompt = generate_prompt(instruction=instruction)

            t = timer('Tokenizing prompt')
            prompt_tokens = self%tokenizer%encode(prompt)
            call t%done()

            write(error_unit, '(a, i0, a)', advance='no') '> ', size(prompt_tokens), ' tokens: '
            call write_i0_arr_1d(error_unit, prompt_tokens)

            t = timer('Preprocessing tokens')
            logits = self%model%forward_batch(prompt_tokens, state)
            call t%done()

            if (self%speculative_sampling_enabled) then
                t = timer('Preprocessing tokens (draft)')
                draft_logits = self%draft_model%forward_batch(prompt_tokens, draft_state)
                call t%done()
            end if

            call set_up_signals_handling()

            tokens_count = 0

            t = timer('Generating text')

            if (self%speculative_sampling_enabled) then
                call self%generate_text_with_speculative_sampling(logits, state, draft_state, tokens_count)
            else
                call self%generate_text(logits, state, tokens_count)
            end if

            write(output_unit, '(a)') new_line('')

            elapsed_time = t%elapsed_time()
            write(output_unit, '(a, i0, a, f0.3, a, f0.3, a)') 'Generated ', tokens_count, ' tokens in ', elapsed_time, ' seconds (', real(tokens_count) / elapsed_time, ' tok/s).'

            if (signal_received) write(error_unit,'(a)') '> Generation stopped by user.'
            call tear_down_signals_handling()

            input_count = input_count + 1
        end do

    contains

        subroutine reset_state()
            state = self%model%init_state()
            if (self%speculative_sampling_enabled) then
                draft_state = self%draft_model%init_state()
            end if
        end subroutine

    end subroutine

    subroutine inference_generate_text(self, input_logits, state, tokens_count)
        class(inference), intent(inout) :: self
        real(sp), intent(in) :: input_logits(:)
        real(sp), intent(inout) :: state(:,:,:)  ! d_model, n_components, n_layers
        integer, intent(out) :: tokens_count

        real(sp) :: occurrence(size(input_logits))
        real(sp) :: logits(size(input_logits))
        integer :: i, token_id
        logical :: end_of_generation

        occurrence = 0
        logits = input_logits

        do i = 1, self%options%generation%max_token_limit
            tokens_count = tokens_count + 1
            token_id = generate_next_token(logits, occurrence, self%options%generation, end_of_generation)
            if (end_of_generation .or. signal_received) exit
            call self%print_token(token_id)
            logits = self%model%forward(token_id, state)
        end do
    end subroutine

    subroutine inference_generate_text_with_speculative_sampling(self, input_logits, state, draft_state, tokens_count)
        class(inference), intent(in) :: self
        real(sp), intent(in) :: input_logits(self%model%vocab_size)
        real(sp), dimension(:,:,:), intent(inout) :: state, draft_state  ! d_model, n_components, n_layers
        integer, intent(out) :: tokens_count

        real(sp), dimension(:,:,:,:), allocatable :: target_states, draft_states  ! d_model, n_components, n_states, n_layers

        real(sp) :: draft_logits(size(input_logits))
        integer :: draft_token_ids(self%options%speculative_sampling_lookahead)
        real(sp) :: draft_tokens_probs(size(input_logits), self%options%speculative_sampling_lookahead)

        real(sp) :: target_logits(size(input_logits), self%options%speculative_sampling_lookahead+1)
        real(sp) :: todo_occurrence(size(input_logits))

        integer :: k, last_token_id, lookahead, max_tokens
        logical :: end_of_generation, all_tokens_accepted
        integer, allocatable :: sampled_tokens_id(:)

        lookahead = self%options%speculative_sampling_lookahead
        max_tokens = self%options%generation%max_token_limit

        draft_states = self%draft_model%init_states(lookahead)
        target_states = self%model%init_states(lookahead+1)

        todo_occurrence = 0
        last_token_id = generate_next_token(input_logits, todo_occurrence, self%options%generation, end_of_generation)
        if (end_of_generation) return
        call self%print_token(last_token_id)

        tokens_count = 0
        main_loop: do while(tokens_count < max_tokens)
            if (signal_received) exit main_loop

            draft_states(:,:,1,:) = draft_state

            call draft_tokens_generation(self, last_token_id, lookahead, draft_states, draft_token_ids, draft_tokens_probs)

            target_logits = self%model%forward_batch_with_hidden_states([last_token_id, draft_token_ids], state, target_states)
            sampled_tokens_id = inference_sample_tokens(self, draft_token_ids, draft_tokens_probs, target_logits, all_tokens_accepted, k)
            last_token_id = sampled_tokens_id(size(sampled_tokens_id))
            tokens_count = tokens_count + size(sampled_tokens_id)

            draft_state = draft_states(:,:,k,:)

            call self%print_tokens(sampled_tokens_id, skip_end_token=.true.)

            if (.not. all_tokens_accepted) then
                state = target_states(:,:,k,:)
                if (last_token_id == 0) exit main_loop
                cycle
            end if

            if (last_token_id == 0) exit main_loop

            draft_logits = self%draft_model%forward(last_token_id, draft_state)

            todo_occurrence = 0 ! TODO
            state = target_states(:,:,lookahead+1,:)

            last_token_id = generate_next_token(target_logits(:, lookahead+1), todo_occurrence, self%options%generation, end_of_generation)
            tokens_count = tokens_count + 1

            if (end_of_generation) exit main_loop
            call self%print_token(last_token_id)
        end do main_loop

    end subroutine

    function inference_sample_tokens(self, draft_token_ids, draft_tokens_probs, target_logits, all_draft_tokens_accepted, k) result(sampled_tokens_id)
        class(inference), intent(in) :: self
        integer, intent(in) :: draft_token_ids(self%options%speculative_sampling_lookahead)
        real(sp), intent(in) :: draft_tokens_probs(self%model%vocab_size, self%options%speculative_sampling_lookahead)
        real(sp), intent(in) :: target_logits(self%model%vocab_size, self%options%speculative_sampling_lookahead+1)
        logical, intent(out) :: all_draft_tokens_accepted
        integer, intent(out) :: k
        integer, allocatable :: sampled_tokens_id(:)

        integer :: sampled_tokens_id_(size(draft_token_ids))
        logical :: accept_draft
        integer :: t

        all_draft_tokens_accepted = .true.

        k = 0
        do t = 1, self%options%speculative_sampling_lookahead
            k = k + 1

            sampled_tokens_id_(t) = self%sample_token(draft_token_ids(t), draft_tokens_probs(:, t), target_logits(:, t), accept_draft)

            if (.not. accept_draft) then
                all_draft_tokens_accepted = .false.
                exit
            end if

            if (sampled_tokens_id_(t) == 0) exit
        end do

        sampled_tokens_id = sampled_tokens_id_(1:k)
    end function

    function inference_sample_token(self, draft_token_id, draft_probs, target_logits, accept_draft) result(sampled_token_id)
        class(inference), intent(in) :: self
        integer, intent(in) :: draft_token_id
        real(sp), dimension(self%model%vocab_size), intent(in) :: draft_probs, target_logits
        logical, intent(out) :: accept_draft
        integer :: sampled_token_id

        real(sp) :: target_probs(size(target_logits))
        real(sp) :: todo_occurrence(size(target_logits))
        real(sp) :: r, draft_prob, target_prob
        integer :: target_token_id
        logical :: end_of_generation

        todo_occurrence = 0 ! TODO
        target_token_id = generate_next_token(target_logits, todo_occurrence, self%options%generation, end_of_generation, output_probs=target_probs)

        target_prob = target_probs(draft_token_id+1)
        draft_prob = draft_probs(draft_token_id+1)

        call random_number(r)

        accept_draft = r < min(1.0, target_prob / draft_prob)

        if (accept_draft) then
            sampled_token_id = draft_token_id
            return
        end if

        sampled_token_id = sample_once_from_multinomial(make_probs_for_resampling(target_probs, draft_probs)) - 1
    end function

    subroutine draft_tokens_generation(self, start_token_id, lookahead, draft_states, draft_token_ids, draft_tokens_probs)
        class(inference), intent(in) :: self
        integer, intent(in) :: start_token_id
        integer, intent(in) :: lookahead
        real(sp), intent(inout) :: draft_states(:,:,:,:)  ! d_model, n_components, n_states, n_layers
        integer, intent(out) :: draft_token_ids(lookahead)
        real(sp), intent(out) :: draft_tokens_probs(self%model%vocab_size, lookahead)

        real(sp) :: draft_logits(self%model%vocab_size)
        real(sp) :: todo_occurrence(self%model%vocab_size)
        integer :: last_token_id
        logical :: end_of_generation
        integer :: t

        last_token_id = start_token_id

        do t = 1, lookahead
            if (signal_received) exit
            if (t > 1) draft_states(:,:,t,:) = draft_states(:,:,t-1,:)

            draft_logits = self%draft_model%forward(last_token_id, draft_states(:,:,t,:))
            todo_occurrence = 0 ! TODO
            draft_token_ids(t) = generate_next_token(draft_logits, todo_occurrence, self%options%generation, end_of_generation, output_probs=draft_tokens_probs(:, t))
            last_token_id = draft_token_ids(t)

            if (end_of_generation) exit
        end do
    end subroutine

    subroutine inference_print_token(self, token_id)
        class(inference), intent(in) :: self
        integer, intent(in) :: token_id
        call print_token(self%tokenizer%decode([token_id]))
    end subroutine

    subroutine inference_print_tokens(self, token_ids, skip_end_token)
        class(inference), intent(in) :: self
        integer, intent(in) :: token_ids(:)
        logical, intent(in), optional :: skip_end_token

        integer :: i

        if (present(skip_end_token)) then
            if (skip_end_token) then
                do i = 1, size(token_ids)
                    if (token_ids(i) /= 0) then
                        call print_token(self%tokenizer%decode([token_ids(i)]))
                    end if
                end do
                return
            end if
        end if

        call print_token(self%tokenizer%decode(token_ids))
    end subroutine

    pure function make_probs_for_resampling(target_probs, draft_probs) result(res)
        real(sp), intent(in) :: target_probs(:)
        real(sp), intent(in) :: draft_probs(size(target_probs))
        real(sp) :: res(size(target_probs))
        res = max(0.0, target_probs - draft_probs)
        res = res / sum(res)
    end function

    subroutine set_up_signals_handling()
        signal_received = .false.
        call signal(unix_sigint, handle_signal, status=default_sigint_handler)
        call signal(unix_sigquit, handle_signal, status=default_sigquit_handler)
    end subroutine

    subroutine tear_down_signals_handling()
        call signal(unix_sigint, default_sigint_handler)
        call signal(unix_sigquit, default_sigquit_handler)
    end subroutine
    
    subroutine handle_signal(signal_number)
        integer, optional, intent(in) :: signal_number
        if (present(signal_number)) then
            signal_received = .true.
        end if
    end subroutine

    type(rwkv_lm_type) function load_model(name, filename, warmup_inputs) result(model)
        character(*), intent(in) :: name, filename
        integer, intent(in) :: warmup_inputs(:)
        type(timer) :: t
        
        t = timer('Loading ' // name // ' ' // filename)
        model = load_rwkv_lm_model(filename)
        call t%done()

        call display_model_info(name, model)
        call precompute_layer_norm_embeddings(name, model)
        call warm_up_model(name, model, warmup_inputs)
    end function

    subroutine display_model_info(name, model)
        character(*), intent(in) :: name
        type(rwkv_lm_type), intent(in) :: model
        write(error_unit,'(3a)') '> ', name ,' parameters:'
        write(error_unit,'(a, i0)') '    d_model: ', model%d_model
        write(error_unit,'(a, i0)') '    vocab_size: ', model%vocab_size
        write(error_unit,'(a, i0)') '    n_layers: ', model%n_layers
    end subroutine

    subroutine precompute_layer_norm_embeddings(name, model)
        character(*), intent(in) :: name
        type(rwkv_lm_type), intent(inout) :: model
        type(timer) :: t
        t = timer('Precomputing ' // name // ' layer normalization for embeddings')
        call model%precompute_layer_norm_embeddings()
        call t%done()
    end subroutine

    subroutine warm_up_model(name, model, inputs)
        character(*), intent(in) :: name
        type(rwkv_lm_type), intent(in) :: model
        integer, intent(in) :: inputs(:)
        type(timer) :: t
        real(sp), allocatable :: state(:,:,:)
        real(sp), allocatable :: logits(:)
        
        t = timer('Warming up ' // name)
        state = model%init_state()
        logits = model%forward_batch(inputs, state)
        call t%done()
    end subroutine

    subroutine read_user_input(text, is_eof)
        character(:), allocatable, intent(out) :: text
        logical, intent(out) :: is_eof
        integer :: iostat

        call readline('» ', text, iostat)

        is_eof = is_iostat_end(iostat)
        if (is_eof) then
            write (output_unit, '(/, a)') '> End of input.'
            return
        end if

        if (iostat /= 0) error stop 'Input error.'
    end subroutine

    subroutine print_token(token)
        character(*), intent(in) :: token
        write(output_unit, '(a)', advance='no') token
    end subroutine

end module
