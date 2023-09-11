! Copyright 2023 FortAI-Hub contributors.
! Released under the MIT License. See LICENSE file for full license information.

module mod_pipeline
    use iso_fortran_env, only: error_unit, output_unit
    use mod_essentials, only : write_i0_arr_1d
    use mod_functions, only : layer_norm_2d
    use mod_generation, only: generation_options, generate_next_token
    use mod_prompt_utils, only: generate_prompt
    use mod_readline, only: readline
    use mod_real_precision, only: sp
    use mod_rwkv_lm, only: rwkv_lm_type, load_rwkv_lm_model
    use mod_state, only: state_type, copy_state, swap_states, finalize_states
    use mod_stats, only: sample_from_multinomial
    use mod_timer, only: timer
    use mod_trie_tokenizer, only: trie_tokenizer, load_trie_tokenizer

    implicit none

    private
    public :: pipeline_options, run_pipeline

    type :: pipeline_options
        character(:), allocatable :: tokenizer_filename
        character(:), allocatable :: model_filename
        character(:), allocatable :: draft_model_filename
        type(generation_options) :: generation
        integer :: speculative_sampling_lookahead = 2
    end type

    type :: pipeline
        type(pipeline_options) :: options
        type(trie_tokenizer) :: tokenizer
        type(rwkv_lm_type) :: model
        type(rwkv_lm_type), allocatable :: draft_model
        logical :: speculative_sampling_enabled
        type(state_type) :: state
        type(state_type) :: draft_state
    contains
        procedure :: load_tokenizer => pipeline_load_tokenizer
        procedure :: run => pipeline_run
        procedure :: init_states => pipeline_init_states
        procedure :: process_and_generate_text => pipeline_process_and_generate_text
        procedure :: generate_text => pipeline_generate_text
        procedure :: generate_text_without_speculative_sampling => pipeline_generate_text_without_speculative_sampling
        procedure :: generate_text_with_speculative_sampling => pipeline_generate_text_with_speculative_sampling
        procedure, private :: print_token => pipeline_print_token
    end type

    interface pipeline
        module procedure :: pipeline_constructor
    end interface

    ! Variables for signals handling
    integer, parameter :: unix_sigint = 2, unix_sigquit = 3
    integer :: default_sigint_handler, default_sigquit_handler
    logical :: signal_received

contains

    subroutine run_pipeline(options)
        type(pipeline_options), intent(in) :: options
        type(pipeline) :: p
        p = pipeline(options)
        call p%run()
    end subroutine

    type(pipeline) function pipeline_constructor(options) result(self)
        type(pipeline_options), intent(in) :: options
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

    subroutine pipeline_load_tokenizer(self)
        class(pipeline), intent(inout) :: self
        type(timer) :: t
        t = timer('Loading tokenizer ' // self%options%tokenizer_filename)
        self%tokenizer = load_trie_tokenizer(self%options%tokenizer_filename)
        call t%done()
    end subroutine

    subroutine pipeline_run(self)
        class(pipeline), intent(inout) :: self
        character(:), allocatable :: instruction
        character(:), allocatable :: prompt
        logical :: is_eof
        
        call self%init_states()

        do
            call read_user_input(instruction, is_eof)
            if (is_eof) exit
            if (len(instruction) == 0) cycle

            prompt = generate_prompt(instruction=instruction)
            call self%init_states() ! reset the state at every input
            call self%process_and_generate_text(prompt)
        end do
    end subroutine

    subroutine pipeline_init_states(self)
        class(pipeline), intent(inout) :: self
        self%state = self%model%init_state()
        if (self%speculative_sampling_enabled) self%draft_state = self%draft_model%init_state()
    end subroutine

    subroutine pipeline_process_and_generate_text(self, prompt)
        class(pipeline), intent(inout) :: self
        character(:), allocatable, intent(in) :: prompt        
        integer, allocatable :: prompt_tokens(:)
        real(sp), allocatable :: logits(:), draft_logits(:)
        type(timer) :: t
        
        t = timer('Tokenizing prompt')
        prompt_tokens = self%tokenizer%encode(prompt)
        call t%done()

        write(error_unit, '(a, i0, a)', advance='no') '> ', size(prompt_tokens), ' tokens: '
        call write_i0_arr_1d(error_unit, prompt_tokens)

        t = timer('Preprocessing tokens')
        if (self%speculative_sampling_enabled) then
            !$omp parallel sections
            !$omp section
            logits = self%model%forward_batch(prompt_tokens, self%state)
            !$omp section
            draft_logits = self%draft_model%forward_batch(prompt_tokens, self%draft_state)
            !$omp end parallel sections
        else
            logits = self%model%forward_batch(prompt_tokens, self%state)
        end if
        call t%done()

        t = timer('Generating')
        call self%generate_text(logits)
        call t%done()
        
    end subroutine

    subroutine pipeline_generate_text(self, logits)
        class(pipeline), intent(inout) :: self
        real(sp), intent(in) :: logits(:)

        call set_up_signals_handling()

        if (self%speculative_sampling_enabled) then
            call self%generate_text_with_speculative_sampling(logits)
        else
            call self%generate_text_without_speculative_sampling(logits)
        end if

        write(output_unit, '(a)') new_line('')
        if (signal_received) write(error_unit,'(a)') '> Generation stopped by user.'

        call tear_down_signals_handling()
    end subroutine

    subroutine pipeline_generate_text_without_speculative_sampling(self, input_logits)
        class(pipeline), intent(inout) :: self
        real(sp), intent(in) :: input_logits(:)
        real(sp) :: occurrence(size(self%model%proj, 1)) ! corresponding to the number of logits
        real(sp), allocatable :: logits(:)
        integer :: i, token_id
        character(:), allocatable :: token
        logical :: end_of_generation

        occurrence = 0
        logits = input_logits

        do i = 1, self%options%generation%max_token_limit
            token_id = generate_next_token(logits, occurrence, self%options%generation, end_of_generation)
            if (end_of_generation .or. signal_received) exit
            call self%print_token(token_id)
            logits = self%model%forward(token_id, self%state)
        end do
    end subroutine

    subroutine pipeline_generate_text_with_speculative_sampling(self, input_logits)
        class(pipeline), intent(inout) :: self
        real(sp), intent(in) :: input_logits(:)

        real(sp), dimension(size(input_logits)) :: todo_occurrence ! TODO: occurrence handling
        real(sp), dimension(size(input_logits)) :: logits
        
        type(state_type) :: draft_states(self%options%speculative_sampling_lookahead)
        integer :: draft_token_ids(self%options%speculative_sampling_lookahead)
        real(sp) :: draft_tokens_probs(self%options%speculative_sampling_lookahead, size(input_logits))
        
        type(state_type) :: target_states(self%options%speculative_sampling_lookahead + 1)
        integer :: target_input_token_ids(self%options%speculative_sampling_lookahead + 1)
        integer :: target_output_token_ids(self%options%speculative_sampling_lookahead + 1)
        real(sp) :: target_tokens_probs(self%options%speculative_sampling_lookahead + 1, size(input_logits))

        integer :: n, t, last_token_id
        logical :: end_of_generation, all_tokens_accepted
        integer :: lookahead, draft_token_id
        real(sp) :: r, draft_prob, target_prob

        integer, parameter :: num_samples = 1
        integer :: sampled_indices(num_samples)

        lookahead = self%options%speculative_sampling_lookahead

        todo_occurrence = 0
        last_token_id = generate_next_token(input_logits, todo_occurrence, self%options%generation, end_of_generation)
        if (end_of_generation) return
        call self%print_token(last_token_id)

        n = 0
        main_loop: do while(n < self%options%generation%max_token_limit)
            target_input_token_ids(1) = last_token_id
            call copy_state(self%draft_state, draft_states(1))
            do t = 1, lookahead
                if (signal_received) exit main_loop
                if (t > 1) call copy_state(draft_states(t-1), draft_states(t))
                
                logits = self%draft_model%forward(target_input_token_ids(t), draft_states(t))
                todo_occurrence = 0
                draft_token_ids(t) = generate_next_token(logits, todo_occurrence, self%options%generation, end_of_generation, output_probs=draft_tokens_probs(t, :))
                target_input_token_ids(t+1) = draft_token_ids(t)

                if (end_of_generation) exit
            end do

            !$omp parallel do private(t, logits, todo_occurrence, end_of_generation) shared(self, target_states, target_input_token_ids, target_output_token_ids, target_tokens_probs)
            do t = 1, lookahead+1
                call copy_state(self%state, target_states(t))
                logits = self%model%forward_batch(target_input_token_ids(1:t), target_states(t))
                todo_occurrence = 0
                target_output_token_ids(t) = generate_next_token(logits, todo_occurrence, self%options%generation, end_of_generation, output_probs=target_tokens_probs(t, :))
            end do
            !$omp end parallel do

            all_tokens_accepted = .true.
            do t = 1, lookahead
                draft_token_id = draft_token_ids(t)
                
                call random_number(r)
                draft_prob = draft_tokens_probs(t, draft_token_id + 1)
                target_prob = target_tokens_probs(t, draft_token_id + 1)

                call swap_states(draft_states(t), self%draft_state)
                call swap_states(target_states(t), self%state)
                if (r < min(1.0, target_prob / draft_prob)) then
                    last_token_id = draft_token_id
                    if (last_token_id == 0) exit main_loop
                    call self%print_token(last_token_id)
                    n = n + 1
                else
                    sampled_indices = sample_from_multinomial(make_probs_for_resampling(target_tokens_probs(t, :), draft_tokens_probs(t, :)), num_samples)
                    last_token_id = sampled_indices(1) - 1
                    if (last_token_id == 0) exit main_loop
                    call self%print_token(last_token_id)
                    n = n + 1
                    all_tokens_accepted = .false.
                    exit
                end if
            end do

            if (all_tokens_accepted) then
                ! make sure draft state is aligned
                logits = self%draft_model%forward(last_token_id, self%draft_state)

                call swap_states(target_states(lookahead+1), self%state)
                last_token_id = target_output_token_ids(lookahead+1)
                if (last_token_id == 0) exit main_loop
                call self%print_token(last_token_id)
                n = n + 1
                ! TODO: handle draft model here?
            end if
        end do main_loop

        call finalize_states(draft_states)
        call finalize_states(target_states)
    end subroutine
    
    subroutine pipeline_print_token(self, token_id)
        class(pipeline), intent(in) :: self
        integer, intent(in) :: token_id
        call print_token(self%tokenizer%decode([token_id]))
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
        integer, intent(in) :: signal_number
        signal_received = .true.
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
        type(state_type) :: state
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

        call readline('>>> ', text, iostat)

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
