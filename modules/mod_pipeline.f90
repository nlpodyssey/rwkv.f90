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
    use mod_state, only: state_type
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
        real(sp), allocatable :: logits(:)
        type(timer) :: t
        
        t = timer('Tokenizing prompt')
        prompt_tokens = self%tokenizer%encode(prompt)
        call t%done()

        write(error_unit, '(a, i0, a)', advance='no') '> ', size(prompt_tokens), ' tokens: '
        call write_i0_arr_1d(error_unit, prompt_tokens)

        t = timer('Preprocessing tokens')
        logits = self%model%forward_batch(prompt_tokens, self%state)
        call t%done()

        t = timer('Generating')
        call self%generate_text(logits)
        call t%done()
        
    end subroutine

    subroutine pipeline_generate_text(self, input_logits)
        class(pipeline), intent(inout) :: self
        real(sp), intent(in) :: input_logits(:)

        call set_up_signals_handling()

        call self%generate_text_without_speculative_sampling(input_logits)

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

            token = self%tokenizer%decode([token_id])
            call print_token(token)

            logits = self%model%forward(token_id, self%state)
        end do
    end subroutine

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

        if (iostat /= 0) stop 'Input error.'
    end subroutine

    subroutine print_token(token)
        character(*), intent(in) :: token
        write(output_unit, '(a)', advance='no') token
    end subroutine

end module
