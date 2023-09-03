! Copyright 2023 FortAI-Hub contributors.
! Released under the MIT License. See LICENSE file for full license information.

module lib_rwkv
    use, intrinsic :: iso_c_binding
    use mod_essentials
    use mod_real_precision
    use mod_state
    use mod_trie_tokenizer, only : load_tokenizer => load_trie_tokenizer, rwkv_tokenizer => trie_tokenizer
    use mod_rwkv_lm, load_model => load_rwkv_lm_model
    use mod_prompt_utils
    use mod_generation
    implicit none
    private
    public :: c_new_rwkv_tokenizer, deallocate_rwkv_tokenizer, c_tokenize, c_detokenize
    public :: c_layer_state_type, c_state_type, c_init_state, c_deallocate_c_state
    public :: c_new_rwkv_model, deallocate_rwkv_model, c_get_d_model, c_get_num_layers, c_get_logits_size, c_forward_batch, c_forward_single
    public :: c_generation_options, generation_context_type, c_new_generation_context, c_generate_next_token

    ! -------------------------------
    ! Generation Bindings
    !---------------------------------

    type, bind(C) :: c_generation_options
        real(c_float) :: temp
        real(c_float) :: alpha_frequency
        real(c_float) :: alpha_presence
        real(c_float) :: alpha_decay
        integer(c_int) :: max_token_limit
        integer(c_int) :: use_multinomial
    end type

    type :: generation_context_type
        real(sp), allocatable :: occurrence(:)
    end type

    ! -------------------------------
    ! State C Bindings
    ! -------------------------------

    type, bind(C) :: c_layer_state_type
        type(c_ptr) :: ffn_xx
        type(c_ptr) :: att_xx
        type(c_ptr) :: att_aa
        type(c_ptr) :: att_bb
        type(c_ptr) :: att_pp
    end type c_layer_state_type

    type, bind(C) :: c_state_type
        type(c_ptr) :: layers
        integer(c_int) :: num_layers
        integer(c_int) :: d_model
    end type c_state_type

    interface c_state_type
        module procedure :: c_state_constructor
    end interface

contains

    ! -------------------------------
    ! RWKV Tokenizer Functions
    ! -------------------------------

    type(c_ptr) function c_new_rwkv_tokenizer(filename, filename_len) bind(C, name="c_new_rwkv_tokenizer")
        character(kind=c_char, len=1), dimension(*), intent(in) :: filename
        integer(c_int), value, intent(in) :: filename_len

        type(rwkv_tokenizer), pointer :: tokenizer_ptr
        allocate(tokenizer_ptr)
        tokenizer_ptr = load_tokenizer(c_to_f_string(filename, filename_len))
        c_new_rwkv_tokenizer = c_loc(tokenizer_ptr)
    end function

    subroutine deallocate_rwkv_tokenizer(c_tokenizer_ptr) bind(C, name="c_deallocate_rwkv_tokenizer")
        type(c_ptr), value, intent(in) :: c_tokenizer_ptr

        type(rwkv_tokenizer), pointer :: tokenizer

        call c_f_pointer(c_tokenizer_ptr, tokenizer)
        if (associated(tokenizer)) then
            deallocate(tokenizer)
        end if
    end subroutine

    subroutine c_tokenize(c_tokenizer_ptr, input, input_len, max_tokens_len, tokens, n_tokens) bind(C, name="c_tokenize")
        type(c_ptr), value, intent(in) :: c_tokenizer_ptr
        character(kind=c_char, len=1), dimension(*), intent(in) :: input
        integer(c_int), value, intent(in) :: input_len
        integer(c_int), value, intent(in) :: max_tokens_len
        integer(c_int), dimension(max_tokens_len), intent(inout) :: tokens
        integer(c_int), intent(out) :: n_tokens

        type(rwkv_tokenizer), pointer :: tokenizer
        integer, dimension(:), allocatable :: temp_tokens

        call c_f_pointer(c_tokenizer_ptr, tokenizer)
        if (.not. associated(tokenizer)) then
            print *, 'Error: Pointer not associated.'
        end if

        temp_tokens = tokenizer%encode(c_to_f_string(input, input_len))

        n_tokens = size(temp_tokens)

        tokens(1:n_tokens) = temp_tokens
    end subroutine c_tokenize

    subroutine c_detokenize(c_tokenizer_ptr, tokens, tokens_len, output, output_len) bind(C, name="c_detokenize")
        type(c_ptr), value, intent(in) :: c_tokenizer_ptr
        integer(c_int), intent(in) :: tokens(*)
        integer(c_int), value, intent(in) :: tokens_len
        character(c_char), intent(out) :: output(*)
        integer(c_int), intent(out) :: output_len

        integer :: i
        character(:), allocatable :: content
        integer :: tokens_f(tokens_len)
        type(rwkv_tokenizer), pointer :: tokenizer

        call c_f_pointer(c_tokenizer_ptr, tokenizer)
        if (.not. associated(tokenizer)) then
            print *, 'Error: tokenizer pointer not associated.'
        end if

        do i = 1, tokens_len
            tokens_f(i) = tokens(i)
        end do

        content = tokenizer%decode(tokens_f)

        output_len = len(content)
        call f_to_c_string(content, output)
    end subroutine

    ! -------------------------------
    ! RWKV Model Functions
    ! -------------------------------

    type(c_ptr) function c_new_rwkv_model(filename, filename_len) bind(C, name="c_new_rwkv_model")
        character(kind=c_char, len=1), dimension(*), intent(in) :: filename
        integer(c_int), value, intent(in) :: filename_len

        type(rwkv_lm_type), pointer :: model_ptr

        allocate(model_ptr)
        model_ptr = load_model(c_to_f_string(filename, filename_len))

        c_new_rwkv_model = c_loc(model_ptr)
    end function

    subroutine deallocate_rwkv_model(c_model_ptr) bind(C, name="c_deallocate_rwkv_model")
        type(c_ptr), value, intent(in) :: c_model_ptr

        type(rwkv_lm_type), pointer :: model

        call c_f_pointer(c_model_ptr, model)
        if (associated(model)) then
            deallocate(model)
        end if
    end subroutine

    type(integer(c_int)) function c_get_num_layers(c_model_ptr) bind(C, name="c_get_num_layers")
        type(c_ptr), value, intent(in) :: c_model_ptr

        type(rwkv_lm_type), pointer :: model

        call c_f_pointer(c_model_ptr, model)
        if (.not. associated(model)) then
            print *, 'Error: Pointer not associated.'
        end if

        c_get_num_layers = size(model%layers)
    end function

    type(integer(c_int)) function c_get_logits_size(c_model_ptr) bind(C, name="c_get_logits_size")
        type(c_ptr), value, intent(in) :: c_model_ptr

        type(rwkv_lm_type), pointer :: model

        call c_f_pointer(c_model_ptr, model)
        if (.not. associated(model)) then
            print *, 'Error: Pointer not associated.'
        end if

        c_get_logits_size = size(model%proj, 1)
    end function

    type(integer(c_int)) function c_get_d_model(c_model_ptr) bind(C, name="c_get_d_model")
        type(c_ptr), value, intent(in) :: c_model_ptr

        type(rwkv_lm_type), pointer :: model

        call c_f_pointer(c_model_ptr, model)
        if (.not. associated(model)) then
            print *, 'Error: Pointer not associated.'
        end if

        c_get_d_model = model%d_model
    end function

    subroutine c_forward_batch(c_model_ptr, c_state_ptr, tokens_len, tokens, logits_len, logits) bind(C, name="c_forward_batch")
        type(c_ptr), value, intent(in) :: c_model_ptr
        type(c_ptr), value, intent(in) :: c_state_ptr
        integer(c_int), value, intent(in) :: tokens_len
        integer(c_int), dimension(tokens_len), intent(in) :: tokens
        integer(c_int), value, intent(in) :: logits_len
        real(c_float), dimension(logits_len), intent(inout) :: logits

        type(rwkv_lm_type), pointer :: model
        type(c_state_type), pointer :: c_state
        type(state_type) :: state
        real(c_float), dimension(:), allocatable :: logits_f
        integer :: x(tokens_len)
        integer :: i

        call c_f_pointer(c_model_ptr, model)
        if (.not. associated(model)) then
            print *, 'Error: model_ptr pointer not associated.'
        end if

        call c_f_pointer(c_state_ptr, c_state)
        if (.not. associated(c_state)) then
            print *, 'Error: c_state pointer not associated.'
        end if

        call c_to_f_state(c_state, state)

        do i = 1, tokens_len
            x(i) = tokens(i)
        end do

        logits_f = model%forward_batch(x, state)

        if (logits_len > size(logits)) stop 'Error: logits array in caller is too small.'

        logits(1:logits_len) = logits_f
    end subroutine

    subroutine c_forward_single(c_model_ptr, c_state_ptr, token, logits_len, logits) bind(C, name="c_forward_single")
        type(c_ptr), value, intent(in) :: c_model_ptr
        type(c_ptr), value, intent(in) :: c_state_ptr
        integer(c_int), value, intent(in) :: token
        integer(c_int), value, intent(in) :: logits_len
        real(c_float), dimension(logits_len), intent(inout) :: logits

        real(c_float), dimension(:), allocatable :: logits_f
        type(rwkv_lm_type), pointer :: model
        type(c_state_type), pointer :: c_state
        type(state_type) :: state

        call c_f_pointer(c_model_ptr, model)
        if (.not. associated(model)) then
            print *, 'Error: model_ptr pointer not associated.'
        end if

        call c_f_pointer(c_state_ptr, c_state)
        if (.not. associated(c_state)) then
            print *, 'Error: c_state pointer not associated.'
        end if

        call c_to_f_state(c_state, state)

        logits_f = model%forward_single(token, state)

        if (logits_len > size(logits)) stop 'Error: logits array in caller is too small.'

        logits(1:logits_len) = logits_f
    end subroutine

    ! -------------------------------
    ! State Functions
    ! -------------------------------

    type(c_ptr) function c_init_state(c_model_ptr) bind(C, name="c_init_state")
        type(c_ptr), value, intent(in) :: c_model_ptr

        type(rwkv_lm_type), pointer :: model
        type(c_state_type), pointer :: c_state_type_ptr

        call c_f_pointer(c_model_ptr, model)
        if (.not. associated(model)) then
            print *, 'Error: Pointer not associated.'
        end if

        allocate(c_state_type_ptr)

        c_state_type_ptr = c_state_type(model%init_state(), model%d_model)
        c_init_state = c_loc(c_state_type_ptr)
    end function

    type(c_state_type) function c_state_constructor(state, d_model) result(self)
        use, intrinsic :: iso_c_binding
        implicit none
        type(state_type), target, intent(in) :: state
        integer, intent(in) :: d_model

        type(c_layer_state_type), pointer :: temp_layers(:)
        integer :: i

        self%d_model = d_model
        self%num_layers = size(state%layers)
        self%layers = c_loc(state%layers)

        call c_f_pointer(self%layers, temp_layers, [self%num_layers])

        do i = 1, self%num_layers
            temp_layers(i)%ffn_xx = c_loc(state%layers(i)%ffn_xx)
            temp_layers(i)%att_xx = c_loc(state%layers(i)%att_xx)
            temp_layers(i)%att_aa = c_loc(state%layers(i)%att_aa)
            temp_layers(i)%att_bb = c_loc(state%layers(i)%att_bb)
            temp_layers(i)%att_pp = c_loc(state%layers(i)%att_pp)
        end do
    end function c_state_constructor

    subroutine c_to_f_state(c_state, state)
        type(c_state_type), intent(in) :: c_state
        type(state_type), intent(inout) :: state

        type(c_layer_state_type), pointer :: c_layers(:)
        integer :: i

        call c_f_pointer(c_state%layers, c_layers, [c_state%num_layers])

        allocate(state%layers(c_state%num_layers))

        do i = 1, c_state%num_layers
            call c_f_pointer(c_layers(i)%ffn_xx, state%layers(i)%ffn_xx, [c_state%d_model])
            call c_f_pointer(c_layers(i)%att_xx, state%layers(i)%att_xx, [c_state%d_model])
            call c_f_pointer(c_layers(i)%att_aa, state%layers(i)%att_aa, [c_state%d_model])
            call c_f_pointer(c_layers(i)%att_bb, state%layers(i)%att_bb, [c_state%d_model])
            call c_f_pointer(c_layers(i)%att_pp, state%layers(i)%att_pp, [c_state%d_model])
        end do
    end subroutine

    subroutine c_deallocate_c_state(c_state_ptr) bind(C, name="c_deallocate_c_state")
        type(c_ptr), value, intent(in) :: c_state_ptr

        type(c_state_type), pointer :: c_state
        type(state_type) :: state

        call c_f_pointer(c_state_ptr, c_state)
        if (.not. associated(c_state)) return

        call c_to_f_state(c_state, state)
        call finalize_state(state)

        deallocate(c_state)
        nullify(c_state)
    end subroutine

    ! -------------------------------
    ! Generation Functions
    ! -------------------------------

    type(c_ptr) function c_new_generation_context(c_model_ptr) bind(C, name="c_new_generation_context")
        type(c_ptr), value, intent(in) :: c_model_ptr

        type(rwkv_lm_type), pointer :: model
        type(generation_context_type), pointer :: generation_context_ptr

        call c_f_pointer(c_model_ptr, model)
        if (.not. associated(model)) then
            print *, 'Error: Pointer not associated.'
        end if

        allocate(generation_context_ptr)
        allocate(generation_context_ptr%occurrence(size(model%proj, 1)))

        generation_context_ptr%occurrence = 0

        c_new_generation_context = c_loc(generation_context_ptr)
    end function

    subroutine c_to_f_generation_options(c_gen_options, gen_options)
        use, intrinsic :: iso_c_binding, only: c_float, c_int
        implicit none
        type(c_generation_options), intent(in) :: c_gen_options
        type(generation_options), intent(out) :: gen_options

        gen_options%temp = c_gen_options%temp
        gen_options%alpha_frequency = c_gen_options%alpha_frequency
        gen_options%alpha_presence = c_gen_options%alpha_presence
        gen_options%alpha_decay = c_gen_options%alpha_decay
        gen_options%max_token_limit = c_gen_options%max_token_limit
        gen_options%use_multinomial = c_gen_options%use_multinomial /= 0
    end subroutine

    function c_generate_next_token(c_generation_context_ptr, logits, logits_len, end_of_generation, c_gen_opts) result(token_id) bind(C, name="c_generate_next_token")
        type(c_ptr), value, intent(in) :: c_generation_context_ptr
        integer(c_int), value, intent(in) :: logits_len
        real(c_float), intent(in) :: logits(logits_len)
        logical(c_bool), intent(out) :: end_of_generation
        type(c_generation_options), value, intent(in) :: c_gen_opts

        logical :: f_end_of_generation
        integer(c_int) :: token_id

        type(generation_context_type), pointer :: generation_context
        type(generation_options) :: gen_opts

        call c_f_pointer(c_generation_context_ptr, generation_context)
        if (.not. associated(generation_context)) then
            print *, 'Error: generation_context pointer not associated.'
        end if

        call c_to_f_generation_options(c_gen_opts, gen_opts)

        token_id = generate_next_token(logits, generation_context%occurrence, gen_opts, f_end_of_generation)
        end_of_generation = f_to_c_logical(f_end_of_generation)
    end function

    ! -------------------------------
    ! Fortran<>C Conversion Functions
    ! -------------------------------

    subroutine f_to_c_string(f_string, c_string)
        character(len=*), intent(in) :: f_string
        character(c_char), intent(out) :: c_string(len(f_string))
        integer :: i, n

        n = len(f_string)

        do i = 1, n
            c_string(i) = f_string(i:i)
        end do

        c_string(n+1) = c_null_char
    end subroutine

    function c_to_f_string(c_string, len) result(f_string)
        use, intrinsic :: iso_c_binding
        character(kind=c_char, len=1), dimension(*), intent(in) :: c_string
        integer(c_int), value, intent(in) :: len
        character(len=len) :: f_string
        integer :: i

        do i = 1, len
            if (c_string(i) == c_null_char) exit
            f_string(i:i) = c_string(i)
        end do
    end function

    function f_to_c_logical(f_val) result(c_val)
        use iso_c_binding, only: c_bool
        logical, intent(in) :: f_val
        logical(c_bool) :: c_val

        if (f_val) then
            c_val = .true.
        else
            c_val = .false.
        end if
    end function

end module
