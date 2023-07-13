! Copyright 2023 FortAI-Hub contributors.
! Released under the MIT License. See LICENSE file for full license information.

module mod_rwkv_lm
    use mod_real_precision
    use mod_state
    use mod_layer_norm
    use mod_rwkv_layer
    implicit none
    private
    public rwkv_lm_type, make_rwkv_lm, load_rwkv_lm_model

    real, parameter :: layer_norm_eps = 1.0e-5
    integer, parameter :: rescale_layer = 6

    type :: rwkv_lm_type
        integer :: d_model, vocab_size, n_layers
        logical :: precomputed_ln_emb
        real(sp), allocatable :: emb(:,:)
        type(layer_norm_type)  :: ln_emb
        type(rwkv_layer_type), allocatable :: layers(:)
        type(layer_norm_type) :: ln_out
        real(sp), allocatable :: proj(:,:)
    contains
        procedure :: read_params
        procedure :: init_state
        procedure, pass :: forward_single
        procedure, pass :: forward_batch
        generic :: forward => forward_single, forward_batch
    end type rwkv_lm_type

contains

    function make_rwkv_lm(d_model, vocab_size, n_layers) result(rwkv_lm)
        integer, intent(in) :: n_layers
        integer, intent(in) :: vocab_size
        integer, intent(in) :: d_model

        type(rwkv_lm_type) :: rwkv_lm
        integer :: i

        rwkv_lm%precomputed_ln_emb = .false.
        rwkv_lm%d_model = d_model
        rwkv_lm%n_layers = n_layers
        rwkv_lm%vocab_size = vocab_size

        rwkv_lm%ln_emb = layer_norm_type(d_model, 1e-5)

        allocate(rwkv_lm%layers(n_layers))
        do i = 1, n_layers
            rwkv_lm%layers(i) = make_rwkv_layer(d_model)
        end do

        rwkv_lm%ln_out = layer_norm_type(d_model, layer_norm_eps)

        allocate(rwkv_lm%emb(d_model, vocab_size))
        allocate(rwkv_lm%proj(vocab_size, d_model))
    end function make_rwkv_lm

    subroutine read_params(self, file_u, iostat)
        class(rwkv_lm_type), intent(inout) :: self
        integer, intent(in) :: file_u
        integer, intent(out) :: iostat

        integer :: i

        read(file_u, iostat=iostat) self%emb
        if (iostat /= 0) return

        call self%ln_emb%read_params(file_u, iostat)
        if (iostat /= 0) return

        do i = 1, size(self%layers)
            call self%layers(i)%read_params(file_u, iostat)
            if (iostat /= 0) return
        end do

        call self%ln_out%read_params(file_u, iostat)
        if (iostat /= 0) return

        read(file_u, iostat=iostat) self%proj
        if (iostat /= 0) return
    end subroutine read_params

    function load_rwkv_lm_model(filename) result(model)
        character(*), intent(in) :: filename

        type(rwkv_lm_type) :: model
        type(integer) :: d_model, vocab_size, n_layers
        integer :: u, status

        open(newunit=u, file=filename, form="unformatted", access="stream", status="old", iostat=status)
        if (status /= 0) then
            write(*, *) "Error opening file:", filename, status
            return
        end if

        read(u) d_model
        read(u) vocab_size
        read(u) n_layers

        model = make_rwkv_lm(d_model, vocab_size, n_layers)

        call model%read_params(u, status)

        if (status /= 0) then
            write(*, *) "Error reading file:", filename, status
            return
        end if
        close(u)
    end function load_rwkv_lm_model

    function init_state(self) result(state)
        class(rwkv_lm_type), intent(in) :: self
        type(state_type) :: state
        state = state_type(self%d_model, self%n_layers)
    end function init_state

    function forward_single(self, x, state) result(output)
        class(rwkv_lm_type), intent(in) :: self
        integer, intent(in) :: x
        type(state_type), intent(inout) :: state
        real(sp), allocatable :: encoded(:), output(:)
        integer :: i

        encoded = self%emb(:, x+1)

        if (.not. self%precomputed_ln_emb) then
            encoded = self%ln_emb%forward(encoded)
        end if

        do i = 1, size(self%layers)
            encoded = self%layers(i)%forward(encoded, state%layers(i))

            if (mod(i, rescale_layer) == 0) then
                encoded = encoded * 0.5
            end if
        end do

        output = matmul(self%proj, self%ln_out%forward(encoded))
    end function forward_single

    function forward_batch(self, x, state) result(output)
        class(rwkv_lm_type), intent(in) :: self
        integer, intent(in) :: x(:)
        type(state_type), intent(inout) :: state

        real(sp) :: encoded(self%d_model,size(x))
        real(sp) :: last_encoded(self%d_model)
        real(sp), allocatable :: output(:)

        integer i,j

        do concurrent (i=1:size(x))
            encoded(:,i) = self%emb(:, x(i)+1)
        end do

        if (.not. self%precomputed_ln_emb) then
            encoded = self%ln_emb%forward(encoded)
        end if

        do i = 1, size(self%layers)
            encoded = self%layers(i)%forward(encoded, state%layers(i))

            if (mod(i, rescale_layer) == 0) then
                encoded = encoded * 0.5
            end if
        end do

        last_encoded = encoded(:, size(encoded, 2))

        output = matmul(self%proj, self%ln_out%forward(last_encoded))
    end function forward_batch

end module mod_rwkv_lm
