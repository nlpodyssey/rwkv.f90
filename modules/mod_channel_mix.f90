! Copyright 2023 FortAI-Hub contributors.
! Released under the MIT License. See LICENSE file for full license information.

module mod_channel_mix
    use mod_real_precision
    use mod_arr_ops_broadcasting
    use mod_state
    use mod_token_shift
    implicit none
    private
    public :: channel_mix_type

    type :: channel_mix_type
        integer :: dm, hidden
        real(sp), allocatable :: wk(:,:) ! Keys weights
        real(sp), allocatable :: wv(:,:) ! Values weights
        real(sp), allocatable :: wr(:,:) ! Receptance weights
        real(sp), allocatable :: mk(:) ! Key mixing parameter
        real(sp), allocatable :: mr(:) ! Receptance mixing parameter
    contains
        procedure :: read_params
        procedure, pass :: forward_single
        procedure, pass :: forward_batch
        procedure, pass :: forward_batch_with_hidden_states
        generic :: forward => forward_single, forward_batch, forward_batch_with_hidden_states
    end type

    interface channel_mix_type
        module procedure :: channel_mix_constructor
    end interface

contains

    pure type(channel_mix_type) function channel_mix_constructor(d_model) result(self)
        integer, intent(in) :: d_model
        integer :: hidden

        hidden = 4 * d_model

        self%dm = d_model
        self%hidden = hidden

        allocate(self%wk(hidden, d_model))
        allocate(self%wv(d_model, hidden))
        allocate(self%wr(d_model, d_model))
        allocate(self%mk(d_model))
        allocate(self%mr(d_model))
    end function

    subroutine read_params(self, file_u, iostat)
        class(channel_mix_type), intent(inout) :: self
        integer, intent(in) :: file_u
        integer, intent(out) :: iostat
        read(file_u, iostat=iostat) self%wk
        if (iostat /= 0) return
        read(file_u, iostat=iostat) self%wv
        if (iostat /= 0) return
        read(file_u, iostat=iostat) self%wr
        if (iostat /= 0) return
        read(file_u, iostat=iostat) self%mk
        if (iostat /= 0) return
        read(file_u, iostat=iostat) self%mr
        if (iostat /= 0) return
    end subroutine

    function forward_single(self, x, state) result(rkv)
        use mod_functions, only: relu, sigmoid

        class(channel_mix_type), intent(in) :: self
        real(sp), intent(in) :: x(:)
        real(sp), intent(inout) :: state(:, :) ! d_model, n_components

        real(sp), dimension(size(x)) :: r, kv, rkv
        real(sp) :: k(size(self%wk, 1))

        !$omp parallel sections
        !$omp section
        k = matmul(self%wk, self%mk * x + (1.0 - self%mk) * state(:, ffn_xx))
        !$omp section
        r = matmul(self%wr, self%mr * x + (1.0 - self%mr) * state(:, ffn_xx))
        !$omp end parallel sections

        kv = matmul(self%wv, relu(k) ** 2)
        rkv = sigmoid(r) * kv

        state(:, ffn_xx) = x ! update state
    end function

    function forward_batch(self, x, state) result(rkv)
        use mod_functions, only: relu, sigmoid

        class(channel_mix_type), intent(in) :: self
        real(sp), intent(in) :: x(:,:) ! d_model, batch_size
        real(sp), intent(inout) :: state(:, :) ! d_model, n_components

        real(sp), dimension(self%dm, size(x, 2)) :: xx, kv, rkv
        real(sp) :: r(self%dm, size(x, 2)), k(self%hidden, size(x, 2))
        integer :: n

        n = size(x, 2)

        xx = token_shift(state(:, ffn_xx), x)

        k = matmul(self%wk, self%mk * x + (1.0 - self%mk) * xx)
        r = matmul(self%wr, self%mr * x + (1.0 - self%mr) * xx)
        kv = matmul(self%wv, relu(k) ** 2)
        rkv = sigmoid(r) * kv

        state(:, ffn_xx) = x(:, n) ! Update state
    end function

    function forward_batch_with_hidden_states(self, x, init_state, hidden_states) result(rkv)
        use mod_functions, only: relu, sigmoid

        class(channel_mix_type), intent(in) :: self
        real(sp), intent(in) :: x(:,:) ! d_model, batch_size
        real(sp), intent(in) :: init_state(:, :) ! d_model, n_components
        real(sp), intent(inout) :: hidden_states(:, :, :) ! d_model, n_components, n_states

        real(sp), dimension(self%dm, size(x, 2)) :: xx, kv, rkv
        real(sp) :: r(self%dm, size(x, 2)), k(self%hidden, size(x, 2))
        integer :: i, n

        n = size(x, 2)

        xx = token_shift(init_state(:, ffn_xx), x)

        k = matmul(self%wk, self%mk * x + (1.0 - self%mk) * xx)
        r = matmul(self%wr, self%mr * x + (1.0 - self%mr) * xx)
        kv = matmul(self%wv, relu(k) ** 2)
        rkv = sigmoid(r) * kv

        hidden_states(:, ffn_xx, :) = x
    end function

end module