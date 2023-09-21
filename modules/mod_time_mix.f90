! Copyright 2023 FortAI-Hub contributors.
! Released under the MIT License. See LICENSE file for full license information.

module mod_time_mix
    use mod_real_precision
    use mod_arr_ops_broadcasting
    use mod_state
    use mod_token_shift
    implicit none
    private
    public :: time_mix_type

    type :: time_mix_type
        integer :: dm
        real(sp), allocatable :: wk(:,:) ! Keys weights
        real(sp), allocatable :: wv(:,:) ! Values weights
        real(sp), allocatable :: wr(:,:) ! Receptance weights
        real(sp), allocatable :: wo(:,:) ! Output weights
        real(sp), allocatable :: mk(:) ! Key mix parameter
        real(sp), allocatable :: mv(:) ! Value mix parameter
        real(sp), allocatable :: mr(:) ! Receptance mix parameter
        real(sp), allocatable :: td(:) ! Time decay parameter
        real(sp), allocatable :: tf(:) ! First-time parameter
    contains
        procedure read_params
        procedure, pass :: forward_single
        procedure, pass :: forward_batch
        procedure, pass :: forward_batch_with_hidden_states
        generic :: forward => forward_single, forward_batch, forward_batch_with_hidden_states
    end type

    interface time_mix_type
        module procedure :: time_mix_constructor
    end interface

contains

    pure type(time_mix_type) function time_mix_constructor(d_model) result(self)
        integer, intent(in) :: d_model

        self%dm = d_model

        allocate(self%wk(d_model, d_model))
        allocate(self%wv(d_model, d_model))
        allocate(self%wr(d_model, d_model))
        allocate(self%wo(d_model, d_model))
        allocate(self%mk(d_model))
        allocate(self%mv(d_model))
        allocate(self%mr(d_model))
        allocate(self%td(d_model))
        allocate(self%tf(d_model))
    end function

    subroutine read_params(self, file_u, iostat)
        class(time_mix_type), intent(inout) :: self
        integer, intent(in) :: file_u
        integer, intent(out) :: iostat
        read(file_u, iostat=iostat) self%wk
        if (iostat /= 0) return
        read(file_u, iostat=iostat) self%wv
        if (iostat /= 0) return
        read(file_u, iostat=iostat) self%wr
        if (iostat /= 0) return
        read(file_u, iostat=iostat) self%wo
        if (iostat /= 0) return
        read(file_u, iostat=iostat) self%mk
        if (iostat /= 0) return
        read(file_u, iostat=iostat) self%mv
        if (iostat /= 0) return
        read(file_u, iostat=iostat) self%mr
        if (iostat /= 0) return
        read(file_u, iostat=iostat) self%td
        if (iostat /= 0) return
        read(file_u, iostat=iostat) self%tf
        if (iostat /= 0) return
    end subroutine

    function forward_single(self, x, state, layer_index) result(out)
        use mod_functions, only: sigmoid

        class(time_mix_type), intent(in) :: self
        real(sp), intent(in) :: x(:)
        type(state_type), intent(inout) :: state
        integer, intent(in) :: layer_index

        real(sp), dimension(size(x)) :: k, v, r, ww, p, e1, e2, a, b, rwkv, out

        !$omp parallel sections
        !$omp section
        k = matmul(self%wk, self%mk * x + (1.0 - self%mk) * state%att_xx(:, layer_index))
        !$omp section
        v = matmul(self%wv, self%mv * x + (1.0 - self%mv) * state%att_xx(:, layer_index))
        !$omp section
        r = matmul(self%wr, self%mr * x + (1.0 - self%mr) * state%att_xx(:, layer_index))
        !$omp end parallel sections

        ww = k + self%tf
        p = max(state%att_pp(:, layer_index), ww)
        e1 = exp(state%att_pp(:, layer_index) - p)
        e2 = exp(ww - p)
        a = e1 * state%att_aa(:, layer_index) + e2 * v
        b = e1 * state%att_bb(:, layer_index) + e2
        rwkv = sigmoid(r) * (a / b)
        out = matmul(self%wo, rwkv)

        ww = state%att_pp(:, layer_index) + self%td
        p = max(ww, k)
        e1 = exp(ww - p)
        e2 = exp(k - p)

        state%att_xx(:, layer_index) = x
        state%att_aa(:, layer_index) = e1 * state%att_aa(:, layer_index) + e2 * v
        state%att_bb(:, layer_index) = e1 * state%att_bb(:, layer_index) + e2
        state%att_pp(:, layer_index) = p
    end function

    function forward_batch(self, x, state, layer_index) result(out)
        use mod_functions, only: sigmoid

        class(time_mix_type), intent(in) :: self
        real(sp), intent(in) :: x(:,:)
        type(state_type), intent(inout) :: state
        integer, intent(in) :: layer_index

        integer :: i, n
        real(sp), dimension(self%dm, size(x, 2)) :: xx, k, v, r, sx, out
        real(sp), dimension(self%dm) :: ww, p, e1, e2, a, b, aa, bb, pp

        xx = token_shift(state%att_xx(:, layer_index), x)

        k = matmul(self%wk, self%mk * x + (1.0 - self%mk) * xx)
        v = matmul(self%wv, self%mv * x + (1.0 - self%mv) * xx)
        r = matmul(self%wr, self%mr * x + (1.0 - self%mr) * xx)

        aa = state%att_aa(:, layer_index)
        bb = state%att_bb(:, layer_index)
        pp = state%att_pp(:, layer_index)

        n = size(x, 2)

        do i = 1, n
            ww = k(:,i) + self%tf
            p = max(pp, ww)
            e1 = exp(pp - p)
            e2 = exp(ww - p)
            a = e1 * aa + e2 * v(:,i)
            b = e1 * bb + e2
            sx(:,i) =  a / b

            ww = pp + self%td
            p = max(ww, k(:,i))
            e1 = exp(ww - p)
            e2 = exp(k(:,i) - p)

            aa = e1 * aa + e2 * v(:,i)
            bb = e1 * bb + e2
            pp = p
        end do

        ! Update state
        state%att_xx(:, layer_index) = x(:, n)
        state%att_aa(:, layer_index) = aa
        state%att_bb(:, layer_index) = bb
        state%att_pp(:, layer_index) = pp

        out = matmul(self%wo, sigmoid(r) * sx)
    end function

    function forward_batch_with_hidden_states(self, x, init_state, hidden_states, layer_index) result(out)
        use mod_functions, only: sigmoid

        class(time_mix_type), intent(in) :: self
        real(sp), intent(in) :: x(:,:)
        type(state_type), intent(in) :: init_state
        type(state_type), intent(inout) :: hidden_states(size(x, 2))
        integer, intent(in) :: layer_index

        integer :: i, n
        real(sp), dimension(self%dm, size(x, 2)) :: xx, k, v, r, sx, out
        real(sp), dimension(self%dm) :: ww, p, e1, e2, a, b, aa, bb, pp

        xx = token_shift(init_state%att_xx(:, layer_index), x)

        k = matmul(self%wk, self%mk * x + (1.0 - self%mk) * xx)
        v = matmul(self%wv, self%mv * x + (1.0 - self%mv) * xx)
        r = matmul(self%wr, self%mr * x + (1.0 - self%mr) * xx)

        aa = init_state%att_aa(:, layer_index)
        bb = init_state%att_bb(:, layer_index)
        pp = init_state%att_pp(:, layer_index)

        n = size(x, 2)

        do i = 1, n
            ww = k(:,i) + self%tf
            p = max(pp, ww)
            e1 = exp(pp - p)
            e2 = exp(ww - p)
            a = e1 * aa + e2 * v(:,i)
            b = e1 * bb + e2
            sx(:,i) =  a / b

            ww = pp + self%td
            p = max(ww, k(:,i))
            e1 = exp(ww - p)
            e2 = exp(k(:,i) - p)

            aa = e1 * aa + e2 * v(:,i)
            bb = e1 * bb + e2
            pp = p

            hidden_states(i)%att_xx(:, layer_index) = x(:, i)
            hidden_states(i)%att_aa(:, layer_index) = aa
            hidden_states(i)%att_bb(:, layer_index) = bb
            hidden_states(i)%att_pp(:, layer_index) = pp
        end do

        out = matmul(self%wo, sigmoid(r) * sx)
    end function

end module
