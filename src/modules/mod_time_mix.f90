! Copyright 2023 FortAI-Hub contributors.
! Released under the MIT License. See LICENSE file for full license information.

module mod_time_mix
    use mod_real_precision
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
        generic :: forward => forward_single, forward_batch
    end type time_mix_type

    interface time_mix_type
        module procedure :: time_mix_constructor
    end interface time_mix_type

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
    end function time_mix_constructor

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
    end subroutine read_params

    function forward_single(self, x, state) result(out)
        use mod_functions, only: sigmoid

        class(time_mix_type), intent(in) :: self
        real(sp), intent(in) :: x(:)
        type(layer_state_type), intent(inout) :: state

        real(sp), allocatable :: out(:)
        real(sp), allocatable :: k(:), v(:), r(:), ww(:), p(:), e1(:), e2(:), a(:), b(:), rwkv(:)

        !$omp parallel sections
        !$omp section
        k = matmul(self%wk, self%mk * x + (1.0 - self%mk) * state%att_xx)
        !$omp section
        v = matmul(self%wv, self%mv * x + (1.0 - self%mv) * state%att_xx)
        !$omp section
        r = matmul(self%wr, self%mr * x + (1.0 - self%mr) * state%att_xx)
        !$omp end parallel sections

        ww = k + self%tf
        p = max(state%att_pp, ww)
        e1 = exp(state%att_pp - p)
        e2 = exp(ww - p)
        a = e1 * state%att_aa + e2 * v
        b = e1 * state%att_bb + e2
        rwkv = sigmoid(r) * (a / b)
        out = matmul(self%wo, rwkv)

        ww = state%att_pp + self%td
        p = max(ww, k)
        e1 = exp(ww - p)
        e2 = exp(k - p)

        state%att_xx = x
        state%att_aa = e1 * state%att_aa + e2 * v
        state%att_bb = e1 * state%att_bb + e2
        state%att_pp = p
    end function forward_single

    function forward_batch(self, x, state) result(out)
        use mod_functions, only: sigmoid

        class(time_mix_type), intent(in) :: self
        real(sp), intent(in) :: x(:,:)
        type(layer_state_type), intent(inout) :: state

        integer :: i, n
        real(sp), dimension(self%dm, size(x, 2)) :: xx, k, v, r, rwkv, out
        real(sp), dimension(self%dm) :: ww, p, e1, e2, a, b

        n = size(x, 2)

        xx = token_shift(state%att_xx, x)

        !$omp parallel do default(shared) private(i)
        do i = 1, n
            k(:,i) = matmul(self%wk, self%mk * x(:,i) + (1.0 - self%mk) * xx(:,i))
            v(:,i) = matmul(self%wv, self%mv * x(:,i) + (1.0 - self%mv) * xx(:,i))
            r(:,i) = matmul(self%wr, self%mr * x(:,i) + (1.0 - self%mr) * xx(:,i))
        end do
        !$omp end parallel do

        do i = 1, n
            ww = k(:,i) + self%tf
            p = max(state%att_pp, ww)
            e1 = exp(state%att_pp - p)
            e2 = exp(ww - p)
            a = e1 * state%att_aa + e2 * v(:,i)
            b = e1 * state%att_bb + e2
            rwkv(:,i) = sigmoid(r(:,i)) * (a / b)

            ww = state%att_pp + self%td
            p = max(ww, k(:,i))
            e1 = exp(ww - p)
            e2 = exp(k(:,i) - p)

            ! Update state
            state%att_xx = x(:,i)
            state%att_aa = e1 * state%att_aa + e2 * v(:,i)
            state%att_bb = e1 * state%att_bb + e2
            state%att_pp = p
        end do

        out = matmul(self%wo, rwkv)

    end function forward_batch

end module mod_time_mix
