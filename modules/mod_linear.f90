! Copyright 2023 FortAI-Hub contributors.
! Released under the MIT License. See LICENSE file for full license information.

module mod_linear
    use mod_real_precision
    implicit none
    private
    public :: linear_type

    type :: linear_type
        real(sp), allocatable :: w(:,:)
        real(sp), allocatable :: b(:)
    contains
        procedure :: read_params
        procedure, pass :: forward_single
        procedure, pass :: forward_batch
        generic :: forward => forward_single, forward_batch
    end type

    interface linear_type
        module procedure :: linear_constructor
    end interface

contains

    pure type(linear_type) function linear_constructor(in, out) result(self)
        integer, intent(in) :: in, out
        allocate(self%w(out, in))
        allocate(self%b(out))
        self%w = 0.0
        self%b = 0.0
    end function linear_constructor

    subroutine read_params(self, file_u, iostat)
        class(linear_type), intent(inout) :: self
        integer, intent(in) :: file_u
        integer, intent(out) :: iostat
        read(file_u, iostat=iostat) self%w
        if (iostat /= 0) return
        read(file_u, iostat=iostat) self%b
        if (iostat /= 0) return
    end subroutine read_params

    pure function forward_single(self, x) result(y)
        class(linear_type), intent(in) :: self
        real(sp), intent(in) :: x(:)
        real(sp) :: y(size(self%b))

        y = matmul(self%w, x) + self%b
    end function forward_single

    pure function forward_batch(self, x) result(y)
        class(linear_type), intent(in) :: self
        real(sp), intent(in) :: x(:,:)
        real(sp) :: y(size(self%b,1),size(x,2))
        integer :: i

        y = matmul(self%w, x)
        do concurrent (i = 1:size(y, 2))
            y(:, i) = y(:, i) + self%b
        end do
    end function forward_batch

end module mod_linear
