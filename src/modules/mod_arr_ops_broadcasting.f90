! Copyright 2023 FortAI-Hub contributors.
! Released under the MIT License. See LICENSE file for full license information.

! This module, mod_arr_ops_broadcasting, extends the basic algebraic operations (*/+-) with broadcasting.
! Broadcasting is a technique that allows arithmetic operations between arrays of different shapes.
module mod_arr_ops_broadcasting
    use mod_real_precision
    implicit none
    private

    public :: operator(*), operator(/), operator(+), operator(-)

    interface operator(*)
        module procedure prod_1d_2d
        module procedure prod_2d_1d
    end interface

    interface operator(/)
        module procedure div_1d_2d
        module procedure div_2d_1d
    end interface

    interface operator(+)
        module procedure add_1d_2d
        module procedure add_2d_1d
    end interface

    interface operator(-)
        module procedure sub_1d_2d
        module procedure sub_2d_1d
    end interface

contains

    pure function prod_1d_2d(a, b) result(c)
        real(sp), intent(in) :: a(:)
        real(sp), intent(in) :: b(:,:)
        real(sp) :: c(size(b, 1), size(b, 2))

        integer :: i, n
        n = size(b, 2)

        do concurrent (i=1:n)
            c(:,i) = a * b(:,i)
        end do
    end function

    pure function prod_2d_1d(a, b) result(c)
        real(sp), intent(in) :: a(:,:)
        real(sp), intent(in) :: b(:)
        real(sp) :: c(size(a, 1), size(a, 2))

        integer :: i, n
        n = size(a, 2)

        do concurrent (i=1:n)
            c(:,i) = a(:,i) * b
        end do
    end function

    pure function div_1d_2d(a, b) result(c)
        real(sp), intent(in) :: a(:)
        real(sp), intent(in) :: b(:,:)
        real(sp) :: c(size(b, 1), size(b, 2))

        integer :: i, n
        n = size(b, 2)

        do concurrent (i=1:n)
            c(:,i) = a / b(:,i)
        end do
    end function

    pure function div_2d_1d(a, b) result(c)
        real(sp), intent(in) :: a(:,:)
        real(sp), intent(in) :: b(:)
        real(sp) :: c(size(a, 1), size(a, 2))

        integer :: i, n
        n = size(a, 2)

        do concurrent (i=1:n)
            c(:,i) = a(:,i) / b
        end do
    end function

    pure function add_1d_2d(a, b) result(c)
        real(sp), intent(in) :: a(:)
        real(sp), intent(in) :: b(:,:)
        real(sp) :: c(size(b, 1), size(b, 2))

        integer :: i, n
        n = size(b, 2)

        do concurrent (i=1:n)
            c(:,i) = a + b(:,i)
        end do
    end function

    pure function add_2d_1d(a, b) result(c)
        real(sp), intent(in) :: a(:,:)
        real(sp), intent(in) :: b(:)
        real(sp) :: c(size(a, 1), size(a, 2))

        integer :: i, n
        n = size(a, 2)

        do concurrent (i=1:n)
            c(:,i) = a(:,i) + b
        end do
    end function

    pure function sub_1d_2d(a, b) result(c)
        real(sp), intent(in) :: a(:)
        real(sp), intent(in) :: b(:,:)
        real(sp) :: c(size(b, 1), size(b, 2))

        integer :: i, n
        n = size(b, 2)

        do concurrent (i=1:n)
            c(:,i) = a - b(:,i)
        end do
    end function

    pure function sub_2d_1d(a, b) result(c)
        real(sp), intent(in) :: a(:,:)
        real(sp), intent(in) :: b(:)
        real(sp) :: c(size(a, 1), size(a, 2))

        integer :: i, n
        n = size(a, 2)

        do concurrent (i=1:n)
            c(:,i) = a(:,i) - b
        end do
    end function

end module
