! Copyright 2023 FortAI-Hub contributors.
! Released under the MIT License. See LICENSE file for full license information.

module mod_layer_norm
    use mod_real_precision
    implicit none
    private
    public :: layer_norm_type

    type :: layer_norm_type
        real(sp), allocatable :: g(:) ! Scaling factor
        real(sp), allocatable :: b(:) ! Bias term
        real(sp) :: eps
    contains
        procedure :: read_params
        procedure, pass :: forward_single
        procedure, pass :: forward_batch
        generic :: forward => forward_single, forward_batch
    end type

    interface layer_norm_type
        module procedure :: layer_norm_constructor
    end interface

contains

    pure type(layer_norm_type) function layer_norm_constructor(d_model, eps) result(self)
        integer, intent(in) :: d_model
        real(sp), intent(in) :: eps

        self%eps = eps

        allocate(self%g(d_model))
        allocate(self%b(d_model))
        self%g = 1.0
        self%b = 0.0
    end function

    subroutine read_params(self, file_u, iostat)
        class(layer_norm_type), intent(inout) :: self
        integer, intent(in) :: file_u
        integer, intent(out) :: iostat
        read(file_u, iostat=iostat) self%g
        if (iostat /= 0) return
        read(file_u, iostat=iostat) self%b
        if (iostat /= 0) return
    end subroutine

    function forward_single(self, x) result(y)
        use mod_functions, only: layer_norm_1d

        class(layer_norm_type), intent(in) :: self
        real(sp), intent(in) :: x(:)
        real(sp), allocatable :: y(:)

        y = layer_norm_1d(x, self%g, self%b, self%eps)
    end function

    function forward_batch(self, x) result(y)
        use mod_functions, only: layer_norm_2d

        class(layer_norm_type), intent(in) :: self
        real(sp), intent(in) :: x(:,:)
        real(sp) :: y(size(x, 1), size(x, 2))

        y = layer_norm_2d(x, self%g, self%b, self%eps)
    end function

end module
