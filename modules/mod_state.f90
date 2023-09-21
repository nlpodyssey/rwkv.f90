! Copyright 2023 FortAI-Hub contributors.
! Released under the MIT License. See LICENSE file for full license information.

module mod_state
    use mod_real_precision
    implicit none
    private
    public :: state_type, copy_state

    type :: state_type
        ! d_model, n_layers
        real(sp), allocatable :: ffn_xx(:, :)
        real(sp), allocatable :: att_xx(:, :)
        real(sp), allocatable :: att_aa(:, :)
        real(sp), allocatable :: att_bb(:, :)
        real(sp), allocatable :: att_pp(:, :)
    end type

    interface state_type
        module procedure :: state_constructor
    end interface
    
contains

    pure type(state_type) function state_constructor(d_model, n_layers) result(self)
        integer, intent(in) :: d_model, n_layers

        if (d_model <= 0 .or. n_layers <= 0) then
            error stop "Error: n_layers and d_model must be positive integers."
        end if

        allocate(self%ffn_xx(d_model, n_layers), self%att_xx(d_model, n_layers), &
                 self%att_aa(d_model, n_layers), self%att_bb(d_model, n_layers), &
                 self%att_pp(d_model, n_layers))

        self%ffn_xx = 0.0
        self%att_xx = 0.0
        self%att_aa = 0.0
        self%att_bb = 0.0
        self%att_pp = -1e30
    end function

    pure subroutine copy_state(source, dest)
        type(state_type), intent(in) :: source
        type(state_type), intent(inout) :: dest

        dest%ffn_xx = source%ffn_xx
        dest%att_xx = source%att_xx
        dest%att_aa = source%att_aa
        dest%att_bb = source%att_bb
        dest%att_pp = source%att_pp
    end subroutine
end module
