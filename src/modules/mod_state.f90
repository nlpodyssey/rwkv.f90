! Copyright 2023 FortAI-Hub contributors.
! Released under the MIT License. See LICENSE file for full license information.

module mod_state
    use mod_real_precision
    implicit none
    private
    public :: state_type, layer_state_type

    type :: layer_state_type
        real(sp), allocatable :: ffn_xx(:)
        real(sp), allocatable :: att_xx(:)
        real(sp), allocatable :: att_aa(:)
        real(sp), allocatable :: att_bb(:)
        real(sp), allocatable :: att_pp(:)
    end type

    type :: state_type
        type(layer_state_type), allocatable :: layers(:)
    end type

    interface state_type
        module procedure :: state_constructor
    end interface
    
contains

    type(state_type) function state_constructor(d_model, n_layers) result(self)
        integer, intent(in) :: d_model, n_layers
        integer :: i

        if (d_model <= 0 .or. n_layers <= 0) then
            stop "Error: n_layers and d_model must be positive integers."
        end if

        allocate(self%layers(n_layers))

        do i = 1, n_layers
            allocate(self%layers(i)%ffn_xx(d_model), self%layers(i)%att_xx(d_model), &
                    self%layers(i)%att_aa(d_model), self%layers(i)%att_bb(d_model), &
                    self%layers(i)%att_pp(d_model))

            self%layers(i)%ffn_xx = 0.0
            self%layers(i)%att_xx = 0.0
            self%layers(i)%att_aa = 0.0
            self%layers(i)%att_bb = 0.0
            self%layers(i)%att_pp = -1e30
        end do
    end function

end module
