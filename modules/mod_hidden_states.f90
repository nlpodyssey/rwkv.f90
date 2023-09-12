! Copyright 2023 FortAI-Hub contributors.
! Released under the MIT License. See LICENSE file for full license information.

module mod_hidden_states
    use mod_real_precision
    implicit none
    private
    public :: hidden_states_type, layer_hidden_states_type

    type :: layer_hidden_states_type
        real(sp), allocatable :: ffn_xx(:, :)
        real(sp), allocatable :: att_xx(:, :)
        real(sp), allocatable :: att_aa(:, :)
        real(sp), allocatable :: att_bb(:, :)
        real(sp), allocatable :: att_pp(:, :)
    end type

    type :: hidden_states_type
        type(layer_hidden_states_type), allocatable :: layers(:)
    end type

    interface hidden_states_type
        module procedure :: hidden_states_constructor
    end interface

contains

    pure type(hidden_states_type) function hidden_states_constructor(d_model, n_layers, n_states) result(self)
        integer, intent(in) :: d_model, n_layers, n_states
        integer :: i

        if (d_model <= 0 .or. n_layers <= 0 .or. n_states <= 0) then
            error stop "Error: n_layers and d_model and n_states must be positive integers."
        end if

        allocate(self%layers(n_layers))

        do i = 1, n_layers
            allocate(self%layers(i)%ffn_xx(d_model, n_states), self%layers(i)%att_xx(d_model, n_states), &
                    self%layers(i)%att_aa(d_model, n_states), self%layers(i)%att_bb(d_model, n_states), &
                    self%layers(i)%att_pp(d_model, n_states))

            self%layers(i)%ffn_xx = 0.0
            self%layers(i)%att_xx = 0.0
            self%layers(i)%att_aa = 0.0
            self%layers(i)%att_bb = 0.0
            self%layers(i)%att_pp = -1e30
        end do
    end function
end module
