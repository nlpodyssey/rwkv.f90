! Copyright 2023 FortAI-Hub contributors.
! Released under the MIT License. See LICENSE file for full license information.

module mod_hidden_states
    use mod_real_precision
    use mod_state, only : state_type, layer_state_type
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
    contains
        procedure :: copy_to_state => copy_hidden_state_to_state
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

    subroutine copy_hidden_state_to_state(self, index, dest)
        class(hidden_states_type), intent(in) :: self
        integer, intent(in) :: index
        type(state_type), intent(inout) :: dest
        integer :: i

        do concurrent (i = 1:size(self%layers))
            call copy_layer_hidden_state_to_layer_state(self%layers(i), index, dest%layers(i))
        end do
    end subroutine

    pure subroutine copy_layer_hidden_state_to_layer_state(source, index, dest)
        type(layer_hidden_states_type), intent(in) :: source
        integer, intent(in) :: index
        type(layer_state_type), intent(inout) :: dest

        dest%ffn_xx = source%ffn_xx(:, index)
        dest%att_xx = source%att_xx(:, index)
        dest%att_aa = source%att_aa(:, index)
        dest%att_bb = source%att_bb(:, index)
        dest%att_pp = source%att_pp(:, index)
    end subroutine

end module
