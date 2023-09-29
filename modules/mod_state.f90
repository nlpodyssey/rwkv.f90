! Copyright 2023 FortAI-Hub contributors.
! Released under the MIT License. See LICENSE file for full license information.

module mod_state
    use mod_real_precision
    implicit none
    private
    public :: make_state, make_states, ffn_xx, att_xx, att_aa, att_bb, att_pp

    enum, bind(C)
        ! state components
        enumerator :: ffn_xx = 1
        enumerator :: att_xx
        enumerator :: att_aa
        enumerator :: att_bb
        enumerator :: att_pp
    end enum

    integer, parameter :: n_components = 5

contains

    pure function make_state(d_model, n_layers) result(state)
        integer, intent(in) :: d_model, n_layers
        real(sp), allocatable :: state(:, :, :)  ! d_model, n_components, n_layers

        if (d_model <= 0 .or. n_layers <= 0) then
            error stop "Error: n_layers, d_model, n_states must be positive integers."
        end if

        allocate(state(d_model, n_components, n_layers))

        state = 0.0
        state(:,att_pp,:) = -1e30
    end function

    pure function make_states(d_model, n_layers, n_states) result(states)
        integer, intent(in) :: d_model, n_layers
        integer, intent(in), optional :: n_states
        real(sp), allocatable :: states(:, :, :, :) ! d_model, n_components, n_states, n_layers

        if (d_model <= 0 .or. n_layers <= 0 .or. n_states <= 0) then
            error stop "Error: n_layers, d_model, n_states must be positive integers."
        end if

        allocate(states(d_model, n_components, n_states, n_layers))

        states = 0.0
        states(:,att_pp,:,:) = -1e30
    end function
end module
