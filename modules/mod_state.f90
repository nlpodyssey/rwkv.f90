! Copyright 2023 FortAI-Hub contributors.
! Released under the MIT License. See LICENSE file for full license information.

module mod_state
    use mod_real_precision
    implicit none
    private
    public :: state_type, layer_state_type, copy_state, swap_states, finalize_states

    type :: layer_state_type
        real(sp), pointer :: ffn_xx(:) => null()
        real(sp), pointer :: att_xx(:) => null()
        real(sp), pointer :: att_aa(:) => null()
        real(sp), pointer :: att_bb(:) => null()
        real(sp), pointer :: att_pp(:) => null()
    contains
        procedure, private :: finalize => layer_state_finalize
    end type

    type :: state_type
        type(layer_state_type), pointer :: layers(:) => null()
    contains
        procedure :: finalize => state_finalize
    end type

    interface state_type
        module procedure :: state_constructor
    end interface
    
contains

    pure type(state_type) function state_constructor(d_model, n_layers) result(self)
        integer, intent(in) :: d_model, n_layers
        integer :: i

        if (d_model <= 0 .or. n_layers <= 0) then
            error stop "Error: n_layers and d_model must be positive integers."
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

    pure subroutine state_finalize(self)
        class(state_type), intent(inout) :: self
        integer :: i
        if (.not. associated(self%layers)) return
        do concurrent (i = 1:size(self%layers))
            call self%layers(i)%finalize()
        end do
        deallocate(self%layers)
        nullify(self%layers)
    end subroutine

    pure subroutine layer_state_finalize(self)
        class(layer_state_type), intent(inout) :: self
        call finalize_layer_state_member(self%ffn_xx)
        call finalize_layer_state_member(self%att_xx)
        call finalize_layer_state_member(self%att_aa)
        call finalize_layer_state_member(self%att_bb)
        call finalize_layer_state_member(self%att_pp)
    end subroutine
    
    pure subroutine finalize_layer_state_member(p)
        real(sp), pointer, intent(inout) :: p(:)
        deallocate(p)
        nullify(p)
    end subroutine

    pure subroutine copy_state(source, dest)
        type(state_type), intent(in) :: source
        type(state_type), intent(inout) :: dest
        integer :: i

        if (.not. associated(source%layers)) then
            call dest%finalize()
            return
        end if

        if (.not. associated(dest%layers)) then
            dest = state_type(size(source%layers(1)%ffn_xx), size(source%layers))
        end if

        do concurrent (i = 1:size(source%layers))
            dest%layers(i)%ffn_xx = source%layers(i)%ffn_xx
            dest%layers(i)%att_xx = source%layers(i)%att_xx
            dest%layers(i)%att_aa = source%layers(i)%att_aa
            dest%layers(i)%att_bb = source%layers(i)%att_bb
            dest%layers(i)%att_pp = source%layers(i)%att_pp
        end do
    end subroutine

    pure subroutine swap_states(a, b)
        type(state_type), intent(inout) :: a, b
        type(state_type) :: t
        t = a
        a = b
        b = t
    end subroutine

    pure subroutine finalize_states(states)
        type(state_type), intent(inout) :: states(:)
        integer :: i
        do concurrent (i = 1:size(states))
            call states(i)%finalize()
        end do
    end subroutine
end module
