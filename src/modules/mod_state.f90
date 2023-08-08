! Copyright 2023 FortAI-Hub contributors.
! Released under the MIT License. See LICENSE file for full license information.

module mod_state
    use mod_real_precision
    implicit none
    private
    public :: state_type, layer_state_type

    type :: layer_state_type
        real(sp), pointer :: ffn_xx(:) => null()
        real(sp), pointer :: att_xx(:) => null()
        real(sp), pointer :: att_aa(:) => null()
        real(sp), pointer :: att_bb(:) => null()
        real(sp), pointer :: att_pp(:) => null()
    end type

    type :: state_type
        type(layer_state_type), pointer :: layers(:) => null()
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

    subroutine finalize_state(self)
        type(state_type), intent(inout) :: self
        integer :: i

        if (associated(self%layers)) then
            do i = 1, size(self%layers)
                if (associated(self%layers(i)%ffn_xx)) then
                    deallocate(self%layers(i)%ffn_xx)
                    nullify(self%layers(i)%ffn_xx)
                end if
                if (associated(self%layers(i)%att_xx)) then
                    deallocate(self%layers(i)%att_xx)
                    nullify(self%layers(i)%att_xx)
                end if
                if (associated(self%layers(i)%att_aa)) then
                    deallocate(self%layers(i)%att_aa)
                    nullify(self%layers(i)%att_aa)
                end if
                if (associated(self%layers(i)%att_bb)) then
                    deallocate(self%layers(i)%att_bb)
                    nullify(self%layers(i)%att_bb)
                end if
                if (associated(self%layers(i)%att_pp)) then
                    deallocate(self%layers(i)%att_pp)
                    nullify(self%layers(i)%att_pp)
                end if
            end do
            deallocate(self%layers)
            nullify(self%layers)
        end if
    end subroutine

end module
