! Copyright 2023 FortAI-Hub contributors.
! Released under the MIT License. See LICENSE file for full license information.

module mod_rwkv_layer
    use mod_real_precision
    use mod_layer_norm
    use mod_channel_mix
    use mod_time_mix
    implicit none
    private
    public :: rwkv_layer_type

    type :: rwkv_layer_type
        type(layer_norm_type)  :: ln1
        type(layer_norm_type)  :: ln2
        type(channel_mix_type) :: channel_mix
        type(time_mix_type)    :: time_mix
    contains
        procedure read_params
        procedure, pass :: forward_single
        procedure, pass :: forward_batch
        procedure, pass :: forward_batch_with_hidden_states
        generic :: forward => forward_single, forward_batch, forward_batch_with_hidden_states
    end type

    interface rwkv_layer_type
        module procedure :: rwkv_layer_constructor
    end interface

contains

    pure type(rwkv_layer_type) function rwkv_layer_constructor(d_model) result(self)
        integer, intent(in) :: d_model

        self%ln1 = layer_norm_type(d_model, 1e-5)
        self%ln2 = layer_norm_type(d_model, 1e-5)
        self%channel_mix = channel_mix_type(d_model)
        self%time_mix = time_mix_type(d_model)
    end function

    subroutine read_params(self, file_u, iostat)
        class(rwkv_layer_type), intent(inout) :: self
        integer, intent(in) :: file_u
        integer, intent(out) :: iostat
        call self%ln1%read_params(file_u, iostat)
        if (iostat /= 0) return
        call self%ln2%read_params(file_u, iostat)
        if (iostat /= 0) return
        call self%channel_mix%read_params(file_u, iostat)
        if (iostat /= 0) return
        call self%time_mix%read_params(file_u, iostat)
        if (iostat /= 0) return
    end subroutine

    function forward_single(self, x, state) result(y)
        class(rwkv_layer_type), intent(in) :: self
        real(sp), intent(in) :: x(:)
        real(sp), intent(inout) :: state(:, :) ! d_model, n_components
        real(sp), allocatable :: y(:)

        y = x + self%time_mix%forward(self%ln1%forward(x), state)
        y = y + self%channel_mix%forward(self%ln2%forward(y), state)
    end function

    function forward_batch(self, x, state) result(y)
        class(rwkv_layer_type), intent(in) :: self
        real(sp), intent(in) :: x(:,:)
        real(sp), intent(inout) :: state(:, :) ! d_model, n_components
        real(sp), allocatable :: y(:,:)

        y = x + self%time_mix%forward(self%ln1%forward(x), state)
        y = y + self%channel_mix%forward(self%ln2%forward(y), state)
    end function

    function forward_batch_with_hidden_states(self, x, init_state, hidden_states) result(y)
        class(rwkv_layer_type), intent(in) :: self
        real(sp), intent(in) :: x(:,:)
        real(sp), intent(in) :: init_state(:, :) ! d_model, n_components
        real(sp), intent(inout) :: hidden_states(:, :, :) ! d_model, n_components, n_states
        real(sp), allocatable :: y(:,:)

        y = x + self%time_mix%forward(self%ln1%forward(x), init_state, hidden_states)
        y = y + self%channel_mix%forward(self%ln2%forward(y), init_state, hidden_states)
    end function

end module
