! Copyright 2023 FortAI-Hub contributors.
! Released under the MIT License. See LICENSE file for full license information.

module mod_rwkv_layer
    use mod_real_precision
    use mod_layer_norm
    use mod_channel_mix
    use mod_time_mix
    use mod_state, only: layer_state_type
    implicit none
    private
    public :: rwkv_layer_type, make_rwkv_layer

    type :: rwkv_layer_type
        type(layer_norm_type)  :: ln1
        type(layer_norm_type)  :: ln2
        type(channel_mix_type) :: channel_mix
        type(time_mix_type)    :: time_mix
    contains
        procedure read_params
        procedure, pass :: forward_single
        procedure, pass :: forward_batch
        generic :: forward => forward_single, forward_batch
    end type rwkv_layer_type

contains

    function make_rwkv_layer(d_model) result(rwkv_layer)
        integer, intent(in) :: d_model
        type(rwkv_layer_type) :: rwkv_layer

        rwkv_layer%ln1 = layer_norm_type(d_model, 1e-5)
        rwkv_layer%ln2 = layer_norm_type(d_model, 1e-5)
        rwkv_layer%channel_mix = channel_mix_type(d_model)
        rwkv_layer%time_mix = time_mix_type(d_model)
    end function make_rwkv_layer

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
    end subroutine read_params

    function forward_single(self, x, state) result(y)
        class(rwkv_layer_type), intent(in) :: self
        real(sp), intent(in) :: x(:)
        type(layer_state_type), intent(inout) :: state
        real(sp), allocatable :: y(:)

        y = x + self%time_mix%forward(self%ln1%forward(x), state)
        y = y + self%channel_mix%forward(self%ln2%forward(y), state)
    end function forward_single

    function forward_batch(self, x, state) result(y)
        class(rwkv_layer_type), intent(in) :: self
        real(sp), intent(in) :: x(:,:)
        type(layer_state_type), intent(inout) :: state
        real(sp), allocatable :: y(:,:)

        y = x + self%time_mix%forward(self%ln1%forward(x), state)
        y = y + self%channel_mix%forward(self%ln2%forward(y), state)
    end function forward_batch

end module mod_rwkv_layer
