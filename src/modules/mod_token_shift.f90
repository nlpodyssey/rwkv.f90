! Copyright 2023 FortAI-Hub contributors.
! Released under the MIT License. See LICENSE file for full license information.

module mod_token_shift
    use mod_real_precision
    implicit none
    private
    public :: token_shift

contains

    pure function token_shift(last_x, x) result(xx)
        real(sp), intent(in) :: last_x(:), x(:,:)
        real(sp) :: xx(size(x, 1), size(x, 2))

        xx(:,1) = last_x
        xx(:,2:) = x(:,1:size(x, 2)-1)
    end function token_shift

end module mod_token_shift