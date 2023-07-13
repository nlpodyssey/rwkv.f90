! Copyright 2023 FortAI-Hub contributors.
! Released under the MIT License. See LICENSE file for full license information.

module mod_real_precision
    integer, parameter :: dp = kind(1.d0) ! double precision
    integer, parameter :: sp = selected_real_kind(p=6) ! single precision
end module mod_real_precision
