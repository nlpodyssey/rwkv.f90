! Copyright 2023 FortAI-Hub contributors.
! Released under the MIT License. See LICENSE file for full license information.

module mod_real_precision
    use iso_fortran_env, only: real32, real64
    integer, parameter :: dp = real64 ! double precision
    integer, parameter :: sp = real32 ! single precision
end module
