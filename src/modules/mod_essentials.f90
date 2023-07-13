module mod_essentials
    use, intrinsic :: iso_fortran_env, only : stdin=>input_unit, &
            stdout=>output_unit, &
            stderr=>error_unit

    implicit none

contains

    function real_to_str(d)
        real, intent(in) :: d
        character(:), allocatable :: real_to_str
        character(len=32) :: temp_str
        write(temp_str,'(f8.3)') d
        real_to_str = trim(adjustl(temp_str))
    end function real_to_str

    subroutine write_i0_arr_1d(unit, arr)
        integer, intent(in) :: unit
        integer, dimension(:), intent(in) :: arr
        integer :: i, n

        n = size(arr)
        do i = 1, n
            write(unit, '(i0)', advance='no') arr(i)
            if (i /= n) then
                write(unit, '(a)', advance='no') ', '
            end if
        end do
        write(unit, *)
    end subroutine write_i0_arr_1d

end module mod_essentials