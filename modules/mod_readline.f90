! Copyright 2023 FortAI-Hub contributors.
! Released under the MIT License. See LICENSE file for full license information.

module mod_readline
    use iso_fortran_env, only: error_unit, input_unit, output_unit
    implicit none

    private
    public :: readline

contains

    subroutine readline(prompt, line, iostat)
        character(*), intent(in) :: prompt
        character(:), allocatable, intent(out) :: line
        integer, intent(out) :: iostat
        character(1024) :: buffer
        integer :: l
        
        line = ''
        call show_prompt(prompt)

        do
            read(input_unit, '(a)', iostat=iostat, advance='no', size=l) buffer
            if (l > 0) line = line // buffer(:l)

            ! handle line continuation in case of trailing '\'
            if (is_iostat_eor(iostat)) then
                l = len(line)
                if (l /= 0) then
                    if (line(l:l) == '\') then
                        line = line(:l - 1)
                        cycle
                    endif
                endif

                iostat = 0
                exit
            end if

            if (iostat /= 0) exit
        enddo
        
        line = replace_newlines(trim(adjustl(line)))
    end subroutine readline

    subroutine show_prompt(prompt)
        character(*), intent(in) :: prompt
        if (len(prompt) == 0) return
        write(output_unit, '(a)', advance='no') prompt
        flush(output_unit)
    end subroutine

    pure function replace_newlines (text) result(res)
        character(*), intent(in) :: text
        character(1), parameter :: newline = new_line('')
        character(:), allocatable :: res
        integer :: i
        res = text
        do
           i = index(res, '\n')
           if (i == 0) exit
           res = res(:i-1) // newline // res(i+2:)
        end do
    end function

end module
