! Copyright 2023 FortAI-Hub contributors.
! Released under the MIT License. See LICENSE file for full license information.

module mod_prompt_utils
    implicit none
    private
    public :: generate_prompt

contains

    ! generate_prompt generates a prompt optimized for rwkv-4-world
    pure function generate_prompt(instruction, input)
        character(len=*), intent(in) :: instruction
        character(len=*), optional, intent(in) :: input
        character(:), allocatable :: generate_prompt
        if (present(input)) then
            generate_prompt = "Instruction: " // trim(instruction) // achar(10) // achar(10) // &
                    "Input: " // trim(input) // achar(10) // achar(10) // &
                    "Response:"
        else
            generate_prompt = "Question: " // trim(instruction) // achar(10) // achar(10) // &
                    "Response:"
        end if
    end function

end module