! Copyright 2023 FortAI-Hub contributors.
! Released under the MIT License. See LICENSE file for full license information.

module mod_command_arguments
    use mod_pipeline, only: pipeline_options
    implicit none

    private
  
    public :: command_arguments, parse_arguments

    type :: command_arguments
        type(pipeline_options) :: pipeline
    end type

contains

    type(command_arguments) function parse_arguments() result(args)
        integer :: count, i
        character(:), allocatable :: arg
        
        count = command_argument_count()
        i = 1 ! skip program name
        
        do while (i <= count)
            arg = get_argument(i)

            select case (arg)
                case ('-tokenizer')
                    if (i == count) error stop 'Missing value for argument ' // arg
                    args%pipeline%tokenizer_filename = get_argument(i + 1)
                    i = i + 2
                case ('-model')
                    if (i == count) error stop 'Missing value for argument ' // arg
                    args%pipeline%model_filename = get_argument(i + 1)
                    i = i + 2
                case ('-draft')
                    if (i == count) error stop 'Missing value for argument ' // arg
                    args%pipeline%draft_model_filename = get_argument(i + 1)
                    i = i + 2
                case default
                    error stop 'Unknown command argument: ' // arg
            end select
        end do

        if (.not. allocated(args%pipeline%tokenizer_filename)) error stop 'Missing argument: -tokenizer'
        if (.not. allocated(args%pipeline%model_filename)) error stop 'Missing argument: -model'
    end function

    function get_argument(number) result (arg)
        integer, intent(in) :: number
        character(:), allocatable :: arg
        integer :: length, status
        character(4096) :: raw

        call get_command_argument(number, raw, length, status)
        if (status /= 0) error stop 'Failed to get command line arguments'
        arg = raw(:length)
    end function

end module
