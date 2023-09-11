! Copyright 2023 FortAI-Hub contributors.
! Released under the MIT License. See LICENSE file for full license information.

program main
    use mod_command_arguments, only: command_arguments, parse_arguments
    use mod_inference, only: run_inference
    implicit none

    type(command_arguments) :: args

    args = parse_arguments()
    call run_inference(args%inference)
end program
