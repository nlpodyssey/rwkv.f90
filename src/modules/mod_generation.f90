! Copyright 2023 FortAI-Hub contributors.
! Released under the MIT License. See LICENSE file for full license information.

module mod_generation
    use mod_essentials
    use mod_real_precision
    use mod_functions, only: softmax_1d, argmax
    use mod_stats, only: sample_from_multinomial
    implicit none
    private
    public :: generate_next_token, generation_options

    type :: generation_options
        real(sp) :: temp = 0.7
        real(sp) :: alpha_frequency = 0.4 ! Frequency Penalty
        real(sp) :: alpha_presence= 0.4 ! Presence Penalty
        real(sp) :: alpha_decay = 0.996 ! Gradually decay the penalty
        integer :: max_token_limit = 100
        logical :: use_multinomial = .true.
    end type

contains

    function generate_next_token(input_logits, occurrence, opts, end_of_generation) result(token_id)
        real(sp), intent(in) :: input_logits(:)
        real(sp), intent(inout) :: occurrence(:) ! corresponding to the logits
        type(generation_options), intent(in) :: opts
        logical, intent(out) :: end_of_generation

        integer :: token_id
        real(sp), allocatable :: logits(:)
        integer, allocatable :: sampled_indices(:)
        integer :: non_zeros

        end_of_generation = .false.
        logits = input_logits

        where (occurrence /= 0)
            logits = logits - (opts%alpha_presence + occurrence * opts%alpha_frequency)
        end where

        if (opts%temp /= 1) then
            call apply_temperature(logits, opts%temp)
        end if

        if (opts%use_multinomial) then
            sampled_indices = sample_from_multinomial(softmax_1d(logits), 1)
            token_id = sampled_indices(1) - 1
        else
            token_id = argmax(softmax_1d(logits)) - 1
        end if

        if (token_id == 0) then
            end_of_generation = .true.
            return
        end if

        where (occurrence /= 0)
            occurrence = occurrence * opts%alpha_decay
        end where

        occurrence(token_id) = occurrence(token_id) + 1
    end function

    pure subroutine apply_temperature(logits, temp)
        real(sp), intent(inout) :: logits(:)
        real(sp), intent(in) :: temp
        character(len=256) :: errmsg

        integer :: i

        if (temp < 0.0 .or. temp > 1.0) then
            errmsg = real_to_str(temp)
            error stop "Invalid temperature value: "//trim(adjustl(errmsg))//". Must be between 0 and 1"
        end if

        if (temp == 0) then
            logits = logits / 0.01
        else
            logits = logits / temp
        end if
    end subroutine

end module
