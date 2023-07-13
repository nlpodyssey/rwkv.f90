! Copyright 2023 FortAI-Hub contributors.
! Released under the MIT License. See LICENSE file for full license information.

module mod_generation
    use mod_essentials
    use mod_real_precision
    use mod_rwkv_lm
    use mod_state, only: state_type
    use mod_functions, only: softmax_1d, argmax
    use mod_stats, only: sample_from_multinomial
    use mod_trie_tokenizer, only : rwkv_tokenizer => trie_tokenizer
    implicit none
    private
    public :: generate_text, generation_options, generated_token_handler, stop_generation_requested, in_generation

    logical :: in_generation, stop_generation_requested

    type :: generation_options
        real(sp) :: temp = 0.7
        real(sp) :: alpha_frequency = 0.4 ! Frequency Penalty
        real(sp) :: alpha_presence= 0.4 ! Presence Penalty
        real(sp) :: alpha_decay = 0.996 ! Gradually decay the penalty
        integer :: max_token_limit = 100
        logical :: use_multinomial = .true.
    end type generation_options

    abstract interface
        subroutine generated_token_handler(token_string)
            implicit none
            character(len=*), intent(in) :: token_string
        end subroutine generated_token_handler
    end interface

contains

    subroutine generate_text(model, tokenizer, state, input_logits, opts, callback)
        type(rwkv_lm_type), intent(in) :: model
        type(rwkv_tokenizer), intent(in) :: tokenizer
        type(state_type), intent(inout) :: state
        real(sp), intent(in) :: input_logits(:)
        type(generation_options), intent(in) :: opts
        procedure(generated_token_handler), pointer, intent(in) :: callback

        real(sp) :: occurrence(size(model%proj, 1)) ! corresponding to the number of logits
        real(sp), allocatable :: logits(:)
        integer, allocatable :: sampled_indices(:)
        integer :: i, token_id
        character(:), allocatable :: token

        occurrence = 0
        in_generation = .true.
        stop_generation_requested = .false.

        logits = input_logits

        do i = 1, opts%max_token_limit
            if (stop_generation_requested) then
                stop_generation_requested = .false.
                write(stderr,'(a)') "> Stop generation due to user request."
                exit ! forced end of generation
            end if

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
                exit ! end of generation
            end if

            where (occurrence /= 0)
                occurrence = occurrence * opts%alpha_decay
            end where

            occurrence(token_id) = occurrence(token_id) + 1

            token = tokenizer%decode([token_id])

            if (i > 1 .or. .not. is_whitespace(token)) then
                call callback(token)
            end if

            logits = model%forward(token_id, state)
        end do

        in_generation = .false.
    end subroutine generate_text

    subroutine apply_temperature(logits, temp)
        real(sp), intent(inout) :: logits(:)
        real(sp), intent(in) :: temp

        integer :: i

        if (temp < 0.0 .or. temp > 1.0) then
            print*, "Invalid temperature value: ", temp, ". Must be between 0 and 1"
            error stop
        end if

        if (temp == 0) then
            logits = logits / 0.01
        else
            logits = logits / temp
        end if
    end subroutine apply_temperature

    pure logical function is_whitespace(string)
        implicit none
        character(len=*), intent(in) :: string
        integer :: i

        is_whitespace = .true.

        do i = 1, len_trim(string)
            if (string(i:i) /= ' ' .and. string(i:i) /= char(9) .and. string(i:i) /= char(10) .and. string(i:i) /= char(13)) then
                is_whitespace = .false.
                exit
            end if
        end do
    end function is_whitespace

end module mod_generation
