! Copyright 2023 FortAI-Hub contributors.
! Released under the MIT License. See LICENSE file for full license information.

module mod_stats
    use mod_real_precision
    implicit none
    private
    public :: sample_from_multinomial

contains

    function sample_from_multinomial(probs, num_samples) result(sampled_indices)
        implicit none
        real(sp), dimension(:), intent(in) :: probs
        integer, intent(in) :: num_samples
        integer, allocatable :: sampled_indices(:)
        integer :: n, i, j
        real(sp) :: p, cum_sum
        logical, dimension(size(probs)) :: already_sampled

        n = size(probs)

        if (num_samples > n) then
            print*, 'Error: num_samples (', num_samples, ') must be less than or equal to the size of the input (', n, ').'
            stop
        end if

        allocate(sampled_indices(num_samples))
        already_sampled = .false.

        do i = 1, num_samples
            call random_number(p)  ! generate a random number between 0 and 1

            cum_sum = 0.0
            do j = 1, n
                cum_sum = cum_sum + probs(j)
                if (cum_sum >= p .and. .not. already_sampled(j)) then
                    sampled_indices(i) = j
                    already_sampled(j) = .true.
                    exit
                end if
            end do
        end do
    end function

end module