subroutine linear_interp_kick(beam_dt, beam_dE, voltage_array, bin_centers, charge, &
                             n_slices, n_macroparticles, acc_kick)
    implicit none
    integer, intent(in) :: n_slices, n_macroparticles
    real(8), intent(in) :: beam_dt(n_macroparticles)
    real(8), intent(inout) :: beam_dE(n_macroparticles)
    real(8), intent(in) :: voltage_array(n_slices)
    real(8), intent(in) :: bin_centers(n_slices)
    real(8), intent(in) :: charge, acc_kick

    integer, parameter :: STEP = 64
    real(8) :: inv_bin_width
    real(8), allocatable :: voltageKick(:), factor(:)
    integer :: i, j, loop_count
    integer :: fbin(STEP)

    ! Calculate inverse bin width:
    inv_bin_width = real(n_slices - 1, 8) / (bin_centers(n_slices) - bin_centers(1))

    allocate(voltageKick(n_slices - 1))
    allocate(factor(n_slices - 1))

    !$omp parallel default(shared) private(i, j, loop_count, fbin)

    !$omp single
    do i = 1, n_slices - 1
        voltageKick(i) = charge * (voltage_array(i+1) - voltage_array(i)) * inv_bin_width
        factor(i) = (charge * voltage_array(i) - bin_centers(i) * voltageKick(i)) + acc_kick
    end do
    !$omp end single

    !$omp do schedule(static)
    do i = 1, n_macroparticles, STEP
        loop_count = min(STEP, n_macroparticles - i + 1)

        do j = 1, loop_count
            fbin(j) = int(floor((beam_dt(i + j - 1) - bin_centers(1)) * inv_bin_width))
        end do

        do j = 1, loop_count
            if (fbin(j) >= 0 .and. fbin(j) < (n_slices - 1)) then
                beam_dE(i + j - 1) = beam_dE(i + j - 1) + &
                    beam_dt(i + j - 1) * voltageKick(fbin(j) + 1) + factor(fbin(j) + 1)
            end if
        end do
    end do
    !$omp end do

    !$omp end parallel

    deallocate(voltageKick)
    deallocate(factor)
end subroutine linear_interp_kick
