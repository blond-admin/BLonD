subroutine linear_interp_kick(beam_dt, beam_dE, voltage_array, bin_centers, charge, &
                              n_slices, n_macroparticles, acc_kick)
   implicit none
   integer, intent(in) :: n_slices, n_macroparticles
   real(4), intent(in) :: beam_dt(n_macroparticles)
   real(4), intent(inout) :: beam_dE(n_macroparticles)
   real(4), intent(in) :: voltage_array(n_slices)
   real(4), intent(in) :: bin_centers(n_slices)
   real(4), intent(in) :: charge, acc_kick

   real(4) :: inv_bin_width
   real(4), allocatable :: voltageKick(:), factor(:)
   integer :: i, bin
   real(4) :: dt

   ! Precompute inverse bin width
   inv_bin_width = real(n_slices - 1, 4)/(bin_centers(n_slices) - bin_centers(1))

   allocate (voltageKick(n_slices - 1))
   allocate (factor(n_slices - 1))

   ! Compute voltageKick and factor arrays
   !$omp parallel do default(shared) private(i)
   do i = 1, n_slices - 1
      voltageKick(i) = charge*(voltage_array(i + 1) - voltage_array(i))*inv_bin_width
      factor(i) = (charge*voltage_array(i) - bin_centers(i)*voltageKick(i)) + acc_kick
   end do
   !$omp end parallel do

   ! Main interpolation loop
   !$omp parallel do default(shared) private(i, dt, bin)
   do i = 1, n_macroparticles
      dt = beam_dt(i)
      bin = int((dt - bin_centers(1))*inv_bin_width)

      if (bin >= 0 .and. bin < n_slices - 1) then
         beam_dE(i) = beam_dE(i) + dt*voltageKick(bin + 1) + factor(bin + 1)
      end if
   end do
   !$omp end parallel do

   deallocate (voltageKick)
   deallocate (factor)
end subroutine linear_interp_kick
