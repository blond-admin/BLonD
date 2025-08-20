subroutine histogram(array_in, array_out, cut_left, cut_right, n_slices, n_macroparticles)
  use omp_lib
  implicit none

  ! Inputs
  real(8), intent(in)  :: array_in(:)
  real(8), intent(in)  :: cut_left, cut_right
  integer, intent(in)  :: n_slices, n_macroparticles

  ! Output
  real(8), intent(inout) :: array_out(:)

  ! Constants and derived values
  real(8)              :: inv_bin_width
  integer              :: i, j, t, tid, nthreads, bin, loop_count

  ! Arrays
  integer, parameter   :: STEP = 16
  integer, allocatable :: histo(:,:), ibin(:)

  inv_bin_width = real(n_slices, 8) / (cut_right - cut_left)

  nthreads = omp_get_max_threads()
  allocate(histo(n_slices, nthreads))
  histo = 0

  allocate(ibin(STEP))

  !$omp parallel private(i, j, tid, ibin, loop_count, bin)
    tid = omp_get_thread_num()

    !$omp do schedule(static)
    do i = 1, n_macroparticles, STEP
      loop_count = min(STEP, n_macroparticles - i + 1)

      ! Compute bin indices only if in range
      do j = 1, loop_count
        bin = int((array_in(i + j - 1) - cut_left) * inv_bin_width)
        if (bin >= 0 .and. bin < n_slices) then
          histo(bin + 1, tid + 1) = histo(bin + 1, tid + 1) + 1
        end if
      end do
    end do
    !$omp end do

  !$omp end parallel

  ! Combine thread-local histograms
  array_out = 0.0
  do i = 1, n_slices
    do t = 1, nthreads
      array_out(i) = array_out(i) + histo(i, t)
    end do
  end do

  deallocate(histo)
  deallocate(ibin)

end subroutine histogram
