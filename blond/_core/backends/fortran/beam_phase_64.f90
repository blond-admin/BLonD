module beam_phase_module
  implicit none
  private
  public :: beam_phase

contains

  function trapz_const_delta(y, delta, n) result(integral)
    implicit none
    integer, intent(in) :: n
    real(kind=8), intent(in) :: y(n)
    real(kind=8), intent(in) :: delta
    real(kind=8) :: integral
    integer :: i

    integral = 0.5_8 * (y(1) + y(n))
    do i = 2, n - 1
        integral = integral + y(i)
    end do
    integral = integral * delta
  end function trapz_const_delta

  function beam_phase(bin_centers, profile, alpha, omega_rf, phi_rf, bin_size, n_bins) result(phase)
    implicit none
    integer, intent(in) :: n_bins
    real(kind=8), intent(in) :: bin_centers(n_bins), profile(n_bins)
    real(kind=8), intent(in) :: alpha, omega_rf, phi_rf, bin_size
    real(kind=8) :: base(n_bins), array1(n_bins), array2(n_bins)
    real(kind=8) :: scoeff, ccoeff, phase
    integer :: i

    !$omp parallel do
    do i = 1, n_bins
        base(i) = exp(alpha * bin_centers(i)) * profile(i)
    end do

    !$omp parallel do
    do i = 1, n_bins
        array1(i) = base(i) * sin(omega_rf * bin_centers(i) + phi_rf)
        array2(i) = base(i) * cos(omega_rf * bin_centers(i) + phi_rf)
    end do

    scoeff = trapz_const_delta(array1, bin_size, n_bins)
    ccoeff = trapz_const_delta(array2, bin_size, n_bins)

    phase = scoeff / ccoeff
  end function beam_phase

end module beam_phase_module
