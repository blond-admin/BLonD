! drift.f90
subroutine drift_simple(dt, dE, T, eta_0, beta, energy, n)
    implicit none
    integer, intent(in) :: n
    real(4), intent(inout) :: dt(n)
    real(4), intent(in) :: dE(n)
    real(4), intent(in) :: T, eta_0, beta, energy
    real(4) :: coeff
    integer :: i

    coeff = eta_0 / (beta * beta * energy)

    !$omp parallel do private(i) shared(dt, dE, coeff, T)
    do i = 1, n
        dt(i) = dt(i) + T * coeff * dE(i)
    end do
    !$omp end parallel do
end subroutine drift_simple
