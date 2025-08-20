! kick.f90
subroutine kick_single_harmonic(dt, dE, voltage, omega_rf, phi_rf, charge, acceleration_kick, n)
    implicit none
    integer, intent(in) :: n
    real(4), intent(in) :: dt(n)
    real(4), intent(inout) :: dE(n)
    real(4), intent(in) :: voltage, omega_rf, phi_rf, charge, acceleration_kick
    real(4) :: voltage_kick
    integer :: i

    voltage_kick = charge * voltage

    !$omp parallel do private(i) shared(dt, dE, voltage_kick, omega_rf, phi_rf, acceleration_kick)
    do i = 1, n
        dE(i) = dE(i) + voltage_kick * sin(omega_rf * dt(i) + phi_rf) + acceleration_kick
    end do
    !$omp end parallel do
end subroutine kick_single_harmonic

subroutine kick_multi_harmonic(dt, dE, n_rf, charge, voltage, omega_RF, phi_RF, n_macroparticles, acc_kick)
    use omp_lib
    implicit none

    ! Inputs
    integer, intent(in) :: n_rf, n_macroparticles
    real(kind=4), intent(in) :: dt(n_macroparticles)
    real(kind=4), intent(in) :: charge, voltage(n_rf), omega_RF(n_rf), phi_RF(n_rf), acc_kick

    ! In/out
    real(kind=4), intent(inout) :: dE(n_macroparticles)

    ! Locals
    integer :: i, j
    real(kind=4) :: dE_sum, dti

    select case (n_rf)

    case (1)
        !$omp parallel do private(i, dti, dE_sum) shared(dt, dE, voltage, omega_RF, phi_RF, charge, acc_kick)
        do i = 1, n_macroparticles
            dti = dt(i)
            dE(i) = dE(i) + charge * voltage(1) * sin(omega_RF(1) * dt(i) + phi_RF(1)) + acc_kick
        end do
        !$omp end parallel do

    case (2)
        !$omp parallel do private(i, dti, dE_sum) shared(dt, dE, voltage, omega_RF, phi_RF, charge, acc_kick)
        do i = 1, n_macroparticles
            dti = dt(i)
            dE_sum = voltage(1) * sin(omega_RF(1) * dti + phi_RF(1)) + &
                     voltage(2) * sin(omega_RF(2) * dti + phi_RF(2))
            dE(i) = dE(i) + charge * dE_sum + acc_kick
        end do
        !$omp end parallel do

    case (3)
        !$omp parallel do private(i, dti, dE_sum) shared(dt, dE, voltage, omega_RF, phi_RF, charge, acc_kick)
        do i = 1, n_macroparticles
            dti = dt(i)
            dE_sum = voltage(1) * sin(omega_RF(1) * dti + phi_RF(1)) + &
                     voltage(2) * sin(omega_RF(2) * dti + phi_RF(2)) + &
                     voltage(3) * sin(omega_RF(3) * dti + phi_RF(3))
            dE(i) = dE(i) + charge * dE_sum + acc_kick
        end do
        !$omp end parallel do

    case (4)
        !$omp parallel do private(i, dti, dE_sum) shared(dt, dE, voltage, omega_RF, phi_RF, charge, acc_kick)
        do i = 1, n_macroparticles
            dti = dt(i)
            dE_sum = voltage(1) * sin(omega_RF(1) * dti + phi_RF(1)) + &
                     voltage(2) * sin(omega_RF(2) * dti + phi_RF(2)) + &
                     voltage(3) * sin(omega_RF(3) * dti + phi_RF(3)) + &
                     voltage(4) * sin(omega_RF(4) * dti + phi_RF(4))
            dE(i) = dE(i) + charge * dE_sum + acc_kick
        end do
        !$omp end parallel do

    case default
        !$omp parallel do private(i, dti, dE_sum) shared(dt, dE, voltage, omega_RF, phi_RF, charge, acc_kick, n_rf)
        do i = 1, n_macroparticles
            dti = dt(i)
            dE_sum = 0.0_4
            do j = 1, n_rf
                dE_sum = dE_sum + voltage(j) * sin(omega_RF(j) * dti + phi_RF(j))
            end do
            dE(i) = dE(i) + charge * dE_sum + acc_kick
        end do
        !$omp end parallel do

    end select

end subroutine kick_multi_harmonic
