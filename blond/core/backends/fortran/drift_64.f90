! Copyright CERN. This software is distributed under the
! terms of the GNU General Public Licence version 3 (GPL Version 3),
! copied verbatim in the file LICENCE.txt.
! In applying this licence, CERN does not waive the privileges and immunities
! granted to it by virtue of its status as an Intergovernmental Organization or
! submit itself to any jurisdiction.
! Project website: http://blond.web.cern.ch/

! drift.f90
subroutine drift_simple(dt, dE, T, eta_0, beta, energy, n)
   implicit none
   integer, intent(in) :: n
   real(8), intent(inout) :: dt(n)
   real(8), intent(in) :: dE(n)
   real(8), intent(in) :: T, eta_0, beta, energy
   real(8) :: coeff
   integer :: i

   coeff = eta_0/(beta*beta*energy)

   !$omp parallel do private(i) shared(dt, dE, coeff, T)
   do i = 1, n
      dt(i) = dt(i) + T*coeff*dE(i)
   end do
   !$omp end parallel do
end subroutine drift_simple
