! Copyright CERN. This software is distributed under the
! terms of the GNU General Public Licence version 3 (GPL Version 3),
! copied verbatim in the file LICENCE.txt.
! In applying this licence, CERN does not waive the privileges and immunities
! granted to it by virtue of its status as an Intergovernmental Organization or
! submit itself to any jurisdiction.
! Project website: http://blond.web.cern.ch/

! loss_box.f90
subroutine loss_box(e_max, e_min, t_min, t_max, dt, dE, flags, n)
   implicit none
   integer, intent(in) :: n
   integer(kind=4), intent(inout) :: flags(n)
   real(4), intent(in) :: dE(n)
   real(4), intent(in) :: dt(n)
   real(4), intent(in) :: e_max, e_min, t_min, t_max
   integer :: i

   !$omp parallel do private(i) shared(e_max, e_min, t_min, t_max, dt, dE, flags)
   do i = 1, n
      if (dE(i) > e_max .or. dE(i) < e_min .or. dt(i) < t_min .or. dt(i) > t_max) then
         flags(i) = -500 ! assume (BeamFlags.LOST.value)
      end if
   end do
   !$omp end parallel do
end subroutine loss_box
