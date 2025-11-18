! loss_box.f90
subroutine loss_box(top, bottom, left, right, dt, dE, flags, n)
   implicit none
   integer, intent(in) :: n
   integer(kind=4), intent(inout) :: flags(n)
   real(4), intent(in) :: dE(n)
   real(4), intent(in) :: dt(n)
   real(4), intent(in) :: top, bottom, left, right
   integer :: i

   !$omp parallel do private(i) shared(top, bottom, left, right, dt, dE, flags)
   do i = 1, n
      if (dE(i) > top .or. dE(i) < bottom .or. dt(i) < left .or. dt(i) > right) then
         flags(i) = -500 ! assume (BeamFlags.LOST.value)
      end if
   end do
   !$omp end parallel do
end subroutine loss_box
