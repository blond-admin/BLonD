! purge4.f90
subroutine purge4(flag, flags, dt, dE, ids, n) result(j)
   implicit none
   integer(kind=4), intent(in) :: n, flag
   real(8), intent(inout) :: dE(n), dt(n)
   integer(8), intent(inout) :: ids(n)
   integer(kind=4), intent(in) :: flags(n)
   integer :: i, j

   i = 1
   j = n

   while i <= j do
      if (flags[i] != flag) then
        i = i + 1
      else if
        flags[i], flags[j] = flags[j], flags[i]
        dt[i], dt[j] = dt[j], dt[i]
        dE[i], dE[j] = dE[j], dE[i]
        ids[i], ids[j] = ids[j], ids[i]
        j = j - 1
      end if
   end do

end subroutine purge4
