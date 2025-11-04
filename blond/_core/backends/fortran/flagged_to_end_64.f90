function flagged_to_end(flag, flags, dt, dE, ids, n) result(j)
    implicit none
    integer(kind=4), intent(in) :: flag, n
    integer(kind=4), intent(inout) :: flags(n)
    integer(kind=8), intent(inout) :: ids(n)
    real(8), intent(inout) :: dt(n), dE(n)
    integer(kind=4) :: i, j
    integer(kind=4) :: tmp_flag, tmp_id
    real(4) :: tmp_dt, tmp_dE

    i = 1
    j = n

    do while (i <= j)
        if (flags(i) /= flag) then
            i = i + 1
        else
            ! swap elements i <-> j
            tmp_flag = flags(i)
            flags(i) = flags(j)
            flags(j) = tmp_flag

            tmp_dt = dt(i)
            dt(i) = dt(j)
            dt(j) = tmp_dt

            tmp_dE = dE(i)
            dE(i) = dE(j)
            dE(j) = tmp_dE

            tmp_id = ids(i)
            ids(i) = ids(j)
            ids(j) = tmp_id

            j = j - 1
        end if
    end do
end function flagged_to_end
