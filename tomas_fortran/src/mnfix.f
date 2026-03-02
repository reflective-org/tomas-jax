
C     **************************************************
C     *  mnfix                                         *
C     **************************************************

C     WRITTEN BY Peter Adams, September 2000

C     This subroutine examines the mass and number distributions and
C     determines if any bins have an average mass outside their normal
C     range.  I have seen this happen because the GCM advection seems
C     to treat the mass and number tracers inconsistently.  If any bins
C     are out of range, I shift some mass and number to a new bin in
C     a way that conserves both.

C-----INPUTS------------------------------------------------------------

C     Nkx and Mkx are the number and mass distributions

C-----OUTPUTS-----------------------------------------------------------

      SUBROUTINE mnfix(Nkx,Mkx)

      IMPLICIT NONE

C-----INCLUDE FILES-----------------------------------------------------

      include 'sizecode.COM'

C-----ARGUMENT DECLARATIONS---------------------------------------------

      double precision Nkx(ibins), Mkx(ibins,icomp)

C-----VARIABLE DECLARATIONS---------------------------------------------

      integer k,j,kk           !counters
      integer newbin           !bin number into which mass is shifted
      double precision xold, xnew  !average masses of old and new bins
      double precision drymass !dry mass of in a bin
      double precision avg     !average dry mass of particles in bin
      double precision number  !number of particles initially in problem bin
      double precision nshift  !number to shift to new bin
      double precision mshift  !mass to shift to new bin
      double precision fj      !fraction of mass that is component j
      double precision eps     !small number
      double precision Neps

C     VARIABLE COMMENTS...

C-----ADJUSTABLE PARAMETERS---------------------------------------------

      parameter(eps=1.d-40,Neps=1E-5)

C-----CODE--------------------------------------------------------------

C If N=0 and M is tiny, set M to zero
      do k=1,ibins
         if (Nkx(k) .lt. Neps) then
            do j=1,icomp-idiag
               if (Mkx(k,j) .lt. 1.d-5) then
                  Mkx(k,j)=0.d0
               else
                  write(*,*) 'N=0 but M>1.d-5 in mnfix'
                  write(*,*) 'bin=',k
                  write(*,*) 'N=',Nkx(k)
                  write(*,*) 'M=',Mkx(k,j)
                  STOP
               endif
            enddo
            Nkx(k)=Neps
            Mkx(k,srtso4)=Neps*sqrt(xk(k)*xk(k+1))
         endif
      enddo

C Check to see if any bins are completely out of bounds for any bin
      do k=1,ibins

         drymass=0.d0
         do j=1,icomp-idiag
            drymass=drymass+Mkx(k,j)
         enddo
         avg=drymass/(Nkx(k)+eps)

         if (avg .gt. xk(ibins+1)) then
            !out of bin range - remove some mass
            mshift=Nkx(k)*xk(ibins+1)/1.2
            do j=1,icomp
               Mkx(k,j)=Mkx(k,j)*mshift/(drymass+eps)
            enddo
         endif
         if (avg .lt. xk(1)) then
            !out of bin range - remove some number
            Nkx(k)=drymass/(xk(1)*1.2)
         endif
      enddo

      do k=1,ibins

         drymass=0.d0
         do j=1,icomp-idiag
            drymass=drymass+Mkx(k,j)
         enddo
         avg=drymass/(Nkx(k)+eps)
         if (Nkx(k) .eq. 0.0) then
            if (drymass .gt. 0.0) then
               write(*,*) 'N=0 but M>0 in mnfix'
               STOP
            else
               avg=sqrt(xk(k)*xk(k+1))
            endif
         endif

         if (avg .gt. xk(k+1)) then
            !average mass is too high - shift to higher bin
            kk=k+1
            xnew=xk(kk+1)/1.1
 100        if (xnew .le. avg) then
               kk=kk+1
               xnew=xk(kk+1)/1.1
               goto 100
            endif
            xold=sqrt(xk(k)*xk(k+1))
            number=Nkx(k)
            nshift=(drymass-xold*number)/(xnew-xold)
            mshift=xnew*nshift
            Nkx(k)=Nkx(k)-nshift
            Nkx(kk)=Nkx(kk)+nshift
            do j=1,icomp
               fj=Mkx(k,j)/(drymass+eps)
               Mkx(k,j)=xold*Nkx(k)*fj
               Mkx(kk,j)=Mkx(kk,j)+mshift*fj
            enddo
         endif

         if (avg .lt. xk(k)) then
            !average mass is too low - shift to lower bin
            kk=k-1
            xnew=xk(kk)*1.1
 200        if (xnew .ge. avg) then
               kk=kk-1
               xnew=xk(kk)*1.1
               goto 200
            endif
            xold=sqrt(xk(k)*xk(k+1))
            number=Nkx(k)
            nshift=(drymass-xold*number)/(xnew-xold)
            mshift=xnew*nshift
            Nkx(k)=Nkx(k)-nshift
            Nkx(kk)=Nkx(kk)+nshift
            do j=1,icomp
               fj=Mkx(k,j)/(drymass+eps)
               Mkx(k,j)=xold*Nkx(k)*fj
               Mkx(kk,j)=Mkx(kk,j)+mshift*fj
            enddo
         endif

      enddo

      RETURN
      END
