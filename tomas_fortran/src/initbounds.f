
C     **************************************************
C     *  initbounds                                    *
C     **************************************************

C     WRITTEN BY Peter Adams, November 1999

C     This subroutine initializes the array, xk, which describes the
C     boundaries between the aerosol size bins.  xk is in terms of dry
C     single-particle mass (kg).  The aerosol microphysics algorithm
C     used here assumes mass doubling such that each xk is twice the
C     previous.

      SUBROUTINE initbounds()

      IMPLICIT NONE

C-----INCLUDE FILES-----------------------------------------------------

      include 'sizecode.COM'

C-----VARIABLE DECLARATIONS---------------------------------------------

      integer k
      double precision Mo     !lower mass bound for first size bin (kg)

C-----ADJUSTABLE PARAMETERS---------------------------------------------

c      parameter(Mo=9.765625d-25) ! so 11th bin is the same as first bin normally

C-----CODE--------------------------------------------------------------

      Mo = 1.0d-21*2.d0**(-6)

      do k=1,ibins+1
         xk(k)=Mo*2.d0**(k-1)
      enddo

      RETURN
      END
