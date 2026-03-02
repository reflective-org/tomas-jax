
C     **************************************************
C     *  loginit                                       *
C     **************************************************

C     WRITTEN BY Peter Adams, November 1999

C     Initializes aerosol number and mass distributions with sulfate
C     only and a lognormal size distribution.

C-----INPUTS------------------------------------------------------------

C     No - total number concentration (#/cm3)
C     Dp - lognormal mean diameter (microns)
C     sigma - width parameter

C-----OUTPUTS-----------------------------------------------------------

      SUBROUTINE loginit(N1, Dp1, sigma1, N2, Dp2, sigma2,
     &                   orgfrac, dens_init)

      IMPLICIT NONE
      
C     INCLUDE FILES
      
      include 'sizecode.COM'

C-----ARGUMENT DECLARATIONS---------------------------------------------

      double precision N1, Dp1, sigma1, N2, Dp2, sigma2
      integer idtcomp    !indicates chemical component to be added

C-----VARIABLE DECLARATIONS---------------------------------------------

      integer b,n     !bin and species counters
      integer I
      double precision Dl, Dh   !lower and upper bounds of bin (microns)
      double precision Dk       !mean diameter of current bin (microns)
      double precision np       !number of particles in size bin in GCM cell
      double precision pi
      double precision No1, No2
      double precision orgfrac
      double precision dens_init

C     VARIABLE COMMENTS...

C-----ADJUSTABLE PARAMETERS---------------------------------------------

      parameter(pi=3.14159)

C-----CODE--------------------------------------------------------------

C Convert No from #/cm3 to #/box
      No1=N1*boxvol
      No2=N2*boxvol

      print*, 'N1=', N1
      print*, 'N2=', N2
      print*, 'Dp1=', Dp1
      print*, 'Dp2=', Dp2
      print*, 'sigma1=', sigma1
      print*, 'sigma2=', sigma2
      print*, 'orgfrac', orgfrac
      print*, 'inorgfrac', 1-orgfrac
      !Loop over number of size bins
      DO N=1,IBINS

         !Calculate diameter of this size bin
         Dl=1.0e+6*((6.0*xk(n))/(dens_init*pi))**0.3333
         Dh=1.0e+6*((6.0*xk(n+1))/(dens_init*pi))**0.3333
         Dk=sqrt(Dl*Dh)
c         write(*,*) 'Dk(',n,')= ',Dk,' microns'

         !Calculate number concentration
         np=(No1/(sqrt(2.*pi)*Dk*log(sigma1))*exp(-( (log(Dk/Dp1))**2./
     &         (2.*(log(sigma1))**2.) )) * (Dh-Dl)) +
     &      (No2/(sqrt(2.*pi)*Dk*log(sigma2))*exp(-( (log(Dk/Dp2))**2./
     &         (2.*(log(sigma2))**2.) )) * (Dh-Dl))

         Nk(n) = Nk(n) + np
         print*,'n=', N, 'Nk(n)=',Nk(n)

         !Calculate component mass
CCC      AliA \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
         Mk(n,srtso4) = Mk(n,srtso4) + np*sqrt(xk(n))*sqrt(xk(n+1)) *
     &                                 (1-orgfrac)
         print*, 'Mk(n,srtso4)=', Mk(n,srtso4)
         Mk(n,srtorglast) = Mk(n,srtorglast) +
     &                      np*sqrt(xk(n))*sqrt(xk(n+1)) * orgfrac
         print*, 'Mk(n,srtorglast)=', Mk(n,srtorglast)
         
C         DO I=1,ICOMP
C            Mk(n,I) = Mk(n,I) +
C     &                np*sqrt(xk(n))*sqrt(xk(n+1))*P_FRAC(I)
C         END DO
CCC      ////////////////////////////////////////////////////////////////// AliA

      END DO

      RETURN
      END
