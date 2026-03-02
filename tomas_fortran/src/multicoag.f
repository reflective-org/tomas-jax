
C     **************************************************
C     *  multicoag                                     *
C     **************************************************

C     WRITTEN BY Peter Adams, June 1999
C     Modified to allow for multicomponent aerosols, February 2000

C     This routine performs coagulation on the aerosol size distribution
C     defined by Nk and Mk (number and mass).  See "An Efficient
C     Numerical Solution to the Stochastic Collection Equation", S.
C     Tzivion, G. Feingold, and Z. Levin, J Atmos Sci, 44, no 21, 3139-
C     3149, 1987.  Unless otherwise noted, all equation references refer
C     to this paper.  Some equations are taken from "Atmospheric Chemistry
C     and Physics: From Air Pollution to Climate Change" by Seinfeld
C     and Pandis (S&P).  

C     This routine uses a "moving sectional" approach in which the
C     aerosol size bins are defined in terms of dry aerosol mass.
C     Addition or loss of water, therefore, does not affect which bin
C     a particle falls into.  As a result, this routine does not
C     change Mk(water), although water masses are needed to compute
C     particle sizes and, therefore, coagulation coefficients.  Aerosol
C     water masses in each size bin will need to be updated later
C     (in another routine) to reflect changes that result from
C     coagulation.

C-----INPUTS------------------------------------------------------------

C     The user must supply the mass and number distributions, Mk and Nk,
C     as well as the time step, dt.

C-----OUTPUTS-----------------------------------------------------------

C     The program updates Nk and Mk.

      SUBROUTINE multicoag(dt)

      IMPLICIT NONE

C     INCLUDE FILES...

      include 'sizecode.COM'

C-----ARGUMENT DECLARATIONS---------------------------------------------

      double precision dt         !time step (s)

C-----VARIABLE DECLARATIONS---------------------------------------------

      integer k,j,i,jj    !counters
      double precision dNdt(ibins), dMdt(ibins,icomp-1)
      double precision xbar(ibins), phi(ibins), eff(ibins)

C kij represents the coagulation coefficient (cm3/s) normalized by the
C volume of the GCM grid cell (boxvol, cm3) such that its units are (s-1)
      double precision kij(ibins,ibins)
      double precision Dpk(ibins)             !diameter (m) of particles in bin k
      double precision Dk(ibins)              !Diffusivity (m2/s) of bin k particles
      double precision ck(ibins)              !Mean velocity (m/2) of bin k particles
      double precision olddiff                !used to iterate to find diffusivity
      double precision density                !density (kg/m3) of particles
      double precision mu                     !viscosity of air (kg/m s)
      double precision mfp                    !mean free path of air molecule (m)
      double precision Kn                     !Knudsen number of particle
      double precision mp         !particle mass (kg)
      double precision beta                   !correction for coagulation coeff.
      double precision aerodens
      external aerodens

      !temporary summation variables
      double precision k1m(icomp-1),k1mx(icomp-1),k1mx2(icomp-1)
      double precision k1mtot,k1mxtot
      double precision sk2mtot, sk2mxtot
      double precision sk2m(icomp-1), sk2mx(icomp-1), sk2mx2(icomp-1)
      double precision in
      double precision mtotal
      double precision Mktot

      double precision zeta                      !see reference, eqn 6
      double precision tlimit, dtlimit, itlimit  !fractional change in M/N allowed in one time step
      double precision dts  !internal time step (<dt for stability)
      double precision tsum !time so far
      double precision Neps !minimum value for Nk
cdbg      character*12 limit        !description of what limits time step

      double precision mi, mf   !initial and final masses
      double precision orgmass ! total organic mass

C     VARIABLE COMMENTS...

C     dNdt and dMdt are the rates of change of Nk and Mk.  xk contains
C     the mass boundaries of the size bins.  xbar is the average mass
C     of a given size bin (it varies with time in this algorithm).  phi
C     and eff are defined in the reference, equations 13a and b.

C-----ADJUSTABLE PARAMETERS---------------------------------------------

      parameter(zeta=1.0625, dtlimit=0.25, itlimit=10.)
      double precision pi, kB  !kB is Boltzmann constant (J/K)
      double precision R       !gas constant (J/ mol K)
      parameter (pi=3.141592654, kB=1.38e-23, R=8.314, Neps=1.0e-3)

 1    format(16E15.3)

C-----CODE--------------------------------------------------------------

      tsum = 0.0

C If any Nk are zero, then set them to a small value to avoid division by zero
      do k=1,ibins
         if (Nk(k) .lt. Neps) then
            Nk(k)=Neps
            do j=1,icomp
               Mk(k,j)=0.d0
            enddo
            Mk(k,srtso4)=Neps*1.4*xk(k) !make the added particles SO4
         endif
      enddo

C Calculate air viscosity and mean free path

      mu=2.5277e-7*temp**0.75302
      mfp=2.0*mu/(pres*sqrt(8.0*0.0289/(pi*R*temp)))  !S&P eqn 8.6

C Calculate particle sizes and diffusivities
      do k=1,ibins

CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
CC  Density subroutine needs to be customized depending on chemical species!!!
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
         orgmass=0.d0
         do j=1,iorg
            orgmass=orgmass+Mk(k,srtorg1+j-1) 
         enddo
         density=aerodens(Mk(k,srtso4)+orgmass,0.d0,
     &        Mk(k,srtnh4),0.d0,  !assume bisulfate and org is sulf acid
     &        Mk(k,srth2o))

         Mktot=0.d0 ! account for ammonia
         do j=1,icomp
            Mktot=Mktot+Mk(k,j)
         enddo
         mp = Mktot/Nk(k)

         Dpk(k)=((mp/density)*(6./pi))**(0.333)
         Kn=2.0*mfp/Dpk(k)                            !S&P Table 12.1
         Dk(k)=kB*temp/(3.0*pi*mu*Dpk(k))             !S&P Table 12.1
     &   *((5.0+4.0*Kn+6.0*Kn**2+18.0*Kn**3)/(5.0-Kn+(8.0+pi)*Kn**2))
         ck(k)=sqrt(8.0*kB*temp/(pi*mp))              !S&P Table 12.1
      enddo

C Calculate coagulation coefficients

      do i=1,ibins
         do j=1,ibins
            Kn=4.0*(Dk(i)+Dk(j))          
     &        /(sqrt(ck(i)**2+ck(j)**2)*(Dpk(i)+Dpk(j))) !S&P eqn 12.51
            beta=(1.0+Kn)/(1.0+2.0*Kn*(1.0+Kn))          !S&P eqn 12.50
            !This is S&P eqn 12.46 with non-continuum correction, beta
            kij(i,j)=2.0*pi*(Dpk(i)+Dpk(j))*(Dk(i)+Dk(j))*beta
            kij(i,j)=kij(i,j)*1.0e6/boxvol  !normalize by grid cell volume
         enddo
      enddo


 10   continue     !repeat process here if multiple time steps are needed

C Calculate xbar, phi and eff

      do k=1,ibins

         xbar(k)=0.0
         do j=1,icomp-idiag
            xbar(k)=xbar(k)+Mk(k,j)/Nk(k)            !eqn 8b
         enddo

         eff(k)=2.*Nk(k)/xk(k)*(2.-xbar(k)/xk(k))    !eqn 13a
         phi(k)=2.*Nk(k)/xk(k)*(xbar(k)/xk(k)-1.)    !eqn 13b
         
         !Constraints in equation 15
         if (xbar(k) .lt. xk(k)) then
            eff(k)=2.*Nk(k)/xk(k)
            phi(k)=0.0
         else if (xbar(k) .gt. xk(k+1)) then
            phi(k)=2.*Nk(k)/xk(k)
            eff(k)=0.0
         endif
      enddo

C Necessary initializations
         sk2mtot=0.0
         sk2mxtot=0.0
         do j=1,icomp-idiag
            sk2m(j)=0.0
            sk2mx(j)=0.0
            sk2mx2(j)=0.0
         enddo

C Calculate rates of change for Nk and Mk

      do k=1,ibins

         !Initialize to zero
         do j=1,icomp-idiag
            k1m(j)=0.0
            k1mx(j)=0.0
            k1mx2(j)=0.0
         enddo
         in=0.0
         k1mtot=0.0
         k1mxtot=0.0

         !Calculate sums
         do j=1,icomp-idiag
            if (k .gt. 1) then
               do i=1,k-1
                  k1m(j)=k1m(j)+kij(k,i)*Mk(i,j)
                  k1mx(j)=k1mx(j)+kij(k,i)*Mk(i,j)*xbar(i)
                  k1mx2(j)=k1mx2(j)+kij(k,i)*Mk(i,j)*xbar(i)**2
               enddo
            endif
            k1mtot=k1mtot+k1m(j)
            k1mxtot=k1mxtot+k1mx(j)
         enddo
         if (k .lt. ibins) then
            do i=k+1,ibins
               in=in+Nk(i)*kij(k,i)
            enddo
         endif

         !Calculate rates of change
         dNdt(k)= 
     &           -kij(k,k)*Nk(k)**2
     &           -phi(k)*k1mtot
     &           -zeta*(eff(k)-phi(k))/(2*xk(k))*k1mxtot
     &           -Nk(k)*in
         if (k .gt. 1) then
         dNdt(k)=dNdt(k)+
     &           0.5*kij(k-1,k-1)*Nk(k-1)**2
     &           +phi(k-1)*sk2mtot
     &           +zeta*(eff(k-1)-phi(k-1))/(2*xk(k-1))*sk2mxtot
         endif

         do j=1,icomp-idiag
            dMdt(k,j)= 
     &           +Nk(k)*k1m(j)
     &           -kij(k,k)*Nk(k)*Mk(k,j)
     &           -Mk(k,j)*in
     &           -phi(k)*xk(k+1)*k1m(j)
     &           -0.5*zeta*eff(k)*k1mx(j)
     &           +zeta**3*(phi(k)-eff(k))/(2*xk(k))*k1mx2(j)
            if (k .gt. 1) then
               dMdt(k,j)=dMdt(k,j)+
     &           kij(k-1,k-1)*Nk(k-1)*Mk(k-1,j)
     &           +phi(k-1)*xk(k)*sk2m(j)
     &           +0.5*zeta*eff(k-1)*sk2mx(j)
     &           -zeta**3*(phi(k-1)-eff(k-1))/(2*xk(k-1))*sk2mx2(j)
            endif
cdbg            if (j. eq. srtso4) then
cdbg               if (k. gt. 1) then
cdbg                  write(*,1) Nk(k)*k1m(j), kij(k,k)*Nk(k)*Mk(k,j),
cdbg     &               Mk(k,j)*in, phi(k)*xk(k+1)*k1m(j),
cdbg     &               0.5*zeta*eff(k)*k1mx(j),
cdbg     &               zeta**3*(phi(k)-eff(k))/(2*xk(k))*k1mx2(j),
cdbg     &               kij(k-1,k-1)*Nk(k-1)*Mk(k-1,j),
cdbg     &               phi(k-1)*xk(k)*sk2m(j),
cdbg     &               0.5*zeta*eff(k-1)*sk2mx(j),
cdbg     &               zeta**3*(phi(k-1)-eff(k-1))/(2*xk(k-1))*sk2mx2(j)
cdbg               else
cdbg                  write(*,1) Nk(k)*k1m(j), kij(k,k)*Nk(k)*Mk(k,j),
cdbg     &               Mk(k,j)*in, phi(k)*xk(k+1)*k1m(j),
cdbg     &               0.5*zeta*eff(k)*k1mx(j),
cdbg     &               zeta**3*(phi(k)-eff(k))/(2*xk(k))*k1mx2(j)
cdbg               endif
cdbg            endif
         enddo

cdbg         write(*,*) 'k,dNdt,dMdt: ', k, dNdt(k), dMdt(k,srtso4)

         !Save the summations that are needed for the next size bin
         sk2mtot=k1mtot
         sk2mxtot=k1mxtot
         do j=1,icomp-idiag
            sk2m(j)=k1m(j)
            sk2mx(j)=k1mx(j)
            sk2mx2(j)=k1mx2(j)
         enddo

      enddo  !end of main k loop

C Update Nk and Mk according to rates of change and time step

      !If any Mkj are zero, add a small amount to achieve finite
      !time steps
      do k=1,ibins
         do j=1,icomp-idiag
            if (Mk(k,j) .eq. 0.d0) then
               !add a small amount of mass
               mtotal=0.d0
               do jj=1,icomp-idiag
                  mtotal=mtotal+Mk(k,jj)
               enddo
               Mk(k,j)=1.d-10*mtotal
            endif
         enddo
      enddo

      !Choose time step
      dts=dt-tsum      !try to take entire remaining time step
cdbg      limit='comp'
      do k=1,ibins
         if (Nk(k) .gt. Neps) then
            !limit rates of change for this bin
            if (dNdt(k) .lt. 0.0) tlimit=dtlimit
            if (dNdt(k) .gt. 0.0) tlimit=itlimit
            if (abs(dNdt(k)*dts) .gt. Nk(k)*tlimit) then 
               dts=Nk(k)*tlimit/abs(dNdt(k))
cdbg               limit='number'
cdbg               write(limit(8:9),'(I2)') k
cdbg               write(*,*) Nk(k), dNdt(k)
            endif
            do j=1,icomp-idiag
               if (abs(dMdt(k,j)*dts) .gt. Mk(k,j)*tlimit) then 
               mtotal=0.d0
               do jj=1,icomp-idiag
                  mtotal=mtotal+Mk(k,jj)
               enddo
               !only use this criteria if this species is significant
               if ((Mk(k,j)/mtotal) .gt. 1.d-5) then
                  dts=Mk(k,j)*tlimit/abs(dMdt(k,j))
               else
                  if (dMdt(k,j) .lt. 0.0) then
                     !set dmdt to 0 to avoid very small mk going negative
                     dMdt(k,j)=0.0
                  endif
               endif
cdbg                  limit='mass'
cdbg                  write(limit(6:7),'(I2)') k
cdbg                  write(limit(9:9),'(I1)') j
cdbg                  write(*,*) Mk(k,j), dMdt(k,j)
               endif
            enddo
         else
            !nothing in this bin - don't let it affect time step
            Nk(k)=Neps
            Mk(k,srtso4)=Neps*1.4*xk(k) !make the added particles SO4
            !make sure mass/number don't go negative
            if (dNdt(k) .lt. 0.0) dNdt(k)=0.0
            do j=1,icomp-idiag
               if (dMdt(k,j) .lt. 0.0) dMdt(k,j)=0.0
            enddo
         endif
      enddo

      !Change Nk and Mk
cdbg      write(*,*) 't=',tsum+dts,' ',limit
      do k=1,ibins
         Nk(k)=Nk(k)+dNdt(k)*dts
         do j=1,icomp-idiag
            Mk(k,j)=Mk(k,j)+dMdt(k,j)*dts
         enddo
      enddo

      call mnfix(Nk,Mk)

      !Update time and repeat process if necessary
      tsum=tsum+dts
      if (tsum .lt. dt) goto 10

      RETURN
      END
