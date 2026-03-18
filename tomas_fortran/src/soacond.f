
C     **************************************************
C     *  soacond                                       *
C     **************************************************

C     WRITTEN BY Jeff Pierce, Sep. 2010

C     This routine determines the condensational driving force for
C     mass transfer of sulfate between gas and aerosol phases.  It then calls
C     a mass- and number-conserving algorithm for condensation (or
C     evaporation) of aerosol.

C     An adaptive time step is used to prevent numerical difficulties.
C     To account for the changing gas phase concentration of sulfuric
C     acid, its decrease during a condensational time step is well-
C     approximated by an exponential decay with a constant, sK (Hz).
C     sK is calculated from the mass and number distribution of the
C     aerosol.  Not only does this approach accurately take into account
C     the changing sulfuric acid concentration, it is also used to
C     predict (and limit) the final sulfuric acid concentration.  This
C     approach is more accurate and faster (allows longer condensational
C     time steps) than assuming a constant supersaturation of sulfate.

C-----INPUTS------------------------------------------------------------

C     Initial values of
C     =================

C     Nki(ibins) - number of particles per size bin in grid cell
C     Mki(ibins, icomp) - mass of a given species per size bin/grid cell
C     Gci(icomp-1) - amount (kg/grid cell) of all species present in the
C                    gas phase except water
C     dt - total model time step to be taken (s)

C-----OUTPUTS-----------------------------------------------------------

C     Nkf, Mkf, Gcf - same as above, but final values

      SUBROUTINE soacond(Nki,Mki,Gci,Nkf,Mkf,Gcf,dt,tau_p,
     &     Dbk,kc)

      IMPLICIT NONE

C-----INCLUDE FILES-----------------------------------------------------

      include 'sizecode.COM'

C-----ARGUMENT DECLARATIONS---------------------------------------------

      double precision Nki(ibins), Mki(ibins, icomp), Gci(icomp-1)
      double precision Nkf(ibins), Mkf(ibins, icomp), Gcf(icomp-1)
      double precision dt

C-----VARIABLE DECLARATIONS---------------------------------------------

      double precision dp(ibins)  !Driving force for condensation (Pa)
      double precision tau(ibins)          !condensation parameter (see cond.f)
      double precision atau(ibins)  !same as tau, but all species
      double precision atauc(ibins) !same as atau, but for const dp
      double precision time                !amount of time (s) that has been simulated
      double precision cdt                 !internal, adaptive time step
      double precision mu                  !viscosity of air (kg/m s)
      double precision mfp                 !mean free path of air molecule (m)
      double precision Kn                  !Knudsen number of particle
      double precision Dpk(ibins)          !diameter (m) of particles in bin k
      double precision Rpk(ibins)          !Radius (m) of particles in bin k
      double precision density             !density (kg/m3) of particles
      integer k,j,jj,jjj,jo           !counters
      double precision tj, tk(ibins)  !factors used for calculating tau
      double precision sK      !exponential decay const for gas species(g)
      double precision pi, R   !constants
      double precision zeta13  !from Eqn B30 of Tzivion et al. (1989)
      double precision Di, Dgk !diffusivity of gas in air (m2/s)
Ckpc  expand density to all specie
!      double precision gmw(icomp-idiag)        !molecular weight of condensing gas
!      double precision Sv(icomp-idiag)         !parameter used for estimating diffusivity
      double precision beta(ibins)              !correction for non-continuum
      double precision mp,mps(ibins) !particle mass (kg)
      double precision Nko(ibins), Mko(ibins, icomp), Gco(icomp-idiag) !output of cond routine
      double precision mi, mf  !initial and final aerosol masses (updates Gc)
      double precision tr      ! used to calculate time step limits
      double precision mc, ttr
      double precision Neps     !value below which Nk is insignificant
      double precision cthresh  !determines minimum gas conc. for cond.
cdbg      character*12 limit        !description of what limits time step
      double precision tdt      !the value 2/3
      double precision Ntotf, Ntoto, dNerr  !used to track number cons.
      double precision Mktot !Ckpc: total mass for all species           
cdbg      integer numcalls          !number of times cond routine is called
      double precision zeros(ibins) ! array of zeros
      double precision tot_n
      double precision CS       !condensation sink [s-1]
      double precision betat(ibins)
      double precision Mkdry    !dry mass
      double precision WR(ibins) !ratio of wet mass to dry mass of particle
      double precision orgmass ! total mass of organics
      double precision totphase ! total amount of mass for organics to condense into
      double precision pamb ! ambient vapor pressure of species (Pa)
      double precision psat ! saturation vapor pressure of species (Pa)
      double precision totmass, Gcfs
      double precision madd(ibins) ! mass to add or remove from particle
      double precision scalefactor ! scale factor for Psat change by surface tension
      double precision sinkfrac(ibins) ! fraction of condensation sink from each bin
      double precision ms ! mean speed [m/s]
      double precision masseqm ! mass of species in particle at eqm
      double precision masscheqm ! change in mass of species to
                                        ! reach eqm
      double precision maddEQ(ibins) ! change in mass of species per particle
                              ! to reach eqm
      double precision fodc ! first order decay constant towards eqm
	  
      double precision tau_p ! report tau, added by Emily
      double precision test_time
      double precision kB, Na, morg(iorg), mair, dorg, dair	  

      DOUBLE PRECISION ACCOM       ! accomodation coefficient
      DOUBLE PRECISION FC(IBINS)   ! Fuch's correction factor
      DOUBLE PRECISION kgk(IBINS)  ! gas-side mass transfer coefficient, m/s
      DOUBLE PRECISION Kgkk(IBINS) ! gas-side mass transfer coefficient, m/s
      DOUBLE PRECISION kpk(IBINS)  !
      DOUBLE PRECISION Dbk         ! particle-phase diffusion coefficient [m2/s]
      DOUBLE PRECISION kc          ! first-order loss rate of species in the particle phase, 1/s
      DOUBLE PRECISION qk(IBINS)   ! diffuso-reactive term, unitless
      DOUBLE PRECISION coth(IBINS) ! hyperbolic cotangent
      DOUBLE PRECISION Qkk         ! ratio of particle concentration to surface 
                                   ! concentration at steady state [unitless]

C     VARIABLE COMMENTS...

C-----EXTERNAL FUNCTIONS------------------------------------------------
      double precision aerodens, gasdiff
      external aerodens, gasdiff

C-----ADJUSTABLE PARAMETERS---------------------------------------------

      parameter(pi=3.141592654, R=8.314) !pi and gas constant (J/mol K)
      parameter(Neps=1.0d-5, zeta13=0.98483, cthresh=1.d-16)
      	 
Ckpc  give the initial density of sulfate and other species

C-----CODE--------------------------------------------------------------

      do k=1,ibins
         zeros(k) = 0.d0
      enddo

cdbg      numcalls=0
      dNerr=0.0
      tdt=2.d0/3.d0
c      call tomasinit(gmw,Sv,alpha)

C Initialize values of Nkf, Mkf, Gcf, and time
      do j=1,icomp-1
         Gcf(j)=Gci(j)
      enddo
      do k=1,ibins
         Nkf(k)=Nki(k)
         do j=1,icomp
            Mkf(k,j)=Mki(k,j)
         enddo
      enddo

C Calculate tj and tk factors needed to calculate tau values
C      mu=2.5277e-7*temp**0.75302 !original
       mu = 1.458e-6*temp**1.5/(110.4 + temp)
      !mfp=2.0*mu/(pres*sqrt(8.0*0.0289/(pi*R*temp)))  !S&P eqn 8.6
       mair = (0.79*28. + 0.21*32.)/1000./Na  ! molecular weight of air, kg/mol, Andy
       kB = 1.38e-23 ! Boltzmann constant, kg-m2/s2-K,Andy
       Na = 6.023e23 ! Avogadro's number, molec/mol,Andy	   
      
       dorg = 10e-10            ! diameter of organic molecule, m,Andy
       dair = (0.79*1.09e-10 + 0.21*1.21e-10) ! diameter of air molecule, m,Andy	  

      !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      ! start loop over species!
      !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      do jo=1,iorg
      
      time=0.0  !go to next species when time=dt
C      mworg(jo)=(0.434 - 0.045*log10(cstar(jo)))*1000. !keep consistent with wall loss module
      morg(jo) = mworg(jo)/Na/1000.0		  !keep consistent with wall loss module

      j=srtorg1+jo-1
      ms=sqrt(8.0*R*temp/(pi*mworg(jo)*1.0d-3)) !S&P2 eqn 9.2
C      Di=diorg(jo)*(temp/T1)**ditemporg(jo)      !J's diffusivity of organic vapors
	  Di = (2./3.)*sqrt(kB*temp/pi*0.5*(1./morg(jo)+1/mair))*1./ 
     &	       (pi*(0.5*(dorg + dair))**2.)/Na*(R*temp/pres) ! Andy's diffusivity of organic vapors, m2/s
C      print *, 'Di'
C   	  print *, Di
      mfp=2.d0*Di/ms ! for use with Dahneke correction factor
      tj=2.*pi*mworg(jo)*1.0d-3/(R*temp)
      
C Repeat from this point if multiple internal time steps are needed
 10   continue

      call mnfix(Nkf,Mkf)
      
      sK=0.0d0
      do k=1,ibins
	     
         if (Nkf(k) .gt. Neps) then

Ckpc  calculate the total mass   
Ckpc   factor of 1.2 assumes ammonium bisulfate.
Ckpc  Add 0.2x first, and then add 1.0x in the loop
            Mktot=0.d0
            Mkdry=0.d0
            do jj=1,icomp
               Mktot=Mktot+Mkf(k,jj)
            enddo
            do jj=1,icomp-idiag
               Mkdry=Mkdry+Mkf(k,jj)
            enddo
            WR(k)=Mktot/Mkdry
Ckpc  Density should be changed due to more species involed. 
            orgmass=0.d0
            do jj=1,iorg
               orgmass=orgmass+Mk(k,srtorg1+jj-1) 
            enddo
            density=aerodens(Mk(k,srtso4)+orgmass,0.d0,
     &         Mk(k,srtnh4),0.d0,  !assume bisulfate and org is sulf acid
     &         Mk(k,srth2o))
 
            mp=Mktot/Nkf(k)
         else
            !nothing in this bin - set to "typical value"
            density=1500.
            mp=1.4*xk(k)
         endif
         mps(k)=mp
         Dpk(k)=((mp/density)*(6./pi))**(0.333)
C         print*, '-------------------------'
C         print*, 'Dpk(k) =', Dpk(k)
C         print*, '-------------------------'
         Kn=2.0*mfp/Dpk(k)      !S&P eqn 11.35 (text)

!     AliA - calculate Kgk
CCCCC ============================================================================================
!     basic properties; can be eliminated if defined in TOMAS prior to this
         Rpk(k) = Dpk(k)/2.0    ! radius, m
         Dgk = Di               ! gas-phase diffusion coefficient, m2/s (constant? if yes, can we vary this based on information about the SOM species)
         accom = alpha          ! mass accommodation coefficient, unitless
         FC(k) = 0.75 * accom * (1.0+Kn) / (Kn*(1.0+Kn)
     &        + 0.238*accom*Kn + 0.75*accom) ! Fuchs correction, unitless, eqn (14)
C         FC(k) = (1.+Kn)/(1.+2.*Kn*(1.+Kn)/alpha) !S&P eqn 11.35         
C         Dbk = 1.0e-10          ! particle-phase diffusion coefficient [m2/s]
C         kc = 0.0               ! first-order loss rate of species in the particle phase [1/s]

! Zaveri et al. (2014) specific properties
         kgk(k) = Dgk / Rpk(k) * FC(k) ! gas-side mass transfer coefficient [m/s]
         if (kc.eq.0.0) then
            Qkk = 1.0
            kpk(k) = 5.0 * Dbk / Rpk(k)
C            print*, 'jo     =', jo
C            print*, 'k      =', k
C            print*, 'density=', density
C            print*, 'xk(k)  =', xk(k)
C            print*, 'mp     =', mp
C            print*, 'Dgk    =', Dgk
C            print*, 'accom  =', accom
C            print*, 'FC(k)  =', FC(k)
C            print*, 'kc     =', kc
C            print*, 'Dbk    =', Dbk
C            print*, 'kgk(k) =', kgk(k)
C            print*, 'Dpk(k) =', Dpk(k)
C            print*, 'Rpk(k) =', Rpk(k)
c            print*, 'kpk(k) =', kpk(k)
C            print*, 'Cstar  =', cstar(jo)
C            print*, 'mworg  =', mworg(jo)
C            print*, 'mfp    =', mfp
         else
            qk(k) = Rpk(k) * (kc/Dbk)**0.5 ! diffuso-reactive term, unitless
            coth(k) = 1.0/(tanh(qk(k))) ! hyperbolic cotangent
            Qkk = 3.0 * (qk(k) * coth(k) - 1.0) / qk(k)**2.0
            ! ratio of particle concentration to surface concentration at steady state [unitless]
            if (Qkk.gt.0.99) then
               Qkk = 1.0
               kpk(k) = 5.0 * Dbk / Rpk(k)
            else
               kpk(k) = Dbk / Rpk(k) * (qk(k) * coth(k) - 1.0) /
     &              (1.0-Qkk)
               ! particle-side mass transfer coefficient [m/s]
            end if
         end if
         
         Kgkk(k) = Rpk(k) * 1.0 / (1.0/kgk(k) + 1.0/kpk(k) * ! ???????????????????????????????????
     &        (cstar(jo)/density/1.0e9))
CCCCC ============================================================================================
!     AliA - calculate Kgk
         
Ckpc  adjust beta, tk, sK and alpha for all species
         beta(k)=(1.+Kn)/(1.+2.*Kn*(1.+Kn)/alpha) !S&P eqn 11.35
C         print*, 'Kgkk(k) =', Kgkk(k)
C         print*, 'beta(k)*Dgk =', beta(k)*Dgk
C         print*, '**************************************'

!AliA ------------------------------------------------------------------------            
CAliA         beta(k) = 0.75 * alpha * (1.0+Kn) / (Kn*(1.0+Kn) +
CAliA     &        0.238*alpha*Kn + 0.75*alpha) ! Fuchs correction, unitless, eqn (14)
!     AliA ------------------------------------------------------------------------
CAliA         tk(k)=(6./(pi*density))**(1./3.)*beta(k)
         tk(k)=(6./(pi*density))**(1./3.)*Kgkk(k)
         if (Nkf(k) .gt. 0.0) then
Ckpc  include all species here***
!            sK=sK+tk(k)*
!     &         Nkf(k)*(mp)**(1.d0/3.d0)
CAliA            sK=sK+Dpk(k)*Nkf(k)*beta(k)
            sK=sK+Dpk(k)*Nkf(k)*Kgkk(k)
         endif
      enddo
      do k=1,ibins
!         sinkfrac(k)=tk(k)*Nkf(k)*mps(k)**(1.d0/3.d0)/sK
         sinkfrac(k)=Dpk(k)*Nkf(k)*kgkk(k)/sK
      enddo
!      sK=sK*zeta13*tj*R*temp/(mworg(jo)*1.d-3)/(boxvol*1.d-6)
      sK = 2.d0*pi*sK/(dble(boxvol)*1D-6)
!      print*,'beta org',beta
!      print*,'CS org',sK
!	  print*,'j=',j
!      print*,'sinkfrac',sinkfrac
       tau_p = 1.0/sK

C Set dp for nonvolatile species
      do k=1,ibins
         totphase=0.d0
         do jj=1,icomp
            if ((jj.ge.srtorg1).and.(jj.lt.srtorg1+iorg))then
               totphase=totphase+Mkf(k,jj)
            else
               totphase=totphase+Mkf(k,jj)*nonorgscale
            endif
         enddo

C         scalefactor=exp((storg*0.2)/(8.314*temp*1200.*Dpk(k)))  !! kelvin effect?
C ORIGINAL (missing factor of 4 — see Seinfeld & Pandis 2016 eqn 9.40):
C      scalefactor=exp((storg*0.001*mworg(j))/(8.314*temp*1200.*Dpk(k)))
C CORRECTED by Akherati & Claude (2026-03-17): Ke = exp(4*sigma*MW/(R*T*rho*Dp))
       scalefactor=exp((4.d0*storg*0.001*mworg(j))/
     &      (8.314*temp*1200.*Dpk(k)))
C         print *, 'scalefactor=',scalefactor
         pamb=(Gcf(j)/mworg(jo))/(boxmass/28.9)*pres 
C         print *, 'mworg=',mworg(jo)		 
                       ! ambient pressure (Pa)
         if (totphase.gt.0.d0) then
            psat=psatorg(jo)*Mkf(k,j)/totphase*scalefactor 
            ! equillibrium pressure (Pa)
            ! VBS is MASS FRACTION BASED!!!
            ! If we want to use real EQM pressure, then must be mole based!
         else
            psat=0.d0
         endif
         dp(k)=pamb-psat
!		 print*,Gcf(j),mworg(jo)
		 

         !!! do not let evaporate for now !!!
         !!! There are big issues with evaporation for multiple species with multiple size bins!
         !!! I need to find a way to limit the timestep so that species don't entirely evaporate,
         !!! and the system does not get overly stiff
!         if (dp(k).lt.0.d0) then
!            dp(k) = 0.d0
!         endif
         
         ! Determine if equillibrium amount of mass of this species exists 
         ! (holding all else constant).  What mass is it?
         ! For now assume surface tension scalefactor is constant
         ! Equillibrium occurs when dp(k) = 0
        if (totphase .EQ. 0.0) then
		 masseqm = 0.0
		 masscheqm = 0.0
		 maddEQ(k) = 0.0
       else
         masseqm=pamb/(psatorg(jo)/totphase*scalefactor)
         masscheqm=masseqm-Mkf(k,j) ! change in mass to reach eqm
         maddEQ(k)=masscheqm/Nkf(k)
       endif
!         print*, j,k,dp(k),Mkf(k,j),masseqm,masscheqm,maddEQ(k),Nkf(k)
!         if ((jo.eq.5).and.(k.eq.15)) then
!		    print*,'test',k,pamb,psat,masseqm,Mkf(k,j)/Nkf(k),maddEQ(k)
!	     endif
      enddo

C Choose appropriate time step
C      print *,'dt=',dt,'cdt=',cdt,'time=',time
      !Try to take full time step
      cdt=dt-time

      !Not more than 15 minutes
      if (cdt .gt. 900.) then
         cdt=900.
cdbg         limit='max'
      endif
!      print*,'cdt',cdt,'sK=',sK
!      print *, exp(-1.d0*sK*cdt)
      !First, limit timestep by changing gas concentration
      if (exp(-1.d0*sK*cdt) .lt. 0.01d0) then
         cdt=cdt/(-2.d0*cdt*sK/log(0.01d0))

      endif
!      print*, 'j',j,'cdt',cdt

      do k=1,ibins
         !!! NAN CHECK !!!
         if (isnan(sK)) then
            print*,'TAU sK in soacond'
            print*,'j=',j
            print*,'time=',time
            print*,'sK=',sK
            print*,'tk=',tk
            print*,'tj=',tj
            print*,'dp=',dp
            print*,'cdt=',cdt
            print*,'WR=',WR
            print*,'Nk=',Nk
            print*,'Mk=',Mk
            stop
         endif
         if (sK .gt. 0.0) then
               atau(k)=tj*tk(k)*dp(k)/sK
     &             *(1.d0-exp(-1.d0*sK*cdt))*WR(k)
         else
           !!nothing to condense onto
            atau(k)=0.0
         endif

         mc=0.0d0
         do jj=1,icomp
            mc=mc+Mkf(k,jj)/Nkf(k)
         enddo
         if (atau(k)/1.5d0+mc**tdt .ge. 0.d0) then
            madd(k)=((atau(k)/1.5d0+mc**tdt)**(1.d0/tdt)-mc)/WR(k)
         else
            madd(k)=maddEQ(k)

         endif
		 
!		 if ((jo.eq.5).and.(k.eq.15)) then
!		    print*,'test2',k,atau(k),madd(k),mc,WR(k)
!	     endif

!         print*,'k',k,'madd',madd(k),'maddEQ',maddEQ(k)
!         print*,'atau',atau(k),'mc',mc,'WR',WR(k)
!         print*,'Nkf',Nkf(k)
!         print*,'Mkf',Mkf(k,:)

         if (Nkf(k).lt.Neps)then
            atau(k)=0.d0
            madd(k)=0.d0
         else
            if((((maddEQ(k).gt.0.d0).and.(madd(k)).lt.0.d0).or.
     &       ((maddEQ(k).lt.0.d0).and.(madd(k).gt.0.d0))).and.
     &       (dp(k)/pamb.gt.1.d-4))then
               print*,'maddEQ and madd have opposite signs in 
     &h2oconda'
               print*,'k',k,'j',j
               print*,'maddEQ',maddEQ(k),'madd',madd(k)
               print*,'dp',dp(k),'pamb',pamb
c               STOP
            endif
            if (madd(k)/maddEQ(k).gt.1E-30)then
                                                     ! 1st order model
                                                     ! 2nd condition
                                                     ! tests for NAN
               fodc=1.d0/(cdt*(maddEQ(k)/madd(k))) ! first
                   ! order decay constant towards equil
               madd(k)=maddEQ(k)*(1.d0-exp(-fodc*cdt)) ! change
                         ! to new mass to add
c               print*,'k',k,'atau1',atau(k,srth2o)
               atau(k)=1.5d0*((mc+madd(k)*WR(k))
     &              **(2.d0/3.d0)-mc**(2.d0/3.d0)) ! find new tau
                             !value
c               print*,'k',k,'atau2',atau(k,srth2o)
            endif
         endif

!         print*,'k',k,'madd',madd(k),'atau',atau(k)

!         if (-madd(k).gt.Mkf(k,j)/Nkf(k)*1.000001)then  !original 
          if ((-madd(k)-Mkf(k,j)/Nkf(k)) .GT. 1.00d-17) then !modified by Emily
            print*,'too much evaporation in SOACOND!'
            print*,'k',k,'j',j
            print*,'madd',madd(k)
            print*,'mp',Mkf(k,j)/Nkf(k)
            print*,'Mk',Mkf(k,j),'NKf',Nkf(k)
            stop
         endif

      enddo

! 20   continue   !if time step is shortened, repeat from here
!
!      !Calculate tau values for all species/bins
!      do k=1,ibins
!         !!! NAN CHECK !!!
!         if (isnan(sK)) then
!            print*,'TAU sK in soacond'
!            print*,'j=',j
!            print*,'time=',time
!            print*,'sK=',sK
!            print*,'tk=',tk
!            print*,'tj=',tj
!            print*,'dp=',dp
!            print*,'cdt=',cdt
!            print*,'WR=',WR
!            print*,'Nk=',Nk
!            print*,'Mk=',Mk
!            stop
!         endif
!         if (sK .gt. 0.0) then
!               atau(k)=tj*tk(k)*dp(k)/sK
!     &             *(1.d0-exp(-1.d0*sK*cdt))*WR(k)
!         else
!           !nothing to condense onto
!            atau(k)=0.0
!         endif
!      enddo
!
!      !The following sections limit the condensation time step
!      !when necessary.  tr is a factor that describes by
!      !how much to reduce the time step.
!      tr=1.0
!
!      !Make mass doesn't change too much
!      do k=1,ibins
!         if (Nkf(k) .gt. Neps) then
!            mc=0.0d0
!            do jj=1,icomp
!               mc=mc+Mkf(k,jj)/Nkf(k)
!            enddo
!            totmass=0.d0
!            do jj=1,icomp-idiag
!               totmass=totmass+Mkf(k,jj)
!            enddo
!            !if (mc/xk(k) .gt. 1.0d-3) then
!            !if (Mkf(k,j)/totmass .gt. 1.0d-3) then
!            if (Mkf(k,j)/totmass .gt. 1.0d-3) then
!               !species has significant mass in particle - limit change
!               if (abs(atau(k))/mc**(2./3.) .gt. 0.1) then
!                  ttr=abs(atau(k))/mc**(2./3.)/0.05
!               !if (abs(atau(k))/(Mkf(k,j)/Nk(k))**(2./3.) .gt. 0.1) then
!               !   ttr=abs(atau(k))/(Mkf(k,j)/Nk(k))**(2./3.)/0.05
!                  if (ttr . gt. tr) then 
!                     tr=ttr
!                  endif
!               endif
!               !! limit timestep for evaporation so species doesn't go negative!
!               madd(k)=((atau(k)/1.5d0+mc**tdt)**(1.d0/tdt)-mc)/WR(k)
!               if (madd(k).lt.0.d0) then
!                  !print*,'test',madd(k)
!                  if (abs(madd(k)).gt.(Mkf(k,j)/Nk(k))*1.d0)then
!                  !   print*, 'limit atau'
!                  !   print*,atau(k)
!                     atau(k)=1.5d0*((mc-(Mkf(k,j)/Nk(k))*WR(k))**tdt
!     &                         -mc**tdt)
!                  !   print*,atau(k)
!                  elseif (abs(madd(k)).gt.(Mkf(k,j)/Nk(k))*0.5d0)then
!                     !print*, 'limit madd'
!                     !print*,j,k,madd(k),Mkf(k,j)/Nk(k)*0.5d0
!                     ttr=abs(madd(k))/((Mkf(k,j)/Nk(k))*0.5d0) ! don't let more than half of the mass evaporate
!                     if (ttr . gt. tr) then 
!                        tr=ttr
!                     endif
!                  endif
!               endif
!            else
!               !species is new to particle - set max time step
!               if ((cdt/tr .gt. 0.1) .and. (atau(k).gt. 0.0)) then 
!                  tr=cdt/0.1
!                     !tr=cdt/0.01
!                     !print*,'limit nspec' 
!                     !print*,'limit amass'
!                     !print*,'k',k,'j',j
!                     !print*,'atau',atau(k,j)
!                     !print*,'Mk',Mkf(k,j)
!               elseif (atau(k).lt. 0.d0) then
!                  atau(k)=0.d0
!               endif
!            endif
!         endif
!      enddo
!      !Make sure gas phase concentrations don't change too much
!      if (exp(-1.d0*sK*cdt) .lt. 0.25) then
!         ttr=-2.d0*cdt*sK/log(0.25)
!         if (ttr .gt. tr) then 
!            print*,'limit gphas'
!            tr=ttr
!         endif
!      endif
!
!      !Never shorten timestep by less than half
!      if (tr .gt. 1.d0) tr=max(tr,2.d0)
!
!      !Repeat for shorter time step if necessary
!      if (tr .gt. 1.0) then
!         cdt=cdt/tr
!         !print*,'cdt',cdt
!         goto 20
!      endif

C Call condensation subroutine to do mass transfer


       !Swap tau values for this species into array for cond
      do k=1,ibins
         tau(k)=atau(k)
         !!! NAN CHECK !!!
         if (isnan(tau(k))) then
            print*,'TAU NaN in soacond'
            print*,'j=',j,'k=',k
            print*,'time=',time
            print*,'tau=',atau
            print*,'sK=',sK
            print*,'tk=',tk
            print*,'tj=',tj
            print*,'dp=',dp
            print*,'cdt=',cdt
            print*,'WR=',WR
            print*,'Nk=',Nk
            print*,'Mk=',Mk
            stop
         endif
      enddo


      !Call condensation routine
      Ntotf=0.0
      do k=1,ibins
         Ntotf=Ntotf+Nkf(k)
         !!! NAN CHECK !!!
         if (isnan(Nkf(k))) then
            print*,'Nk NaN in soacond before tmcond'
            print*,'j=',j,'k=',k
            print*,'time=',time
            print*,'tau=',atau
            print*,'sK=',sK
            print*,'tk=',tk
            print*,'tj=',tj
            print*,'dp=',dp
            print*,'cdt=',cdt
            print*,'WR=',WR
            print*,'Nkf=',Nkf
            print*,'Mkf=',Mkf
            stop
         endif
      enddo
      call tmcond(tau,xk,Mkf,Nkf,Mko,Nko,j,zeros)
 !     call tmcond(tau,xk,Mkf,Nkf,Mko,Nko,j,madd)

      !Check for number conservation
      Ntoto=0.0
      do k=1,ibins
         Ntoto=Ntoto+Nko(k)
         !!! NAN CHECK !!!
         if (isnan(Nkf(k))) then
            print*,'Nk NaN in soacond after tmcond'
            print*,'j=',j,'k=',k
            print*,'time=',time
            print*,'tau=',atau
            print*,'sK=',sK
            print*,'tk=',tk
            print*,'tj=',tj
            print*,'dp=',dp
            print*,'cdt=',cdt
            print*,'WR=',WR
            print*,'Nkf=',Nkf
            print*,'Mkf=',Mkf
            stop
         endif
      enddo
         
      dNerr=dNerr+Ntotf-Ntoto
      
      if (abs(dNerr/Ntoto) .gt. 1.e-4) then
         write(*,*) 'ERROR in soacond: Number not conserved'
         write(*,*) 'time=',time
         write(*,*) 'j=', j
         write(*,*) Ntoto, Ntotf
         write(*,*) (Nkf(k),k=1,ibins)
      endif

      do k=1,ibins
         if (Nko(k).lt.0.d0) then
            print*, 'Nk < 0 in soacond'
            print*, 'k=',k,'j=',j
            print*,'Nk=',Nko(k)
            print*,'Mk=',Mko(k,:)
            stop
         endif
         do jj=1,icomp
            if (Mko(k,jj).lt.-1.0d-10) then
               print*, 'Mk < 0 in soacond'
               print*, 'k=',k,'j=',j,'jj=',jj
               print*,'Nkf=',Nkf(k)
               print*,'Mkf=',Mkf(k,:)
               print*,'Nko=',Nko(k)
               print*,'Mko=',Mko(k,:)
               mc=0.0d0
               do jjj=1,icomp
                  mc=mc+Mkf(k,jjj)/Nkf(k)
               enddo
               print*,'mc',mc
               print*,'mp',Mkf(k,jj)/Nkf(k)
               print*,'madd1',madd(k)
               print*,'madd2',((atau(k)/1.5d0+mc**tdt)**(1.d0/tdt)-mc)
     &                         /WR(k)
               print*,'time=',time
               print*,'tau=',atau(k)
               print*,'sK=',sK
               print*,'tk=',tk
               print*,'tj=',tj
               print*,'dp=',dp
               print*,'cdt=',cdt
               print*,'WR=',WR
               stop
               !Mko(k,jj)=0.d0
            endif
         enddo
      enddo

      !Update gas phase concentration
      mi=0.0
      mf=0.0
      do k=1,ibins
         mi=mi+Mkf(k,j)
         mf=mf+Mko(k,j)
      enddo
      Gcfs=Gcf(j)
      Gcf(j)=Gcf(j)+(mi-mf)
      if (Gcf(j).lt.0.d0)then
         if ((abs(Gcf(j)/mi).lt.1d-8).or.(abs(Gcf(j)).lt.1d-8))then
            Gcf(j)=0.d0
         else
            print*,'soacond Gc < 0'
            print*,'j',j
            print*,Gcfs,Gcf(j),(mi-mf),mi,mf
            print*,'cdt',cdt
            print*,'tau',tau
            stop
         endif
      endif

      !Swap into Nkf, Mkf
      do k=1,ibins
         Nkf(k)=Nko(k)
         do jj=1,icomp
            Mkf(k,jj)=Mko(k,jj)
         enddo
      enddo

      !Update ammonia concentrations
      call eznh3eqm(Gcf,Mkf)

      !Update water concentrations
      call ezwatereqm(Mkf)

C Update time
      time=time+cdt
      
C Repeat process if necessary
C      if (time .lt. dt) goto 10

      test_time = time - dt
      if (abs(test_time).lt.1.00d-8) then
        test_time = 0.00d0
      endif
 
       if (test_time.lt.0.00d0) goto 10

      !!!!!!!!!!!!!!!!!!!!!!!!!!!!
      enddo ! loop over components
      !!!!!!!!!!!!!!!!!!!!!!!!!!!!
      
cdbg      write(*,*) 'Cond routine called ',numcalls,' times'
cdbg      write(*,*) 'Number cons. error was ', dNerr

 100  continue   !skip to here if there is no gas phase to condense
      
      RETURN
      END






