
C     **************************************************
C     *  ezcond                                        *
C     **************************************************

C     WRITTEN BY Jeff Pierce, May 2007

C     This subroutine takes a given amount of mass and condenses it
C     across the bins accordingly.  

C     ADD MORE HERE!

C-----INPUTS------------------------------------------------------------

C     Initial values of
C     =================

C     Nki(ibins) - number of particles per size bin in grid cell
C     Mki(ibins, icomp) - mass of a given species per size bin/grid cell [kg]
C     mcond - mass of species to condense [kg/grid cell]
C     spec - the number of the species to condense

C-----OUTPUTS-----------------------------------------------------------

C     Nkf, Mkf - same as above, but final values

      SUBROUTINE ezcond(Nki,Mki,mcondi,spec,Nkf,Mkf)

      IMPLICIT NONE

C-----INCLUDE FILES-----------------------------------------------------

      include 'sizecode.COM'

C-----ARGUMENT DECLARATIONS---------------------------------------------

      double precision Nki(ibins), Mki(ibins, icomp)
      double precision Nkf(ibins), Mkf(ibins, icomp)
      double precision mcondi
      integer spec

C-----VARIABLE DECLARATIONS---------------------------------------------

      integer i,j,k,c           ! counters
      double precision mcond
      double precision pi, R    ! pi and gas constant (J/mol K)
      double precision CS       ! condensation sink [s^-1]
      double precision sinkfrac(ibins+1) ! fraction of CS in size bin
      double precision Nk1(ibins), Mk1(ibins, icomp)
      double precision Nk2(ibins), Mk2(ibins, icomp)
      double precision madd     ! mass to add to each bin [kg]
      double precision maddp(ibins)    ! mass to add per particle [kg]
      double precision mconds ! mass to add per step [kg]
      integer nsteps            ! number of condensation steps necessary
      integer floor, ceil       ! the floor and ceiling (temporary)
      double precision eps     ! small number
      double precision tdt      !the value 2/3
      double precision mpo,mpw  !dry and "wet" mass of particle
      double precision WR       !wet ratio
      double precision tau(ibins) !driving force for condensation
      double precision totsinkfrac ! total sink fraction not including nuc bin
      double precision CSeps    ! lower limit for condensation sink
      double precision tot_m,tot_s    !total mass, total sulfate mass
      double precision ratio    ! used in mass correction
      double precision fracch(ibins,icomp)
      double precision totch
      double precision zeros(ibins)
      double precision tot_i,tot_f,tot_fa ! used for conservation of mass check

C     VARIABLE COMMENTS...

C-----EXTERNAL FUNCTIONS------------------------------------------------


C-----ADJUSTABLE PARAMETERS---------------------------------------------

      parameter(pi=3.141592654, R=8.314) !pi and gas constant (J/mol K)
      parameter(eps=1.d-40)
      parameter(CSeps=1.d-20)

C-----CODE--------------------------------------------------------------

      do k=1,ibins
         zeros(k) = 0.d0
      enddo

      tdt=2.d0/3.d0

      mcond=mcondi
!      print*,'what?'
      ! initialize variables
      do k=1,ibins
         Nk1(k)=Nki(k)
         do j=1,icomp
            Mk1(k,j)=Mki(k,j)
         enddo
      enddo

      call mnfix(Nk1,Mk1)

      ! get the sink fractions
      call getCondSink(Nk1,Mk1,spec,CS,sinkfrac) ! set Nnuc to zero for this calc

      ! make sure that condensation sink isn't too small
      if (CS.lt.CSeps) then     ! just make particles in first bin
         Mkf(1,spec) = Mk1(1,spec) + mcond
         Nkf(1) = Nk1(1) + mcond/sqrt(xk(1)*xk(2))
         do j=1,icomp
            if (icomp.ne.spec) then
               Mkf(1,j) = Mk1(1,j)
            endif
         enddo
         do k=2,ibins
            Nkf(k) = Nk1(k)
            do j=1,icomp
               Mkf(k,j) = Mk1(k,j)
            enddo
         enddo
!         print*,CS,CSeps
         return
      endif	
c      print*,'CS',CS
c      print*,'sinkfrac',sinkfrac
c      print*,'mcond',mcond

      ! determine how much mass to add to each size bin
      ! also determine how many condensation steps we need
      totsinkfrac = 0.d0
      do k=1,ibins
         totsinkfrac = totsinkfrac + sinkfrac(k) ! get sink frac total not including nuc bin
      enddo
      nsteps = 1
      do k=1,ibins
         if (sinkfrac(k).lt.1.0D-20)then
            madd = 0.d0
         else
            madd = mcond*sinkfrac(k)/totsinkfrac
         endif
         mpo=0.0
         do j=1,icomp-idiag
            mpo=mpo + Mk1(k,j)
         enddo
         floor = int(madd*0.00001/mpo)
         ceil = floor + 1
         nsteps = max(nsteps,ceil) ! don't let the mass increase by more than 10%
      enddo
!      print*,'nsteps',nsteps

      ! mass to condense each step
      mconds = mcond/nsteps

      ! do steps of condensation
      do i=1,nsteps
         if (i.ne.1) then
            call getCondSink(Nk1,Mk1,spec,
     &        CS,sinkfrac)      ! set Nnuc to zero for this calculation
            totsinkfrac = 0.d0
            do k=1,ibins
	         totsinkfrac = totsinkfrac + sinkfrac(k) ! get sink frac total not including nuc bin
	      enddo
         endif      
         
         tot_m=0.d0
         tot_s=0.d0
         do k=1,ibins
            do j=1,icomp-idiag
               tot_m = tot_m + Mk1(k,j)
               if (j.eq.spec) then
                  tot_s = tot_s + Mk1(k,j)
               endif
            enddo
         enddo

         if (mconds.gt.tot_m*1.0D-3) then
!            print*,'test1'
            do k=1,ibins
               mpo=0.0
               mpw=0.0
                                !WIN'S CODE MODIFICATION 6/19/06
                                !THIS MUST CHANGED WITH THE NEW dmdt_int.f
               do j=1,icomp-idiag
                  mpo = mpo+Mk1(k,j) !accumulate dry mass
               enddo
               do j=1,icomp
                  mpw = mpw+Mk1(k,j) ! have wet mass include amso4
               enddo
               WR = mpw/mpo     !WR = wet ratio = total mass/dry mass
               if (Nk1(k) .gt. 0.d0) then
                  maddp(k) = mconds*sinkfrac(k)/totsinkfrac/Nk1(k)
                  mpw=mpw/Nk1(k)
c     print*,'mpw',mpw,'maddp',maddp(k),'WR',WR
                  tau(k)=1.5d0*((mpw+maddp(k)*WR)**tdt-mpw**tdt) !added WR to moxid term (win, 5/15/06)
c     tau(k)=0.d0
c     maddp(k)=0.d0
               else
                                !nothing in this bin - set tau to zero
                  tau(k)=0.d0
                  maddp(k) = 0.d0
               endif
            enddo
c     print*,'tau',tau
            call mnfix(Nk1,Mk1)
                                ! do condensation
            !call tmcond(tau,xk,Mk1,Nk1,Mk2,Nk2,spec,maddp)
            call tmcond(tau,xk,Mk1,Nk1,Mk2,Nk2,spec,zeros)
C     jrp         totch=0.0
C     jrp         do k=1,ibins
C     jrp            do j=1,icomp
C     jrp               fracch(k,j)=(Mk2(k,j)-Mk1(k,j))
C     jrp               totch = totch + (Mk2(k,j)-Mk1(k,j))
C     jrp            enddo
C     jrp         enddo
c     print*,'fracch',fracch,'totch',totch
            

         elseif (mconds.gt.tot_s*1.0D-12) then
!            print*,'test2'
            do k=1,ibins
               if (Nk1(k) .gt. 0.d0) then
                  maddp(k) = mconds*sinkfrac(k)/totsinkfrac
               else
                  maddp(k) = 0.d0
               endif
               Mk2(k,spec)=Mk1(k,spec)+maddp(k)
               do j=1,icomp
                  if (j.ne.spec) then
                     Mk2(k,j)=Mk1(k,j)
                  endif
               enddo
               Nk2(k)=Nk1(k)
            enddo
            call mnfix(Nk2,Mk2)
         else ! do nothing
!            print*,'test3'
            mcond = 0.d0
            do k=1,ibins
               Nk2(k)=Nk1(k)
               do j=1,icomp
                  Mk2(k,j)=Mk1(k,j)
               enddo
            enddo
         endif
         if (i.ne.nsteps)then
            do k=1,ibins
               Nk1(k)=Nk2(k)
               do j=1,icomp
                  Mk1(k,j)=Mk2(k,j)
               enddo
            enddo            
         endif

      enddo

      do k=1,ibins
         Nkf(k)=Nk2(k)
         do j=1,icomp
            Mkf(k,j)=Mk2(k,j)
         enddo
      enddo

      ! check for conservation of mass
      tot_i = 0.d0
      tot_fa = mcond
      tot_f = 0.d0
      do k=1,ibins
         tot_i=tot_i+Mki(k,spec)
         tot_f=tot_f+Mkf(k,spec)
         tot_fa=tot_fa+Mki(k,spec)
      enddo
      if (mcond.gt.0.d0.and.
     &    abs((mcond-(tot_f-tot_i))/mcond).gt.0.d0) then
         if (abs((mcond-(tot_f-tot_i))/mcond).lt.1.d0) then
            ! do correction of mass
            ratio = (tot_f-tot_i)/mcond
            do k=1,ibins
               Mkf(k,spec)=Mki(k,spec)+
     &              (Mkf(k,spec)-Mki(k,spec))/ratio
            enddo
            call mnfix(Nkf,Mkf)
         else
            print*,'ERROR in ezcond'
            print*,'Condensation error',(mcond-(tot_f-tot_i))/mcond
            print*,'mcond',mcond,'change',tot_f-tot_i
            print*,'tot_i',tot_i,'tot_fa',tot_fa,'tot_f',tot_f
            print*,'Nki',Nki
            print*,'Nkf',Nkf
            print*,'Mki',Mki
            print*,'Mkf',Mkf
            STOP
         endif
      endif

Cjrp      if (abs(tot_f-tot_fa)/tot_i.gt.1.0D-8)then
Cjrp         print*,'No S conservation in ezcond.f'
Cjrp         print*,'initial',tot_fa
Cjrp         print*,'final',tot_f
Cjrp         print*,'mcond',mcond,'change',tot_f-tot_i
Cjrp         print*,'ERROR',(mcond-(tot_f-tot_i))/mcond
Cjrp      endif

      ! check for conservation of mass
      tot_i = 0.d0
      tot_f = 0.d0
      do k=1,ibins
         tot_i=tot_i+Mki(k,srtnh4)
         tot_f=tot_f+Mkf(k,srtnh4)
      enddo
      if (abs(tot_f-tot_i)/tot_i.gt.1.0D-8)then
         print*,'No N conservation in ezcond.f'
         print*,'initial',tot_i
         print*,'final',tot_f
      endif


      return
      end
