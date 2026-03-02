C     **************************************************
C     *  TOMAS Benchmark Harness                       *
C     **************************************************
C
C     Standalone program that duplicates multicoag.f calculations
C     step-by-step and writes intermediate results to CSV files
C     for validation against the JAX implementation.
C
C     Levels 1-10: Coagulation benchmarks (single configuration)
C     Levels 11-13: Condensation benchmarks (5 test cases each)
C
C     Coagulation test configuration:
C       - temp=298K, pres=101325Pa, boxvol=1e6 cm^3
C       - Single lognormal mode: N=1e5 #/cm^3, Dp=100nm, sigma=1.6
C       - Pure sulfate aerosol

      PROGRAM benchmark

      IMPLICIT NONE
      include 'sizecode.COM'

C-----VARIABLE DECLARATIONS---------------------------------------------
      integer k, j, i, jj
      double precision dt
      double precision dNdt(ibins), dMdt(ibins,icomp-1)
      double precision xbar(ibins), phi(ibins), eff(ibins)
      double precision kij_mat(ibins,ibins)
      double precision Dpk(ibins), Dk(ibins), ck(ibins)
      double precision density_arr(ibins)
      double precision mu, mfp
      double precision Kn, mp, beta, density
      double precision aerodens
      external aerodens
      double precision orgmass, Mktot

      double precision k1m(icomp-1), k1mx(icomp-1), k1mx2(icomp-1)
      double precision k1mtot, k1mxtot
      double precision sk2mtot, sk2mxtot
      double precision sk2m(icomp-1), sk2mx(icomp-1), sk2mx2(icomp-1)
      double precision in_term
      double precision mtotal

      double precision zeta, dtlimit, itlimit
      double precision dts, tsum, tlimit
      double precision Neps

      double precision pi, kB, R
      parameter(zeta=1.0625, dtlimit=0.25, itlimit=10.)
      parameter(pi=3.141592654, kB=1.38e-23, R=8.314, Neps=1.0e-3)

C     Saved states for before/after comparisons
      double precision Nk_before(ibins), Mk_before(ibins,icomp)
      double precision Nk_after(ibins), Mk_after(ibins,icomp)

C     For lognormal initialization
      double precision N_total, Dp_gmd, sigma_gsd
      double precision Dl, Dh, Dk_init, np_init
      double precision dens_init

C     Variables for condensation levels (11-13)
      integer icase, ncases
      parameter(ncases=5)
      character*1 casech(5)
      data casech /'A','B','C','D','E'/
      double precision case_temp(5), case_pres(5)
      double precision case_N(5), case_Dp(5), case_sigma(5)
      double precision case_Gcso4(5), case_rh(5)
      data case_temp /298.d0, 298.d0, 298.d0, 240.d0, 310.d0/
      data case_pres /101325.d0,101325.d0,101325.d0,50000.d0,101325.d0/
      data case_N /1.d5, 1.d5, 1.d6, 1.d4, 500.d0/
      data case_Dp /0.1d0, 0.1d0, 0.02d0, 0.05d0, 0.3d0/
      data case_sigma /1.6d0, 1.6d0, 1.8d0, 1.4d0, 2.0d0/
      data case_Gcso4 /1.d-13, 1.d-11, 1.d-13, 1.d-14, 1.d-12/
      data case_rh /0.50d0, 0.50d0, 0.80d0, 0.30d0, 0.90d0/
      double precision Nkf(ibins), Mkf(ibins,icomp)
      double precision CS, sinkfrac(ibins)
      double precision Di_gas, ms_gas, mfp_gas
      double precision mcond_so4
      character*100 fname

C-----CODE--------------------------------------------------------------

      write(*,*) '========================================'
      write(*,*) 'TOMAS Coagulation Benchmark Harness'
      write(*,*) '========================================'

C     Set environment
      temp = 298.0d0
      pres = 101325.0d0
      boxvol = 1.0d6
      dt = 60.0d0

C     ==================== LEVEL 1: Bin Boundaries ====================
      write(*,*) 'Level 1: Computing bin boundaries...'
      call initbounds()

      open(unit=10, file='output/level01_xk.csv', status='replace')
      do k=1,ibins+1
         write(10,'(E25.16)') xk(k)
      enddo
      close(10)
      write(*,*) '  Written level01_xk.csv'

C     ==================== Initialize State ====================
C     Lognormal: N=1e5 #/cm^3, Dp=100nm=0.1um, sigma=1.6, pure sulfate
      N_total = 1.0d5
      Dp_gmd = 0.1d0      ! microns
      sigma_gsd = 1.6d0
      dens_init = 1770.0d0  ! kg/m^3

C     Clear arrays
      do k=1,ibins
         Nk(k) = 0.0d0
         do j=1,icomp
            Mk(k,j) = 0.0d0
         enddo
      enddo

C     Initialize using loginit-style calculation
      do k=1,ibins
         Dl = 1.0d6*((6.0d0*xk(k))/(dens_init*pi))**0.3333d0
         Dh = 1.0d6*((6.0d0*xk(k+1))/(dens_init*pi))**0.3333d0
         Dk_init = sqrt(Dl*Dh)

         np_init = (N_total*boxvol) /
     &        (sqrt(2.0d0*pi)*Dk_init*log(sigma_gsd)) *
     &        exp(-((log(Dk_init/Dp_gmd))**2.0d0 /
     &        (2.0d0*(log(sigma_gsd))**2.0d0))) * (Dh-Dl)

         Nk(k) = np_init
         Mk(k,srtso4) = np_init * sqrt(xk(k)) * sqrt(xk(k+1))
      enddo

C     Write initial state
      open(unit=10, file='output/initial_Nk.csv', status='replace')
      do k=1,ibins
         write(10,'(E25.16)') Nk(k)
      enddo
      close(10)

      open(unit=10, file='output/initial_Mk.csv', status='replace')
      do k=1,ibins
         do j=1,icomp
            if (j .lt. icomp) then
               write(10,'(E25.16,A)', advance='no') Mk(k,j), ','
            else
               write(10,'(E25.16)') Mk(k,j)
            endif
         enddo
      enddo
      close(10)

      open(unit=10, file='output/params.csv', status='replace')
      write(10,'(E25.16)') temp
      write(10,'(E25.16)') pres
      write(10,'(E25.16)') boxvol
      close(10)

C     ==================== LEVEL 2: Air Properties ====================
      write(*,*) 'Level 2: Computing air properties...'
      mu = 2.5277d-7 * temp**0.75302d0
      mfp = 2.0d0*mu/(pres*sqrt(8.0d0*0.0289d0/(pi*R*temp)))

      open(unit=10, file='output/level02_air.csv', status='replace')
      write(10,'(E25.16)') mu
      write(10,'(E25.16)') mfp
      close(10)
      write(*,*) '  mu=', mu, ' mfp=', mfp

C     ==================== Preprocessing (set empty bins) ====================
      do k=1,ibins
         if (Nk(k) .lt. Neps) then
            Nk(k) = Neps
            do j=1,icomp
               Mk(k,j) = 0.d0
            enddo
            Mk(k,srtso4) = Neps*1.4d0*xk(k)
         endif
      enddo

C     ==================== LEVEL 3: Density ====================
      write(*,*) 'Level 3: Computing density per bin...'
      open(unit=10, file='output/level03_density.csv', status='replace')
      do k=1,ibins
         orgmass = 0.d0
         do j=1,iorg
            orgmass = orgmass + Mk(k,srtorg1+j-1)
         enddo
         density = aerodens(Mk(k,srtso4)+orgmass, 0.d0,
     &        Mk(k,srtnh4), 0.d0, Mk(k,srth2o))
         density_arr(k) = density
         write(10,'(E25.16)') density
      enddo
      close(10)
      write(*,*) '  Written level03_density.csv'

C     ==================== LEVEL 4: Particle Properties ====================
      write(*,*) 'Level 4: Computing Dpk, Dk, ck, mp...'
      open(unit=10, file='output/level04_properties.csv',
     &     status='replace')
      write(10,'(A)') '# Dpk, Dk, ck, mp'
      do k=1,ibins
         orgmass = 0.d0
         do j=1,iorg
            orgmass = orgmass + Mk(k,srtorg1+j-1)
         enddo
         density = aerodens(Mk(k,srtso4)+orgmass, 0.d0,
     &        Mk(k,srtnh4), 0.d0, Mk(k,srth2o))

         Mktot = 0.d0
         do j=1,icomp
            Mktot = Mktot + Mk(k,j)
         enddo
         mp = Mktot / Nk(k)

         Dpk(k) = ((mp/density)*(6.d0/pi))**(0.333d0)
         Kn = 2.0d0*mfp/Dpk(k)
         Dk(k) = kB*temp/(3.0d0*pi*mu*Dpk(k))
     &        *((5.0d0+4.0d0*Kn+6.0d0*Kn**2+18.0d0*Kn**3)
     &        /(5.0d0-Kn+(8.0d0+pi)*Kn**2))
         ck(k) = sqrt(8.0d0*kB*temp/(pi*mp))

         write(10,'(4(E25.16,A))') Dpk(k),',',Dk(k),',',
     &        ck(k),',',mp,' '
      enddo
      close(10)
      write(*,*) '  Written level04_properties.csv'

C     ==================== LEVEL 5: Coagulation Kernel ====================
      write(*,*) 'Level 5: Computing kij matrix...'
      do i=1,ibins
         do j=1,ibins
            Kn = 4.0d0*(Dk(i)+Dk(j))
     &        /(sqrt(ck(i)**2+ck(j)**2)*(Dpk(i)+Dpk(j)))
            beta = (1.0d0+Kn)/(1.0d0+2.0d0*Kn*(1.0d0+Kn))
            kij_mat(i,j) = 2.0d0*pi*(Dpk(i)+Dpk(j))*(Dk(i)+Dk(j))
     &           *beta
            kij_mat(i,j) = kij_mat(i,j)*1.0d6/boxvol
         enddo
      enddo

      open(unit=10, file='output/level05_kij.csv', status='replace')
      do i=1,ibins
         do j=1,ibins
            if (j .lt. ibins) then
               write(10,'(E25.16,A)', advance='no') kij_mat(i,j), ','
            else
               write(10,'(E25.16)') kij_mat(i,j)
            endif
         enddo
      enddo
      close(10)
      write(*,*) '  Written level05_kij.csv'

C     ==================== LEVEL 6: xbar, phi, eff ====================
      write(*,*) 'Level 6: Computing xbar, phi, eff...'
      do k=1,ibins
         xbar(k) = 0.0d0
         do j=1,icomp-idiag
            xbar(k) = xbar(k) + Mk(k,j)/Nk(k)
         enddo
         eff(k) = 2.d0*Nk(k)/xk(k)*(2.d0-xbar(k)/xk(k))
         phi(k) = 2.d0*Nk(k)/xk(k)*(xbar(k)/xk(k)-1.d0)
         if (xbar(k) .lt. xk(k)) then
            eff(k) = 2.d0*Nk(k)/xk(k)
            phi(k) = 0.0d0
         else if (xbar(k) .gt. xk(k+1)) then
            phi(k) = 2.d0*Nk(k)/xk(k)
            eff(k) = 0.0d0
         endif
      enddo

      open(unit=10, file='output/level06_tfl.csv', status='replace')
      write(10,'(A)') '# xbar, phi, eff'
      do k=1,ibins
         write(10,'(3(E25.16,A))') xbar(k),',',phi(k),',',eff(k),' '
      enddo
      close(10)
      write(*,*) '  Written level06_tfl.csv'

C     ==================== LEVEL 7: dNdt, dMdt ====================
      write(*,*) 'Level 7: Computing dNdt, dMdt...'

      sk2mtot = 0.0d0
      sk2mxtot = 0.0d0
      do j=1,icomp-idiag
         sk2m(j) = 0.0d0
         sk2mx(j) = 0.0d0
         sk2mx2(j) = 0.0d0
      enddo

      do k=1,ibins
         do j=1,icomp-idiag
            k1m(j) = 0.0d0
            k1mx(j) = 0.0d0
            k1mx2(j) = 0.0d0
         enddo
         in_term = 0.0d0
         k1mtot = 0.0d0
         k1mxtot = 0.0d0

         do j=1,icomp-idiag
            if (k .gt. 1) then
               do i=1,k-1
                  k1m(j) = k1m(j) + kij_mat(k,i)*Mk(i,j)
                  k1mx(j) = k1mx(j) + kij_mat(k,i)*Mk(i,j)*xbar(i)
                  k1mx2(j) = k1mx2(j) +
     &                 kij_mat(k,i)*Mk(i,j)*xbar(i)**2
               enddo
            endif
            k1mtot = k1mtot + k1m(j)
            k1mxtot = k1mxtot + k1mx(j)
         enddo
         if (k .lt. ibins) then
            do i=k+1,ibins
               in_term = in_term + Nk(i)*kij_mat(k,i)
            enddo
         endif

         dNdt(k) =
     &        -kij_mat(k,k)*Nk(k)**2
     &        -phi(k)*k1mtot
     &        -zeta*(eff(k)-phi(k))/(2*xk(k))*k1mxtot
     &        -Nk(k)*in_term
         if (k .gt. 1) then
            dNdt(k) = dNdt(k) +
     &           0.5d0*kij_mat(k-1,k-1)*Nk(k-1)**2
     &           +phi(k-1)*sk2mtot
     &           +zeta*(eff(k-1)-phi(k-1))/(2*xk(k-1))*sk2mxtot
         endif

         do j=1,icomp-idiag
            dMdt(k,j) =
     &           +Nk(k)*k1m(j)
     &           -kij_mat(k,k)*Nk(k)*Mk(k,j)
     &           -Mk(k,j)*in_term
     &           -phi(k)*xk(k+1)*k1m(j)
     &           -0.5d0*zeta*eff(k)*k1mx(j)
     &           +zeta**3*(phi(k)-eff(k))/(2*xk(k))*k1mx2(j)
            if (k .gt. 1) then
               dMdt(k,j) = dMdt(k,j) +
     &              kij_mat(k-1,k-1)*Nk(k-1)*Mk(k-1,j)
     &              +phi(k-1)*xk(k)*sk2m(j)
     &              +0.5d0*zeta*eff(k-1)*sk2mx(j)
     &              -zeta**3*(phi(k-1)-eff(k-1))/(2*xk(k-1))*sk2mx2(j)
            endif
         enddo

         sk2mtot = k1mtot
         sk2mxtot = k1mxtot
         do j=1,icomp-idiag
            sk2m(j) = k1m(j)
            sk2mx(j) = k1mx(j)
            sk2mx2(j) = k1mx2(j)
         enddo
      enddo

      open(unit=10, file='output/level07_dNdt.csv', status='replace')
      do k=1,ibins
         write(10,'(E25.16)') dNdt(k)
      enddo
      close(10)

      open(unit=10, file='output/level07_dMdt.csv', status='replace')
      do k=1,ibins
         do j=1,icomp-idiag
            if (j .lt. icomp-idiag) then
               write(10,'(E25.16,A)', advance='no') dMdt(k,j), ','
            else
               write(10,'(E25.16)') dMdt(k,j)
            endif
         enddo
      enddo
      close(10)
      write(*,*) '  Written level07_dNdt.csv, level07_dMdt.csv'

C     ==================== LEVEL 8: MNFIX Test ====================
      write(*,*) 'Level 8: Testing MNFIX...'

C     Save state before MNFIX
      do k=1,ibins
         Nk_before(k) = Nk(k)
         do j=1,icomp
            Mk_before(k,j) = Mk(k,j)
         enddo
      enddo

C     Create artificially drifted state for MNFIX test
C     Shift some mass to make xbar drift outside bin boundaries
      do k=5,15
         Mk(k,srtso4) = Mk(k,srtso4) * 3.0d0
      enddo
      do k=20,25
         Nk(k) = Nk(k) * 3.0d0
      enddo

C     Write pre-MNFIX state
      open(unit=10, file='output/level08_pre_mnfix_Nk.csv',
     &     status='replace')
      do k=1,ibins
         write(10,'(E25.16)') Nk(k)
      enddo
      close(10)

      open(unit=10, file='output/level08_pre_mnfix_Mk.csv',
     &     status='replace')
      do k=1,ibins
         do j=1,icomp
            if (j .lt. icomp) then
               write(10,'(E25.16,A)', advance='no') Mk(k,j), ','
            else
               write(10,'(E25.16)') Mk(k,j)
            endif
         enddo
      enddo
      close(10)

C     Apply MNFIX
      call mnfix(Nk, Mk)

C     Write post-MNFIX state
      open(unit=10, file='output/level08_post_mnfix_Nk.csv',
     &     status='replace')
      do k=1,ibins
         write(10,'(E25.16)') Nk(k)
      enddo
      close(10)

      open(unit=10, file='output/level08_post_mnfix_Mk.csv',
     &     status='replace')
      do k=1,ibins
         do j=1,icomp
            if (j .lt. icomp) then
               write(10,'(E25.16,A)', advance='no') Mk(k,j), ','
            else
               write(10,'(E25.16)') Mk(k,j)
            endif
         enddo
      enddo
      close(10)
      write(*,*) '  Written level08 MNFIX files'

C     ==================== Restore state for levels 9-10 ====================
      do k=1,ibins
         Nk(k) = Nk_before(k)
         do j=1,icomp
            Mk(k,j) = Mk_before(k,j)
         enddo
      enddo

C     ==================== LEVEL 9: Single Euler Step ====================
      write(*,*) 'Level 9: Computing single Euler step...'

C     Compute adaptive dt (same logic as multicoag.f)
      dts = dt
      do k=1,ibins
         if (Nk(k) .gt. Neps) then
            if (dNdt(k) .lt. 0.0d0) tlimit = dtlimit
            if (dNdt(k) .gt. 0.0d0) tlimit = itlimit
            if (abs(dNdt(k)*dts) .gt. Nk(k)*tlimit) then
               dts = Nk(k)*tlimit/abs(dNdt(k))
            endif
            do j=1,icomp-idiag
               if (Mk(k,j) .eq. 0.d0) then
                  mtotal = 0.d0
                  do jj=1,icomp-idiag
                     mtotal = mtotal + Mk(k,jj)
                  enddo
                  Mk(k,j) = 1.d-10*mtotal
               endif
               if (abs(dMdt(k,j)*dts) .gt. Mk(k,j)*tlimit) then
                  mtotal = 0.d0
                  do jj=1,icomp-idiag
                     mtotal = mtotal + Mk(k,jj)
                  enddo
                  if ((Mk(k,j)/mtotal) .gt. 1.d-5) then
                     dts = Mk(k,j)*tlimit/abs(dMdt(k,j))
                  else
                     if (dMdt(k,j) .lt. 0.0d0) then
                        dMdt(k,j) = 0.0d0
                     endif
                  endif
               endif
            enddo
         else
            Nk(k) = Neps
            Mk(k,srtso4) = Neps*1.4d0*xk(k)
            if (dNdt(k) .lt. 0.0d0) dNdt(k) = 0.0d0
            do j=1,icomp-idiag
               if (dMdt(k,j) .lt. 0.0d0) dMdt(k,j) = 0.0d0
            enddo
         endif
      enddo

      write(*,*) '  Adaptive dt = ', dts

C     Apply single Euler step
      do k=1,ibins
         Nk(k) = Nk(k) + dNdt(k)*dts
         do j=1,icomp-idiag
            Mk(k,j) = Mk(k,j) + dMdt(k,j)*dts
         enddo
      enddo

      call mnfix(Nk, Mk)

      open(unit=10,file='output/level09_euler_Nk.csv',status='replace')
      do k=1,ibins
         write(10,'(E25.16)') Nk(k)
      enddo
      close(10)

      open(unit=10,file='output/level09_euler_Mk.csv',status='replace')
      do k=1,ibins
         do j=1,icomp
            if (j .lt. icomp) then
               write(10,'(E25.16,A)', advance='no') Mk(k,j), ','
            else
               write(10,'(E25.16)') Mk(k,j)
            endif
         enddo
      enddo
      close(10)

      open(unit=10, file='output/level09_dt.csv', status='replace')
      write(10,'(E25.16)') dts
      close(10)
      write(*,*) '  Written level09 Euler step files'

C     ==================== LEVEL 10: Full multicoag ====================
      write(*,*) 'Level 10: Running full multicoag(dt=60s)...'

C     Restore initial state
      do k=1,ibins
         Nk(k) = Nk_before(k)
         do j=1,icomp
            Mk(k,j) = Mk_before(k,j)
         enddo
      enddo

C     Run full multicoag
      call multicoag(dt)

      open(unit=10,file='output/level10_final_Nk.csv',status='replace')
      do k=1,ibins
         write(10,'(E25.16)') Nk(k)
      enddo
      close(10)

      open(unit=10,file='output/level10_final_Mk.csv',status='replace')
      do k=1,ibins
         do j=1,icomp
            if (j .lt. icomp) then
               write(10,'(E25.16,A)', advance='no') Mk(k,j), ','
            else
               write(10,'(E25.16)') Mk(k,j)
            endif
         enddo
      enddo
      close(10)
      write(*,*) '  Written level10 full coag files'

C     ==================================================================
C     CONDENSATION BENCHMARK LEVELS (11-13)
C     Each level runs 5 test cases with different physical conditions.
C     ==================================================================

C     ==================== LEVEL 11: Gas Properties + Cond Sink ====================
      write(*,*) ''
      write(*,*) 'Level 11: Gas Properties + Condensation Sink...'

      do icase=1,ncases

C        --- Initialize case ---
         temp = case_temp(icase)
         pres = case_pres(icase)
         boxvol = 1.0d6
         rh = case_rh(icase)
         alpha = 1.0d0

C        Clear arrays
         do k=1,ibins
            Nk(k) = 0.0d0
            do j=1,icomp
               Mk(k,j) = 0.0d0
            enddo
         enddo
         do j=1,icomp-1
            Gc(j) = 0.0d0
         enddo
         Gc(srtso4) = case_Gcso4(icase)

C        Initialize lognormal
         N_total = case_N(icase)
         Dp_gmd = case_Dp(icase)
         sigma_gsd = case_sigma(icase)
         dens_init = 1770.0d0

         call initbounds()

         do k=1,ibins
            Dl = 1.0d6*((6.0d0*xk(k))/(dens_init*pi))**0.3333d0
            Dh = 1.0d6*((6.0d0*xk(k+1))/(dens_init*pi))**0.3333d0
            Dk_init = sqrt(Dl*Dh)
            np_init = (N_total*boxvol) /
     &           (sqrt(2.0d0*pi)*Dk_init*log(sigma_gsd)) *
     &           exp(-((log(Dk_init/Dp_gmd))**2.0d0 /
     &           (2.0d0*(log(sigma_gsd))**2.0d0))) * (Dh-Dl)
            Nk(k) = np_init
            Mk(k,srtso4) = np_init * sqrt(xk(k)) * sqrt(xk(k+1))
         enddo

C        Neps preprocessing
         do k=1,ibins
            if (Nk(k) .lt. Neps) then
               Nk(k) = Neps
               do j=1,icomp
                  Mk(k,j) = 0.d0
               enddo
               Mk(k,srtso4) = Neps*1.4d0*xk(k)
            endif
         enddo

C        Gas diffusivity
         call gasdiff(temp, pres, 98.0d0, 42.88d0, Di_gas)
         ms_gas = sqrt(8.0d0*R*temp/(pi*0.098d0))
         mfp_gas = 2.0d0*Di_gas/ms_gas

C        Condensation sink
         call getCondSink(Nk, Mk, srtso4, CS, sinkfrac)

C        Write gas properties
         write(fname,'(A,A1,A)')
     &        'output/level11_case',casech(icase),'_gas_props.csv'
         open(unit=10, file=fname, status='replace')
         write(10,'(E25.16)') Di_gas
         write(10,'(E25.16)') ms_gas
         write(10,'(E25.16)') mfp_gas
         close(10)

C        Write condensation sink (CS + sinkfrac)
         write(fname,'(A,A1,A)')
     &        'output/level11_case',casech(icase),'_condsink.csv'
         open(unit=10, file=fname, status='replace')
         write(10,'(E25.16)') CS
         do k=1,ibins
            write(10,'(E25.16)') sinkfrac(k)
         enddo
         close(10)

         write(*,*) '  Case ', casech(icase), ' done'
      enddo
      write(*,*) '  Written level11 files'

C     ==================== LEVEL 12: Isolated Condensation ====================
      write(*,*) ''
      write(*,*) 'Level 12: Isolated Condensation...'

      do icase=1,ncases

C        --- Initialize case (fresh) ---
         temp = case_temp(icase)
         pres = case_pres(icase)
         boxvol = 1.0d6
         rh = case_rh(icase)
         alpha = 1.0d0

         do k=1,ibins
            Nk(k) = 0.0d0
            do j=1,icomp
               Mk(k,j) = 0.0d0
            enddo
         enddo
         do j=1,icomp-1
            Gc(j) = 0.0d0
         enddo
         Gc(srtso4) = case_Gcso4(icase)

         N_total = case_N(icase)
         Dp_gmd = case_Dp(icase)
         sigma_gsd = case_sigma(icase)
         dens_init = 1770.0d0

         call initbounds()

         do k=1,ibins
            Dl = 1.0d6*((6.0d0*xk(k))/(dens_init*pi))**0.3333d0
            Dh = 1.0d6*((6.0d0*xk(k+1))/(dens_init*pi))**0.3333d0
            Dk_init = sqrt(Dl*Dh)
            np_init = (N_total*boxvol) /
     &           (sqrt(2.0d0*pi)*Dk_init*log(sigma_gsd)) *
     &           exp(-((log(Dk_init/Dp_gmd))**2.0d0 /
     &           (2.0d0*(log(sigma_gsd))**2.0d0))) * (Dh-Dl)
            Nk(k) = np_init
            Mk(k,srtso4) = np_init * sqrt(xk(k)) * sqrt(xk(k+1))
         enddo

         do k=1,ibins
            if (Nk(k) .lt. Neps) then
               Nk(k) = Neps
               do j=1,icomp
                  Mk(k,j) = 0.d0
               enddo
               Mk(k,srtso4) = Neps*1.4d0*xk(k)
            endif
         enddo

C        Get condensation sink and compute mcond
         call getCondSink(Nk, Mk, srtso4, CS, sinkfrac)
         mcond_so4 = Gc(srtso4) * (1.d0 - exp(-CS*60.d0))

C        Write mcond and CS
         write(fname,'(A,A1,A)')
     &        'output/level12_case',casech(icase),'_mcond.csv'
         open(unit=10, file=fname, status='replace')
         write(10,'(E25.16)') mcond_so4
         write(10,'(E25.16)') CS
         close(10)

C        Run ezcond
         call ezcond(Nk, Mk, mcond_so4, srtso4, Nkf, Mkf)

C        Copy back to common block arrays for equilibrium routines
         do k=1,ibins
            Nk(k) = Nkf(k)
            do j=1,icomp
               Mk(k,j) = Mkf(k,j)
            enddo
         enddo

C        NH3 equilibrium (uses Gc from common block)
         call eznh3eqm(Gc, Mk)

C        Water equilibrium (uses rh from common block)
         call ezwatereqm(Mk)

C        MNFIX
         call mnfix(Nk, Mk)

C        Write final Nk
         write(fname,'(A,A1,A)')
     &        'output/level12_case',casech(icase),'_Nk.csv'
         open(unit=10, file=fname, status='replace')
         do k=1,ibins
            write(10,'(E25.16)') Nk(k)
         enddo
         close(10)

C        Write final Mk
         write(fname,'(A,A1,A)')
     &        'output/level12_case',casech(icase),'_Mk.csv'
         open(unit=10, file=fname, status='replace')
         do k=1,ibins
            do j=1,icomp
               if (j .lt. icomp) then
                  write(10,'(E25.16,A)', advance='no') Mk(k,j), ','
               else
                  write(10,'(E25.16)') Mk(k,j)
               endif
            enddo
         enddo
         close(10)

         write(*,*) '  Case ', casech(icase), ' done'
      enddo
      write(*,*) '  Written level12 files'

C     ==================== LEVEL 13: Combined Coag + Cond ====================
      write(*,*) ''
      write(*,*) 'Level 13: Combined Coagulation + Condensation...'

      do icase=1,ncases

C        --- Initialize case (fresh) ---
         temp = case_temp(icase)
         pres = case_pres(icase)
         boxvol = 1.0d6
         rh = case_rh(icase)
         alpha = 1.0d0

         do k=1,ibins
            Nk(k) = 0.0d0
            do j=1,icomp
               Mk(k,j) = 0.0d0
            enddo
         enddo
         do j=1,icomp-1
            Gc(j) = 0.0d0
         enddo
         Gc(srtso4) = case_Gcso4(icase)

         N_total = case_N(icase)
         Dp_gmd = case_Dp(icase)
         sigma_gsd = case_sigma(icase)
         dens_init = 1770.0d0

         call initbounds()

         do k=1,ibins
            Dl = 1.0d6*((6.0d0*xk(k))/(dens_init*pi))**0.3333d0
            Dh = 1.0d6*((6.0d0*xk(k+1))/(dens_init*pi))**0.3333d0
            Dk_init = sqrt(Dl*Dh)
            np_init = (N_total*boxvol) /
     &           (sqrt(2.0d0*pi)*Dk_init*log(sigma_gsd)) *
     &           exp(-((log(Dk_init/Dp_gmd))**2.0d0 /
     &           (2.0d0*(log(sigma_gsd))**2.0d0))) * (Dh-Dl)
            Nk(k) = np_init
            Mk(k,srtso4) = np_init * sqrt(xk(k)) * sqrt(xk(k+1))
         enddo

         do k=1,ibins
            if (Nk(k) .lt. Neps) then
               Nk(k) = Neps
               do j=1,icomp
                  Mk(k,j) = 0.d0
               enddo
               Mk(k,srtso4) = Neps*1.4d0*xk(k)
            endif
         enddo

C        Coagulation step (dt=60s)
         call multicoag(60.0d0)

C        Condensation step
         call getCondSink(Nk, Mk, srtso4, CS, sinkfrac)
         mcond_so4 = Gc(srtso4) * (1.d0 - exp(-CS*60.d0))
         Gc(srtso4) = Gc(srtso4) - mcond_so4

         call ezcond(Nk, Mk, mcond_so4, srtso4, Nkf, Mkf)

C        Copy back
         do k=1,ibins
            Nk(k) = Nkf(k)
            do j=1,icomp
               Mk(k,j) = Mkf(k,j)
            enddo
         enddo

C        Equilibria + MNFIX
         call eznh3eqm(Gc, Mk)
         call ezwatereqm(Mk)
         call mnfix(Nk, Mk)

C        Write final Nk
         write(fname,'(A,A1,A)')
     &        'output/level13_case',casech(icase),'_Nk.csv'
         open(unit=10, file=fname, status='replace')
         do k=1,ibins
            write(10,'(E25.16)') Nk(k)
         enddo
         close(10)

C        Write final Mk
         write(fname,'(A,A1,A)')
     &        'output/level13_case',casech(icase),'_Mk.csv'
         open(unit=10, file=fname, status='replace')
         do k=1,ibins
            do j=1,icomp
               if (j .lt. icomp) then
                  write(10,'(E25.16,A)', advance='no') Mk(k,j), ','
               else
                  write(10,'(E25.16)') Mk(k,j)
               endif
            enddo
         enddo
         close(10)

C        Write final Gc
         write(fname,'(A,A1,A)')
     &        'output/level13_case',casech(icase),'_Gc.csv'
         open(unit=10, file=fname, status='replace')
         do j=1,icomp-1
            write(10,'(E25.16)') Gc(j)
         enddo
         close(10)

         write(*,*) '  Case ', casech(icase), ' done'
      enddo
      write(*,*) '  Written level13 files'

      write(*,*) ''
      write(*,*) '========================================'
      write(*,*) 'Benchmark complete! Output in output/'
      write(*,*) '========================================'

      END PROGRAM
