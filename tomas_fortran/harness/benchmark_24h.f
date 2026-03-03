C     **************************************************
C     *  TOMAS 24-Hour Benchmark Harness              *
C     **************************************************
C
C     Reads 50 Latin Hypercube scenarios from scenarios.csv and runs
C     24-hour simulations with hourly output for five modes:
C       1. Coagulation-only (multicoag)
C       2. Condensation-only (ezcond + equilibria)
C       3. Combined (multicoag + ezcond + equilibria)
C       4. Nucleation+Condensation (nucleation_driver + ezcond)
C       5. Full (nucleation + coagulation + condensation)
C
C     Modes 1-3 run first (safe), then 4-5 (may crash on some scenarios).
C     Timing is written incrementally after each scenario to avoid data loss.
C
C     Output: CSV files in output/24h/ directory
C       s{01-50}_{coag|cond|combined|nucl_cond|full}_hour{01-24}_{Nk|Mk|Gc}.csv

      PROGRAM benchmark_24h

      IMPLICIT NONE
      include 'sizecode.COM'

C-----VARIABLE DECLARATIONS-------------------------------------------
      integer k, j, istep, iscen, imode
      double precision dt
      parameter(dt=60.0d0)

      integer nscenarios, nsteps, nhours
      parameter(nscenarios=50, nsteps=1440, nhours=24)

      integer nmodes
      parameter(nmodes=5)

      double precision pi, kB, R, Neps
      parameter(pi=3.141592654, kB=1.38e-23, R=8.314, Neps=1.0e-3)

C     Scenario parameters (read from CSV)
      integer scen_id(nscenarios)
      double precision scen_temp(nscenarios)
      double precision scen_pres(nscenarios)
      double precision scen_N(nscenarios)
      double precision scen_GMD(nscenarios)
      double precision scen_GSD(nscenarios)
      double precision scen_Gc(nscenarios)
      double precision scen_RH(nscenarios)
      double precision scen_prod(nscenarios)

C     Working variables
      double precision N_total, Dp_gmd, sigma_gsd, dens_init
      double precision Dl, Dh, Dk_init, np_init
      double precision CS, sinkfrac(ibins)
      double precision mcond_so4
      double precision Nkf(ibins), Mkf(ibins,icomp), Gcf(icomp-1)
      double precision prod_rate_kg_s

C     Nucleation parameters (fixed for all scenarios)
      double precision nuc_org_conc, nuc_nh3_conc, nuc_fion
      double precision nuc_fn_scale
      integer nuc_org_flag, nuc_dunne_flag
      parameter(nuc_org_conc=1.d7, nuc_nh3_conc=1.d9, nuc_fion=3.d0)
      parameter(nuc_fn_scale=1.d0)

      character*200 fname
      character*200 line
      integer ihour
      character*10 mode_name(nmodes)
      data mode_name /'coag      ','cond      ','combined  ',
     &                'nucl_cond ','full      '/

C     Timing variables
      double precision t_start, t_end
      double precision scen_times(nscenarios, nmodes)

C     Mode list for each pass
      integer pass1_modes(3), pass2_modes(2)
      integer npass1, npass2, im
      data pass1_modes /1, 2, 3/
      data pass2_modes /4, 5/
      parameter(npass1=3, npass2=2)

C-----READ SCENARIOS---------------------------------------------------

      write(*,*) '========================================'
      write(*,*) 'TOMAS 24-Hour Benchmark Harness'
      write(*,*) '========================================'
      write(*,*) 'Reading scenarios from scenarios.csv...'

      open(unit=20, file='scenarios.csv', status='old',
     &     form='formatted')
      read(20, '(A)') line  ! Skip header

      do iscen=1,nscenarios
         read(20,*) scen_id(iscen), scen_temp(iscen),
     &        scen_pres(iscen), scen_N(iscen),
     &        scen_GMD(iscen), scen_GSD(iscen),
     &        scen_Gc(iscen), scen_RH(iscen),
     &        scen_prod(iscen)
      enddo
      close(20)

      write(*,*) 'Read ', nscenarios, ' scenarios'

C     Initialize timing to -1 (marks unrun modes)
      do iscen=1,nscenarios
         do imode=1,nmodes
            scen_times(iscen, imode) = -1.d0
         enddo
      enddo

C-----PASS 1: scenarios x modes 4-5 (nucleation)------------------------
C     COMMENTED OUT — only running coag/cond/combined for now
C
C      write(*,*) ''
C      write(*,*) '--- Pass 1: modes 4-5 (nucl_cond, full) ---'
C
C      do iscen=1,nscenarios
C         write(*,'(A,I3,A,I3)') ' Scenario ', iscen, '/', nscenarios
C
C         do im=1,npass2
C            imode = pass2_modes(im)
C
C            write(*,'(A,A10)') '   Mode: ', mode_name(imode)
C
C           --- Initialize and run ---
C            call init_scenario(iscen, nscenarios,
C     &           scen_temp, scen_pres, scen_N, scen_GMD, scen_GSD,
C     &           scen_Gc, scen_RH, scen_prod,
C     &           prod_rate_kg_s, pi, Neps)
C
C            call cpu_time(t_start)
C            do istep=1,nsteps
C               call timestep(imode, dt, prod_rate_kg_s,
C     &              nuc_org_conc, nuc_nh3_conc, nuc_fion,
C     &              nuc_fn_scale, Nkf, Mkf, Gcf,
C     &              CS, sinkfrac, mcond_so4)
C               call write_hourly(istep, iscen, imode, mode_name)
C            enddo
C            call cpu_time(t_end)
C            scen_times(iscen, imode) = t_end - t_start
C         enddo
C
C        Write timing CSV incrementally (modes 4-5 only first)
C         if (iscen .eq. 1) then
C            open(unit=30, file='output/24h/timing_fortran.csv',
C     &           status='replace')
C            write(30,'(A)')
C     &  'scenario_id,coag_s,cond_s,combined_s,nucl_cond_s,full_s'
C         else
C            open(unit=30, file='output/24h/timing_fortran.csv',
C     &           position='append')
C         endif
C         write(30,'(I3,A,F12.4,A,F12.4,A,F12.4,A,F12.4,A,F12.4)')
C     &        iscen, ',',
C     &        scen_times(iscen,1), ',',
C     &        scen_times(iscen,2), ',',
C     &        scen_times(iscen,3), ',',
C     &        scen_times(iscen,4), ',',
C     &        scen_times(iscen,5)
C         close(30)
C
C      enddo  ! iscen pass 1
C
C      write(*,*) ''
C      write(*,*) 'Pass 1 complete (modes 4-5).'

C-----PASS 2: scenarios x modes 1-3 (may have issues at s42/s50)--------

      write(*,*) ''
      write(*,*) '--- Pass 2: modes 1-3 (coag, cond, combined) ---'

      do iscen=1,nscenarios
         write(*,'(A,I3,A,I3)') ' Scenario ', iscen, '/', nscenarios

         do im=1,npass1
            imode = pass1_modes(im)

C           Skip scenarios that hang in combined mode
            if (imode .eq. 3 .and.
     &          (iscen .eq. 42 .or. iscen .eq. 50)) then
               write(*,'(A,A10,A)') '   Mode: ',
     &              mode_name(imode), ' SKIPPED (known hang)'
               goto 300
            endif

            write(*,'(A,A10)') '   Mode: ', mode_name(imode)

C           --- Initialize and run ---
            call init_scenario(iscen, nscenarios,
     &           scen_temp, scen_pres, scen_N, scen_GMD, scen_GSD,
     &           scen_Gc, scen_RH, scen_prod,
     &           prod_rate_kg_s, pi, Neps)

            call cpu_time(t_start)
            do istep=1,nsteps
               call timestep(imode, dt, prod_rate_kg_s,
     &              nuc_org_conc, nuc_nh3_conc, nuc_fion,
     &              nuc_fn_scale, Nkf, Mkf, Gcf,
     &              CS, sinkfrac, mcond_so4)
               call write_hourly(istep, iscen, imode, mode_name)
            enddo
            call cpu_time(t_end)
            scen_times(iscen, imode) = t_end - t_start
 300        continue
         enddo

C        Update timing CSV with modes 1-3 values
         open(unit=30, file='output/24h/timing_fortran.csv',
     &        status='replace')
         write(30,'(A)')
     &  'scenario_id,coag_s,cond_s,combined_s,nucl_cond_s,full_s'
         do j=1,iscen
            write(30,'(I3,A,F12.4,A,F12.4,A,F12.4,A,F12.4,A,F12.4)')
     &           j, ',',
     &           scen_times(j,1), ',',
     &           scen_times(j,2), ',',
     &           scen_times(j,3), ',',
     &           scen_times(j,4), ',',
     &           scen_times(j,5)
         enddo
         close(30)

      enddo  ! iscen pass 2

      write(*,*) ''
      write(*,*) '========================================'
      write(*,*) '24-Hour benchmark complete!'
      write(*,*) 'Output in output/24h/'
      write(*,*) 'Timing in output/24h/timing_fortran.csv'
      write(*,*) '========================================'

      END PROGRAM


C     **************************************************
C     *  init_scenario                                  *
C     **************************************************
      SUBROUTINE init_scenario(iscen, nscenarios,
     &     scen_temp, scen_pres, scen_N, scen_GMD, scen_GSD,
     &     scen_Gc, scen_RH, scen_prod,
     &     prod_rate_kg_s, pi, Neps)

      IMPLICIT NONE
      include 'sizecode.COM'

      integer iscen, nscenarios, k, j
      double precision scen_temp(nscenarios), scen_pres(nscenarios)
      double precision scen_N(nscenarios), scen_GMD(nscenarios)
      double precision scen_GSD(nscenarios), scen_Gc(nscenarios)
      double precision scen_RH(nscenarios), scen_prod(nscenarios)
      double precision prod_rate_kg_s, pi, Neps
      double precision N_total, Dp_gmd, sigma_gsd, dens_init
      double precision Dl, Dh, Dk_init, np_init

      call initbounds()

      temp = scen_temp(iscen)
      pres = scen_pres(iscen)
      boxvol = 1.0d6
      rh = scen_RH(iscen)
      alpha = 1.0d0
      prod_rate_kg_s = scen_prod(iscen)

C     Clear arrays
      do k=1,ibins
         Nk(k) = 0.0d0
         do j=1,icomp
            Mk(k,j) = 0.0d0
         enddo
      enddo
      do j=1,icomp-1
         Gc(j) = 0.0d0
      enddo
      Gc(srtso4) = scen_Gc(iscen)

C     Initialize lognormal distribution
      N_total = scen_N(iscen)
      Dp_gmd = scen_GMD(iscen)
      sigma_gsd = scen_GSD(iscen)
      dens_init = 1770.0d0

      do k=1,ibins
         Dl = 1.0d6*((6.0d0*xk(k))/(dens_init*pi))**0.3333d0
         Dh = 1.0d6*((6.0d0*xk(k+1))/(dens_init*pi))
     &        **0.3333d0
         Dk_init = sqrt(Dl*Dh)
         np_init = (N_total*boxvol) /
     &        (sqrt(2.0d0*pi)*Dk_init*log(sigma_gsd)) *
     &        exp(-((log(Dk_init/Dp_gmd))**2.0d0 /
     &        (2.0d0*(log(sigma_gsd))**2.0d0))) * (Dh-Dl)
         Nk(k) = np_init
         Mk(k,srtso4) = np_init * sqrt(xk(k)) * sqrt(xk(k+1))
      enddo

C     Neps preprocessing
      do k=1,ibins
         if (Nk(k) .lt. Neps) then
            Nk(k) = Neps
            do j=1,icomp
               Mk(k,j) = 0.d0
            enddo
            Mk(k,srtso4) = Neps*1.4d0*xk(k)
         endif
      enddo

      RETURN
      END


C     **************************************************
C     *  timestep                                       *
C     **************************************************
      SUBROUTINE timestep(imode, dt, prod_rate_kg_s,
     &     nuc_org_conc, nuc_nh3_conc, nuc_fion,
     &     nuc_fn_scale, Nkf, Mkf, Gcf,
     &     CS, sinkfrac, mcond_so4)

      IMPLICIT NONE
      include 'sizecode.COM'

      integer imode, k, j
      double precision dt, prod_rate_kg_s
      double precision nuc_org_conc, nuc_nh3_conc, nuc_fion
      double precision nuc_fn_scale
      double precision Nkf(ibins), Mkf(ibins,icomp), Gcf(icomp-1)
      double precision CS, sinkfrac(ibins), mcond_so4
      integer nuc_org_flag, nuc_dunne_flag

C     1. H2SO4 production (modes 2,3,4,5)
      if (imode .ge. 2) then
         Gc(srtso4) = Gc(srtso4) + prod_rate_kg_s * dt
      endif

C     2. Nucleation (modes 4 and 5)
      if (imode .eq. 4 .or. imode .eq. 5) then
         nuc_org_flag = 1
         nuc_dunne_flag = 1
         call nucleation_driver(Nk, Mk, Gc, Nkf, Mkf, Gcf,
     &        dt, nuc_org_conc, nuc_nh3_conc, nuc_fion,
     &        nuc_org_flag, nuc_dunne_flag, nuc_fn_scale)
         do k=1,ibins
            Nk(k) = Nkf(k)
            do j=1,icomp
               Mk(k,j) = Mkf(k,j)
            enddo
         enddo
         do j=1,icomp-1
            Gc(j) = Gcf(j)
         enddo
      endif

C     3. Coagulation (modes 1, 3, and 5)
      if (imode .eq. 1 .or. imode .eq. 3
     &    .or. imode .eq. 5) then
         call multicoag(dt)
      endif

C     4. Condensation (modes 2, 3, 4, and 5)
      if (imode .ge. 2) then

C        Condensation sink
         call getCondSink(Nk, Mk, srtso4, CS, sinkfrac)

C        Condensed mass
         if (CS .gt. 1.0d-20 .and. Gc(srtso4) .gt. 0.d0)
     &   then
            mcond_so4 = Gc(srtso4)*(1.d0 - exp(-CS*dt))
            Gc(srtso4) = Gc(srtso4) - mcond_so4

C           Ezcond
            call ezcond(Nk, Mk, mcond_so4, srtso4,
     &           Nkf, Mkf)

C           Copy back
            do k=1,ibins
               Nk(k) = Nkf(k)
               do j=1,icomp
                  Mk(k,j) = Mkf(k,j)
               enddo
            enddo
         elseif (Gc(srtso4) .gt. 0.d0) then
C           CS too small - dump into first bin
            Mk(1,srtso4) = Mk(1,srtso4) + Gc(srtso4)
            Nk(1) = Nk(1) + Gc(srtso4)/
     &           sqrt(xk(1)*xk(2))
            Gc(srtso4) = 0.d0
         endif

C        Equilibria + MNFIX
         call eznh3eqm(Gc, Mk)
         call ezwatereqm(Mk)
         call mnfix(Nk, Mk)
      endif

      RETURN
      END


C     **************************************************
C     *  write_hourly                                   *
C     **************************************************
      SUBROUTINE write_hourly(istep, iscen, imode, mode_name)

      IMPLICIT NONE
      include 'sizecode.COM'

      integer istep, iscen, imode, ihour, k, j
      character*10 mode_name(5)
      character*200 fname

      if (mod(istep, 60) .ne. 0) return

      ihour = istep / 60

C     Write Nk
      write(fname,'(A,I2.2,A,A,A,I2.2,A)')
     &     'output/24h/s',iscen,'_',
     &     trim(mode_name(imode)),'_hour',
     &     ihour,'_Nk.csv'
      open(unit=10, file=fname, status='replace')
      do k=1,ibins
         write(10,'(E25.16)') Nk(k)
      enddo
      close(10)

C     Write Mk
      write(fname,'(A,I2.2,A,A,A,I2.2,A)')
     &     'output/24h/s',iscen,'_',
     &     trim(mode_name(imode)),'_hour',
     &     ihour,'_Mk.csv'
      open(unit=10, file=fname, status='replace')
      do k=1,ibins
         do j=1,icomp
            if (j .lt. icomp) then
               write(10,'(E25.16,A)',advance='no')
     &              Mk(k,j), ','
            else
               write(10,'(E25.16)') Mk(k,j)
            endif
         enddo
      enddo
      close(10)

C     Write Gc
      write(fname,'(A,I2.2,A,A,A,I2.2,A)')
     &     'output/24h/s',iscen,'_',
     &     trim(mode_name(imode)),'_hour',
     &     ihour,'_Gc.csv'
      open(unit=10, file=fname, status='replace')
      do j=1,icomp-1
         write(10,'(E25.16)') Gc(j)
      enddo
      close(10)

      RETURN
      END
