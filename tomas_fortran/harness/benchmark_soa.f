C     **************************************************
C     *  TOMAS SOA Benchmark Harness                   *
C     **************************************************
C
C     SOA-only benchmark: organic VBS condensation onto inert sulfate
C     particles.  No H2SO4 condensation — only soacond() + mnfix().
C
C     3 scenarios:
C       A. Pure condensation (288K, 1atm, high gas, low C*)
C       B. Mixed condensation/evaporation (270K, 800hPa, uniform gas)
C       C. Warm evaporative (310K, 1atm, low gas)
C
C     Each timestep: soacond() then mnfix().
C     Initial particles: pure sulfate lognormal + 10% organic seed.
C     Gc(srtso4) = 0 (no H2SO4 gas).
C
C     Output: hourly CSV files in output/soa/

      PROGRAM benchmark_soa

      IMPLICIT NONE
      include 'sizecode.COM'

C-----VARIABLE DECLARATIONS-------------------------------------------
      integer k, j, istep, iscen
      double precision dt
      parameter(dt=10.0d0)

      integer nscenarios, nsteps, nhours, snap_interval
      parameter(nscenarios=3, nsteps=8640, nhours=24, snap_interval=360)

      double precision pi, kBoltz, Rgas, Neps
      parameter(pi=3.141592654d0, kBoltz=1.38d-23)
      parameter(Rgas=8.314d0, Neps=1.0d-3)

      double precision dens_init
      parameter(dens_init=1770.0d0)

C     Scenario parameters
      double precision sc_temp(3), sc_pres(3), sc_rh(3)
      double precision sc_N(3), sc_GMD(3), sc_GSD(3)
      double precision sc_Gc_org(6,3)
      character*2 sc_label(3)

C     Working variables
      double precision N_total, Dl, Dh, Dk_init, np_init
      double precision gmd_um
      double precision Nkf(ibins), Mkf(ibins,icomp), Gcf(icomp-1)
      double precision tau_p, Dbk, kc

C     Timing
      double precision t_start, t_end, wall_time

C     File output
      character*200 fname
      integer ihour, iminute

C-----SCENARIO DEFINITIONS---------------------------------------------
      sc_label(1) = 'sA'
      sc_label(2) = 'sB'
      sc_label(3) = 'sC'

C     Scenario A: Pure condensation (surface, low C*, high gas)
      sc_temp(1) = 288.0d0
      sc_pres(1) = 101325.0d0
      sc_rh(1) = 0.5d0
      sc_N(1) = 1.0d4
      sc_GMD(1) = 50.0d-9
      sc_GSD(1) = 1.6d0
      sc_Gc_org(1,1) = 1.0d-10
      sc_Gc_org(2,1) = 1.0d-10
      sc_Gc_org(3,1) = 1.0d-9
      sc_Gc_org(4,1) = 3.0d-9
      sc_Gc_org(5,1) = 1.0d-8
      sc_Gc_org(6,1) = 2.0d-8

C     Scenario B: Mixed condensation/evaporation (cold, moderate)
      sc_temp(2) = 270.0d0
      sc_pres(2) = 80000.0d0
      sc_rh(2) = 0.3d0
      sc_N(2) = 5.0d3
      sc_GMD(2) = 80.0d-9
      sc_GSD(2) = 1.5d0
      sc_Gc_org(1,2) = 1.0d-10
      sc_Gc_org(2,2) = 1.0d-10
      sc_Gc_org(3,2) = 1.0d-9
      sc_Gc_org(4,2) = 3.0d-9
      sc_Gc_org(5,2) = 1.0d-8
      sc_Gc_org(6,2) = 2.0d-8

C     Scenario C: Warm evaporative (high T, low gas)
      sc_temp(3) = 310.0d0
      sc_pres(3) = 101325.0d0
      sc_rh(3) = 0.6d0
      sc_N(3) = 2.0d4
      sc_GMD(3) = 30.0d-9
      sc_GSD(3) = 1.8d0
      sc_Gc_org(1,3) = 2.0d-9
      sc_Gc_org(2,3) = 2.0d-9
      sc_Gc_org(3,3) = 2.0d-9
      sc_Gc_org(4,3) = 2.0d-9
      sc_Gc_org(5,3) = 2.0d-9
      sc_Gc_org(6,3) = 2.0d-9

C-----MAIN LOOP---------------------------------------------------------

      write(*,*) '========================================'
      write(*,*) 'TOMAS SOA Benchmark (SOA-only)'
      write(*,*) '  No H2SO4 condensation'
      write(*,*) '  soacond() + mnfix() per timestep'
      write(*,*) '========================================'

      do iscen=1,nscenarios

         write(*,'(A,A2,A,I1,A,I1)') ' Scenario ', sc_label(iscen),
     &        ' (', iscen, '/', nscenarios, ')'

C        --- Initialize bin boundaries ---
         call initbounds()

C        --- Set thermodynamic state ---
         temp = sc_temp(iscen)
         pres = sc_pres(iscen)
         boxvol = 1.0d6
         boxmass = pres * boxvol * 1.0d-6 * 0.0289d0 / (Rgas * temp)
         rh = sc_rh(iscen)
         alpha = 1.0d0

C        --- Initialize /org/ common block ---
         cstar(1) = 0.01d0
         cstar(2) = 0.1d0
         cstar(3) = 1.0d0
         cstar(4) = 10.0d0
         cstar(5) = 100.0d0
         cstar(6) = 1000.0d0

         do j=1,iorg
            mworg(j) = 200.0d0
         enddo

         Hvap(1) = 150.0d0
         Hvap(2) = 136.0d0
         Hvap(3) = 122.0d0
         Hvap(4) = 108.0d0
         Hvap(5) = 94.0d0
         Hvap(6) = 80.0d0

         storg = 0.025d0
         nonorgscale = 0.0d0

C        Zero out unused organic species (7-41)
         do j=7,iorg
            cstar(j) = 1.0d10
            Hvap(j) = 80.0d0
         enddo

C        Compute temperature-corrected psatorg (Clausius-Clapeyron)
C        Fixed: was using Rgas*298 instead of Rgas*temp
         do j=1,iorg
            psatorg(j) = (cstar(j)*1.0d-9) / (mworg(j)*1.0d-3)
     &                   * Rgas * temp
     &                   * exp((-Hvap(j)*1000.d0/Rgas)
     &                         * (1.d0/temp - 1.d0/298.d0))
         enddo

C        Dbk and kc (particle-phase diffusion and loss)
         Dbk = 1.0d-10
         kc = 0.0d0

C        --- Clear arrays ---
         do k=1,ibins
            Nk(k) = 0.0d0
            do j=1,icomp
               Mk(k,j) = 0.0d0
            enddo
         enddo
         do j=1,icomp-1
            Gc(j) = 0.0d0
         enddo

C        --- Initialize lognormal distribution (pure sulfate) ---
         N_total = sc_N(iscen)
         gmd_um = sc_GMD(iscen) * 1.0d6

         do k=1,ibins
            Dl = 1.0d6*((6.d0*xk(k))/(dens_init*pi))**0.3333d0
            Dh = 1.0d6*((6.d0*xk(k+1))/(dens_init*pi))**0.3333d0
            Dk_init = sqrt(Dl*Dh)
            np_init = (N_total*boxvol)
     &           / (sqrt(2.0d0*pi)*Dk_init*log(sc_GSD(iscen)))
     &           * exp(-((log(Dk_init/gmd_um))**2.d0
     &           / (2.d0*(log(sc_GSD(iscen)))**2.d0))) * (Dh-Dl)
            Nk(k) = np_init
            Mk(k,srtso4) = np_init * sqrt(xk(k)) * sqrt(xk(k+1))
         enddo

C        Seed particles with 10% organic mass (VBS bin 1) to enable
C        Raoult partitioning in soacond.
         do k=1,ibins
            Mk(k,srtorg1) = 0.1d0 * Mk(k,srtso4)
         enddo

C        Neps preprocessing
         do k=1,ibins
            if (Nk(k) .lt. Neps) then
               Nk(k) = Neps
               do j=1,icomp
                  Mk(k,j) = 0.d0
               enddo
               Mk(k,srtso4) = Neps * 1.4d0 * xk(k)
            endif
         enddo

C        --- Set gas-phase concentrations ---
C        NO H2SO4 gas — Gc(srtso4) stays at 0
C        Organic gas: directly in kg/cell
         do j=1,6
            Gc(srtorg1+j-1) = sc_Gc_org(j,iscen)
         enddo

C        Zero gas for unused organic species
         do j=7,iorg
            Gc(srtorg1+j-1) = 0.0d0
         enddo

C        --- Write initial state ---
         call write_soa_minute(0, sc_label(iscen))
         call write_soa_hourly(0, sc_label(iscen))

C        --- Time stepping loop ---
         call cpu_time(t_start)

         do istep=1,nsteps

C           SOA condensation only (no H2SO4 condensation)
            call soacond(Nk, Mk, Gc, Nkf, Mkf, Gcf, dt, tau_p,
     &           Dbk, kc)
            do k=1,ibins
               Nk(k) = Nkf(k)
               do j=1,icomp
                  Mk(k,j) = Mkf(k,j)
               enddo
            enddo
            do j=1,icomp-1
               Gc(j) = Gcf(j)
            enddo

C           MNFIX
            call mnfix(Nk, Mk)

C           Write Nk every minute (every 6 steps at dt=10s)
            if (mod(istep, 6) .eq. 0) then
               iminute = istep / 6
               call write_soa_minute(iminute, sc_label(iscen))
            endif

C           Write full state (Nk+Mk+Gc) every snap_interval minutes
            if (mod(istep, snap_interval) .eq. 0) then
               ihour = istep / snap_interval
               call write_soa_hourly(ihour, sc_label(iscen))
            endif

         enddo

         call cpu_time(t_end)
         wall_time = t_end - t_start
         write(*,'(A,F8.3,A)') '   Wall time: ', wall_time, ' s'

      enddo  ! iscen

      write(*,*) ''
      write(*,*) '========================================'
      write(*,*) 'SOA benchmark complete!'
      write(*,*) 'Output in output/soa/'
      write(*,*) '========================================'

      END PROGRAM


C     **************************************************
C     *  write_soa_minute — Nk only (for banana plots) *
C     **************************************************
      SUBROUTINE write_soa_minute(imin, label)

      IMPLICIT NONE
      include 'sizecode.COM'

      integer imin, k
      character*2 label
      character*200 fname

C     Write Nk only (Mk too large for every-minute output)
      write(fname,'(A,A2,A,I4.4,A)')
     &     'output/soa/',label,'_soa_min',imin,'_Nk.csv'
      open(unit=10, file=fname, status='replace')
      do k=1,ibins
         write(10,'(E25.16)') Nk(k)
      enddo
      close(10)

      RETURN
      END


C     **************************************************
C     *  write_soa_hourly                              *
C     **************************************************
      SUBROUTINE write_soa_hourly(ihour, label)

      IMPLICIT NONE
      include 'sizecode.COM'

      integer ihour, k, j
      character*2 label
      character*200 fname

C     Write Nk
      write(fname,'(A,A2,A,I2.2,A)')
     &     'output/soa/',label,'_soa_hour',ihour,'_Nk.csv'
      open(unit=10, file=fname, status='replace')
      do k=1,ibins
         write(10,'(E25.16)') Nk(k)
      enddo
      close(10)

C     Write Mk (all 44 species)
      write(fname,'(A,A2,A,I2.2,A)')
     &     'output/soa/',label,'_soa_hour',ihour,'_Mk.csv'
      open(unit=10, file=fname, status='replace')
      do k=1,ibins
         do j=1,icomp
            if (j .lt. icomp) then
               write(10,'(E25.16,A)',advance='no') Mk(k,j), ','
            else
               write(10,'(E25.16)') Mk(k,j)
            endif
         enddo
      enddo
      close(10)

C     Write Gc
      write(fname,'(A,A2,A,I2.2,A)')
     &     'output/soa/',label,'_soa_hour',ihour,'_Gc.csv'
      open(unit=10, file=fname, status='replace')
      do j=1,icomp-1
         write(10,'(E25.16)') Gc(j)
      enddo
      close(10)

      RETURN
      END
