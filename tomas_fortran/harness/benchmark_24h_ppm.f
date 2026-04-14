C     **************************************************
C     *  TOMAS 24-Hour PPM Benchmark Harness           *
C     **************************************************
C
C     Same as benchmark_24h.f but uses PPM condensation (ezcond_ppm)
C     instead of TFL condensation (ezcond).
C
C     Reads 50 Latin Hypercube scenarios from scenarios.csv and runs
C     24-hour simulations with hourly output for three modes:
C       1. Coagulation-only (multicoag)     — same as TFL
C       2. Condensation-only (ezcond_ppm)   — PPM instead of TFL
C       3. Combined (multicoag + ezcond_ppm)
C
C     Output: CSV files in output/24h/ directory
C       ppm_s{01-50}_{coag|cond|combined}_hour{01-24}_{Nk|Mk|Gc}.csv

      PROGRAM benchmark_24h_ppm

      IMPLICIT NONE
      include 'sizecode.COM'

C-----VARIABLE DECLARATIONS-------------------------------------------
      integer k, j, istep, iscen, imode
      double precision dt
      parameter(dt=60.0d0)

      integer nscenarios, nsteps, nhours
      parameter(nscenarios=50, nsteps=1440, nhours=24)

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
      double precision Nkf(ibins), Mkf(ibins,icomp)
      double precision prod_rate_kg_s

      character*200 fname
      character*200 line
      integer ihour
      character*8 mode_name(3)
      data mode_name /'coag    ','cond    ','combined'/

C     Timing variables
      double precision t_start, t_end
      double precision scen_times(nscenarios, 3)

C-----READ SCENARIOS---------------------------------------------------

      write(*,*) '========================================'
      write(*,*) 'TOMAS 24-Hour PPM Benchmark Harness'
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

C-----MAIN LOOP: scenarios x modes-------------------------------------

      do iscen=1,nscenarios
         write(*,'(A,I3,A,I3)') ' Scenario ', iscen, '/', nscenarios

         do imode=1,3

            write(*,'(A,A8)') '   Mode: ', mode_name(imode)

C           --- Initialize state ---
            call initbounds()

            temp = scen_temp(iscen)
            pres = scen_pres(iscen)
            boxvol = 1.0d6
            rh = scen_RH(iscen)
            alpha = 1.0d0
            prod_rate_kg_s = scen_prod(iscen)

C           Clear arrays
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

C           Initialize lognormal distribution
            N_total = scen_N(iscen)
            Dp_gmd = scen_GMD(iscen)
            sigma_gsd = scen_GSD(iscen)
            dens_init = 1770.0d0

            do k=1,ibins
               Dl = 1.0d6*((6.0d0*xk(k))/(dens_init*pi))**0.3333d0
               Dh = 1.0d6*((6.0d0*xk(k+1))/(dens_init*pi))
     &              **0.3333d0
               Dk_init = sqrt(Dl*Dh)
               np_init = (N_total*boxvol) /
     &              (sqrt(2.0d0*pi)*Dk_init*log(sigma_gsd)) *
     &              exp(-((log(Dk_init/Dp_gmd))**2.0d0 /
     &              (2.0d0*(log(sigma_gsd))**2.0d0))) * (Dh-Dl)
               Nk(k) = np_init
               Mk(k,srtso4) = np_init * sqrt(xk(k)) * sqrt(xk(k+1))
            enddo

C           Neps preprocessing
            do k=1,ibins
               if (Nk(k) .lt. Neps) then
                  Nk(k) = Neps
                  do j=1,icomp
                     Mk(k,j) = 0.d0
                  enddo
                  Mk(k,srtso4) = Neps*1.4d0*xk(k)
               endif
            enddo

C           --- Time stepping loop ---
            call cpu_time(t_start)
            do istep=1,nsteps

C              1. Coagulation (modes 1 and 3)
               if (imode .eq. 1 .or. imode .eq. 3) then
                  call multicoag(dt)
               endif

C              2. Condensation (modes 2 and 3) — PPM version
               if (imode .eq. 2 .or. imode .eq. 3) then

C                 Add H2SO4 production
                  Gc(srtso4) = Gc(srtso4) + prod_rate_kg_s * dt

C                 Condensation sink
                  call getCondSink(Nk, Mk, srtso4, CS, sinkfrac)

C                 Condensed mass
                  if (CS .gt. 1.0d-20 .and. Gc(srtso4) .gt. 0.d0)
     &            then
                     mcond_so4 = Gc(srtso4)*(1.d0 - exp(-CS*dt))
                     Gc(srtso4) = Gc(srtso4) - mcond_so4

C                    PPM Ezcond (instead of TFL ezcond)
                     call ezcond_ppm(Nk, Mk, mcond_so4, srtso4,
     &                    Nkf, Mkf)

C                    Copy back
                     do k=1,ibins
                        Nk(k) = Nkf(k)
                        do j=1,icomp
                           Mk(k,j) = Mkf(k,j)
                        enddo
                     enddo
                  elseif (Gc(srtso4) .gt. 0.d0) then
C                    CS too small - dump into first bin
                     Mk(1,srtso4) = Mk(1,srtso4) + Gc(srtso4)
                     Nk(1) = Nk(1) + Gc(srtso4)/
     &                    sqrt(xk(1)*xk(2))
                     Gc(srtso4) = 0.d0
                  endif

C                 Equilibria + MNFIX
                  call eznh3eqm(Gc, Mk)
                  call ezwatereqm(Mk)
                  call mnfix(Nk, Mk)
               endif

C              --- Hourly output (ppm_ prefix) ---
               if (mod(istep, 60) .eq. 0) then
                  ihour = istep / 60

C                 Write Nk
                  write(fname,'(A,I2.2,A,A,A,I2.2,A)')
     &                 'output/24h/ppm_s',iscen,'_',
     &                 trim(mode_name(imode)),'_hour',
     &                 ihour,'_Nk.csv'
                  open(unit=10, file=fname, status='replace')
                  do k=1,ibins
                     write(10,'(E25.16)') Nk(k)
                  enddo
                  close(10)

C                 Write Mk
                  write(fname,'(A,I2.2,A,A,A,I2.2,A)')
     &                 'output/24h/ppm_s',iscen,'_',
     &                 trim(mode_name(imode)),'_hour',
     &                 ihour,'_Mk.csv'
                  open(unit=10, file=fname, status='replace')
                  do k=1,ibins
                     do j=1,icomp
                        if (j .lt. icomp) then
                           write(10,'(E25.16,A)',advance='no')
     &                          Mk(k,j), ','
                        else
                           write(10,'(E25.16)') Mk(k,j)
                        endif
                     enddo
                  enddo
                  close(10)

C                 Write Gc
                  write(fname,'(A,I2.2,A,A,A,I2.2,A)')
     &                 'output/24h/ppm_s',iscen,'_',
     &                 trim(mode_name(imode)),'_hour',
     &                 ihour,'_Gc.csv'
                  open(unit=10, file=fname, status='replace')
                  do j=1,icomp-1
                     write(10,'(E25.16)') Gc(j)
                  enddo
                  close(10)

               endif

            enddo  ! istep
            call cpu_time(t_end)
            scen_times(iscen, imode) = t_end - t_start

         enddo  ! imode

C        Write timing CSV incrementally (in case of crash later)
         if (iscen .eq. 1) then
            open(unit=30,file='output/24h/timing_fortran_ppm.csv',
     &           status='replace')
            write(30,'(A)') 'scenario_id,coag_s,cond_s,combined_s'
         else
            open(unit=30,file='output/24h/timing_fortran_ppm.csv',
     &           position='append')
         endif
         write(30,'(I3,A,F12.4,A,F12.4,A,F12.4)')
     &        iscen, ',',
     &        scen_times(iscen,1), ',',
     &        scen_times(iscen,2), ',',
     &        scen_times(iscen,3)
         close(30)

      enddo  ! iscen

      write(*,*) ''
      write(*,*) '========================================'
      write(*,*) '24-Hour PPM benchmark complete!'
      write(*,*) 'Output in output/24h/'
      write(*,*) 'Timing in output/24h/timing_fortran_ppm.csv'
      write(*,*) '========================================'

      END PROGRAM
