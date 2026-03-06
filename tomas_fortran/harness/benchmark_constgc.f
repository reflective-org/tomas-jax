C     **************************************************
C     *  Constant-Gas Condensation/Combined Benchmark  *
C     **************************************************
C
C     Runs condensation-only or coag+cond with constant H2SO4 gas for 24h.
C     Tests multiple timestep values (20, 30, 60 seconds).
C     Uses standard TOMAS grid for compatibility with JAX convergence test.
C
C     Parameters are hardcoded (not read from CSV):
C       N=1000 #/cm3, GMD=0.02 um, GSD=1.6
C       T=298 K, P=101325 Pa, RH=0.30
C       H2SO4 = 1e7 molec/cm3 (held constant)
C
C     Output: CSV files in output/constgc/ directory
C       constgc_dt{20|30|60}_final_{Nk|Mk}.csv          (cond-only)
C       constgc_combined_dt{20|30|60}_final_{Nk|Mk}.csv (combined)
C
C     Set do_coag = .true. to enable coagulation before condensation.

      PROGRAM benchmark_constgc

      IMPLICIT NONE
      include 'sizecode.COM'

C-----VARIABLE DECLARATIONS-------------------------------------------
      integer k, j, istep, idt
      double precision dt

      integer ndt
      parameter(ndt=3)
      double precision dt_vals(ndt)
      data dt_vals /20.0d0, 30.0d0, 60.0d0/

      integer nsteps

      double precision pi, Neps
      parameter(pi=3.141592654d0, Neps=1.0d-3)

C     Physical parameters
      double precision temp_val, pres_val, rh_val
      double precision N_total, Dp_gmd, sigma_gsd, dens_init
      parameter(temp_val=298.0d0)
      parameter(pres_val=101325.0d0)
      parameter(rh_val=0.30d0)
      parameter(N_total=1.0d4)
      parameter(Dp_gmd=0.02d0)
      parameter(sigma_gsd=1.6d0)
      parameter(dens_init=1770.0d0)

C     H2SO4 gas: 1e7 molec/cm3 -> kg/cell
C     = 1e7 * 1e6 * (98/1000) / 6.022e23
      double precision h2so4_kg
      double precision MW_H2SO4, AVOGADRO, BOXVOL_VAL
      parameter(MW_H2SO4=98.0d0)
      parameter(AVOGADRO=6.02214076d23)
      parameter(BOXVOL_VAL=1.0d6)

C     Grid start (standard TOMAS via initbounds)
      double precision XK0_17NM

C     Working variables
      double precision Dl, Dh, Dk_init, np_init
      double precision CS, sinkfrac(ibins)
      double precision mcond_so4
      double precision Nkf(ibins), Mkf(ibins,icomp)
      double precision Gc_init

C     Coagulation toggle
      logical do_coag
      parameter(do_coag=.true.)

C     Timing
      double precision t_start, t_end

      character*200 fname
      character*20 prefix

C-----INITIALIZATION---------------------------------------------------

      if (do_coag) then
         write(*,*) '========================================'
         write(*,*) 'Constant-Gas Coag+Cond Benchmark'
         write(*,*) '========================================'
         prefix = 'constgc_combined'
      else
         write(*,*) '========================================'
         write(*,*) 'Constant-Gas Condensation Benchmark'
         write(*,*) '========================================'
         prefix = 'constgc'
      endif

C     Compute H2SO4 in kg/cell
      h2so4_kg = 1.0d7 * BOXVOL_VAL * (MW_H2SO4/1000.0d0) / AVOGADRO

C     Standard TOMAS grid: Mo = 1e-21 * 2^(-6) = 1.5625e-23
      XK0_17NM = 1.0d-21 * 2.0d0**(-6)
      write(*,'(A,E12.4,A)') '  XK0 = ', XK0_17NM, ' kg (standard)'
      write(*,'(A,E12.4,A)') '  H2SO4 = ', h2so4_kg, ' kg/cell'
      write(*,'(A,L1)') '  Coagulation: ', do_coag
      write(*,*) ''

C-----LOOP OVER DT VALUES---------------------------------------------

      do idt=1,ndt
         dt = dt_vals(idt)
         nsteps = nint(86400.0d0 / dt)

         write(*,'(A,F5.1,A,I6,A)')
     &        '--- dt = ', dt, 's, nsteps = ', nsteps, ' ---'

C        Initialize custom grid (1.7nm start, mass doubling)
         do k=1,ibins+1
            xk(k) = XK0_17NM * 2.0d0**(k-1)
         enddo

C        Set environment
         temp = temp_val
         pres = pres_val
         boxvol = BOXVOL_VAL
         rh = rh_val
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

C        Initialize lognormal distribution
         do k=1,ibins
            Dl = 1.0d6*((6.0d0*xk(k))/(dens_init*pi))**0.3333d0
            Dh = 1.0d6*((6.0d0*xk(k+1))/(dens_init*pi))
     &           **0.3333d0
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
                  Mk(k,j) = 0.0d0
               enddo
               Mk(k,srtso4) = Neps*1.4d0*xk(k)
            endif
         enddo

C        Set constant gas
         Gc(srtso4) = h2so4_kg
         Gc_init = h2so4_kg

C-----TIME LOOP (constant gas)----------------------------------------

         call cpu_time(t_start)

         do istep=1,nsteps

C           Reset gas to constant value
            Gc(srtso4) = Gc_init

C           Coagulation (before condensation, matching JAX operator split)
            if (do_coag) then
               call multicoag(dt)
               call mnfix(Nk, Mk)
            endif

C           Condensation sink
            call getCondSink(Nk, Mk, srtso4, CS, sinkfrac)

C           Condensed mass
            if (CS .gt. 1.0d-20 .and. Gc(srtso4) .gt. 0.d0) then
               mcond_so4 = Gc(srtso4)*(1.d0 - exp(-CS*dt))
               Gc(srtso4) = Gc(srtso4) - mcond_so4

C              Ezcond
               call ezcond(Nk, Mk, mcond_so4, srtso4, Nkf, Mkf)

C              Copy back
               do k=1,ibins
                  Nk(k) = Nkf(k)
                  do j=1,icomp
                     Mk(k,j) = Mkf(k,j)
                  enddo
               enddo
            elseif (Gc(srtso4) .gt. 0.d0) then
C              CS too small - dump into first bin
               Mk(1,srtso4) = Mk(1,srtso4) + Gc(srtso4)
               Nk(1) = Nk(1) + Gc(srtso4)/sqrt(xk(1)*xk(2))
               Gc(srtso4) = 0.d0
            endif

C           Equilibria + MNFIX
            call eznh3eqm(Gc, Mk)
            call ezwatereqm(Mk)
            call mnfix(Nk, Mk)

         enddo

         call cpu_time(t_end)

         write(*,'(A,F8.3,A)') '  Time: ',t_end-t_start,' s'

C-----WRITE OUTPUT-----------------------------------------------------

C        Write final Nk
         write(fname,'(A,A,A,I2.2,A)')
     &        'output/constgc/',trim(prefix),'_dt',nint(dt),
     &        '_final_Nk.csv'
         open(unit=10, file=fname, status='replace')
         do k=1,ibins
            write(10,'(E25.16)') Nk(k)
         enddo
         close(10)
         write(*,'(A,A)') '  Wrote: ', trim(fname)

C        Write final Mk
         write(fname,'(A,A,A,I2.2,A)')
     &        'output/constgc/',trim(prefix),'_dt',nint(dt),
     &        '_final_Mk.csv'
         open(unit=10, file=fname, status='replace')
         do k=1,ibins
            do j=1,icomp
               if (j .lt. icomp) then
                  write(10,'(E25.16,A)',advance='no')
     &                 Mk(k,j), ','
               else
                  write(10,'(E25.16)') Mk(k,j)
               endif
            enddo
         enddo
         close(10)
         write(*,'(A,A)') '  Wrote: ', trim(fname)

C        Write xk boundaries (for verification)
         write(fname,'(A)')
     &        'output/constgc/constgc_xk.csv'
         open(unit=10, file=fname, status='replace')
         do k=1,ibins+1
            write(10,'(E25.16)') xk(k)
         enddo
         close(10)

         write(*,*) ''

      enddo  ! idt

      write(*,*) '========================================'
      write(*,*) 'Benchmark complete!'
      write(*,*) 'Output in output/constgc/'
      write(*,*) '========================================'

      END PROGRAM
