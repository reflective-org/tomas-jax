
C     **************************************************
C     *  ezcond_ppm                                    *
C     **************************************************
C
C     PPM-aware condensation driver. Same structure as ezcond.f
C     (3-path decision tree, CS calculation, mass conservation check)
C     but Path 1 calls PPM_CONDENSATION_STEP instead of TMCOND,
C     then adds condensed mass proportionally via sinkfrac.
C
C     Uses dt=1.0 since TAU encodes the full growth forcing.
C     No moxd (c1/c2 corrections).
C
C     Ported from tomas_jax/physics/ezcond_ppm_jax.py + ezcond.f
C
C-----INPUTS--------------------------------------------------------
C     Nki(ibins)        - number of particles per size bin [#/grid cell]
C     Mki(ibins,icomp)  - mass per size bin per species [kg/grid cell]
C     mcondi            - mass of species to condense [kg/grid cell]
C     spec              - index of condensing species
C
C-----OUTPUTS-------------------------------------------------------
C     Nkf(ibins)        - final number per size bin
C     Mkf(ibins,icomp)  - final mass per size bin

      SUBROUTINE ezcond_ppm(Nki,Mki,mcondi,spec,Nkf,Mkf)

      IMPLICIT NONE

C-----INCLUDE FILES-------------------------------------------------
      include 'sizecode.COM'

C-----ARGUMENT DECLARATIONS-----------------------------------------
      double precision Nki(ibins), Mki(ibins, icomp)
      double precision Nkf(ibins), Mkf(ibins, icomp)
      double precision mcondi
      integer spec

C-----VARIABLE DECLARATIONS-----------------------------------------
      integer k, j
      double precision mcond
      double precision pi, R
      double precision CS
      double precision sinkfrac(ibins)
      double precision Nk1(ibins), Mk1(ibins, icomp)
      double precision Nk2(ibins), Mk2(ibins, icomp)
      double precision maddp(ibins)
      double precision eps
      double precision tdt
      double precision mpo, mpw
      double precision WR
      double precision tau(ibins)
      double precision totsinkfrac
      double precision CSeps
      double precision tot_m, tot_s
      double precision ratio
      double precision tot_i, tot_f
      double precision madd

      parameter(pi=3.141592654d0, R=8.314d0)
      parameter(eps=1.d-40)
      parameter(CSeps=1.d-20)

C-----CODE----------------------------------------------------------

      tdt = 2.d0/3.d0
      mcond = mcondi

C     Initialize working arrays
      do k=1,ibins
         Nk1(k) = Nki(k)
         do j=1,icomp
            Mk1(k,j) = Mki(k,j)
         enddo
      enddo

      call mnfix(Nk1, Mk1)

C     Get condensation sink
      call getCondSink(Nk1, Mk1, spec, CS, sinkfrac)

C     CS too small: dump mass in first bin
      if (CS .lt. CSeps) then
         Mkf(1,spec) = Mk1(1,spec) + mcond
         Nkf(1) = Nk1(1) + mcond/sqrt(xk(1)*xk(2))
         do j=1,icomp
            if (j .ne. spec) then
               Mkf(1,j) = Mk1(1,j)
            endif
         enddo
         do k=2,ibins
            Nkf(k) = Nk1(k)
            do j=1,icomp
               Mkf(k,j) = Mk1(k,j)
            enddo
         enddo
         return
      endif

C     Total sink fraction (excluding nucleation bin)
      totsinkfrac = 0.d0
      do k=1,ibins
         totsinkfrac = totsinkfrac + sinkfrac(k)
      enddo

C     Compute total dry and species mass for path selection
      tot_m = 0.d0
      tot_s = 0.d0
      do k=1,ibins
         do j=1,icomp-idiag
            tot_m = tot_m + Mk1(k,j)
         enddo
         tot_s = tot_s + Mk1(k,spec)
      enddo

C     === PATH 1: Full PPM condensation (mcond > tot_m * 1e-3) ===
      if (mcond .gt. tot_m*1.0d-3) then

C        Compute TAU for each bin (same as ezcond.f Path 1)
         do k=1,ibins
            mpo = 0.d0
            mpw = 0.d0
            do j=1,icomp-idiag
               mpo = mpo + Mk1(k,j)
            enddo
            do j=1,icomp
               mpw = mpw + Mk1(k,j)
            enddo
            WR = mpw / max(mpo, 1.d-30)
            if (Nk1(k) .gt. 0.d0 .and. totsinkfrac .gt. 0.d0) then
               maddp(k) = mcond*sinkfrac(k)/totsinkfrac/Nk1(k)
               mpw = mpw / Nk1(k)
               tau(k) = 1.5d0*((mpw+maddp(k)*WR)**tdt - mpw**tdt)
            else
               tau(k) = 0.d0
               maddp(k) = 0.d0
            endif
         enddo

         call mnfix(Nk1, Mk1)

C        Call PPM condensation step (dt=1.0 since TAU encodes full forcing)
         call PPM_CONDENSATION_STEP(Nk1, Mk1, tau, spec,
     &                              1.0d0, Nk2, Mk2)

C        Add condensed mass proportionally to sinkfrac
         do k=1,ibins
            if (sinkfrac(k) .gt. 1.d-20 .and.
     &          totsinkfrac .gt. 0.d0) then
               madd = mcond * sinkfrac(k) / totsinkfrac
            else
               madd = 0.d0
            endif
            Mk2(k,spec) = Mk2(k,spec) + madd
         enddo

C     === PATH 2: Simple proportional add (mcond > tot_s * 1e-12) ===
      elseif (mcond .gt. tot_s*1.0d-12) then

         do k=1,ibins
            if (Nk1(k) .gt. 0.d0 .and. totsinkfrac .gt. 0.d0) then
               maddp(k) = mcond*sinkfrac(k)/totsinkfrac
            else
               maddp(k) = 0.d0
            endif
            Mk2(k,spec) = Mk1(k,spec) + maddp(k)
            do j=1,icomp
               if (j .ne. spec) then
                  Mk2(k,j) = Mk1(k,j)
               endif
            enddo
            Nk2(k) = Nk1(k)
         enddo
         call mnfix(Nk2, Mk2)

C     === PATH 3: No-op (mcond too small) ===
      else
         mcond = 0.d0
         do k=1,ibins
            Nk2(k) = Nk1(k)
            do j=1,icomp
               Mk2(k,j) = Mk1(k,j)
            enddo
         enddo
      endif

C     Copy to output
      do k=1,ibins
         Nkf(k) = Nk2(k)
         do j=1,icomp
            Mkf(k,j) = Mk2(k,j)
         enddo
      enddo

C     Mass conservation check
      tot_i = 0.d0
      tot_f = 0.d0
      do k=1,ibins
         tot_i = tot_i + Mki(k,spec)
         tot_f = tot_f + Mkf(k,spec)
      enddo
      if (mcond .gt. 0.d0 .and.
     &    abs((mcond-(tot_f-tot_i))/mcond) .gt. 0.d0) then
         if (abs((mcond-(tot_f-tot_i))/mcond) .lt. 1.d0) then
C           Do correction of mass
            ratio = (tot_f - tot_i) / mcond
            do k=1,ibins
               Mkf(k,spec) = Mki(k,spec) +
     &              (Mkf(k,spec) - Mki(k,spec)) / ratio
            enddo
            call mnfix(Nkf, Mkf)
         else
            print*,'ERROR in ezcond_ppm'
            print*,'Condensation error',(mcond-(tot_f-tot_i))/mcond
            print*,'mcond',mcond,'change',tot_f-tot_i
            print*,'tot_i',tot_i,'tot_f',tot_f
            STOP
         endif
      endif

C     Check NH4 conservation
      tot_i = 0.d0
      tot_f = 0.d0
      do k=1,ibins
         tot_i = tot_i + Mki(k,srtnh4)
         tot_f = tot_f + Mkf(k,srtnh4)
      enddo
      if (tot_i .gt. 0.d0) then
         if (abs(tot_f-tot_i)/tot_i .gt. 1.0d-8) then
            print*,'No N conservation in ezcond_ppm.f'
            print*,'initial',tot_i
            print*,'final',tot_f
         endif
      endif

      RETURN
      END
