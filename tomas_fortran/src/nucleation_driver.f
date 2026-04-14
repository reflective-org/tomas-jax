
C     **************************************************
C     *  nucleation_driver                              *
C     **************************************************
C
C     Cleaned-up nucleation driver for benchmark use.
C     Adapted from TRACER_SOM-TOMAS nucleation.f.
C
C     Calls ricco_nucl and dunne_inorg_nucl, adds nucleated
C     particles to bin 1 (90% SO4, 10% organic at srtorglast),
C     depletes gas-phase H2SO4.

      SUBROUTINE nucleation_driver(Nki, Mki, Gci, Nkf, Mkf, Gcf,
     &     dt, org_conc, nh3_conc, fion_in,
     &     org_nuc, dunne_nuc, fn_scale)

      IMPLICIT NONE

C-----INCLUDE FILES-----------------------------------------------------
      include 'sizecode.COM'

C-----ARGUMENT DECLARATIONS---------------------------------------------
      double precision Nki(ibins), Mki(ibins, icomp), Gci(icomp-1)
      double precision Nkf(ibins), Mkf(ibins, icomp), Gcf(icomp-1)
      double precision dt
      double precision org_conc      ! organic vapor [molec/cm3]
      double precision nh3_conc      ! NH3 [molec/cm3]
      double precision fion_in       ! ion formation rate [pairs/cm3/s]
      integer org_nuc                ! 1=enable Riccobono, 0=disable
      integer dunne_nuc              ! 1=enable Dunne, 0=disable
      double precision fn_scale      ! scaling factor for total rate

C-----VARIABLE DECLARATIONS---------------------------------------------
      integer j, k
      double precision h2so4         ! gas phase H2SO4 [molec/cm3]
      double precision Mair          ! air concentration [molec/cm3]
      double precision fn            ! total nucleation rate [cm-3 s-1]
      double precision fntemp1       ! Riccobono rate
      double precision fntemp2       ! Dunne rate
      double precision rnuc          ! cluster radius [nm]
      double precision mnuc          ! cluster mass [kg]
      double precision mold          ! saved SO4 mass in nuc bin
      double precision pi
      double precision Jbn2, Jtn2, Jbi2, Jti2
      integer nuc_bin

      parameter(pi=3.14159d0)

C-----CODE--------------------------------------------------------------

C     Copy input to output (default: no change)
      do k=1,ibins
         Nkf(k) = Nki(k)
         do j=1,icomp
            Mkf(k,j) = Mki(k,j)
         enddo
      enddo
      do k=1,icomp-1
         Gcf(k) = Gci(k)
      enddo

C     Convert gas-phase H2SO4: kg/grid_cell -> molec/cm3
      h2so4 = Gci(srtso4)/boxvol*1000.d0/98.d0*6.022d23

C     Air number density [molec/cm3] from ideal gas law
      Mair = 2.69d19*273.15d0/temp*pres/101325.d0

C     Initialize rates
      fn = 0.d0
      fntemp1 = 0.d0
      fntemp2 = 0.d0
      rnuc = 0.85d0

C     Riccobono 2014 organic nucleation
      if (org_nuc .eq. 1) then
         call ricco_nucl(temp, h2so4, org_conc, fntemp1, rnuc)
         fn = fn + fntemp1
      endif

C     Dunne 2016 inorganic nucleation
      if (dunne_nuc .eq. 1) then
         call dunne_inorg_nucl(temp, fion_in, h2so4, nh3_conc, Mair,
     &        fntemp2, rnuc, Jbn2, Jtn2, Jbi2, Jti2)
         fntemp2 = fntemp2 * fn_scale
         fn = fn + fntemp2
      endif

C     If nucleation occurred, add particles
      if (fn .gt. 0.d0) then

C        Cluster mass [kg]
         mnuc = (4.d0/3.d0*pi*(rnuc*1.d-9)**3)*1350.d0

C        Find appropriate bin (where mnuc fits)
         nuc_bin = 1
         do while (mnuc .gt. xk(nuc_bin+1))
            nuc_bin = nuc_bin + 1
         enddo

C        Add mass: 90% SO4, 10% organic
         mold = Mki(nuc_bin, srtso4)
         Mkf(nuc_bin, srtso4) = Mki(nuc_bin, srtso4)
     &        + 0.9d0*fn*mnuc*boxvol*dt
         Mkf(nuc_bin, srtorglast) = Mki(nuc_bin, srtorglast)
     &        + 0.1d0*fn*mnuc*boxvol*dt
         Nkf(nuc_bin) = Nki(nuc_bin) + fn*boxvol*dt

C        Deplete gas-phase H2SO4
         Gcf(srtso4) = Gci(srtso4) - (Mkf(nuc_bin,srtso4) - mold)

C        Clamp: don't let Gc go negative
         if (Gcf(srtso4) .lt. 0.d0) then
            Mkf(nuc_bin, srtso4) = Mki(nuc_bin, srtso4)
     &           + Gci(srtso4)*96.d0/98.d0
            Nkf(nuc_bin) = Nki(nuc_bin)
     &           + Gci(srtso4)*96.d0/98.d0/mnuc
            Gcf(srtso4) = 0.d0
         endif

      endif

      RETURN
      END
