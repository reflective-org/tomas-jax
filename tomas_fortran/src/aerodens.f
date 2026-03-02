
C     **************************************************
C     *  aerodens                                      *
C     **************************************************

C     WRITTEN BY Peter Adams, May 1999
C     November, 2001 - extended to include NaCl and bug fixed
C                      the bug was that species densities (dan, ds0,
C                      etc...) are supposed to be calculated based on
C                      *total* solute concentration, not each species
C                      contribution as it had been.

C     This function calculates the density (kg/m3) of a sulfate-
C     nitrate-ammonium-nacl-water mixture that is assumed to be internally
C     mixed.  

C-----Literature cited--------------------------------------------------
C     I. N. Tang and H. R. Munkelwitz, Water activities, densities, and
C       refractive indices of aqueous sulfates and sodium nitrate droplets
C       of atmospheric importance, JGR, 99, 18,801-18,808, 1994
C     Ignatius N. Tang, Chemical and size effects of hygroscopic aerosols
C       on light scattering coefficients, JGR, 101, 19,245-19,250, 1996
C     Ignatius N. Tang, Thermodynamic and optical properties of mixed-salt
C       aerosols of atmospheric importance, JGR, 102, 1883-1893, 1997

C-----INPUTS------------------------------------------------------------

C     mso4, mno3, mnh4, mh2o, mnacl - These are the masses of each aerosol
C     component.  Since the density is an intensive property,
C     these may be input in a variety of units (ug/m3, mass/cell, etc.).

C-----OUTPUTS-----------------------------------------------------------

      double precision FUNCTION aerodens(mso4, mno3, mnh4, mnacl, mh2o)

      IMPLICIT NONE

C-----ARGUMENT DECLARATIONS---------------------------------------------

      double precision mso4, mno3, mnh4, mnacl, mh2o

C-----VARIABLE DECLARATIONS---------------------------------------------

      double precision so4t, no3t, nh4t, naclt, h2ot  !store initial values
      double precision mwso4, mwno3, mwnh4, mwnacl, mwh2o            !molecular weights
      double precision ntot, mtot                          !total number of moles, mass
      double precision nso4, nno3, nnh4, nnacl, nh2o       !moles of each species
      double precision xso4, xno3, xnh4, xnacl, xh2o       !mole fractions
      double precision rso4, rno3, rnh4, rnacl, rh2o       !partial molar refractions
      double precision ran, rs0, rs1, rs15, rs2       !same, but for solute species
      double precision asr                            !ammonium/sulfate molar ratio
      double precision nan, ns0, ns1, ns15, ns2, nss  !moles of dry solutes (nss = sea salt)
      double precision xan, xs0, xs1, xs15, xs2, xss  !mass % of dry solutes - Tang (1997) eq. 10
      double precision dan, ds0, ds1, ds15, ds2, dss  !binary solution densities - Tang (1997) eq. 10
      double precision mwan, mws0, mws1, mws15, mws2  !molecular weights
      double precision yan, ys0, ys1, ys15, ys2, yss  !mole fractions of dry solutes
      double precision yh2o
      double precision d                              !mixture density
      double precision xtot

C     In the lines above, "an" refers to ammonium nitrate, "s0" to 
C     sulfuric acid, "s1" to ammonium bisulfate, and "s2" to ammonium sulfate.
C     "nacl" or "ss" is sea salt.

C-----ADJUSTABLE PARAMETERS---------------------------------------------
      parameter(mwso4=96., mwno3=62., mwnh4=18., mwh2o=18., 
     &          mwnacl=58.45)
      parameter(mwan=mwnh4+mwno3, mws0=mwso4+2., mws1=mwso4+1.+mwnh4,
     &          mws2=2*mwnh4+mwso4)
C-----CODE--------------------------------------------------------------

C Save initial component masses to restore later
      so4t=mso4
      no3t=mno3
      nh4t=mnh4
      h2ot=mh2o
      naclt=mnacl

C Calculate mole fractions
      mtot = so4t+no3t+nh4t+naclt+h2ot
      if (mtot .lt. 1.e-15) then
      aerodens=1000.
      return
      endif
      nso4 = so4t/mwso4
      nno3 = no3t/mwno3
      nnh4 = nh4t/mwnh4
      nnacl = naclt/mwnacl
      nh2o = h2ot/mwh2o
      ntot = nso4+nno3+nnh4+nnacl+nh2o
      xso4 = nso4/ntot
      xno3 = nno3/ntot
      xnh4 = nnh4/ntot
      xnacl = nnacl/ntot
      xh2o = nh2o/ntot
C      call nanstop(mtot,92,0,0)
C If there are more moles of nitrate than ammonium, treat unneutralized
C HNO3 as H2SO4
      if (nno3 .gt. nnh4) then
         !make the switch
         nso4=nso4+(nno3-nnh4)
         nno3=nnh4
         so4t=nso4*mwso4
         no3t=nno3*mwno3

         !recalculate quantities
         mtot = so4t+no3t+nh4t+naclt+h2ot
         nso4 = so4t/mwso4
         nno3 = no3t/mwno3
         nnh4 = nh4t/mwnh4
         nnacl = naclt/mwnacl
         nh2o = h2ot/mwh2o
         ntot = nso4+nno3+nnh4+nnacl+nh2o
         xso4 = nso4/ntot
         xno3 = nno3/ntot
         xnh4 = nnh4/ntot
         xnacl = nnacl/ntot
         xh2o = nh2o/ntot

      endif

C Calculate the mixture density
C Assume that nitrate exists as ammonium nitrate and that other ammonium
C contributes to neutralizing sulfate
      nan=nno3
      if (nnh4 .gt. nno3) then
         !extra ammonium
         asr=(nnh4-nno3)/nso4
      else
         !less ammonium than nitrate - all sulfate is sulfuric acid
         asr=0.0
      endif
      if (asr .ge. 2.) asr=2.0
      if (asr .ge. 1.) then
         !assume NH4HSO4 and (NH4)2(SO4) mixture
         !NH4HSO4
         ns1=nso4*(2.-asr)
         !(NH4)2SO4
         ns2=nso4*(asr-1.)
         ns0=0.0
      else
         !assume H2SO4 and NH4HSO4 mixture
         !NH4HSO4
         ns1=nso4*asr
         !H2SO4
         ns0=nso4*(1.-asr)
         ns2=0.0
      endif

      !Calculate weight percent of solutes
      xan=nan*mwan/mtot*100.
      xs0=ns0*mws0/mtot*100.
      xs1=ns1*mws1/mtot*100.
      xs2=ns2*mws2/mtot*100.
      xnacl=nnacl*mwnacl/mtot*100.
      xtot=xan+xs0+xs1+xs2+xnacl
C      call nanstop(xtot,153,0,0)
      !Calculate binary mixture densities (Tang, eqn 9)
      dan=0.9971 +4.05e-3*xtot +9.0e-6*xtot**2.
      ds0=0.9971 +7.367e-3*xtot -4.934e-5*xtot**2. +1.754e-6*xtot**3.
     &       -1.104e-8*xtot**4.
      ds1=0.9971 +5.87e-3*xtot -1.89e-6*xtot**2. +1.763e-7*xtot**3.
      ds2=0.9971 +5.92e-3*xtot -5.036e-6*xtot**2. +1.024e-8*xtot**3.
      dss=0.9971 +7.41e-3*xtot -3.741e-5*xtot**2. +2.252e-6*xtot**3.
     &       -2.06e-8*xtot**4.

      !Convert x's (weight percent of solutes) to fraction of dry solute (scale to 1)
      xtot=xan+xs0+xs1+xs2+xnacl
      xan=xan/xtot
      xs0=xs0/xtot
      xs1=xs1/xtot
      xs2=xs2/xtot
      xnacl=xnacl/xtot

      !Calculate mixture density
      d=1./(xan/dan+xs0/ds0+xs1/ds1+xs2/ds2+xnacl/dss)  !Tang, eq. 10
C      call nanstop(d,173,0,0)
      if ((d .gt. 2.) .or. (d .lt. 0.997)) then
         write(*,*) 'ERROR in aerodens'
         write(*,*) so4t,no3t,nh4t,naclt,h2ot
         write(*,*) 'density',d*1000.
         STOP
      endif

C Restore masses passed
c      mso4=so4temp
c      mno3=no3temp
c      mnh4=nh4temp
c      mnacl=nacltemp
c      mh2o=h2otemp

C Return the density
      aerodens=1000.*d    !Convert g/cm3 to kg/m3

      RETURN
      END












