
C     **************************************************
C     *  dunne_inorg_nucl                              *
C     **************************************************

C     WRITTEN BY Jeff Pierce, July 2019

c     This subroutine calculates the binary and ternary (neutral and ion-mediated)
c     nucleation rats from Dunne et al. 2016

c     Dunne et al., Science, DOI: 10.1126/science.aaf2649, 2016
c

      SUBROUTINE dunne_inorg_nucl(tempi,fioni,cnai,nh3i,Mairi,fn,rnuc,
     &                             Jbn,Jtn,Jbi,Jti)

      IMPLICIT NONE

C-----INPUTS------------------------------------------------------------

      double precision tempi                ! temperature of air [Ka]
      double precision fioni                 ! formation rate of ions [pairs cm-3 s-1[
      double precision cnai                 ! concentration of gas phase sulfuric acid [molec cm-3]
      double precision nh3i                 ! concentration of ammonia [molec cm-3]
      double precision Mairi                 ! concentration of air [molec cm-3]

C-----OUTPUTS-----------------------------------------------------------

      double precision fn                   ! nucleation rate [cm-3 s-1]
      double precision rnuc                 ! critical cluster radius [nm]

C-----INCLUDE FILES-----------------------------------------------------

C-----ARGUMENT DECLARATIONS---------------------------------------------

C-----VARIABLE DECLARATIONS---------------------------------------------

      double precision temp                 ! temperature of air [K]
      double precision nh3                  ! concentration of gas phase ammonia [molec cm-3]
      double precision cna                  ! concentration of gas phase sulfuric acid [molec cm-3]
      double precision fion, Mair
      ! Complex temperature dependence
      double precision pbn, ubn, vbn, wbn, ptn, utn, vtn, wtn, pAn, an ! parameters for neutral nuc
      double precision pbi, ubi, vbi, wbi, pti, uti, vti, wti, pAi, ai ! parameters for ion nuc
      double precision kbn, ktn, kbi, kti ! rate constants for 4 nucleation types
      double precision ionc ! steady-state ion conc. [cm-3]
      double precision alpha_ion ! ion recombination coef [cm3/ion/s]
      double precision ffn, ffi ! intermediat calculation for ternary
      double precision Jbn, Jtn, Jbi, Jti ! nucleation rates from 4 mechanisms
      integer i                 ! counter

C-----ADJUSTABLE PARAMETERS---------------------------------------------

      ! Complex temperature dependence
      parameter(pbn=3.95, ubn=9.70, vbn=12.6, wbn=-0.00707, ptn=2.89)
      parameter(utn=182., vtn=1.20, wtn=-4.19, pAn=8.00, an=1.6d-6)
      parameter(pbi=3.37, ubi=-11.5, vbi=25.5, wbi=0.181, pti=3.14)
      parameter(uti=-23.8, vti=37.0, wti=0.227, pAi=3.07, ai=0.00485)


C-----CODE--------------------------------------------------------------

      temp=tempi
      nh3=nh3i*1E-6 ! I think they want it in units of 1E6 molec cm-3
      cna=cnai*1E-6
      fion=fioni
      Mair=Mairi

c CALCULATE ION CONCENTRATION
      alpha_ion = 6d-8*sqrt(300./temp) + 6d-26*Mair*(300./temp)**4 ! Need Mair in air molec per cm3
      ionc = sqrt(fion/alpha_ion) ! assume that ion-ion recombination dominates from conversion with svensmakr, need to verify

c CALCULATE k VALUES
      ! Complex temperature dependence
      kbn = exp(ubn - exp(vbn*(temp/1000. - wbn)))
      ktn = exp(utn - exp(vtn*(temp/1000. - wtn)))
      kbi = exp(ubi - exp(vbi*(temp/1000. - wbi)))
      kti = exp(uti - exp(vti*(temp/1000. - wti)))


c CALCULATE f VALUES
      if (nh3.gt.1d-10)then
         ffn = nh3*cna**ptn/(an + (cna**ptn)/(nh3**pAn))
         ffi = nh3*cna**pti/(ai + (cna**pti)/(nh3**pAi))
      else
         ffn = 0.
         ffi = 0.
      endif

c CALCULATE NUCLEATION RATES
      Jbn = kbn*cna**pbn
      Jtn = ktn*ffn
      Jbi = kbi*ionc*cna**pbi
      Jti = kti*ionc*ffi

      fn = Jbn + Jtn + Jbi + Jti ! sum them

c     cluster radius
      rnuc=0.85d0 ! [nm]

 10   return
      end
