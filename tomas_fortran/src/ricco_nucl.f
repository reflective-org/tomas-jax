
C     **************************************************
C     *  ricco_nucl                                     *
C     **************************************************

C     WRITTEN BY Jeff Pierce, May 2017

C     This subroutine calculates the binary nucleation rate and radius of the
C     critical nucleation cluster using the parameterization of Riccobono et
C     al. (2014) updated with the temperature dependence accorrding to Yu et
C     al. (2017).

c     F Riccobono et al. "Oxidation Products of Biogenic Emissions
C     Contribute to Nucleation of Atmospheric Particles ."
C     Science 344 (6185), 717-721. 2014 May 16.
C
C     Yu et al. "Impact of temperature dependence on the possible contribution
C     of organics to new particle formation in the atmosphere", Atmos. Chem.
C     Phys., 17, 4997-5005, 2017.

c

      SUBROUTINE ricco_nucl(tempi,cnai,orgi,fn,rnuc)

      IMPLICIT NONE

C-----INPUTS------------------------------------------------------------

      double precision tempi                ! temperature of air [K]
      double precision cnai                 ! concentration of gas phase sulfuric acid [molec cm-3]
      double precision orgi                 ! concentration of gas phase BioOxOrg [molec cm-3]

C-----OUTPUTS-----------------------------------------------------------

      double precision fn                   ! nucleation rate [cm-3 s-1]
      double precision rnuc                 ! critical cluster radius [nm]

C-----INCLUDE FILES-----------------------------------------------------

C-----ARGUMENT DECLARATIONS---------------------------------------------

C-----VARIABLE DECLARATIONS---------------------------------------------

      double precision temp                 ! temperature of air [K]
      double precision org                   ! concentration of gas phase BioOxOrg [molec cm-3]
      double precision cna                  ! concentration of gas phase sulfuric acid [molec cm-3]
      double precision kb   ! Boltzmann constant J K-1
      double precision km ! Riccobono nucleation rate constant cm6 s-1
      double precision Jnotemp !Riccobono's nucleation rate, at T=278 K
      double precision ft !Yu's correction factor accounting for the T dependece of Riccobono's nucleation rate
      double precision Jtemp !Yu's T dependece of Riccobono's nucleation rate
      double precision delta_H !enthalpy change, kcal mol-1
      double precision temp0 !temperature of Riccobono's scheme in K
      integer i                 ! counter

C-----ADJUSTABLE PARAMETERS---------------------------------------------

      parameter(kb=1.38d-23, km=3.27d-21)
      parameter(delta_H=2.66d-19)
      parameter(temp0=278.d0)

C-----CODE--------------------------------------------------------------

      temp=tempi
      org=orgi
      cna=cnai

      Jnotemp=km*(cna**2)*org !Riccobono's nucleation rate, at T=278 K
      ft=exp(delta_H/kb*(1.d0/temp-1.d0/temp0))
      Jtemp=Jnotemp*ft !Yu's T dependece of Riccobono's nucleation rate
      fn=Jtemp

c     cluster radius
      rnuc=0.85d0 ! [nm]

 10   return
      end
