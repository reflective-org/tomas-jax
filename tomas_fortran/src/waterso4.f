
C     **************************************************
C     *  waterso4                                      *
C     **************************************************

C     Adaptation of ezwatereqm used in size-resolved sulfate only sim
C     November, 2001
C     ezwatereqm WRITTEN BY Peter Adams, March 2000

C     This function uses the current RH to calculate how much water is 
C     in equilibrium with the sulfate.  Aerosol water concentrations 
C     are assumed to be in equilibrium at all times and the array of 
C     concentrations is updated accordingly.

C-----INPUTS------------------------------------------------------------

C-----OUTPUTS-----------------------------------------------------------

      double precision FUNCTION waterso4(rhe)

      IMPLICIT NONE

C-----INCLUDE FILES-----------------------------------------------------

C-----ARGUMENT DECLARATIONS---------------------------------------------

      double precision rhe   !relative humidity (0-100 scale)

C-----VARIABLE DECLARATIONS---------------------------------------------

C     VARIABLE COMMENTS...

C     waterso4 is the ratio of wet mass to dry mass of a particle.  Instead
C     of calling a thermodynamic equilibrium code, this routine uses a
C     simple curve fit to estimate wr based on the current humidity.
C     The curve fit is based on ISORROPIA results for ammonium bisulfate
C     at 273 K.

C-----ADJUSTABLE PARAMETERS---------------------------------------------

C-----CODE--------------------------------------------------------------

      if (rhe .gt. 99.) rhe=99.
      if (rhe .lt. 1.) rhe=1.

         if (rhe .gt. 96.) then
            waterso4=
     &      0.7540688*rhe**3-218.5647*rhe**2+21118.19*rhe-6.801999e5
         else
         if (rhe .gt. 91.) then
            waterso4=8.517e-2*rhe**2 -15.388*rhe +698.25
         else
         if (rhe .gt. 81.) then
            waterso4=8.2696e-3*rhe**2 -1.3076*rhe +53.697
         else
         if (rhe .gt. 61.) then
            waterso4=9.3562e-4*rhe**2 -0.10427*rhe +4.3155
         else
         if (rhe .gt. 41.) then
            waterso4=1.9149e-4*rhe**2 -8.8619e-3*rhe +1.2535
         else
            waterso4=5.1337e-5*rhe**2 +2.6266e-3*rhe +1.0149
         endif
         endif
         endif
         endif
         endif

         !check for error
         if (waterso4 .gt. 30.) then
            write(*,*) 'ERROR in waterso4'
            write(*,*) rhe,waterso4
            STOP
         endif

      RETURN
      END
