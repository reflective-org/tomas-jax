
C     **************************************************
C     *  waternacl                                     *
C     **************************************************

C     WRITTEN BY Peter Adams, November 2001

C     This function uses the current RH to calculate how much water is 
C     in equilibrium with the seasalt.  Aerosol water concentrations 
C     are assumed to be in equilibrium at all times and the array of 
C     concentrations is updated accordingly.

C-----INPUTS------------------------------------------------------------

C-----OUTPUTS-----------------------------------------------------------

      double precision FUNCTION waternacl(rhe)

      IMPLICIT NONE

C-----INCLUDE FILES-----------------------------------------------------

C-----ARGUMENT DECLARATIONS---------------------------------------------

      double precision rhe   !relative humidity (0-100 scale)

C-----VARIABLE DECLARATIONS---------------------------------------------

C     VARIABLE COMMENTS...

C     waternacl is the ratio of wet mass to dry mass of a particle.  Instead
C     of calling a thermodynamic equilibrium code, this routine uses a
C     simple curve fit to estimate waternacl based on the current humidity.
C     The curve fit is based on ISORROPIA results for sodium sulfate
C     at 273 K.

C-----ADJUSTABLE PARAMETERS---------------------------------------------

C-----CODE--------------------------------------------------------------

      if (rhe .gt. 99.) rhe=99.
      if (rhe .lt. 1.) rhe=1.

         if (rhe .gt. 90.) then
            waternacl=5.1667642e-2*rhe**3-14.153121*rhe**2
     &               +1292.8377*rhe-3.9373536e4
         else
         if (rhe .gt. 80.) then
            waternacl=
     &      1.0629e-3*rhe**3-0.25281*rhe**2+20.171*rhe-5.3558e2
         else
         if (rhe .gt. 50.) then
            waternacl=
     &      4.2967e-5*rhe**3-7.3654e-3*rhe**2+.46312*rhe-7.5731
         else
         if (rhe .gt. 20.) then
            waternacl=
     &      2.9443e-5*rhe**3-2.4739e-3*rhe**2+7.3430e-2*rhe+1.3727
         else
            waternacl=1.17
         endif
         endif
         endif
         endif

         !check for error
         if (waternacl .gt. 45.) then
            write(*,*) 'ERROR in waternacl'
            write(*,*) rhe,waternacl
            STOP
         endif

      RETURN
      END
