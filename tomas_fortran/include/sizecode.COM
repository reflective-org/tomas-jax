C	*************************************************************
C	* Size-Resolved Aerosol Microphysics header file            *
C	*************************************************************

C	This header file includes all the variables used by the
C	size-resolved aerosol microphysics code incorporated into
C	the GISS GCM II' by Peter Adams.  The microphysics algorithm
C	conserves aerosol number and mass using the schemes developed
C	by Graham Feingold and others.

C	Tzivion, S., Feingold, G., and Levin, Z., An Efficient
C	   Numerical Solution to the Stochastic Collection Equation,
C	   J. Atmos. Sci., 44, 3139-3149, 1987.
C	Feingold, G., Tzivion, S., and Levin, Z., Evolution of
C	   Raindrop Spectra. Part I: Solution to the Stochastic
C	   Collection/Breakup Equation Using the Method of Moments,
C	   J. Atmos. Sci., 45, 3387-3399, 1988.
C	Tzivion, S., Feingold, G., and Levin, Z., The Evolution of
C	   Raindrop Spectra. Part II: Collisional Collection/Breakup
C	   and Evaporation in a Rainshaft, J. Atmos. Sci., 46, 3312-
C	   3327, 1989.
C	Feingold, G., Levin, Z., and Tzivion, S., The Evolution of
C	   Raindrop Spectra. Part III: Downdraft Generation in an
C	   Axisymmetrical Rainshaft Model, J. Atmos. Sci., 48, 315-
C	   330, 1991.

C	The algorithms described in these papers have been extended
C	to include multicomponent aerosols and modified for a moving
C	sectional approach.  Using this approach, the boundaries
C	between size bins are defined in terms of dry aerosol mass
C	such that the actual sizes of the sections move as water
C	is added to or lost from the aerosol.

C	All of the subroutines needed for this aerosol microphysics
C	algorithm use only their own internal variables or the ones
C	listed here.  GISS GCM II' variables are not used (a driver
C	subroutine performs the necessary swapping between the GCM
C	and the microphysics code).  The microphysics code is,
C	therefore, completely modular.

C--------------------------------------------------------------------

C ibins is the number of size bins and icomp the number of chemical
C species.  There are 4 aerosol species (SO4--, NO3-, NH4+, and H2O).
C Water is assumed to always be the last component and is mostly
C ignored (see above) because this algorithm uses a moving sectional
C approach.  Most loops are from 1 to icomp-idiag as a result.

      integer ibins, icomp
      integer idiag ! number of diagnostic aerosol species
      integer iorg ! number of organic components
      parameter(ibins=36, icomp=44, idiag=2, iorg=41)

      character*15 parname(icomp)

      integer srtso4, srtnh4, srth2o, srtorg1, srtorglast

      parameter (srtso4=1,
     &           srtorg1=2,
     &           SRTORGLAST=SRTORG1+IORG-1,
     &           srtnh4=iorg+2,
     &           srth2o=iorg+3)
	      
C .................................................................
C Nk and Mk contain the number and mass size distributions of the
C aerosol.  Units are #/grid cell or kg/grid cell, respectively.
C Gc are gas phase concentrations (kg/grid cell) of species
C corresponding to the aerosol species (e.g. H2SO4 for sulfate).
C Nkd and Mkd store values of Nk and Mk for diagnostic purposes.

      double precision Nk(ibins), Mk(ibins,icomp), Gc(icomp-1)
      double precision Nkd(ibins), Mkd(ibins,icomp), Gcd(icomp-1)

C .................................................................
C    The following array contains the mass boundaries of the size
C    bins.  xk(1) is lower boundary of the first size bins in terms
C    of mass (kg).  xk(2) is the boundary between the first and
C    second bins, etc...  The scheme used here assumes mass doubling:

C	xk(k+1) = 2 * xk(k)

      double precision xk(ibins+1)

C .................................................................
C The following variables describe the grid cell in which the
C microphysics is operating.

      double precision boxvol     !volume of grid cell (cm3)
      double precision boxmass    !volume of grid cell (kg)
      double precision temp       !temperature (K) of grid cell
      double precision pres       !air pressure (Pa) of grid cell
      double precision rh         !relative humidity (0-1)
      double precision alpha      ! accommodation coef

C .................................................................
C Physical properties of aerosol components

      double precision molwt(icomp)

      common /sizecode/ Nk, Mk, Gc, Nkd, Mkd, Gcd, xk,
     &                  boxvol, boxmass, temp, pres, rh, alpha
	      
C .................................................................
C Physical properties for organic components
      double precision mworg(iorg) ! g/mol

      double precision cstar(iorg) ! saturation ratios [ug m3]

      double precision T1 ! reference temperature
      data T1 /298./

      double precision Hvap(iorg) ! heat of vaporization [kJ mol^-1]
      ! Vaporization enthalpies of the surrogate compounds at 298 K [J/mol]
      double precision densorg(iorg) ! density of surrogate compunds [kg/mol]
      data densorg /iorg*1400./

      double precision diorg(iorg), ditemporg(iorg)
      ! Diffusion coefficients of the surrogate compounds in air at T_ref [m2/s]
      ! and the temperature-dependent factor
      data diorg /iorg*5.0e-6/
      data ditemporg /iorg*1.75/

      double precision storg ! surface tension of pure compound [N m^-1]

      double precision omfraci(iorg)      !modified at 10/08/2014

      double precision nonorgscale ! scale factor for how much organics can
                                   ! partition into other species

C .................................................................
C Other common organic variables
      double precision pstar(iorg) ! Volatilities of the surrogate compounds at T_ref [Pa]
      double precision psatorg(iorg) ! Volatilities of the surrogate compounds at T_amb [Pa]
      double precision omfrac(iorg)
      common /org/ omfrac, pstar, psatorg, nonorgscale,
     &             storg, cstar, Hvap, mworg, parname
