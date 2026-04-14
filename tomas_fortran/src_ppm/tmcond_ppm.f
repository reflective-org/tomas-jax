
C     **************************************************
C     *  PPM Condensation for TOMAS                    *
C     **************************************************
C
C     Piecewise Parabolic Method (PPM) condensation in log-mass
C     coordinate (xi = ln m). Ported from tomas_jax/physics/condensation_ppm.py.
C
C     References:
C       - Colella & Woodward (1984) "The Piecewise Parabolic Method"
C       - Stevens, Feingold, Cotton (1996) for dmdt_int growth equation
C
C     Contains 8 subroutines:
C       PPM_RECONSTRUCT        - 4th-order interface + C&W limiting
C       PPM_EDGE_VELOCITY      - Upwind velocity at bin edges
C       PPM_NUMBER_FLUX        - Departure-point parabola integration
C       PPM_MASS_FLUX          - Analytical mass-weighted flux
C       PPM_DRY_MASS_ANALYTICAL- Exact dry mass from PPM coefficients
C       PPM_SPECIES_FLUX       - Upwind donor ratio x dry mass flux
C       PPM_COMPUTE_SUBSTEPS   - CFL-limited substep count
C       PPM_CONDENSATION_STEP  - Main orchestrator
C
C-----CONSTANTS (module-level)--------------------------------------
C     DELTA_XI = ln(2) = 0.6931471805599453
C     CMAX = 0.8 (maximum Courant number)
C     EPSN = 1e-30 (safety for division by Nk)
C     TEPS = 1e-40 (threshold for zero TAU)
C
C     Precomputed moment integrals:
C       I_k = integral_0^1 eta^k exp(a*eta) d_eta, a = ln(2)
C       I0 = (exp(a)-1)/a        = 1.4426950408889634
C       I1 = (exp(a)*(a-1)+1)/a^2 = 0.5672269882832156
C       I2 = (exp(a)*(a^2-2a+2)-2)/a^3 = 0.31609640204744345
C
C     Inverse powers of a:
C       INV_A  = 1/ln(2) = 1.4426950408889634
C       INV_A2 = 1/ln(2)^2 = 2.0813689810056077
C       INV_A3 = 1/ln(2)^3 = 3.0043522781638864


C======================================================================
C     PPM_RECONSTRUCT
C======================================================================
C
C     4th-order PPM reconstruction on uniform log-mass grid.
C     Computes parabolic representation within each cell:
C       q(eta) = q_L + eta*(dq + q_6*(1-eta))
C     where eta in [0,1], dq = q_R - q_L
C     and q_6 = 6*(q_bar - (q_L+q_R)/2) ensures integral preservation.
C
C     INPUTS:
C       QBAR(ibins) - cell-average number density = Nk / delta_xi
C
C     OUTPUTS:
C       QL(ibins)   - left edge value
C       QR(ibins)   - right edge value
C       Q6(ibins)   - curvature coefficient

      SUBROUTINE PPM_RECONSTRUCT(QBAR, QL, QR, Q6)

      IMPLICIT NONE
      include 'sizecode.COM'

C-----ARGUMENTS-----------------------------------------------------
      double precision QBAR(ibins)
      double precision QL(ibins), QR(ibins), Q6(ibins)

C-----LOCAL VARIABLES-----------------------------------------------
      integer k
      double precision QEXT(ibins+4)
      double precision QEDGE(ibins+1)
      double precision QLEFT, QRIGHT, QMIN, QMAX
      double precision DQ, ETASTAR, QEXT_VAL
      double precision Q6SAFE, QMIN_B, QMAX_B, QATEX
      logical HAS_EXTR, PROB_EXTR, NEG_INT

C-----CODE----------------------------------------------------------

C     Step 1: Build extended array with boundary padding
C     QEXT(1:2) = QBAR(1), QEXT(3:ibins+2) = QBAR, QEXT(ibins+3:ibins+4) = QBAR(ibins)
      QEXT(1) = QBAR(1)
      QEXT(2) = QBAR(1)
      do k=1,ibins
         QEXT(k+2) = QBAR(k)
      enddo
      QEXT(ibins+3) = QBAR(ibins)
      QEXT(ibins+4) = QBAR(ibins)

C     Step 2: Fourth-order interface interpolation
C     Edge k (1..ibins+1) uses QEXT indices k, k+1, k+2, k+3
      do k=1,ibins+1
         QEDGE(k) = (7.d0/12.d0)*(QEXT(k+1) + QEXT(k+2))
     &            - (1.d0/12.d0)*(QEXT(k) + QEXT(k+3))
      enddo

C     Step 3: Monotonicity limiting on edge values
C     Each edge is clipped to [min,max] of adjacent cell averages
C     Edge 1: between boundary pad and cell 1
C     Edge k: between cell k-1 and cell k (for k=2..ibins)
C     Edge ibins+1: between cell ibins and boundary pad
      do k=1,ibins+1
         if (k .eq. 1) then
            QLEFT = QBAR(1)
            QRIGHT = QBAR(1)
         elseif (k .eq. ibins+1) then
            QLEFT = QBAR(ibins)
            QRIGHT = QBAR(ibins)
         else
            QLEFT = QBAR(k-1)
            QRIGHT = QBAR(k)
         endif
         QMIN = min(QLEFT, QRIGHT)
         QMAX = max(QLEFT, QRIGHT)
         QEDGE(k) = max(QMIN, min(QMAX, QEDGE(k)))
      enddo

C     Step 4: Extract left/right edges and compute q_6
      do k=1,ibins
         QL(k) = QEDGE(k)
         QR(k) = QEDGE(k+1)
         Q6(k) = 6.d0 * (QBAR(k) - 0.5d0*(QL(k) + QR(k)))
      enddo

C     Step 5: Limit parabola (Colella-Woodward + positivity)
      do k=1,ibins
         DQ = QR(k) - QL(k)

C        Case 1: If q_bar not between q_L and q_R, reset to constant
         if ((QBAR(k)-QL(k))*(QBAR(k)-QR(k)) .gt. 0.d0) then
            QL(k) = QBAR(k)
            QR(k) = QBAR(k)
            Q6(k) = 0.d0
            DQ = 0.d0
         endif

C        Case 2: Check for interior extremum
         if (abs(Q6(k)) .gt. 1.d-30) then
            Q6SAFE = Q6(k)
            ETASTAR = (DQ + Q6SAFE) / (2.d0 * Q6SAFE)
            if (ETASTAR .gt. 0.d0 .and. ETASTAR .lt. 1.d0) then
               QATEX = QL(k) + ETASTAR*(DQ + Q6(k)*(1.d0-ETASTAR))
               QMIN_B = min(min(QL(k),QR(k)), QBAR(k))
               QMAX_B = max(max(QL(k),QR(k)), QBAR(k))
               if (QATEX .lt. QMIN_B .or. QATEX .gt. QMAX_B) then
                  Q6(k) = 0.d0
               endif
            endif
         endif

C        Case 3: Positivity limiting
         QL(k) = max(QL(k), 0.d0)
         QR(k) = max(QR(k), 0.d0)
         DQ = QR(k) - QL(k)
         Q6(k) = 6.d0 * (QBAR(k) - 0.5d0*(QL(k) + QR(k)))

C        Check interior minimum with new q_6
         if (abs(Q6(k)) .gt. 1.d-30) then
            Q6SAFE = Q6(k)
            ETASTAR = (DQ + Q6SAFE) / (2.d0 * Q6SAFE)
            if (ETASTAR .gt. 0.d0 .and. ETASTAR .lt. 1.d0) then
               QATEX = QL(k) + ETASTAR*(DQ + Q6(k)*(1.d0-ETASTAR))
               if (QATEX .lt. 0.d0) then
                  QL(k) = QBAR(k)
                  QR(k) = QBAR(k)
                  Q6(k) = 0.d0
               endif
            endif
         endif

      enddo

      RETURN
      END


C======================================================================
C     PPM_EDGE_VELOCITY
C======================================================================
C
C     Compute velocity at bin edges using upwind-consistent approach.
C     Uses DMDT_INT to map bin-edge masses forward, then computes
C     velocity in xi-space: u = ln(m_final/m_edge) / dt_sub
C
C     INPUTS:
C       TAU_SUB(ibins)  - growth forcing for one substep
C       WR(ibins)       - frozen wet/dry ratio per bin
C       DT_SUB          - substep duration [s]
C
C     OUTPUTS:
C       UEDGE(ibins+1)  - velocity in xi-space [1/s]

      SUBROUTINE PPM_EDGE_VELOCITY(TAU_SUB, WR, DT_SUB, UEDGE)

      IMPLICIT NONE
      include 'sizecode.COM'

C-----ARGUMENTS-----------------------------------------------------
      double precision TAU_SUB(ibins)
      double precision WR(ibins)
      double precision DT_SUB
      double precision UEDGE(ibins+1)

C-----LOCAL VARIABLES-----------------------------------------------
      integer k
      double precision DMDT_INT
      external DMDT_INT
      double precision TAU_L(ibins+1), TAU_R(ibins+1)
      double precision WR_L(ibins+1), WR_R(ibins+1)
      double precision MF_L(ibins+1), MF_R(ibins+1)
      double precision U_L(ibins+1), U_R(ibins+1)
      double precision U_AVG

C-----CODE----------------------------------------------------------

C     Extend TAU/WR to edges
C     Left donor: edge k uses bin k-1 (edge 1 uses bin 1)
      TAU_L(1) = TAU_SUB(1)
      WR_L(1) = WR(1)
      do k=2,ibins+1
         TAU_L(k) = TAU_SUB(k-1)
         WR_L(k) = WR(k-1)
      enddo

C     Right donor: edge k uses bin k (edge ibins+1 uses bin ibins)
      do k=1,ibins
         TAU_R(k) = TAU_SUB(k)
         WR_R(k) = WR(k)
      enddo
      TAU_R(ibins+1) = TAU_SUB(ibins)
      WR_R(ibins+1) = WR(ibins)

C     Compute final mass and velocity from each side
      do k=1,ibins+1
         MF_L(k) = DMDT_INT(xk(k), TAU_L(k), WR_L(k))
         MF_R(k) = DMDT_INT(xk(k), TAU_R(k), WR_R(k))

C        Clip to grid bounds
         MF_L(k) = max(xk(1), min(xk(ibins+1), MF_L(k)))
         MF_R(k) = max(xk(1), min(xk(ibins+1), MF_R(k)))

C        Velocity from each side
         U_L(k) = dlog(MF_L(k) / xk(k)) / DT_SUB
         U_R(k) = dlog(MF_R(k) / xk(k)) / DT_SUB

C        Upwind selection
         U_AVG = 0.5d0 * (U_L(k) + U_R(k))
         if (U_AVG .ge. 0.d0) then
            UEDGE(k) = U_L(k)
         else
            UEDGE(k) = U_R(k)
         endif
      enddo

      RETURN
      END


C======================================================================
C     PPM_NUMBER_FLUX
C======================================================================
C
C     Compute number flux at each edge using PPM departure-point method.
C     For u>0: integrate parabola over [1-|C|, 1] in left (donor) cell.
C     For u<0: integrate parabola over [0, |C|] in right (donor) cell.
C
C     INPUTS:
C       QL(ibins), QR(ibins), Q6(ibins) - PPM coefficients
C       UEDGE(ibins+1)  - edge velocities [1/s]
C       DT_SUB          - substep duration [s]
C
C     OUTPUTS:
C       FN(ibins+1)     - number flux at each edge [#/s]

      SUBROUTINE PPM_NUMBER_FLUX(QL, QR, Q6, UEDGE, DT_SUB, FN)

      IMPLICIT NONE
      include 'sizecode.COM'

C-----ARGUMENTS-----------------------------------------------------
      double precision QL(ibins), QR(ibins), Q6(ibins)
      double precision UEDGE(ibins+1)
      double precision DT_SUB
      double precision FN(ibins+1)

C-----LOCAL VARIABLES-----------------------------------------------
      integer k
      double precision DELTA_XI
      parameter(DELTA_XI=0.69314718055994530d0)
      double precision CNUM, CSAFE
      double precision DQ, FPOS, FNEG
      double precision QL_D, QR_D, Q6_D
      double precision ETA_LO, ETA_HI
      double precision ANTID_HI, ANTID_LO

C-----CODE----------------------------------------------------------

      do k=1,ibins+1
C        Courant number
         CNUM = UEDGE(k) * DT_SUB / DELTA_XI
         CSAFE = max(0.d0, min(1.d0, abs(CNUM)))

C        --- Positive velocity: flux from left cell (k-1) ---
         if (k .eq. 1) then
C           No left cell at edge 1
            FPOS = 0.d0
         else
            QL_D = QL(k-1)
            QR_D = QR(k-1)
            Q6_D = Q6(k-1)
            DQ = QR_D - QL_D
C           Integrate over [1-C, 1]
C           Antiderivative: A(eta) = qL*eta + 0.5*(dq+q6)*eta^2 - (q6/3)*eta^3
            ETA_HI = 1.d0
            ETA_LO = 1.d0 - CSAFE
            ANTID_HI = QL_D*ETA_HI
     &         + 0.5d0*(DQ + Q6_D)*ETA_HI*ETA_HI
     &         - (Q6_D/3.d0)*ETA_HI*ETA_HI*ETA_HI
            ANTID_LO = QL_D*ETA_LO
     &         + 0.5d0*(DQ + Q6_D)*ETA_LO*ETA_LO
     &         - (Q6_D/3.d0)*ETA_LO*ETA_LO*ETA_LO
            FPOS = ANTID_HI - ANTID_LO
         endif

C        --- Negative velocity: flux from right cell (k) ---
         if (k .eq. ibins+1) then
C           No right cell at last edge
            FNEG = 0.d0
         else
            QL_D = QL(k)
            QR_D = QR(k)
            Q6_D = Q6(k)
            DQ = QR_D - QL_D
C           Integrate over [0, C]
            ETA_HI = CSAFE
            ETA_LO = 0.d0
            ANTID_HI = QL_D*ETA_HI
     &         + 0.5d0*(DQ + Q6_D)*ETA_HI*ETA_HI
     &         - (Q6_D/3.d0)*ETA_HI*ETA_HI*ETA_HI
            ANTID_LO = 0.d0
            FNEG = ANTID_HI - ANTID_LO
         endif

C        Select based on velocity sign
         if (UEDGE(k) .ge. 0.d0) then
            FN(k) = FPOS * DELTA_XI / DT_SUB
         else
            FN(k) = -FNEG * DELTA_XI / DT_SUB
         endif
      enddo

C     Boundary conditions
      FN(1) = 0.d0
      FN(ibins+1) = 0.d0

      RETURN
      END


C======================================================================
C     PPM_MASS_FLUX
C======================================================================
C
C     Compute dry mass flux at each edge using analytical mass-weighted
C     integrals: integral m(eta)*n(eta) d_eta where m(eta) = m_L*exp(a*eta).
C
C     This correctly accounts for the 2x mass variation across each
C     mass-doubling bin, unlike the naive F_M = F_N * r_avg.
C
C     INPUTS:
C       NL(ibins), NR(ibins), N6(ibins) - PPM coefficients for number
C       UEDGE(ibins+1)  - edge velocities [1/s]
C       DT_SUB          - substep duration [s]
C
C     OUTPUTS:
C       FM_DRY(ibins+1) - dry mass flux at each edge [kg/s]

      SUBROUTINE PPM_MASS_FLUX(NL, NR, N6, UEDGE, DT_SUB, FM_DRY)

      IMPLICIT NONE
      include 'sizecode.COM'

C-----ARGUMENTS-----------------------------------------------------
      double precision NL(ibins), NR(ibins), N6(ibins)
      double precision UEDGE(ibins+1)
      double precision DT_SUB
      double precision FM_DRY(ibins+1)

C-----LOCAL VARIABLES-----------------------------------------------
      integer k
      double precision DELTA_XI, INV_A, INV_A2, INV_A3, AA
      parameter(DELTA_XI=0.69314718055994530d0)
      parameter(AA=0.69314718055994530d0)
      parameter(INV_A=1.4426950408889634d0)
      parameter(INV_A2=2.0813689810056077d0)
      parameter(INV_A3=3.0043522781638864d0)
      double precision CNUM, CSAFE
      double precision B_COEFF, ML_D
      double precision NL_D, NR_D, N6_D
      double precision ETA_HI, ETA_LO
      double precision EA_HI, EA_LO
      double precision A0_HI, A1_HI, A2_HI
      double precision A0_LO, A1_LO, A2_LO
      double precision INTEGRAL
      double precision FPOS, FNEG

C-----CODE----------------------------------------------------------

      do k=1,ibins+1
C        Courant number
         CNUM = UEDGE(k) * DT_SUB / DELTA_XI
         CSAFE = max(0.d0, min(1.d0, abs(CNUM)))

C        --- Positive velocity: flux from left cell (k-1) ---
         if (k .eq. 1) then
            FPOS = 0.d0
         else
            NL_D = NL(k-1)
            NR_D = NR(k-1)
            N6_D = N6(k-1)
            B_COEFF = (NR_D - NL_D) + N6_D
            ML_D = xk(k-1)

C           Integrate m(eta)*n(eta) over [1-C, 1]
C           Antiderivatives of eta^k * exp(a*eta):
C             A0(eta) = exp(a*eta) / a
C             A1(eta) = exp(a*eta) * (eta/a - 1/a^2)
C             A2(eta) = exp(a*eta) * (eta^2/a - 2*eta/a^2 + 2/a^3)
            ETA_HI = 1.d0
            ETA_LO = 1.d0 - CSAFE

            EA_HI = dexp(AA * ETA_HI)
            A0_HI = EA_HI * INV_A
            A1_HI = EA_HI * (ETA_HI * INV_A - INV_A2)
            A2_HI = EA_HI * (ETA_HI*ETA_HI*INV_A
     &                      - 2.d0*ETA_HI*INV_A2 + 2.d0*INV_A3)

            EA_LO = dexp(AA * ETA_LO)
            A0_LO = EA_LO * INV_A
            A1_LO = EA_LO * (ETA_LO * INV_A - INV_A2)
            A2_LO = EA_LO * (ETA_LO*ETA_LO*INV_A
     &                      - 2.d0*ETA_LO*INV_A2 + 2.d0*INV_A3)

            INTEGRAL = NL_D * (A0_HI - A0_LO)
     &               + B_COEFF * (A1_HI - A1_LO)
     &               - N6_D * (A2_HI - A2_LO)

            FPOS = DELTA_XI * ML_D * INTEGRAL
         endif

C        --- Negative velocity: flux from right cell (k) ---
         if (k .eq. ibins+1) then
            FNEG = 0.d0
         else
            NL_D = NL(k)
            NR_D = NR(k)
            N6_D = N6(k)
            B_COEFF = (NR_D - NL_D) + N6_D
            ML_D = xk(k)

C           Integrate m(eta)*n(eta) over [0, C]
            ETA_HI = CSAFE
            ETA_LO = 0.d0

            EA_HI = dexp(AA * ETA_HI)
            A0_HI = EA_HI * INV_A
            A1_HI = EA_HI * (ETA_HI * INV_A - INV_A2)
            A2_HI = EA_HI * (ETA_HI*ETA_HI*INV_A
     &                      - 2.d0*ETA_HI*INV_A2 + 2.d0*INV_A3)

            EA_LO = dexp(AA * ETA_LO)
            A0_LO = EA_LO * INV_A
            A1_LO = EA_LO * (ETA_LO * INV_A - INV_A2)
            A2_LO = EA_LO * (ETA_LO*ETA_LO*INV_A
     &                      - 2.d0*ETA_LO*INV_A2 + 2.d0*INV_A3)

            INTEGRAL = NL_D * (A0_HI - A0_LO)
     &               + B_COEFF * (A1_HI - A1_LO)
     &               - N6_D * (A2_HI - A2_LO)

            FNEG = DELTA_XI * ML_D * INTEGRAL
         endif

C        Select based on velocity sign
         if (UEDGE(k) .ge. 0.d0) then
            FM_DRY(k) = FPOS / DT_SUB
         else
            FM_DRY(k) = -FNEG / DT_SUB
         endif
      enddo

C     Boundary conditions
      FM_DRY(1) = 0.d0
      FM_DRY(ibins+1) = 0.d0

      RETURN
      END


C======================================================================
C     PPM_DRY_MASS_ANALYTICAL
C======================================================================
C
C     Compute exact dry mass per bin from PPM number reconstruction.
C     M_dry[k] = a * m_L[k] * (n_L*I0 + b*I1 - n_6*I2)
C     where I0,I1,I2 are precomputed moment integrals.
C
C     INPUTS:
C       NL(ibins), NR(ibins), N6(ibins) - PPM coefficients for number
C
C     OUTPUTS:
C       MDRY(ibins) - dry mass per bin [kg]

      SUBROUTINE PPM_DRY_MASS_ANALYTICAL(NL, NR, N6, MDRY)

      IMPLICIT NONE
      include 'sizecode.COM'

C-----ARGUMENTS-----------------------------------------------------
      double precision NL(ibins), NR(ibins), N6(ibins)
      double precision MDRY(ibins)

C-----LOCAL VARIABLES-----------------------------------------------
      integer k
      double precision DELTA_XI
      parameter(DELTA_XI=0.69314718055994530d0)
      double precision PPM_I0, PPM_I1, PPM_I2
      parameter(PPM_I0=1.4426950408889634d0)
      parameter(PPM_I1=0.5672269882832156d0)
      parameter(PPM_I2=0.31609640204744345d0)
      double precision B_COEFF, ML

C-----CODE----------------------------------------------------------

      do k=1,ibins
         B_COEFF = (NR(k) - NL(k)) + N6(k)
         ML = xk(k)
         MDRY(k) = DELTA_XI * ML *
     &      (NL(k)*PPM_I0 + B_COEFF*PPM_I1 - N6(k)*PPM_I2)
      enddo

      RETURN
      END


C======================================================================
C     PPM_SPECIES_FLUX
C======================================================================
C
C     Compute mass flux for all species using upwind donor ratio
C     times the analytical dry mass flux.
C
C     F_M_all(k,j) = FM_DRY(k) * Mk(donor,j) / M_dry_analytical(donor)
C
C     INPUTS:
C       FM_DRY(ibins+1)         - analytical dry mass flux [kg/s]
C       MK_IN(ibins,icomp)      - mass of all species per bin [kg]
C       MDRY(ibins)             - analytical dry mass per bin [kg]
C       UEDGE(ibins+1)          - edge velocities (for upwind direction)
C
C     OUTPUTS:
C       FM_ALL(ibins+1,icomp)   - mass flux at edges for all species [kg/s]

      SUBROUTINE PPM_SPECIES_FLUX(FM_DRY, MK_IN, MDRY, UEDGE,
     &                            FM_ALL)

      IMPLICIT NONE
      include 'sizecode.COM'

C-----ARGUMENTS-----------------------------------------------------
      double precision FM_DRY(ibins+1)
      double precision MK_IN(ibins, icomp)
      double precision MDRY(ibins)
      double precision UEDGE(ibins+1)
      double precision FM_ALL(ibins+1, icomp)

C-----LOCAL VARIABLES-----------------------------------------------
      integer k, j, IDONOR
      double precision RATIO(ibins, icomp)
      double precision DONOR_R

C-----CODE----------------------------------------------------------

C     Compute composition ratio for all bins/species
      do k=1,ibins
         if (MDRY(k) .gt. 1.d-30) then
            do j=1,icomp
               RATIO(k,j) = MK_IN(k,j) / MDRY(k)
            enddo
         else
            do j=1,icomp
               RATIO(k,j) = 0.d0
            enddo
         endif
      enddo

C     Compute flux at each edge
      do k=1,ibins+1
C        Donor bin index (upwind)
         if (UEDGE(k) .ge. 0.d0) then
C           Flux from left cell
            IDONOR = max(1, k-1)
         else
C           Flux from right cell
            IDONOR = min(ibins, k)
         endif

         do j=1,icomp
            FM_ALL(k,j) = FM_DRY(k) * RATIO(IDONOR, j)
         enddo
      enddo

      RETURN
      END


C======================================================================
C     PPM_COMPUTE_SUBSTEPS
C======================================================================
C
C     Compute number of substeps needed for CFL stability.
C     Ensures max Courant number <= C_max = 0.8.
C
C     INPUTS:
C       TAU(ibins)  - growth forcing (full step)
C       WR(ibins)   - wet/dry ratio per bin
C
C     OUTPUTS:
C       NSUB        - number of substeps (>=1)

      SUBROUTINE PPM_COMPUTE_SUBSTEPS(TAU, WR, NSUB)

      IMPLICIT NONE
      include 'sizecode.COM'

C-----ARGUMENTS-----------------------------------------------------
      double precision TAU(ibins)
      double precision WR(ibins)
      integer NSUB

C-----LOCAL VARIABLES-----------------------------------------------
      integer k
      double precision DELTA_XI, CMAX
      parameter(DELTA_XI=0.69314718055994530d0)
      parameter(CMAX=0.8d0)
      double precision DMDT_INT
      external DMDT_INT
      double precision TAU_E(ibins+1), WR_E(ibins+1)
      double precision MF, MF_SAFE
      double precision DXIMAX, DXI_K

C-----CODE----------------------------------------------------------

C     Interpolate TAU/WR to edges
      TAU_E(1) = TAU(1)
      WR_E(1) = WR(1)
      do k=2,ibins
         TAU_E(k) = 0.5d0 * (TAU(k-1) + TAU(k))
         WR_E(k) = 0.5d0 * (WR(k-1) + WR(k))
      enddo
      TAU_E(ibins+1) = TAU(ibins)
      WR_E(ibins+1) = WR(ibins)

C     Maximum shift in xi-space
      DXIMAX = 0.d0
      do k=1,ibins+1
         MF = DMDT_INT(xk(k), TAU_E(k), WR_E(k))
         MF_SAFE = max(MF, 0.1d0 * xk(1))
         DXI_K = abs(dlog(MF_SAFE / xk(k)))
         if (DXI_K .gt. DXIMAX) DXIMAX = DXI_K
      enddo

C     Substep count: ceil(dxi_max / (C_max * delta_xi))
      if (DXIMAX .gt. 0.d0) then
         NSUB = int(DXIMAX / (CMAX * DELTA_XI) + 0.9999d0)
      else
         NSUB = 1
      endif
      if (NSUB .lt. 1) NSUB = 1

      RETURN
      END


C======================================================================
C     PPM_CONDENSATION_STEP
C======================================================================
C
C     Main PPM condensation orchestrator.
C     1. Freeze WR at step entry
C     2. Compute substep count from CFL
C     3. Substep loop: edge velocity -> reconstruct -> flux -> advect
C     4. Condensed mass added AFTER this call by ezcond_ppm
C
C     INPUTS:
C       NK_IN(ibins)           - number per bin [#]
C       MK_IN(ibins,icomp)     - mass per bin per species [kg]
C       TAU(ibins)             - growth forcing (full step)
C       CSPEC                  - condensing species index (unused here)
C       DT                     - full timestep [s]
C
C     OUTPUTS:
C       NK_OUT(ibins)          - updated number per bin
C       MK_OUT(ibins,icomp)    - updated mass per bin

      SUBROUTINE PPM_CONDENSATION_STEP(NK_IN, MK_IN, TAU, CSPEC,
     &                                  DT, NK_OUT, MK_OUT)

      IMPLICIT NONE
      include 'sizecode.COM'

C-----ARGUMENTS-----------------------------------------------------
      double precision NK_IN(ibins), MK_IN(ibins, icomp)
      double precision TAU(ibins)
      integer CSPEC
      double precision DT
      double precision NK_OUT(ibins), MK_OUT(ibins, icomp)

C-----LOCAL VARIABLES-----------------------------------------------
      integer k, j, isub, NSUB
      double precision DELTA_XI
      parameter(DELTA_XI=0.69314718055994530d0)
      double precision WR(ibins)
      double precision MKDRY, MKWET
      double precision DT_SUB
      double precision TAU_SUB(ibins)
      double precision UEDGE(ibins+1)
      double precision NBAR(ibins)
      double precision NL(ibins), NR(ibins), N6(ibins)
      double precision FN(ibins+1)
      double precision FM_DRY(ibins+1)
      double precision MDRY_A(ibins)
      double precision FM_ALL(ibins+1, icomp)

C-----CODE----------------------------------------------------------

C     1. Freeze WR
      do k=1,ibins
         MKDRY = 0.d0
         MKWET = 0.d0
         do j=1,icomp-idiag
            MKDRY = MKDRY + MK_IN(k,j)
         enddo
         do j=1,icomp
            MKWET = MKWET + MK_IN(k,j)
         enddo
         if (MKDRY .gt. 1.d-30) then
            WR(k) = MKWET / MKDRY
         else
            WR(k) = 1.d0
         endif
         WR(k) = max(WR(k), 1.d0)
      enddo

C     2. Compute substep count
      call PPM_COMPUTE_SUBSTEPS(TAU, WR, NSUB)
      DT_SUB = DT / dble(NSUB)
      do k=1,ibins
         TAU_SUB(k) = TAU(k) / dble(NSUB)
      enddo

C     Initialize working arrays
      do k=1,ibins
         NK_OUT(k) = NK_IN(k)
         do j=1,icomp
            MK_OUT(k,j) = MK_IN(k,j)
         enddo
      enddo

C     3. Substep loop
      do isub=1,NSUB

C        3a. Compute edge velocities
         call PPM_EDGE_VELOCITY(TAU_SUB, WR, DT_SUB, UEDGE)

C        3b. Reconstruct number density
         do k=1,ibins
            NBAR(k) = NK_OUT(k) / DELTA_XI
         enddo
         call PPM_RECONSTRUCT(NBAR, NL, NR, N6)

C        3c. Compute number flux and advect Nk
         call PPM_NUMBER_FLUX(NL, NR, N6, UEDGE, DT_SUB, FN)
         do k=1,ibins
            NK_OUT(k) = NK_OUT(k) - DT_SUB * (FN(k+1) - FN(k))
         enddo

C        3d. Compute analytical dry mass flux
         call PPM_MASS_FLUX(NL, NR, N6, UEDGE, DT_SUB, FM_DRY)

C        3e. Compute analytical dry mass for normalization
         call PPM_DRY_MASS_ANALYTICAL(NL, NR, N6, MDRY_A)

C        Transport all species
         call PPM_SPECIES_FLUX(FM_DRY, MK_OUT, MDRY_A, UEDGE,
     &                         FM_ALL)
         do k=1,ibins
            do j=1,icomp
               MK_OUT(k,j) = MK_OUT(k,j)
     &            - DT_SUB * (FM_ALL(k+1,j) - FM_ALL(k,j))
            enddo
         enddo

C        3f. Positivity clamp
         do k=1,ibins
            NK_OUT(k) = max(NK_OUT(k), 0.d0)
            do j=1,icomp
               MK_OUT(k,j) = max(MK_OUT(k,j), 0.d0)
            enddo
         enddo

      enddo  ! isub

      RETURN
      END
