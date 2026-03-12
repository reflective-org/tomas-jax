# References

Papers and textbooks referenced in the TOMAS-JAX codebase, organized by topic.

---

## TOMAS Model Development

- Adams, P. J.: Predicting global aerosol size distributions in general circulation models. Ph.D. thesis, California Institute of Technology, 2000.

- Adams, P. J. and Seinfeld, J. H.: Predicting global aerosol size distributions in general circulation models. *J. Geophys. Res.*, 107, 4370, https://doi.org/10.1029/2001JD001010, 2002.

- Lee, Y. H. and Adams, P. J.: A fast and efficient version of the TwO-Moment Aerosol Sectional (TOMAS) global aerosol microphysics model. *Aerosol Sci. Technol.*, 46, 678–689, https://doi.org/10.1080/02786826.2011.643259, 2012.

---

## Nucleation Parameterizations

- Riccobono, F., et al.: Oxidation products of biogenic emissions contribute to nucleation of atmospheric particles. *Science*, 344(6185), 717–721, 2014.

- Yu, F., et al.: Impact of temperature dependence on the possible contribution of increased galactic cosmic rays on the recent increase in aerosol nucleation rates. *Atmos. Chem. Phys.*, 17, 4997–5005, 2017.

- Dunne, E. M., et al.: Global atmospheric particle formation from CERN CLOUD measurements. *Science*, 354(6316), 1119–1124, https://doi.org/10.1126/science.aaf2649, 2016.

- Zhao, B., et al.: Global variability in atmospheric new particle formation mechanisms. *Nature*, 631, 98–105, https://doi.org/10.1038/s41586-024-07547-1, 2024.

- Wang, M., et al.: Synergistic HNO3-H2SO4-NH3 upper tropospheric particle formation. *Nature*, 605, 483–489, 2022.

- Kirkby, J., et al.: Ion-induced nucleation of pure biogenic particles. *Nature*, 533, 521–526, 2016.

- Lehtipalo, K., et al.: Multicomponent new particle formation from sulfuric acid, ammonia, and biogenic vapors. *Sci. Adv.*, 4, eaau5363, 2018.

- Almeida, J., et al.: Molecular understanding of sulphuric acid-amine particle nucleation in the atmosphere. *Nature*, 502, 359–363, 2013.

- Kurten, A., et al.: New particle formation in the sulfuric acid-dimethylamine-water system: reevaluation of CLOUD chamber measurements. *Atmos. Chem. Phys.*, 18, 845–863, 2018.

- Cai, R., et al.: Sulfuric acid-amine nucleation in urban Beijing. *Atmos. Chem. Phys.*, 21, 2457–2468, 2021.

- He, X.-C., et al.: Role of iodine oxoacids in atmospheric aerosol nucleation. *Science*, 371, 589–595, 2021.

---

## Aerosol Physics and Thermodynamics

- Seinfeld, J. H. and Pandis, S. N.: *Atmospheric Chemistry and Physics: From Air Pollution to Climate Change*, 3rd ed., John Wiley & Sons, Hoboken, NJ, 2016.

- Fuchs, N. A.: *The Mechanics of Aerosols*, Pergamon Press, Oxford, 1964.

- Tang, I. N.: Thermodynamic and optical properties of mixed-salt aerosols of atmospheric importance. *J. Geophys. Res.*, 102, 1883–1893, 1997.

- Nenes, A., Pandis, S. N., and Pilinis, C.: ISORROPIA: A new thermodynamic equilibrium model for multiphase multicomponent inorganic aerosols. *Aquat. Geochem.*, 4, 123–152, 1998.

---

## Numerical Methods — Condensation

- Colella, P. and Woodward, P. R.: The Piecewise Parabolic Method (PPM) for gas-dynamical simulations. *J. Comput. Phys.*, 54, 174–201, https://doi.org/10.1016/0021-9991(84)90143-8, 1984.

- Stevens, B., Feingold, G., Cotton, W. R., and Walko, R. L.: Elements of the microphysical structure of numerically simulated nonprecipitating stratocumulus. *J. Atmos. Sci.*, 53, 980–1006, https://doi.org/10.1175/1520-0469(1996)053<0980:EOTMSO>2.0.CO;2, 1996.

- Tzivion, S., Feingold, G., and Levin, Z.: An efficient numerical solution to the stochastic collection equation. *J. Atmos. Sci.*, 44, 3139–3149, https://doi.org/10.1175/1520-0469(1987)044<3139:AENSTT>2.0.CO;2, 1987.

- Tzivion, S., Feingold, G., and Levin, Z.: The evolution of raindrop spectra. Part II: Collisional collection/breakup and evaporation in a rainshaft. *J. Atmos. Sci.*, 46, 3312–3327, https://doi.org/10.1175/1520-0469(1989)046<3312:TEORSP>2.0.CO;2, 1989.

---

## SO2 Gas-Phase Chemistry

- Sun, W., et al.: Kinetics of OH + SO2 + M: The key reaction in atmospheric SO2-to-sulfate conversion. *Atmos. Chem. Phys.*, 22, 4969–4984, https://doi.org/10.5194/acp-22-4969-2022, 2022.

- Buck, A. L.: New equations for computing vapor pressure and enhancement factor. *J. Appl. Meteorol.*, 20, 1527–1532, 1981.

---

## Numerical Methods — Gas-Phase Transport

- Fuller, E. N., Schettler, P. D., and Giddings, J. C.: A new method for prediction of binary gas-phase diffusion coefficients. *Ind. Eng. Chem.*, 58(5), 18–27, 1966. *(Used via Perry's Chemical Engineers' Handbook)*

---

## Computational Frameworks

- Bradbury, J., et al.: JAX: composable transformations of Python+NumPy programs. http://github.com/google/jax, 2018.

- Tsitouras, C.: Runge-Kutta pairs of order 5(4) satisfying only the first column simplifying assumption. *Comput. Math. Appl.*, 62, 770–775, https://doi.org/10.1016/j.camwa.2011.06.002, 2011.

- Kidger, P.: On Neural Differential Equations. Ph.D. thesis, University of Oxford, 2022. *(Diffrax ODE solver library)*

---

## Climate Modeling and Aerosol Effects

- Liu, X., et al.: Toward a minimal representation of aerosols in climate models: description and evaluation in the Community Atmosphere Model CAM5. *Geosci. Model Dev.*, 5, 709–739, https://doi.org/10.5194/gmd-5-709-2012, 2012.

- Pierce, J. R. and Adams, P. J.: Uncertainty in global CCN concentrations from uncertain aerosol nucleation and primary emission rates. *Atmos. Chem. Phys.*, 9, 1339–1356, https://doi.org/10.5194/acp-9-1339-2009, 2009.

- Kodros, J. K. and Pierce, J. R.: Important global and regional differences in aerosol cloud-albedo effect estimates between simulations with and without prognostic aerosol microphysics. *J. Geophys. Res. Atmos.*, 122, 4003–4018, https://doi.org/10.1002/2016JD025886, 2017.

- Schneider, T., Lan, S., Stuart, A., and Teixeira, J.: Earth System Modeling 2.0: A blueprint for models that learn from observations and targeted high-resolution simulations. *Geophys. Res. Lett.*, 44, 12396–12417, https://doi.org/10.1002/2017GL076101, 2017.

---

## Related Software

- Hafner, D., Nuterman, R., and Jochum, M.: Veros v0.1 -- a fast and versatile ocean simulator in pure Python. *Geosci. Model Dev.*, 14, 4605–4631, https://doi.org/10.5194/gmd-14-4605-2021, 2021.

- Bartman, P., Bulenok, O., Unterstrasser, S., and Arabas, S.: PySDM v1: particle-based cloud modeling package for warm-rain microphysics and aqueous chemistry. *J. Open Source Softw.*, 7, 3219, https://doi.org/10.21105/joss.03219, 2022.

---

## Machine Learning for Climate

- Beucler, T., Pritchard, M., Rasp, S., Ott, J., Baldi, P., and Gentine, P.: Enforcing analytic constraints in neural networks emulating physical systems. *Phys. Rev. Lett.*, 126, 098302, https://doi.org/10.1103/PhysRevLett.126.098302, 2021.

- Brenowitz, N. D. and Bretherton, C. S.: Prognostic validation of a neural network unified physics parameterization. *Geophys. Res. Lett.*, 45, 6289–6298, https://doi.org/10.1029/2018GL078510, 2019.

---

## Experimental Design

- McKay, M. D., Beckman, R. J., and Conover, W. J.: A comparison of three methods for selecting values of input variables in the analysis of output from a computer code. *Technometrics*, 21, 239–245, 1979.

---

## Particle Dynamics

- Riemer, N., West, M., Zaveri, R. A., and Easter, R. C.: Simulating the evolution of soot mixing state with a particle-resolved aerosol model. *J. Geophys. Res.*, 114, D09202, https://doi.org/10.1029/2008JD011073, 2009.

---

*37 references. Last updated: 2026-03-12*
