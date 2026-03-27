"""
Star formation efficiency (SFE) prescriptions for molecular clouds.

Implements several analytic and semi-analytic models for the cloud-integrated
star formation efficiency as a function of cloud properties and feedback
parameters, including:

- Grudić et al. (2018): surface-density-based SFE fit
- Kim et al. (2018): ionization- and momentum-driven SFE prescriptions
- Thompson & Krumholz (2016): time-evolving ODE model for the gas, ejected
  gas, and stellar mass fractions in a turbulent cloud with log-normal surface
  density distribution

Author: Lachlan Lancaster
"""

import numpy as np
from astropy import units as u
from astropy import constants as aconsts
from astropy.units import Quantity
from . import quantities
from scipy.special import erf
from scipy.integrate import solve_ivp

def estar_Grudic18(Sigma_cl: Quantity["surface mass density"],
                   Scrit: Quantity["surface mass density"] = 2800*u.Msun/u.pc**2,
                   emax: float = 0.77) -> float:
    """
    Returns the star formation efficiency from Grudić et al. (2018).

    Implements Equation 11 of Grudić et al. (2018).

    Args:
        Sigma_cl: cloud surface density
        Scrit: critical surface density
        emax: maximum star formation efficiency parameter

    Returns:
        Dimensionless star formation efficiency
    """
    return 1./(1./emax + Scrit/Sigma_cl)

def sigma_ion_Kim18(Xi: Quantity,
                    ci: Quantity["speed"] = 10*u.km/u.s,
                    alphaB: Quantity["volumetric flow rate"] = 3.11e-13*(u.cm**3/u.s),
                    muH: float = 1.4
                    ) -> Quantity["surface mass density"]:
    """
    Returns the ionization surface density from Kim et al. (2018).

    Implements Equation 24 of Kim et al. (2018). This is the characteristic
    cloud surface density at which photoionization feedback becomes important.

    Args:
        Xi: ionizing photon rate per unit stellar mass
        ci: ionized gas sound speed
        alphaB: case B recombination rate
        muH: mean molecular weight per hydrogen nucleus

    Returns:
        Ionization surface density in solar masses per square parsec
    """
    sion = muH*aconsts.m_p*ci*(Xi/(8*aconsts.G*alphaB))**(1./2)
    return sion.to("solMass/pc^2")

def phitphiion_Kim18(Sigma_cl: Quantity["surface mass density"]) -> float:
    """
    Returns the product of the dimensionless evaporation rate and timescale.

    This product (phi_t * phi_ion) is defined in Equations 13 & 14 of Kim et al.
    (2018). The fit to simulations used here is given by Equation 16 of the same
    paper.

    Args:
        Sigma_cl: cloud surface density

    Returns:
        Dimensionless product phi_t * phi_ion
    """
    (c1, c2, c3) = (-2.89, 2.11, 25.3)
    S0 = Sigma_cl.to("Msun/pc^2").value
    return c1 + c2*np.log10(S0 + c3)

def estar_ion_Kim18(Sigma_cl: Quantity["surface mass density"],
                    Xi: Quantity,
                    ci: Quantity["speed"] = 10*u.km/u.s,
                    alphaB: Quantity["volumetric flow rate"] = 3.11e-13*(u.cm**3/u.s),
                    muH: float = 1.4
                    ) -> float:
    """
    Returns the photoionization-driven star formation efficiency from Kim et al. (2018).

    Implements Equation 26 of Kim et al. (2018).

    Args:
        Sigma_cl: cloud surface density
        Xi: ionizing photon rate per unit stellar mass
        ci: ionized gas sound speed
        alphaB: case B recombination rate
        muH: mean molecular weight per hydrogen nucleus

    Returns:
        Dimensionless star formation efficiency
    """
    sion = sigma_ion_Kim18(Xi, ci=ci, alphaB=alphaB, muH=muH)
    phitphiion = phitphiion_Kim18(Sigma_cl)
    xi = Sigma_cl/(phitphiion*sion)
    return (2*xi/(1 + np.sqrt(1 + 4*xi**2)))**2


def pstar_mstar(Sigma_cl: Quantity["surface mass density"],
                prefac: Quantity["speed"] = 135*u.km/u.s
                ) -> Quantity["speed"]:
    """
    Returns the momentum input per unit stellar mass as a function of cloud
    surface density.

    Implements Equation 18 of Kim et al. (2018).

    Args:
        Sigma_cl: cloud surface density
        prefac: normalization prefactor for p*/M*

    Returns:
        Momentum per unit stellar mass in km/s
    """
    return prefac*(Sigma_cl/(100*u.Msun/u.pc**2))**-0.74

def estar_Kim18(Sigma_cl: Quantity["surface mass density"],
                vej: Quantity["speed"] = 15*u.km/u.s,
                epsej: float = 0.13,
                prefac: Quantity["speed"] = 135*u.km/u.s
                ) -> float:
    """
    Returns the momentum-driven star formation efficiency from Kim et al. (2018).

    Implements Equation 28 of Kim et al. (2018).

    Args:
        Sigma_cl: cloud surface density
        vej: average velocity of gas ejected from the cloud by turbulence
        epsej: fraction of gas ejected in the initial turbulent phase
        prefac: normalization prefactor for p*/M*

    Returns:
        Dimensionless star formation efficiency
    """
    return (1- epsej)/(1 + pstar_mstar(Sigma_cl, prefac=prefac)/vej)

class TK16():
    """
    Thompson & Krumholz (2016) model for the time evolution of the star
    formation efficiency in a molecular cloud.

    Solves the coupled ODE system describing the evolution of gas, ejected
    gas, and stellar mass fractions as a function of time, driven by stellar
    feedback momentum injection into a turbulent cloud with a log-normal
    surface density distribution.
    """

    def __init__(self,
                 Mcl: Quantity["mass"],
                 Rcl: Quantity["length"],
                 epsff: float,
                 pdot_Mstar: Quantity["speed"] = 30*u.km/u.s/u.Myr,
                 sig_lnS: float = 1.5):
        """
        Args:
            Mcl: cloud mass
            Rcl: cloud radius
            epsff: star formation efficiency per free-fall time
            pdot_Mstar: momentum input rate per unit stellar mass
            sig_lnS: standard deviation of the log-normal surface density
                     distribution
        """
        self.Mcl = Mcl
        self.Rcl = Rcl
        self.Sigma_cl = Mcl/(np.pi*Rcl**2)
        self.rhobar = Mcl/(4*np.pi*Rcl**3/3)
        self.epsff = epsff
        self.pdot_Mstar = pdot_Mstar
        self.sig_lnS = sig_lnS
        self.Gamma = (pdot_Mstar/(4*np.pi*aconsts.G*self.Sigma_cl)).value

        self.tff0 = quantities.Tff(self.rhobar)

        self.solution = self.get_solution()

    def zeta_m(self, x):
        """
        Returns the mass fraction of the cloud above the critical log-surface
        density threshold x.

        Args:
            x: log-surface density threshold

        Returns:
            Mass fraction above threshold
        """
        arg = (self.sig_lnS**2 -2*x)/(2*self.sig_lnS*np.sqrt(2))
        return 0.5*(1- erf(arg))

    def get_solution(self):
        """
        Solves the Thompson & Krumholz (2016) ODE system using scipy's solve_ivp.

        Integrates the evolution equations for the gas, ejected gas, and stellar
        mass fractions (eps_gas, eps_ej, eps_star) in units of the initial
        free-fall time. Integration terminates when the gas fraction drops below
        0.1%.

        Returns:
            OdeSolution object from solve_ivp with dense output enabled
        """
        y0 = [1,0,0]

        def gas_depleted(t, y):
            return y[0] - 1e-3

        def derivs(t, y):
            (eg,eej,est) = y
            xcrit = np.log(4*self.Gamma*est/(3*eg*(est+eg)))
            zetah = self.zeta_m(xcrit)
            deej = zetah*eg*(eg+est)**0.5
            dest = self.epsff*eg*(eg+est)**0.5
            deg = -deej - dest
            return (deg, deej, dest)

        gas_depleted.terminal = True

        sol = solve_ivp(derivs, [0, 500], y0, events=[gas_depleted], dense_output=True)
        return sol

    def eps_gas(self, t: Quantity["time"]) -> float:
        """
        Returns the gas mass fraction as a function of time.

        Args:
            t: time

        Returns:
            Dimensionless gas mass fraction
        """
        return self.solution.sol((t/self.tff0).value)[0]

    def eps_ej(self, t: Quantity["time"]) -> float:
        """
        Returns the ejected gas mass fraction as a function of time.

        Args:
            t: time

        Returns:
            Dimensionless ejected gas mass fraction
        """
        return self.solution.sol((t/self.tff0).value)[1]

    def eps_star(self, t: Quantity["time"]) -> float:
        """
        Returns the cumulative star formation efficiency as a function of time.

        Args:
            t: time

        Returns:
            Dimensionless stellar mass fraction
        """
        return self.solution.sol((t/self.tff0).value)[2]
