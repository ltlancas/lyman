# Various prescriptions for the star formation efficiency (SFE)
# of a molecular cloud
# author: Lachlan Lancaster

import numpy as np
from astropy import units as u
from astropy import constants as aconsts
from astropy.units import Quantity
import quantities
from scipy.special import erf
from scipy.integrate import solve_ivp

def estar_Grudic18(Sigma_cl: Quantity["surface mass density"],
                   Scrit: Quantity["surface mass density"] = 2800*u.Msun/u.pc**2,
                   emax: float = 0.77) -> float:
    # sfe prescription from Equation 11 of Grudić et al. 2018
    # Sigma_cl : cloud surface density
    # Scrit : critical surface density
    # emax : maximum star formation efficiency parameter
    return 1./(1./emax + Scrit/Sigma_cl)

def sigma_ion_Kim18(Xi: Quantity["mass flow rate"],
                    ci: Quantity["speed"] = 10*u.km/u.s,
                    alphaB: Quantity["volumetric flow rate"] = 3.11e-13*(u.cm**3/u.s),
                    muH: float = 1.4
                    ) -> Quantity["surface mass density"]:
    # ionization surface density from Equation 24 of Kim et al. 2018
    # Xi : ionizing photon rate per unit stellar mass
    # ci : ionized gas sound speed
    # alphaB : case B recombination rate
    # muH : mean molecular weight per hydrogen nucleus
    sion = muH*aconsts.m_p*ci*(Xi/(8*aconsts.G*alphaB))**(1./2)
    return sion.to("solMass/pc^2")

def phitphiion_Kim18(Sigma_cl: Quantity["surface mass density"]) -> float:
    # product of dimensionless evaporation rate and evaporation time scale
    # defined in equations 13 & 14 of Kim et al. 2018
    # fit to simulations is given in Equation 16 of Kim et al. 2018
    # Sigma_cl : cloud surface density
    (c1, c2, c3) = (-2.89, 2.11, 25.3)
    S0 = Sigma_cl.to("Msun/pc^2").value
    return c1 + c2*np.log10(S0 + c3)

def estar_ion_Kim18(Sigma_cl: Quantity["surface mass density"],
                    Xi: Quantity["mass flow rate"],
                    ci: Quantity["speed"] = 10*u.km/u.s,
                    alphaB: Quantity["volumetric flow rate"] = 3.11e-13*(u.cm**3/u.s),
                    muH: float = 1.4
                    ) -> float:
    # sfe prescription from Equation 26 of Kim et al. 2018
    # Sigma_cl : cloud surface density
    # Xi : ionizing photon rate per unit stellar mass
    # ci : ionized gas sound speed
    # alphaB : case B recombination rate
    # muH : mean molecular weight per hydrogen nucleus
    sion = sigma_ion_Kim18(Xi, ci=ci, alphaB=alphaB, muH=muH)
    phitphiion = phitphiion_Kim18(Sigma_cl)
    xi = Sigma_cl/(phitphiion*sion)
    return (2*xi/(1 + np.sqrt(1 + 4*xi**2)))**2


def pstar_mstar(Sigma_cl: Quantity["surface mass density"],
                prefac: Quantity["speed"] = 135*u.km/u.s
                ) -> Quantity["speed"]:
    # momentum input per unit stellar mass as a funciton
    # of the cloud surface density
    # from Equation 18 of Kim et al. 2018
    # Sigma_cl : cloud surface density
    # prefac : prefactor of p/Mstar
    return prefac*(Sigma_cl/(100*u.Msun/u.pc**2))**-0.74

def estar_Kim18(Sigma_cl: Quantity["surface mass density"],
                vej: Quantity["speed"] = 15*u.km/u.s,
                epsej: float = 0.13,
                prefac: Quantity["speed"] = 135*u.km/u.s
                ) -> float:
    # sfe prescription from Equation 28 of Kim et al. 2018
    # Sigma_cl : cloud surface density
    # vej : average velocity of gas ejected from the cloud
    # epsej : fraction gas ejected in initial turbulence
    # prefac : prefactor of p/Mstar
    return (1- epsej)/(1 + pstar_mstar(Sigma_cl, prefac=prefac)/vej)

class TK16():
    # Thompson & Krumholz 2016 model for the evolution of the star formation efficiency
    # in a molecular cloud
    def __init__(self,
                 Mcl: Quantity["mass"],
                 Rcl: Quantity["length"],
                 epsff: float,
                 pdot_Mstar: Quantity["speed"] = 30*u.km/u.s/u.Myr,
                 sig_lnS: float = 1.5):
        # Mcl : cloud mass
        # Rcl : cloud radius
        # epsff : star formation efficiency per free-fall time
        # pdot_Mstar : momentum input rate per stellar mass
        # sig_lnS : std dev of the log-normal distribution of surface densities
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
        arg = (self.sig_lnS**2 -2*x)/(2*self.sig_lnS*np.sqrt(2))
        return 0.5*(1- erf(arg))

    def get_solution(self):
        # gets the solution to the Thompson & Krumholz '16 ODE model
        # using scipy's solve_ivp
        # initial conditions eps_gas, eps_ej, eps_star
        y0 = [1,0,0]

        def gas_depleted(t, y):
            # function to determine when to stop integrating
            # based on gas being depleted to less than 0.1%
            return y[0] - 1e-3

        def derivs(t, y):
            # y = [S, Sdot, Sddot]
            (eg,eej,est) = y
            xcrit = np.log(4*self.Gamma*est/(3*eg*(est+eg)))
            zetah = self.zeta_m(xcrit)
            deej = zetah*eg*(eg+est)**0.5
            dest = self.epsff*eg*(eg+est)**0.5
            deg = -deej - dest
            return (deg, deej, dest)

        # make sure to terminate integration on gas depletion
        gas_depleted.terminal = True

        sol = solve_ivp(derivs, [0, 500], y0, events=[gas_depleted], dense_output=True)
        return sol

    def eps_gas(self, t: Quantity["time"]) -> float:
        # get the gas fraction as a function of time
        return self.solution.sol((t/self.tff0).value)[0]

    def eps_ej(self, t: Quantity["time"]) -> float:
        # get the ejected gas as a function of time
        return self.solution.sol((t/self.tff0).value)[1]

    def eps_star(self, t: Quantity["time"]) -> float:
        # get the star formation efficiency at a given time
        return self.solution.sol((t/self.tff0).value)[2]
