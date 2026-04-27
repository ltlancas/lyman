"""
Several different Cooling Functions for multiple uses cases

Author: Lachlan Lancaster
"""

import numpy as np
from astropy import units as u
from astropy import constants as ac
from astropy.units import Quantity
from abc import ABC, abstractmethod
from pathlib import Path

_HERE = Path(__file__).parent


class Cooling(ABC):
    """
    Abstract base class for cooling.

    Subclasses must implement cooling functions
    """

    def __init__(self, **kwargs):
        """
        Args:
            met: metallicity of the cooling function in log_10 realtive to solar
        """
        self._set_parmeters_parent(**kwargs)
        self._check_parameter_units_parent()

    def _set_parmeters_parent(self, **kwargs):
        """
        Sets base parameters from kwargs, applying the default background
        density if not provided.
        """
        for key, value in kwargs.items():
            setattr(self, key, value)

        if "met" not in self.__dict__:
            self.met = 0.0

    def _check_parameter_units_parent(self):
        """Validates the units of the base parameters."""
        if not u.get_physical_type(self.met) == "dimensionless":
            raise ValueError("Units of metallicity are incorrect")

    @abstractmethod
    def heating(self, T: Quantity["temperature"], rho:Quantity["mass density"]
                ) -> Quantity["power density"]:
        """Returns the total heating rate."""
        pass

    @abstractmethod
    def cooling(self, T: Quantity["temperature"], rho:Quantity["mass density"]
                ) -> Quantity["power density"]:
        """Returns the total cooling rate."""
        pass

    @abstractmethod
    def net_cooling(self, T: Quantity["temperature"], rho:Quantity["mass density"]
                ) -> Quantity["power density"]:
        """Returns the net cooling rate."""
        pass

    @abstractmethod
    def cooling_time(self, T: Quantity["temperature"], rho:Quantity["mass density"]
                ) -> Quantity["time"]:
        """Returns the net cooling time"""
        pass

class CBF23(Cooling):
    def __init__(self, **kwargs):
        self.set_parameters(**kwargs)

    def set_parameters(self, **kwargs):
        """
        Sets base parameters from kwargs
        """
        for key, value in kwargs.items():
            setattr(self, key, value)

        if "beta_lo" not in self.__dict__:
            self.beta_lo = -2.
        if "beta_hi" not in self.__dict__:
            self.beta_hi = 3.
        if "Tcold" not in self.__dict__:
            self.Tcold = 1e-2
        if "Thot" not in self.__dict__:
            self.Thot = 1.
        if "Tpeak" not in self.__dict__:
            self.Tpeak = np.power(self.Tcold**2 * self.Thot, 1./3)
        if "T_cutoff" not in self.__dict__:
            self.T_cutoff = self.Thot * 0.85
        if "gamma" not in self.__dict__:
            self.gamma = 5./3.
        if "chi" not in self.__dict__:
            self.chi = self.Thot / self.Tcold
        if "epsilon_T" not in self.__dict__:
            self.epsilon_T = 0.05
        if "pgas_0" not in self.__dict__:
            self.pgas_0 = 1.

    def _pressure(self, T: Quantity["temperature"], rho: Quantity["mass density"]
                  ) -> Quantity["pressure"]:
        return rho * T

    def heating(self, T: Quantity["temperature"], rho: Quantity["mass density"]
                ) -> Quantity["power density"]:
        """Returns the volumetric heating rate."""
        P = self._pressure(T, rho)
        P0 = self.pgas_0
        Thot, Tpeak = self.Thot, self.Tpeak
        beta_hi = self.beta_hi
        epsT = 1 + self.epsilon_T
        a_heat = ((self.beta_lo - beta_hi)
                  * (np.log(self.Tcold / Tpeak) / np.log(self.chi)) - beta_hi)

        edot_h = np.zeros(P.shape)
        edot_h[T < self.T_cutoff] = 1.5 * P0 * self.hc * (P[T < self.T_cutoff] / P0) / self.tcool
        edot_h[T < epsT * Thot] = edot_h[T < epsT * Thot] * (T[T < epsT * Thot] / Tpeak)**a_heat
        edot_h[T >= epsT * Thot] = edot_h[T >= epsT * Thot] * (
            (epsT * Thot / Tpeak)**a_heat
            * (T[T >= epsT * Thot] / (epsT * Thot))**(-beta_hi - 0.5)
        )
        return edot_h

    def cooling(self, T: Quantity["temperature"], rho: Quantity["mass density"]
                ) -> Quantity["power density"]:
        """Returns the volumetric cooling rate."""
        P = self._pressure(T, rho)
        P0 = self.pgas_0
        Tpeak = self.Tpeak
        beta_lo, beta_hi = self.beta_lo, self.beta_hi

        # 1/(gamma-1) = 1.5 for gamma=5/3
        edot_c = np.zeros(P.shape)
        edot_c[T < self.T_cutoff] = 1.5 * P0 * (P[T < self.T_cutoff] / P0)**2 / self.tcool
        edot_c[T < Tpeak] = edot_c[T < Tpeak] * (T[T < Tpeak] / Tpeak)**(-beta_lo)
        edot_c[T >= Tpeak] = edot_c[T >= Tpeak] * (T[T >= Tpeak] / Tpeak)**(-beta_hi)
        return edot_c

    def net_cooling(self, T: Quantity["temperature"], rho: Quantity["mass density"]
                   ) -> Quantity["power density"]:
        """Returns the net volumetric cooling rate (cooling minus heating)."""
        return self.cooling(T, rho) - self.heating(T, rho)

    def cooling_time(self, T: Quantity["temperature"], rho: Quantity["mass density"]
                    ) -> Quantity["time"]:
        """Returns the net cooling time t_cool = u_th / net_cooling."""
        P = self._pressure(T, rho)
        return P / self.net_cooling(T, rho) / (self.gamma - 1)
    
class KIGS(Cooling):
    """
    Implements the Cooling function used in Tan & Fielding 2024 (the "cloud atlas" paper)
    This uses Gnat & Sternberg 2007 at high temperatures and Koyama & Inutsuka 2002 at 
    low temperatures, which is pretty similar to the choices in Tigress Classic
    KIGS = Koyama, Inutsuka, Gnat, & Sternberg
    """
    def __init__(self):
        (X,Z) = (0.7, 0.02)
        (self.X, self.Z) = (X,Z)
        # about 0.6173, which is what was used before
        self.mu = 1.0/(2*X + 0.75*(1-X-Z) + Z/2.)
        self.mu_e = 2./(1+X)
        self.mu_H = 1./X
        self.nH_ntot = self.mu/self.mu_H
        self.ne_ntot = self.mu/self.mu_e
        self.gamma = 5./3

        cooling_data = np.loadtxt(_HERE / "cooling_tables/kigs_cooling.txt").T
        (self.Tarr, self.cooling_arr, self.heating_arr) = cooling_data
        self.Tarr *= u.K
        self.cooling_arr *= u.erg/u.s*(u.cm**3)
        self.heating_arr *= u.erg/u.s

    def _pressure(self, T, rho):
        n = (rho/(self.mu*ac.m_p)).to(u.cm**-3)
        P = n*ac.k_B*T
        return P

    def heating(self, T, rho):
        nH = (rho/self.muH*ac.m_p).to(u.cm**-3)
        T_int = np.log10(T.to(u.K).value)
        Ta_int = np.log10(self.T_arr.value)
        ha_int = np.log10(self.heating_arr.value)
        g_int = np.interp(T_int, Ta_int, ha_int)
        edot_h = nH*np.power(10, g_int)*u.erg/u.s
        return edot_h
    
    def cooling(self, T, rho):
        nH = (rho/self.muH*ac.m_p).to(u.cm**-3)
        ne = (rho/self.mue*ac.m_p).to(u.cm**-3)
        T_int = np.log10(T.to(u.K).value)
        Ta_int = np.log10(self.T_arr.value)
        ca_int = np.log10(self.cooling_arr.value)
        l_int = np.interp(T_int, Ta_int, ca_int)
        edot_c = ne*nH*np.power(10, l_int)*(u.erg/u.s*(u.cm**3))
        return edot_c

    def net_cooling(self, T, rho):
        edot_h = self.heating(T,rho)
        edot_c = self.cooling(T,rho)
        return edot_c - edot_h
    
    def cooling_time(self, T, rho):
        P = self._pressure(T,rho)
        cr = self.net_cooling(T,rho)
        ct = P/(self.gamma - 1) / cr
        return ct.to(u.yr)