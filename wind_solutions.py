# The structure of winds from feedback sources
# author: Lachlan Lancaster

import numpy as np
from astropy import units as u
from astropy import constants as ac
from astropy.units import Quantity
import quantities
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from abc import ABC, abstractmethod

#########################################################################################
###########################   CLASSICAL WIND SOLUTION MODELS   ##########################
#########################################################################################

class WindModel(ABC):
    def __init__(self, **kwargs):
        self._set_parameters_parent(**kwargs)
        self._check_parameter_units_parent()

    def _set_parameters_parent(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

        # following CC85, set the default wind paramters in Msun/yr and 1e43 erg/s
        # this very roughly follows parameters for M82
        if "Mdot" not in self.__dict__:
            self.Mdot = 1.0 * u.Msun / u.yr

        if "Edot" not in self.__dict__:
            self.Edot = 1.0e43 * u.erg / u.s

    def _check_parameter_units_parent(self):
        if not u.get_physical_type(self.Edot) == "power":
            raise ValueError("Units of Edot are incorrect")
        if not u.get_physical_type(self.Mdot*u.s) == "mass":
            raise ValueError("Units of Mdot are incorrect")

    @abstractmethod
    def mach(self, r: Quantity["length"]) -> float:
        pass

    @abstractmethod
    def c(self, r: Quantity["length"]) -> Quantity["speed"]:
        pass

    @abstractmethod
    def u(self, r: Quantity["length"]) -> Quantity["speed"]:
        pass

    @abstractmethod
    def rho(self, r: Quantity["length"]) -> Quantity["mass density"]:
        pass

    @abstractmethod
    def press(self, r: Quantity["length"]) -> Quantity["pressure"]:
        pass

class CC85Wind(WindModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._set_parmeters()
        self._check_parameter_units()
        self._set_derived_parameters()

    def _set_parmeters(self):
        if "gamma" not in self.__dict__:
            self.gamma = 5./3
        if "R" not in self.__dict__:
            # 100 pc based more on CC85 values for M82
            self.R = 100.0 * u.pc

    def _check_parameter_units(self):
        if not u.get_physical_type(self.R) == "length":
            raise ValueError("Units of R are incorrect")
        if not u.get_physical_type(self.gamma) == "dimensionless":
            raise ValueError("gamma should be dimensionless")

    def _set_derived_parameters(self):
        self.vinf = np.sqrt(2*self.Edot/self.Mdot).to(u.km/u.s)

    def mach(self, r: Quantity["length"]) -> float:
        g = self.gamma
        # the below represent Equations 4 & 5 of CC85
        # these give M^-2
        y = (r/self.R).to(" ").value
        if np.isscalar(y):
            y = np.array([y])
            was_scalar = True
        else:
            was_scalar = False
        mm2 = np.zeros_like(y)
        for i in range(len(y)):
            if y[i] < 1:
                p1 = -1.*(3*g+1)
                p2 = 0.5*(g + 1)
                p3 = 5*g + 1
                f = lambda x: ((3*g+x)/(1+3*g))**p1 * ((g-1+2*x)/(g+1))**p2 - y[i]**p3
                mm2[i] = brentq(f,1,16*(y[i]**-2))
            else:
                p1 = 0.5*(g + 1)
                p2 = 2*(g - 1)
                f = lambda x: x**-1 * ((g-1+2*x)/(1+g))**p1 - y[i]**p2
                mm2[i] = brentq(f,1e-15,1)
        if was_scalar:
            return mm2[0]**-0.5
        else:
            return mm2**-0.5

    def c(self, r: Quantity["length"]) -> Quantity["speed"]:
        # sound speed profile
        mm = self.mach(r)
        return self.vinf / np.sqrt(mm**2 + 2./(self.gamma - 1))

    def u(self, r: Quantity["length"]) -> Quantity["speed"]:
        # velocity profile
        mm = self.mach(r)
        cc = self.c(r)
        return mm * cc

    def rho(self, r: Quantity["length"]) -> Quantity["mass density"]:
        uu = self.u(r)
        res = self.Mdot/(4*np.pi*r**2*uu)
        return res.to(u.g/u.cm**3)

    def press(self, r: Quantity["length"]) -> Quantity["pressure"]:
        cc = self.c(r)
        return cc**2 * self.rho(r) / self.gamma
