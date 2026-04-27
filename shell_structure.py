# models specifying the structure of shells around 
# feedback-driven bubbles
# author: Lachlan Lancaster

import numpy as np
from astropy import units as u
from astropy import constants as ac
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from abc import ABC, abstractmethod

import fb_models

class Shell(ABC):
    def __init__(self, bub, **kwargs):
        self.bubble = bub
        self._set_parameters_parent(**kwargs)
        self._check_parameter_units_parent()

    def _set_parameters_parent(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

        if "rho0" not in self.__dict__:
            self.rho0 = 140*ac.m_p/(u.cm**3)

    def _check_parameter_units_parent(self):
        t1 = u.get_physical_type(self.rho0) == "mass density"
        if not t1:
            raise ValueError("Units of n0 are incorrect")
        if not isinstance(self.bubble, fb_models.Bubble):
            raise ValueError("bubble must be an instance of Bubble class")

    # TODO(ltlancas): put these checks in a utils file?
    #                 right now they're duplicated across multiple files
    def _check_time_units(self, t):
        t1 = u.get_physical_type(t)=="time"
        if not t1:
            raise ValueError("Units of t are incorrect")

    def _check_radius_units(self, r):
        r1 = u.get_physical_type(r)=="length"
        if not r1:
            raise ValueError("Units of r are incorrect")

    #######################################################
    ######## ABSTRACT METHODS FOR SHELL STRUCTURE #########
    #######################################################
    @abstractmethod
    def density(self, r, t):
        pass

    @abstractmethod
    def velocity(self, r, t):
        pass

    @abstractmethod
    def pressure(self, r, t):
        pass

class AdiabaticShell(Shell):
    def __init__(self, bub, **kwargs):
        super().__init__(bub, **kwargs)
        self._set_parameters()
        self._check_parameter_units()
        self._shell_solve()

    def _set_parameters(self):
        # adiabatic index
        if "gamma" not in self.__dict__:
            self.gamma = 5./3
        # second order deceleration parameter of the shell
        # related to the power-law index of the shell radius in time
        # Equal to -2/3 for the R ~ t^(3/5) solution
        if "kappa" not in self.__dict__:
            self.kappa = -2./3

    def _check_parameter_units(self):
        t1 = isinstance(self.gamma, (int, float))
        t2 = isinstance(self.kappa, (int, float))
        if not t1:
            raise ValueError("gamma must be dimensionless")
        if not t2:
            raise ValueError("kappa must be dimensionless")
        if self.gamma <= 1.0:
            raise ValueError("gamma must be > 1")
        if self.kappa >= 0.0:
            raise ValueError("kappa must be < 0")
        return None

    def _shell_solve(self):
        """
        Solves the structure equation for the dimensionless parameters of the shell
        surrounding an adiabatic bubble following section 2 of Weaver et al. (1977).
        """
        gamma = self.gamma
        kappa = self.kappa

        def derivs(xi, ys):
            (U, G, P) = ys
            t1 = kappa*G*(U-xi)/P - 2*gamma/xi - 2*kappa/U
            t2 = gamma - (U-xi)**2 *G/P
            Up = U*(t1/t2)
            t1 = Up + 2*U/xi
            t2 = U-xi
            Gp = -G*(t1/t2)
            Pp = P*(gamma*Gp/G - 2*kappa/(U-xi))
            return (Up, Gp, Pp)

        # stop if density goes to 0
        def event_1(t, ys):
            return ys[1]
        event_1.terminal = True

        U0 = 2./(gamma + 1)
        G0 = (gamma + 1)/(gamma - 1)
        P0 = 2/(gamma + 1)
        if not(hasattr(self, "ad_shell_sol")):
            self.ad_shell_sol = solve_ivp(derivs, (1, 0.5), [U0, G0, P0],\
                                          events=[event_1], dense_output=True,\
                                          rtol=1e-12, atol = 1e-12)
        return None

    def density(self, r, t):
        self._check_radius_units(r)
        self._check_time_units(t)
        # radius of the forward shock of the bubble
        r_b = self.bubble.radius(t)
        xi = (r/r_b).to(" ").value
        g_sol = self.ad_shell_sol.sol(xi)[1]
        return self.rho0 * g_sol

    def velocity(self, r, t):
        self._check_radius_units(r)
        self._check_time_units(t)
        r_b = self.bubble.radius(t)
        drb_dt = self.bubble.velocity(t)
        xi = (r/r_b).to(" ").value
        u_sol = self.ad_shell_sol.sol(xi)[0]
        return drb_dt * u_sol

    def pressure(self, r, t):
        self._check_radius_units(r)
        self._check_time_units(t)
        r_b = self.bubble.radius(t)
        drb_dt = self.bubble.velocity(t)
        xi = (r/r_b).to(" ").value
        p_sol = self.ad_shell_sol.sol(xi)[2]
        return self.rho0 * (drb_dt**2) * p_sol

class IsothermalShell(Shell):
    def __init__(self, bub, **kwargs):
        super().__init__(bub, **kwargs)
        self._set_parameters()
        self._check_parameter_units()

    def _set_parameters(self):
        # isothermal sound speed in the shell
        if "cs" not in self.__dict__:
            self.cs = 10*u.km/u.s

    def _check_parameter_units(self):
        t1 = u.get_physical_type(self.cs)=="speed"
        if not t1:
            raise ValueError("Units of cs are incorrect")
        return None

    def density(self, r, t):
        self._check_radius_units(r)
        self._check_time_units(t)
        r_b = self.bubble.radius(t)
        drb_dt = self.bubble.velocity(t)
        # density just inside the shell (post-shock)
        rho_ps = self.rho0 * (drb_dt/self.cs)**2
        # assuming an isothermal shock, the density profile in the shell is exponential
        res = rho_ps * np.exp((r_b - r)*drb_dt/self.cs**2)
        return res.to(u.g/u.cm**3)

    def velocity(self, r, t):
        self._check_radius_units(r)
        self._check_time_units(t)
        r_b = self.bubble.radius(t)
        drb_dt = self.bubble.velocity(t)
        rho_r = self.density(r, t)
        # mass conservation in the shell
        res = (self.rho0 * drb_dt * r_b**2)/(rho_r * r**2)
        return res.to(u.km/u.s)

    def pressure(self, r, t):
        self._check_radius_units(r)
        self._check_time_units(t)
        rho_r = self.density(r, t)
        return rho_r * self.cs**2