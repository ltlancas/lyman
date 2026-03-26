# Models to the evolution of feedback bubbles
# stellar feedback from massive stars
# author: Lachlan Lancaster

import numpy as np
from astropy import units as u
from astropy import constants as ac
from astropy.units import Quantity
import quantities
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from abc import ABC, abstractmethod

import wind_solutions

#########################################################################################
########################### CLASSICAL BUBBLE EVOLUTION MODELS ###########################
#########################################################################################

class Bubble(ABC):
    def __init__(self, **kwargs):
        self._set_parmeters_parent(**kwargs)
        self._check_parameter_units_parent()

    def _set_parmeters_parent(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

        if "rho0" not in self.__dict__:
            self.rho0 = 140*ac.m_p/(u.cm**3)

    def _check_parameter_units_parent(self):
        if not u.get_physical_type(self.rho0) == "mass density":
            raise ValueError("Units of rho0 are incorrect")

    @abstractmethod
    def radius(self, t: Quantity["time"]) -> Quantity["length"]:
        pass

    @abstractmethod
    def velocity(self, t: Quantity["time"]) -> Quantity["speed"]:
        pass

    @abstractmethod
    def momentum(self, t: Quantity["time"]) -> Quantity["momentum"]:
        pass

    @abstractmethod
    def pressure(self, t: Quantity["time"]) -> Quantity["pressure"]:
        pass

class SedovTaylorBW(Bubble):
    # Sedov Taylor Solution for an instantaneous blast wave
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._set_parmeters(**kwargs)
        self._check_parameter_units()

    def _set_parmeters(self, **kwargs):
        if "E" not in self.__dict__:
            self.E = 1e51*u.erg

    def _check_parameter_units(self):
        if not u.get_physical_type(self.E) == "energy":
            raise ValueError("Units of E are incorrect")

    def radius(self, t: Quantity["time"]) -> Quantity["length"]:
        r_ST = 1.15167*(self.E*t**2/(self.rho0))**(1./5)
        return r_ST.to("pc")

    def velocity(self, t: Quantity["time"]) -> Quantity["speed"]:
        v_ST = 0.4*self.radius(t)/t
        return v_ST.to("km/s")

    def momentum(self, t: Quantity["time"]) -> Quantity["momentum"]:
        pr_ST = 4*np.pi*self.rho0*self.radius(t)**3*self.velocity(t)/3
        return pr_ST.to("solMass*km/s")

    def pressure(self, t: Quantity["time"]) -> Quantity["pressure"]:
        # TODO: fill this in with the correct number, this is simply
        #       a placeholder estimate
        press_ST = self.E/(self.radius(t)**3)
        return (press_ST/ac.k_B).to("K/cm3")

class Spitzer(Bubble):
    # Spitzer solution for a photo-ionized gas bubble
    # includes the Hosokawa & Inutsuka (2006) correction
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._set_parmeters(**kwargs)
        self._check_parameter_units()

        self.nbar = self.rho0/(self.muH*ac.m_p)
        self.RSt = quantities.RSt(self.Q0, self.nbar, alphaB=self.alphaB)
        self.tdio = quantities.Tdion(self.Q0, self.nbar, ci=self.ci, alphaB=self.alphaB)

    def _set_parmeters(self, **kwargs):
        if "Q0" not in self.__dict__:
            self.Q0 = 1e50/u.s
        if "ci" not in self.__dict__:
            self.ci = 10*u.km/u.s
        if "alphaB" not in self.__dict__:
            self.alphaB = 3.11e-13*(u.cm**3/u.s)
        if "muH" not in self.__dict__:
            self.muH = 1.4
        if "adj" not in self.__dict__:
            self.adj = True

    def _check_parameter_units(self):
        if not u.get_physical_type(self.Q0) == "frequency":
            raise ValueError("Units of Q0 are incorrect")
        if not u.get_physical_type(self.ci) == "speed":
            raise ValueError("Units of ci are incorrect")
        if not u.get_physical_type(self.alphaB) == "volumetric flow rate":
            raise ValueError("Units of alpha_B are incorrect")
        if not u.get_physical_type(self.muH) == "dimensionless":
            raise ValueError("Units of mu_H are incorrect")
        if not type(self.adj) == bool:
            raise ValueError("adj must be a boolean value")

    def rhoi(self, t: Quantity["time"]) -> Quantity["mass density"]:
        rhoi_sp = self.rho0*(1 + 7*t/(4*self.tdio))**(-3./2)
        return rhoi_sp.to("solMass/pc3")

    def radius(self, t: Quantity["time"]) -> Quantity["length"]:
        r_sp = self.RSt*(1 + 7*t/(4*self.tdio))**(4./7)
        return r_sp.to("pc")

    def velocity(self, t: Quantity["time"]) -> Quantity["speed"]:
        v_sp = (self.RSt/self.tdio)*(1 + 7*t/(4*self.tdio))**(-3./7)
        return v_sp.to("km/s")

    def momentum(self, t: Quantity["time"]) -> Quantity["momentum"]:
        prefac = 4*np.pi*self.rho0*self.RSt**4/(3*self.tdio)
        pr_sp = prefac*(1 + 7*t/(4*self.tdio))**(9./7)
        if self.adj:
            pr_sp *= (1 - (self.RSt/self.radius(t))**1.5)
        return pr_sp.to("solMass*km/s")

    def pressure(self, t: Quantity["time"]) -> Quantity["pressure"]:
        press_sp = self.rhoi(t)*self.ci**2
        return (press_sp/ac.k_B).to("K/cm3")

class EnergyDrivenWind(Bubble):
    # Weaver solution for a wind bubble
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._set_parmeters(**kwargs)
        self._check_parameter_units()

    def _set_parmeters(self, **kwargs):
        if "Lwind" not in self.__dict__:
            self.Lwind = 1e38*u.erg/u.s

    def _check_parameter_units(self):
        if not u.get_physical_type(self.Lwind) == "power":
            raise ValueError("Units of L_wind are incorrect")

    def radius(self, t: Quantity["time"]) -> Quantity["length"]:
        r_we = (125*self.Lwind*(t**3)/(154*np.pi*self.rho0))**(1./5)
        return r_we.to("pc")

    def velocity(self, t: Quantity["time"]) -> Quantity["speed"]:
        v_we = 0.6*self.radius(t)/t
        return v_we.to("km/s")

    def momentum(self, t: Quantity["time"]) -> Quantity["momentum"]:
        pr_we = 4*np.pi*self.rho0*self.radius(t)**3*self.velocity(t)/3
        return pr_we.to("solMass*km/s")

    def pressure(self, t: Quantity["time"]) -> Quantity["pressure"]:
        press_we = (10./33)*self.Lwind*t/((4*np.pi/3)*self.radius(t)**3)
        return (press_we/ac.k_B).to("K/cm3")

class AdiabaticWind(Bubble):
    # Weaver solution Section 2 for an adiabatic wind bubble
    # assumes no radiative losses, even in the shell.
    # We don't treat conduction here either

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._set_parmeters(**kwargs)
        self._check_parameter_units()
        self._ad_shell_solve(-2./3)
        self._set_derived_parameters()

        

        # set free-wind solution
        fw_dict = {"Mdot": self.Mdotw, "Edot": self.Lwind,
                   "R": self.rfb, "gamma":self.gamma}
        self.free_wind = wind_solutions.CC85Wind(**fw_dict)

    def _set_parmeters(self, **kwargs):
        # scaling paramter for dimensional analysis solution
        # given after Equation 13 of Weaver et al. (1977)
        self.alpha = 0.88

        if "Lwind" not in self.__dict__:
            self.Lwind = 1e38*u.erg/u.s
        if "Mdotw" not in self.__dict__:
            self.Mdotw = 1e-4*u.Msun/u.yr
        if "rfb" not in self.__dict__:
            self.rfb = 1.0*u.pc
        if "gamma" not in self.__dict__:
            self.gamma = 5./3
    
    def _check_parameter_units(self):
        if not u.get_physical_type(self.Lwind) == "power":
            raise ValueError("Units of L_wind are incorrect")
        if not u.get_physical_type(self.rfb) == "length":
            raise ValueError("Units of r_fb are incorrect")
        if not u.get_physical_type(self.Mdotw*u.s) == "mass":
            raise ValueError("Units of Mdot_w are incorrect")

    def _set_derived_parameters(self):
        # fraction of the shell's outer radius at which the shell's inner radius lies
        # approximate 0.86, but determined here from the numerical solution
        xic = self.ad_shell_sol.t[-1]
        # the dimensionless value of the pressure in the shell at the inner edge of the
        # shell radius. Approx 0.59 but determined here from the numerical solution
        Pxic = self.ad_shell_sol.y[2,-1]
        g = self.gamma
        # the dimensionless pre-factor in the scaling solution. Approx 0.88 but 
        # determined here for general gamma and the numerical solution
        self.alpha = (125*(g - 1)/(12*np.pi*xic**3*Pxic*(9*g - 4)))**0.2
        (self.xic, self.Pxic) = (xic, Pxic)
        self.Vwind = np.sqrt(2*self.Lwind/self.Mdotw).to("km/s")

    #################################################################
    #################   TOP-LINE DEFAULT FUNCTIONS   ################
    #################################################################

    def radius(self, t: Quantity["time"]) -> Quantity["length"]:
        r_we = self.alpha*(self.Lwind*(t**3)/self.rho0)**(1./5)
        return r_we.to("pc")

    def velocity(self, t: Quantity["time"]) -> Quantity["speed"]:
        v_we = 0.6*self.radius(t)/t
        return v_we.to("km/s")

    def momentum(self, t: Quantity["time"]) -> Quantity["momentum"]:
        pr_we = 4*np.pi*self.rho0*self.radius(t)**3*self.velocity(t)/3
        return pr_we.to("solMass*km/s")

    def pressure(self, t: Quantity["time"]) -> Quantity["pressure"]:
        g = self.gamma
        prefac = 15*(g-1)/(4*np.pi*(9*g-4)*(self.xic*self.alpha)**3)
        press_we = prefac*(self.Lwind**2 * self.rho0**3 / t**4)**(1./5)
        return (press_we/ac.k_B).to("K/cm3")

    def density_profile(self, r: Quantity["length"], t: Quantity["time"]) -> Quantity["mass density"]:
        # returns the density of the bubble at radius r and time t
        # r : radius (generally an array)
        # t : time (should be a scalar)
        g = self.gamma
        r_rs = self.R_rs(t)
        r_b = self.radius(t)
        r_c = self.xic*r_b
        rho_fw = lambda r: self.free_wind.rho(r)
        rho_sw = lambda r: ((g+1)/(g-1))*self.free_wind.rho(r_rs)
        r_sh_ref = r_b*self.ad_shell_sol.t[::-1]
        rho_sh_ref = self.rho0*self.ad_shell_sol.y[1,::-1]
        rho_sh = lambda r: np.interp(r, r_sh_ref, rho_sh_ref)
        rho_bg = lambda r: self.rho0
        rho = np.piecewise(r, [r<r_rs, (r>=r_rs) & (r<r_c), (r>=r_c) & (r<=r_b), r>r_b],\
                           [rho_fw, rho_sw, rho_sh, rho_bg])
        return rho.to("solMass/pc3")
    
    def velocity_profile(self, r: Quantity["length"], t: Quantity["time"]) -> Quantity["speed"]:
        # returns the velocity of the bubble at radius r and time t
        # r : radius (generally an array)
        # t : time (should be a scalar)
        g = self.gamma
        r_rs = self.R_rs(t)
        r_b = self.radius(t)
        r_c = self.xic*r_b
        u_fw = lambda r: self.free_wind.u(r)
        u_sw = lambda r: self._v_sw(r, t)
        r_sh_ref = r_b*self.ad_shell_sol.t[::-1]
        u_sh_ref = self.velocity(t)*self.ad_shell_sol.y[0,::-1]
        u_sh = lambda r: np.interp(r, r_sh_ref, u_sh_ref)
        u_bg = lambda r: 0
        u = np.piecewise(r, [r<r_rs, (r>=r_rs) & (r<r_c), (r>=r_c) & (r<=r_b), r>r_b],\
                           [u_fw, u_sw, u_sh, u_bg])
        return u.to("km/s")
    
    def pressure_profile(self, r: Quantity["length"], t: Quantity["time"]) -> Quantity["pressure"]:
        # returns the pressure of the bubble at radius r and time t
        # r : radius (generally an array)
        # t : time (should be a scalar)
        g = self.gamma
        r_rs = self.R_rs(t)
        r_b = self.radius(t)
        r_c = self.xic*r_b
        p_fw = lambda r: self.free_wind.press(r)
        p_sw = lambda r: self.pressure(t)*ac.k_B
        r_sh_ref = r_b*self.ad_shell_sol.t[::-1]
        p_sh_ref = self.velocity(t)**2*self.rho0*self.ad_shell_sol.y[2,::-1]
        p_sh = lambda r: np.interp(r, r_sh_ref, p_sh_ref)
        p_bg = lambda r: self.rho0*(1*u.km/u.s)**2
        p = np.piecewise(r, [r<r_rs, (r>=r_rs) & (r<r_c), (r>=r_c) & (r<=r_b), r>r_b],\
                           [p_fw, p_sw, p_sh, p_bg])
        return (p/ac.k_B).to("K/cm3")
    
    # TODO: make a separate file for shell structure solutions
    #       so this is similar to the free-wind implementation here 
    #     - Also, replace the string versions of the 
    #       param.to() unit conversion calls
    #################################################################
    ################ FUNCTIONS FOR INTERNAL STRUCTURE ###############
    #################################################################

    def _ad_shell_solve(self, kappa):
        """
        Solves the structure equation for the dimensionless parameters of the shell
        surrounding an adiabatic wind bubble following section 2 of Weaver et al. (1977).
        Args:
            kappa (float): "second order deceleration parameter" of the shell
                           related to the power-law index of the shell radius in time
                           Equal to -2/3 for the R ~ t^(3/5) solution
        Returns:
            None: sets self.ad_shell_sol to the solution object from solve_ivp
        """
        gamma = self.gamma

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
            self.ad_shell_sol = solve_ivp(derivs, (1, 0.75), [U0, G0, P0],\
                                          events=[event_1], dense_output=True,\
                                          rtol=1e-12, atol = 1e-12)
        return None
    
    def _v_sw(self, r: Quantity["length"], t: Quantity["time"]) -> Quantity["speed"]:
        # gives the radial velocity in the shocked wind region
        g = self.gamma
        r_c = self.xic*self.radius(t)
        gfac = (9*g-4)/(15*g)
        t1 = (gfac*r_c**3/(r**2*t)).to("km/s")
        t2 = ((4/(15*g))*(r/t)).to("km/s")
        return t1 + t2
    
    def R_rs(self, t: Quantity["time"]) -> Quantity["length"]:
        # gives the radius of the reverse shock as a function of time in the
        # adiabatic wind bubble solution
        Rc = self.xic*self.radius(t)
        R_ballistic = self.Vwind*t
        g = self.gamma
        gfac = ((g+1)/(g-1)) * ((9*g-4)/(15*g)) * ((g+1)**2/(4*g))**(1/(g-1))
        res = np.sqrt(gfac*Rc**3/R_ballistic)
        return res.to("pc")


class MomentumDrivenWind(Bubble):
    # Momentum-driven bubble solution
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._set_parmeters(**kwargs)
        self._check_parameter_units()

    def _set_parmeters(self, **kwargs):
        if "pdotw" not in self.__dict__:
            self.pdotw = 1e5*u.Msun*u.km/u.s/u.Myr

    def _check_parameter_units(self):
        if not u.get_physical_type(self.pdotw) == "force":
            raise ValueError("Units of pdotw are incorrect")

    def radius(self, t: Quantity["time"]) -> Quantity["length"]:
        r_md = ((3*self.pdotw*t**2)/(2*np.pi*self.rho0))**(1./4)
        return r_md.to("pc")

    def velocity(self, t: Quantity["time"]) -> Quantity["speed"]:
        v_md = 0.5*self.radius(t)/t
        return v_md.to("km/s")

    def momentum(self, t: Quantity["time"]) -> Quantity["momentum"]:
        pr_md = self.pdotw*t
        return pr_md.to("solMass*km/s")

    def pressure(self, t: Quantity["time"]) -> Quantity["pressure"]:
        press_md = self.pdotw/(4*np.pi*self.radius(t)**2)
        return (press_md/ac.k_B).to("K/cm3")


#########################################################################################
################################### CO-EVOLUTION MODELS #################################
#########################################################################################

class MD_CEM(Bubble):
    # Joint solution for the evolution of a photo-ionized gas bubble
    # and a momentum-driven wind bubble in force balance with one another
    # assumes that the bubbles are uncoupled and evolve independently
    # up until t_eq, the equilibration time
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._set_parmeters(**kwargs)
        self._check_parameter_units()
        self._set_derived_parameters()

        # Separate Spitzer solution
        sp_dict = {"rho0": self.rho0, "Q0": self.Q0, "ci": self.ci,\
                   "alphaB": self.alphaB, "muH": self.muH}
        self.spitz_bubble = Spitzer(**sp_dict)
        # separate momentum-driven wind bubble
        md_dict = {"rho0": self.rho0, "pdotw": self.pdotw}
        self.wind_bubble = MomentumDrivenWind(**md_dict)
        # call ODE integrator to get the joint evolution solution
        self.joint_sol = self.joint_evol()

    def _set_parmeters(self, **kwargs):
        if "Q0" not in self.__dict__:
            self.Q0 = 1e50/u.s
        if "pdotw" not in self.__dict__:
            self.pdotw = 1e5*u.Msun*u.km/u.s/u.Myr
        if "ci" not in self.__dict__:
            self.ci = 10*u.km/u.s
        if "alphaB" not in self.__dict__:
            self.alphaB = 3.11e-13*(u.cm**3/u.s)
        if "muH" not in self.__dict__:
            self.muH = 1.4

    def _check_parameter_units(self):
        if not u.get_physical_type(self.Q0) == "frequency":
            raise ValueError("Units of Q0 are incorrect")
        if not u.get_physical_type(self.ci) == "speed":
            raise ValueError("Units of ci are incorrect")
        if not u.get_physical_type(self.pdotw) == "force":
            raise ValueError("Units of pdotw are incorrect")
        if not u.get_physical_type(self.alphaB) == "volumetric flow rate":
            raise ValueError("Units of alpha_B are incorrect")
        if not u.get_physical_type(self.muH) == "dimensionless":
            raise ValueError("Units of mu_H are incorrect")

    def _set_derived_parameters(self):
        (ci, alphaB, muH) = (self.ci, self.alphaB, self.muH)
        (Q0, pdotw, rho0) = (self.Q0, self.pdotw, self.rho0)
        self.nbar = rho0/(muH*ac.m_p)
        self.RSt = quantities.RSt(Q0, self.nbar, alphaB=alphaB)
        self.teq = quantities.Teq_MD(pdotw, rho0, ci=ci)
        self.Req = quantities.Req_MD(pdotw, rho0, ci=ci)
        self.Rch = quantities.Rch(Q0, self.nbar, pdotw, rho0, ci=ci, alphaB=alphaB)
        self.tdio = quantities.Tdion(Q0, self.nbar, ci=ci, alphaB=alphaB)
        self.tff = quantities.Tff(rho0)
        self.pscl = ((4*np.pi/3)*rho0*(self.Req**4)/self.tdio).to("solMass*km/s")
        self.zeta = (self.Req/self.RSt).to(" ").value

        if self.zeta < 1:
            self.tswitch = self.teq
        else:
            tot = self._get_Tot()
            self.tswitch = min(self.teq.value, tot.value)*u.Myr

    def _get_Tot(self):
        # Returns the time at which the WBB overtakes the PIR
        # used as the switch-over time in the zeta > 1 case.
        # root found in dimensionless form, as in Equation C35
        # of Paper 1
        fac1 = (4.5**0.25)*np.sqrt(self.zeta)
        f  = lambda x: fac1*np.sqrt(x) - (1 + 1.75*x)**(4./7)
        # over-take time only matters if it is smaller than t_eq
        chi_eq = (self.teq/self.tdio).to(" ").value
        try:
            chi_ot = brentq(f, 0, chi_eq)
        except:
            chi_ot = chi_eq
        return (chi_ot*self.tdio).to(u.Myr)

    @staticmethod
    def _get_largest_real(roots):
        real_roots = np.real(roots[np.isreal(roots)])
        return np.max(real_roots)

    def get_xiw(self, xii):
        xiw = []
        for xi in xii:
            p = [self.zeta**-3, 1.0, 0., 0., -1*(xi**3)]
            roots = np.roots(p)
            xiw.append(self._get_largest_real(np.roots(p)))
        return np.array(xiw)

    def joint_evol(self):
        # Gives the solution for the joint dynamical evolution of
        # photo-ionized gas and a wind bubble
        # zeta : the Req/RSt ratio, free parameter of the model

        zeta = self.zeta

        if zeta < 1:
            xiw0 = 1
        else:
            xiw0 = self.wind_bubble.radius(self.tswitch)/self.Req
            xiw0 = xiw0.to(" ").value
        xii0 = xiw0*((1+(zeta**-3)*xiw0)**(1./3))
        momentum_tot = self.wind_bubble.momentum(self.tswitch)
        momentum_tot += self.spitz_bubble.momentum(self.tswitch)
        mass_tot = 4*np.pi*self.rho0*((self.Req*xii0)**3)/3
        psi0 = (((momentum_tot/mass_tot)*(self.tdio/self.Req)).to(" ")).value

        # pre-calculate the relationship between xii and xiw
        xii_range = np.linspace(xii0/2,100*xii0,1000)
        xiw_prec = self.get_xiw(xii_range)

        # defin the differential equations
        def derivs(chi,y):
            xii = y[0]
            psi = y[1]
            xiw = np.interp(xii,xii_range,xiw_prec)
            t1 = (2.25*(zeta**-2)*(1 + (zeta**-3)*xiw)**(2./3))/(xii**3)
            t2 = 3*(psi**2)/xii
            return (psi,t1-t2)

        # use solve_ivp to get solution
        return solve_ivp(derivs,[0,100],[xii0,psi0],dense_output=True)

    def radius(self, t: Quantity["time"]) -> Quantity["length"]:
        # Returns the radius of the ionized bubble at time t
        ri = self.spitz_bubble.radius(t)*(t<self.tswitch)
        chi = ((t-self.tswitch)/self.tdio).to(" ").value
        solution =  self.joint_sol.sol(chi)
        ri += solution[0]*self.Req*(t>self.tswitch)
        return ri.to("pc")

    def wind_radius(self, t: Quantity["time"]) -> Quantity["length"]:
        # Returns the radius of the wind bubble at time t
        # up until tswitch the wind bubble follows the normal momentum-driven solution
        rw = self.wind_bubble.radius(t)*(t<self.tswitch)
        # afterwards it follows the joint evolution solution
        chi = ((t-self.tswitch)/self.tdio).to(" ").value
        solution =  self.joint_sol.sol(chi)
        xiw = self.get_xiw(solution[0])
        rw += xiw*self.Req*(t>self.tswitch)
        return rw.to("pc")

    def velocity(self, t: Quantity["time"]) -> Quantity["speed"]:
        # Returns the velocity of the ionized bubble at time t
        # up until tswitch the ionized bubble follows the Spitzer solution
        vi = self.spitz_bubble.velocity(t)*(t<self.tswitch)
        chi = ((t-self.tswitch)/self.tdio).to(" ").value
        solution =  self.joint_sol.sol(chi)
        vi += solution[1]*self.Req/self.tdio*(t>self.tswitch)
        return vi.to("km/s")

    def momentum(self, t: Quantity["time"]) -> Quantity["momentum"]:
        # returns the momentum carried by the joint bubble at time t
        prefac = self.pscl
        chi = ((t-self.tswitch)/self.tdio).to(" ").value
        solution =  self.joint_sol.sol(chi)
        pr = prefac*solution[1]*solution[0]**3*(t>self.tswitch)
        pr += self.spitz_bubble.momentum(t)*(t<self.tswitch)
        pr += self.wind_bubble.momentum(t)*(t<self.tswitch)
        return pr.to("solMass*km/s")

    def momentum_uncoupled(self, t: Quantity["time"]) -> Quantity["momentum"]:
        # returns the momentum carried by the joint bubble at time t
        # if the two constituent bubbles evolved independently
        pr = self.spitz_bubble.momentum(t)
        pr += self.wind_bubble.momentum(t)
        return pr.to("solMass*km/s")

    def pressure(self, t: Quantity["time"]) -> Quantity["pressure"]:
        # returns the pressure of the wind bubble at time t
        press = self.pdotw/(4*np.pi*self.wind_radius(t)**2)
        return (press/ac.k_B).to("K/cm3")

    def pressure_ionized(self, t: Quantity["time"]) -> Quantity["pressure"]:
        # returns the pressure of the ionized bubble at time t
        press = self.spitz_bubble.pressure(t)*(t<self.tswitch)
        press += self.pressure(t)*(t>self.tswitch)
        return press

class ED_CEM(Bubble):
    # Joint solution for the evolution of a photo-ionized gas bubble
    # and a wind bubble in force balance with each other
    # assumes that the bubbles are uncoupled and evolve independently
    # up until t_eq, the equilibration time
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._set_parmeters(**kwargs)
        self._check_parameter_units()
        self._set_derived_parameters()

        # Separate Spitzer solution
        sp_dict = {"rho0": self.rho0, "Q0": self.Q0, "ci": self.ci,\
                   "alphaB": self.alphaB, "muH": self.muH}
        self.spitz_bubble = Spitzer(**sp_dict)
        # separate energy-driven wind bubble
        w_dict = {"rho0": self.rho0, "Lwind": self.Lwind}
        self.wind_bubble = EnergyDrivenWind(**w_dict)
        # call ODE integrator to get the joint evolution solution
        self.joint_sol = self.joint_evol()

    def _set_parmeters(self, **kwargs):
        if "Q0" not in self.__dict__:
            self.Q0 = 1e50/u.s
        if "Lwind" not in self.__dict__:
            self.Lwind = 1e38*u.erg/u.s
        if "ci" not in self.__dict__:
            self.ci = 10*u.km/u.s
        if "alphaB" not in self.__dict__:
            self.alphaB = 3.11e-13*(u.cm**3/u.s)
        if "muH" not in self.__dict__:
            self.muH = 1.4

    def _check_parameter_units(self):
        if not u.get_physical_type(self.Q0) == "frequency":
            raise ValueError("Units of Q0 are incorrect")
        if not u.get_physical_type(self.ci) == "speed":
            raise ValueError("Units of ci are incorrect")
        if not u.get_physical_type(self.Lwind) == "power":
            raise ValueError("Units of L_wind are incorrect")
        if not u.get_physical_type(self.alphaB) == "volumetric flow rate":
            raise ValueError("Units of alpha_B are incorrect")
        if not u.get_physical_type(self.muH) == "dimensionless":
            raise ValueError("Units of mu_H are incorrect")

    def _set_derived_parameters(self):
        (ci, alphaB, muH) = (self.ci, self.alphaB, self.muH)
        (Q0, Lwind, rho0) = (self.Q0, self.Lwind, self.rho0)
        self.nbar = rho0/(muH*ac.m_p)
        self.RSt = quantities.RSt(Q0, self.nbar, alphaB=alphaB)
        self.teq = quantities.Teq_ED(Lwind, rho0, ci=ci)
        self.Req = quantities.Req_ED(Lwind, rho0, ci=ci)
        self.tdio = quantities.Tdion(Q0, self.nbar, ci=ci, alphaB=alphaB)
        self.tff = quantities.Tff(rho0)
        self.pscl = ((4*np.pi/3)*rho0*(self.Req**4)/self.tdio).to("solMass*km/s")

        self.zeta = (self.Req/self.RSt).to(" ").value
        if self.zeta < 1:
            self.tswitch = self.teq
        else:
            tot = self._get_Tot()
            self.tswitch = min(self.teq.value, tot.value)*u.Myr

    def _get_Tot(self):
        # Returns the time at which the WBB overtakes the PIR
        # used as the seitch-over time in the zeta > 1 case.
        # root found in dimensionless form, as in Equation C35
        # of Paper 1
        fac1 = ((2.5*np.sqrt(3./7))**0.6)*(self.zeta**0.4)
        f  = lambda x: fac1*(x**0.6) - (1 + 1.75*x)**(4./7)
        # over-take time only matters if it is smaller than t_eq
        chi_eq = (self.teq/self.tdio).to(" ").value
        try:
            chi_ot = brentq(f, 0, chi_eq)
        except:
            chi_ot = chi_eq
        return (chi_ot*self.tdio).to(u.Myr)

    def joint_evol(self):
        # Gives the solution for the joint dynamical evolution of
        # photo-ionized gas and a wind bubble
        # zeta : the Req/RSt ratio, free parameter of the model

        zeta = self.zeta

        if zeta < 1:
            xiw0 = 1
        else:
            xiw0 = self.wind_bubble.radius(self.tswitch)/self.Req
        xii0 = xiw0*((1+(zeta**-3)*xiw0)**(1./3))
        Pfac = self.wind_bubble.pressure(self.tswitch)*ac.k_B/(self.rho0*self.ci**2)
        Pfac = Pfac.to(" ").value
        Et0 = (2./11)*np.sqrt(7./3)*zeta*(xiw0**3)*Pfac
        # get initial condition for derivative of xii -> Mi
        momentum_tot = self.wind_bubble.momentum(self.tswitch)
        momentum_tot += self.spitz_bubble.momentum(self.tswitch)
        mass_tot = 4*np.pi*self.rho0*((self.Req*xii0)**3)/3
        dxii_dchi0 = (((momentum_tot/mass_tot)*(self.tdio/self.Req)).to(" ")).value
        Mi0 = (2*zeta/np.sqrt(3))*dxii_dchi0

        # define the differential equations
        def derivs(chi,y):
            (xii,Mi,xiw,Et) = y
            Pt = (11./2)*np.sqrt(3/7)*Et*(xiw**-3)/zeta
            A = 2/(3*((xiw*zeta)**3)*(Pt**2))
            dlnxii_dchi = (np.sqrt(3)/(2*zeta))*Mi/xii

            dxii_dchi = dlnxii_dchi*xii
            dMi_dchi = (3*np.sqrt(3)/(2*zeta*xii))*(Pt - Mi**2)
            dlnxiw_dchi = (((xii/xiw)**3)*dlnxii_dchi + A/Et)/(1 + 5*A)
            dxiw_dchi = dlnxiw_dchi*xiw
            dlnEt_chi = 1./Et - 2*dlnxiw_dchi
            dEt_dchi = dlnEt_chi*Et
            return (dxii_dchi,dMi_dchi,dxiw_dchi,dEt_dchi)

        # use solve_ivp to get solution
        return solve_ivp(derivs,[0,100],[xii0,Mi0,xiw0,Et0],dense_output=True)

    def radius(self, t: Quantity["time"]) -> Quantity["length"]:
        # Returns the radius of the ionized bubble at time t
        ri = self.spitz_bubble.radius(t)*(t<self.tswitch)
        chi = ((t-self.tswitch)/self.tdio).to(" ").value
        solution =  self.joint_sol.sol(chi)
        ri += solution[0]*self.Req*(t>self.tswitch)
        return ri.to("pc")

    def wind_radius(self, t: Quantity["time"]) -> Quantity["length"]:
        # Returns the radius of the wind bubble at time t
        # up until tswitch the wind bubble follows the normal momentum-driven solution
        rw = self.wind_bubble.radius(t)*(t<self.tswitch)
        # afterwards it follows the joint evolution solution
        chi = ((t-self.tswitch)/self.tdio).to(" ").value
        solution =  self.joint_sol.sol(chi)
        xiw = solution[2]
        rw += xiw*self.Req*(t>self.tswitch)
        return rw.to("pc")

    def velocity(self, t: Quantity["time"]) -> Quantity["speed"]:
        # Returns the velocity of the ionized bubble at time t
        # up until tswitch the ionized bubble follows the Spitzer solution
        vi = self.spitz_bubble.velocity(t)*(t<self.tswitch)
        chi = ((t-self.tswitch)/self.tdio).to(" ").value
        solution =  self.joint_sol.sol(chi)
        vi += solution[1]*self.ci*(t>self.tswitch)
        return vi.to("km/s")

    def momentum(self, t: Quantity["time"]) -> Quantity["momentum"]:
        # returns the momentum carried by the joint bubble at time t
        prefac = (4*np.pi/3)*self.Req**3*self.rho0*self.ci
        chi = ((t-self.tswitch)/self.tdio).to(" ").value
        solution =  self.joint_sol.sol(chi)
        pr = prefac*solution[1]*solution[0]**3*(t>self.tswitch)
        pr += self.spitz_bubble.momentum(t)*(t<self.tswitch)
        pr += self.wind_bubble.momentum(t)*(t<self.tswitch)
        return pr.to("solMass*km/s")

    def momentum_uncoupled(self, t: Quantity["time"]) -> Quantity["momentum"]:
        # returns the momentum carried by the joint bubble at time t
        # if the two constituent bubbles evolved independently
        pr = self.spitz_bubble.momentum(t)
        pr += self.wind_bubble.momentum(t)
        return pr.to("solMass*km/s")

    def pressure(self, t: Quantity["time"]) -> Quantity["pressure"]:
        # returns the pressure of the wind bubble at time t
        press = self.wind_bubble.pressure(t)*(t<self.tswitch)
        chi = ((t-self.tswitch)/self.tdio).to(" ").value
        solution =  self.joint_sol.sol(chi)
        Pt = (11./2)*np.sqrt(3/7)*solution[3]*(solution[2]**-3)/self.zeta
        Pt = Pt*self.rho0*self.ci**2
        press += Pt*(t>self.tswitch)/ac.k_B
        return (press).to("K/cm3")

    def pressure_ionized(self, t: Quantity["time"]) -> Quantity["pressure"]:
        # returns the pressure of the ionized bubble at time t
        press = self.spitz_bubble.pressure(t)*(t<self.tswitch)
        press += self.pressure(t)*(t>self.tswitch)
        return press
