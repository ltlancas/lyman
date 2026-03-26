"""
Analytical scaling relations for stellar feedback from massive stars.

Provides functions for computing characteristic length and timescales relevant
to the expansion of feedback bubbles in molecular clouds, including the
Strömgren radius, wind equilibration radii (momentum- and energy-driven),
wind shock radius, cloud radius, and the corresponding timescales (free-fall,
recombination, shell cooling, equilibration, and dynamical expansion times).

Author: Lachlan Lancaster
"""

import numpy as np
from astropy import units as u
from astropy import constants as aconsts
from astropy.units import Quantity

#########################################################################################
################################ Length Scale Quantities ################################
#########################################################################################

def RSt(Q0: Quantity["frequency"],
        nbar: Quantity["number density"],
        alphaB: Quantity["volumetric flow rate"] = 3.11e-13*(u.cm**3/u.s)
        ) -> Quantity["length"]:
    """
    Returns the Strömgren radius.

    Args:
        Q0: ionizing photon rate
        nbar: number density of hydrogen in the background
        alphaB: case B recombination rate

    Returns:
        Strömgren radius in parsecs
    """
    r_st = (3*Q0/(4*np.pi*nbar**2*alphaB))**(1./3)
    return r_st.to("pc")

def Req_MD(pdotw: Quantity["force"],
           rhobar: Quantity["mass density"],
           ci: Quantity["speed"] = 10*u.km/u.s
           ) -> Quantity["length"]:
    """
    Returns the equilibration radius for a momentum-driven wind bubble.

    The equilibration radius is where the ram pressure of the wind equals
    the thermal pressure of the photo-ionized gas.

    Args:
        pdotw: wind momentum input rate
        rhobar: background mass density
        ci: ionized gas sound speed

    Returns:
        Equilibration radius in parsecs
    """
    r_eq = (pdotw/(4*np.pi*rhobar*ci**2))**(1./2)
    return r_eq.to("pc")

def Req_ED(Lwind: Quantity["power"],
           rhobar: Quantity["mass density"],
           ci: Quantity["speed"] = 10*u.km/u.s
           ) -> Quantity["length"]:
    """
    Returns the equilibration radius for an energy-driven wind bubble.

    The equilibration radius is where the pressure of the hot wind bubble
    equals the thermal pressure of the photo-ionized gas.

    Args:
        Lwind: wind luminosity
        rhobar: background mass density
        ci: ionized gas sound speed

    Returns:
        Equilibration radius in parsecs
    """
    prefac = np.sqrt(7)/(22*np.pi)
    r_eq = (prefac*Lwind/(rhobar*(ci**3)))**(1./2)
    return r_eq.to("pc")

def Rch(Q0: Quantity["frequency"],
        nbar: Quantity["number density"],
        pdotw: Quantity["force"],
        rhobar: Quantity["mass density"],
        ci: Quantity["speed"] = 10*u.km/u.s,
        alphaB: Quantity["volumetric flow rate"] = 3.11e-13*(u.cm**3/u.s)
        ) -> Quantity["length"]:
    """
    Returns the characteristic radius Rch = Req^4 / RSt^3.

    This is the radius at which the force from the photo-ionized gas equals
    that of the momentum-driven wind bubble.

    Args:
        Q0: ionizing photon rate
        nbar: number density of hydrogen in the background
        pdotw: wind momentum input rate
        rhobar: background mass density
        ci: ionized gas sound speed
        alphaB: case B recombination rate

    Returns:
        Characteristic radius in parsecs
    """
    r_ch = Req_MD(pdotw, rhobar, ci=ci)**4 / RSt(Q0, nbar, alphaB=alphaB)**3
    return r_ch.to("pc")

def Rwshock(Mdotw: Quantity["mass flow rate"],
            rhobar: Quantity["mass density"],
            Vwind: Quantity["speed"]
            ) -> Quantity["length"]:
    """
    Returns the wind shock radius.

    This is the radius at which the inertia of the swept-up surrounding
    material is comparable to the wind momentum, causing the wind to shock.

    Args:
        Mdotw: wind mass loss rate
        rhobar: background mass density
        Vwind: wind terminal velocity

    Returns:
        Wind shock radius in parsecs
    """
    r_wshock = (Mdotw/(4*np.pi*rhobar*Vwind))**(1./2)
    return r_wshock.to("pc")

def Rcl(Mcl: Quantity["mass"],
        nbar: Quantity["number density"],
        muH: float = 1.4
        ) -> Quantity["length"]:
    """
    Returns the radius of a uniform-density spherical cloud.

    Args:
        Mcl: cloud mass
        nbar: background number density of hydrogen
        muH: mean molecular weight per hydrogen nucleus

    Returns:
        Cloud radius in parsecs
    """
    r_cl = (3*Mcl/(4*np.pi*nbar*muH*aconsts.m_p))**(1./3)
    return r_cl.to("pc")

#########################################################################################
################################# Time Scale Quantities #################################
#########################################################################################

def Twshock(Mdotw: Quantity["mass flow rate"],
            rhobar: Quantity["mass density"],
            Vwind: Quantity["speed"]
            ) -> Quantity["time"]:
    """
    Returns the wind shock timescale.

    This is the time for the wind to travel to the wind shock radius.

    Args:
        Mdotw: wind mass loss rate
        rhobar: background mass density
        Vwind: wind terminal velocity

    Returns:
        Wind shock timescale in Myr
    """
    t_wshock = Rwshock(Mdotw, rhobar, Vwind)/Vwind
    return t_wshock.to("Myr")

def Tcool(nbar: Quantity["number density"],
          Lwind: Quantity["power"]
          ) -> Quantity["time"]:
    """
    Returns the shell cooling timescale for a Weaver-like wind-blown bubble.

    Uses Equation 8 from Mac Low & McCray (1988).

    Args:
        nbar: number density of hydrogen in the background
        Lwind: wind luminosity

    Returns:
        Shell cooling timescale in Myr
    """
    t_cool = 2.3e-2*u.Myr
    t_cool *= (nbar/(u.cm**-3))**-0.71
    t_cool *= (Lwind/(1e38*u.erg/u.s))**0.29
    return t_cool

def Trec(nbar: Quantity["number density"],
         alphaB: Quantity["volumetric flow rate"] = 3.11e-13*(u.cm**3/u.s)
         ) -> Quantity["time"]:
    """
    Returns the ionization-recombination timescale.

    Args:
        nbar: number density of hydrogen in the background
        alphaB: case B recombination rate

    Returns:
        Recombination timescale in Myr
    """
    t_ion = (nbar*alphaB)**-1
    return t_ion.to("Myr")

def Tff(rhobar: Quantity["mass density"]) -> Quantity["time"]:
    """
    Returns the free-fall timescale.

    Args:
        rhobar: mass density

    Returns:
        Free-fall timescale in Myr
    """
    t_ff = (3*np.pi/(32*aconsts.G*rhobar))**(1./2)
    return t_ff.to("Myr")

def Teq_MD(pdotw: Quantity["force"],
           rhobar: Quantity["mass density"],
           ci: Quantity["speed"] = 10*u.km/u.s
           ) -> Quantity["time"]:
    """
    Returns the time to reach the equilibration radius for a momentum-driven bubble.

    Args:
        pdotw: wind momentum input rate
        rhobar: background mass density
        ci: ionized gas sound speed

    Returns:
        Equilibration timescale in Myr
    """
    t_eq = (((3*pdotw/(2*np.pi*rhobar))**(1./2)))/(6*ci**2)
    return t_eq.to("Myr")

def Teq_ED(Lwind: Quantity["power"],
           rhobar: Quantity["mass density"],
           ci: Quantity["speed"] = 10*u.km/u.s
           ) -> Quantity["time"]:
    """
    Returns the time to reach the equilibration radius for an energy-driven bubble.

    Args:
        Lwind: wind luminosity
        rhobar: background mass density
        ci: ionized gas sound speed

    Returns:
        Equilibration timescale in Myr
    """
    prefac = 0.2*(7**0.75)/((22*np.pi)**0.5)
    t_eq = prefac*(Lwind/(rhobar*(ci**5)))**(1./2)
    return t_eq.to("Myr")

def TSt_MD(Q0: Quantity["frequency"],
           nbar: Quantity["number density"],
           pdotw: Quantity["force"],
           rhobar: Quantity["mass density"],
           ci: Quantity["speed"] = 10*u.km/u.s,
           alphaB: Quantity["volumetric flow rate"] = 3.11e-13*(u.cm**3/u.s)
           ) -> Quantity["time"]:
    """
    Returns the time for an unimpeded momentum-driven wind bubble to reach the
    Strömgren radius.

    Args:
        Q0: ionizing photon rate
        nbar: number density of hydrogen in the background
        pdotw: wind momentum input rate
        rhobar: background mass density
        ci: ionized gas sound speed
        alphaB: case B recombination rate

    Returns:
        Strömgren crossing timescale in Myr
    """
    r_st = RSt(Q0, nbar, alphaB=alphaB)
    r_eq = Req_MD(pdotw, rhobar, ci=ci)
    t_eq = Teq_MD(pdotw, rhobar, ci=ci)
    t_st = t_eq*(r_st/r_eq)**(2)
    return t_st.to("Myr")

def TSt_ED(Q0: Quantity["frequency"],
           nbar: Quantity["number density"],
           Lwind: Quantity["power"],
           rhobar: Quantity["mass density"],
           ci: Quantity["speed"] = 10*u.km/u.s,
           alphaB: Quantity["volumetric flow rate"] = 3.11e-13*(u.cm**3/u.s)
           ) -> Quantity["time"]:
    """
    Returns the time for an unimpeded energy-driven wind bubble to reach the
    Strömgren radius.

    Args:
        Q0: ionizing photon rate
        nbar: number density of hydrogen in the background
        Lwind: wind luminosity
        rhobar: background mass density
        ci: ionized gas sound speed
        alphaB: case B recombination rate

    Returns:
        Strömgren crossing timescale in Myr
    """
    r_st = RSt(Q0, nbar, alphaB=alphaB)
    t_eq = Teq_ED(Lwind, rhobar, ci=ci)
    r_eq = Req_ED(Lwind, rhobar, ci=ci)
    t_st = t_eq*(r_st/r_eq)**(5./3)
    return t_st.to("Myr")

def Tdion(Q0: Quantity["frequency"],
          nbar: Quantity["number density"],
          ci: Quantity["speed"] = 10*u.km/u.s,
          alphaB: Quantity["volumetric flow rate"] = 3.11e-13*(u.cm**3/u.s)
          ) -> Quantity["time"]:
    """
    Returns the dynamical expansion timescale of a photo-ionized gas bubble.

    This is the sound-crossing time across the Strömgren radius.

    Args:
        Q0: ionizing photon rate
        nbar: number density of hydrogen in the background
        ci: ionized gas sound speed
        alphaB: case B recombination rate

    Returns:
        Dynamical expansion timescale in Myr
    """
    r_st = RSt(Q0, nbar, alphaB=alphaB)
    t_di = np.sqrt(3)*r_st/(2*ci)
    return t_di.to("Myr")
