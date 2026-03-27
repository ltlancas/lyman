"""
lyman
=====

A python library that implements the physical structure of several feedback-driven
bubbles and winds. This is meant to be used to quickly generate derived observable
properties as well as to provide a framework to test numerical simulations against.
"""

from . import fb_models
from . import sfe_prescriptions
from . import quantities
from . import wind_solutions

__version__ = "0.0.1"
__all__ = ['fb_models', 'sfe_prescriptions', 'quantities', 'wind_solutions']