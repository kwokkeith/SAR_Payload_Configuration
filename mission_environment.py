"""
Module defining physical constants and environment parameters for mission simulations.

Author: Kwok Keith
Date: 14 Jan 2026
"""

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class PhysicalConstants:
    c: np.float64 = 299_792_458.0  # m/s, Speed of light in vacuum
    boltzman: np.float64 = 1.380649e-23  # J/K, Boltzmann constant
    r_e: np.float64 = (6_371_000.0,)  # meters, Earth's mean radius


@dataclass(frozen=True)
class EnvironmentParameters:
    nominal_temperature_k: np.float64 = 290.0  # Kelvin
