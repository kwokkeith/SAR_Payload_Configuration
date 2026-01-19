"""
Module defining physical constants and environment parameters for mission simulations.

Author: Kwok Keith
Date: 19 Jan 2026
"""

from dataclasses import dataclass
import numpy as np


C: np.float64 = 299_792_458.0  # m/s, Speed of light in vacuum
BOLTZMAN: np.float64 = 1.380649e-23  # J/K, Boltzmann constant
EARTH_RADIUS_M: np.float64 = 6_371_000.0  # metres, Earth's mean radius


@dataclass(frozen=True)
class EnvironmentParameters:
    nominal_temperature_k: np.float64 = 290.0  # Kelvin
    two_way_atmospheric_loss_db: np.float64 = 0  # db

    @property
    def nominal_temperature_c(self) -> np.float64:
        return self.nominal_temperature_k - 273.15

    @property
    def two_way_atmospheric_loss_linear(self) -> np.float64:
        return 10.0 ** (self.two_way_atmospheric_loss_db / 10.0)
