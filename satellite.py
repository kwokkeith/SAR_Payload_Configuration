"""
Satellite platform parameters and derived properties.

Author: Kwok Keith
Date: 14 Jan 2026
"""

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Satellite:
    platform_velocity_mps: np.float64  # V_p
    orbit_altitude_m: np.float64  # H
    look_angle_from_nadir_deg: np.float64  # θ_look
    radar_loss_db: np.float64  # L_radar
    receiver_noise_factor_db: np.float64  # F_N

    @property
    def graze_angle_deg(self) -> np.float64:
        """Graze angle measured from horizontal."""
        return 90.0 - self.look_angle_from_nadir_deg

    @property
    def slant_range_flat_earth_m(self) -> np.float64:
        """Flat Earth slant range approximation."""
        theta_rad = np.deg2rad(self.look_angle_from_nadir_deg)
        return self.orbit_altitude_m / np.cos(theta_rad)
