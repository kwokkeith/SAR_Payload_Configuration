"""
Satellite platform parameters and derived properties.

Author: Kwok Keith
Date: 27 Jan 2026
"""

from dataclasses import dataclass
import numpy as np
from mission_environment import EARTH_RADIUS_M


@dataclass
class Satellite:
    platform_velocity_mps: np.float64  # V_p
    orbit_altitude_m: np.float64  # H
    look_angle_from_nadir_deg: np.float64  # θ_look
    radar_loss_db: np.float64  # L_radar
    receiver_noise_factor_db: np.float64  # F_N

    @property
    def graze_angle_rad(self) -> np.float64:
        """Graze angle measured from horizontal (radians)."""
        # Calculate the argument for arcsin
        arcsin_arg = (
            (EARTH_RADIUS_M + self.orbit_altitude_m)
            * np.sin(np.deg2rad(self.look_angle_from_nadir_deg))
            / EARTH_RADIUS_M
        )
        # to avoid domain errors in arcsin
        arcsin_arg = np.clip(arcsin_arg, -1.0, 1.0)
        
        return np.abs(np.arcsin(arcsin_arg) - np.deg2rad(90.0))

    @property
    def look_angle_from_nadir_rad(self) -> np.float64:
        """Look angle from nadir (radians)."""
        return np.deg2rad(self.look_angle_from_nadir_deg)

    @property
    def graze_angle_deg(self) -> np.float64:
        """Graze angle measured from horizontal (degrees)."""
        return self.graze_angle_rad * (180.0 / np.pi)

    @property
    def slant_range_flat_earth_m(self) -> np.float64:
        """Flat Earth slant range approximation."""
        theta_rad = np.deg2rad(self.look_angle_from_nadir_deg)
        return self.orbit_altitude_m / np.cos(theta_rad)

    @property
    def radar_loss_linear(self) -> float:
        """Convert radar loss from dB to linear scale."""
        return 10.0 ** (self.radar_loss_db / 10.0)

    @property
    def receiver_noise_factor_linear(self) -> float:
        """Convert receiver noise factor from dB to linear scale."""
        return 10.0 ** (self.receiver_noise_factor_db / 10.0)
