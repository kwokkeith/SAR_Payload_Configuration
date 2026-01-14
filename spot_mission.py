"""
Spot Mission class for radar satellite missions for SPOTlight SAR processing.

Author: Kwok Keith
Date: 14 Jan 2026
"""

import numpy as np
from mission import Mission


class SpotMission(Mission):
    integration_angle_deg: np.float64  # Azimuth angle in degrees

    def azimuth_resolution_m(self):
        """Calculate the azimuth resolution in metres for Spot Mission."""
        nominal_wavelength = self.signal.nominal_wavelength_m
        delta_az = nominal_wavelength / (2.0 * self.integration_angle_rad)

        return delta_az

    @property
    def integration_angle_rad(self) -> np.float64:
        """Convert azimuth angle from degrees to radians."""
        return np.deg2rad(self.integration_angle_deg)
