"""
Module defining the PatchAntennaSignal class, which models the signal
characteristics of a radar system using a patch antenna.

Reference: https://www.antenna-theory.com/antennas/patches/antenna.php

Author: Kwok Keith
Date: 20 Jan 2026
"""

from dataclasses import dataclass
import numpy as np
from radar_signal import Signal
from patch_antenna import PatchAntenna


@dataclass
class PatchAntennaSignal(Signal):
    patched_antenna: PatchAntenna

    @property
    def normalised_radiation_pattern_mag(
        self, theta_rad: np.float64, phi_rad: np.float64
    ) -> np.float64:
        """Calculate the normalised radiation pattern magnitude for the patch antenna (0-1 wrt to peak power).
        Link: https://www.antenna-theory.com/antennas/patches/antenna.php

        Args:
            theta_rad (np.float64): Elevation angle in radians.
            phi_rad (np.float64): Azimuth angle in radians.

        Returns:
            np.float64: Magnitude of the radiation pattern.
        """
        k = 2.0 * np.pi / self.nominal_wavelength_m  # free-space wavenumber
        L = self.patched_antenna.length
        W = self.patched_antenna.width

        common_coeff = np.sin(k * W / 2.0 * np.sin(theta_rad) * np.sin(phi_rad)) / (
            k * W / 2.0 * np.sin(theta_rad) * np.sin(phi_rad)
        )
        common_coeff = common_coeff * np.cos(
            k * L / 2.0 * np.sin(theta_rad) * np.cos(phi_rad)
        )
        E_theta = common_coeff * np.cos(phi_rad)
        E_phi = -common_coeff * np.cos(theta_rad) * np.sin(phi_rad)
        magnitude = np.sqrt(E_theta**2 + E_phi**2)

        return magnitude

    @property
    def gain_pattern_db(self, theta_rad: np.float64, phi_rad: np.float64) -> np.float64:
        """
        Calculate the gain pattern in dB for the patch antenna.

        Args:
            theta_rad (np.float64): Elevation angle in radians.
            phi_rad (np.float64): Azimuth angle in radians.

        Returns:
            np.float64: Gain pattern in dB.
        """
        mag = self.normalised_radiation_pattern_mag(theta_rad, phi_rad)
        mag = np.maximum(mag, 1e-12)  # Avoid log(0)

        G_peak_db = self.peak_antenna_gain_db
        G_pattern_db = G_peak_db + 20.0 * np.log10(mag)
        return G_pattern_db

    @property
    def gain_pattern_linear(
        self, theta_rad: np.float64, phi_rad: np.float64
    ) -> np.float64:
        """
        Calculate the gain pattern in linear scale for the patch antenna.

        Args:
            theta_rad (np.float64): Elevation angle in radians.
            phi_rad (np.float64): Azimuth angle in radians.

        Returns:
            np.float64: Gain pattern in linear scale.
        """
        G_pattern_db = self.gain_pattern_db(theta_rad, phi_rad)
        G_pattern_linear = 10.0 ** (G_pattern_db / 10.0)
        return G_pattern_linear
