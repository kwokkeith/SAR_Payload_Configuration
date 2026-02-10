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
    antenna: PatchAntenna

    def normalised_radiation_pattern_mag(
        self, theta_rad: np.float64, phi_rad: np.float64
    ) -> np.float64:
        """
        Normalised radiation pattern field magnitude (0–1 wrt peak field).
        Uses separable pattern: F_total(theta, phi) = F_elevation(theta) * F_azimuth(phi)

        theta_rad: elevation angle (0 at boresight)
        phi_rad: azimuth angle (0 at boresight)
        """
        k = 2.0 * np.pi / self.nominal_wavelength_m
        L = self.antenna.length
        W = self.antenna.width

        # Elevation pattern (along length L, varies with theta)
        u_elevation = (k * L / 2.0) * np.sin(theta_rad)
        F_elevation = np.sinc(u_elevation / np.pi)

        # Azimuth pattern (along width W, varies with phi)
        u_azimuth = (k * W / 2.0) * np.sin(phi_rad)
        F_azimuth = np.sinc(u_azimuth / np.pi)

        # 3D pattern is product of elevation and azimuth patterns.
        # Take absolute value because the sinc field pattern can be negative;
        # gain should be based on field magnitude (or power), not signed field.
        mag = np.abs(F_elevation * F_azimuth)

        return np.clip(mag, 0.0, 1.0)

    def gain_pattern_db(self, theta_rad: np.float64, phi_rad: np.float64) -> np.float64:
        """
        Calculate the gain pattern in dB (aboslute) for the patch antenna.

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
