"""
Mission base class for radar satellite missions.

Author: Kwok Keith
Date: 14 Jan 2026
"""

from abc import abstractmethod
from dataclasses import dataclass
import numpy as np
from signal import Signal
from satellite import Satellite
from phased_array import PhasedArray
from mission_environment import PhysicalConstants, EnvironmentParameters


@dataclass
class Mission:
    swath_m: np.float64  # Swath width in meters
    signal: Signal
    satellite: Satellite
    phased_array: PhasedArray
    physical_constants: PhysicalConstants
    environment_parameters: EnvironmentParameters

    @property
    def antenna_gain_linear(self) -> np.float64:
        """Calculate the antenna gain in linear scale."""
        wavelength_m = self.signal.nominal_wavelength_m
        A_e = self.phased_array.effective_antenna_area_m2
        G = (4.0 * np.pi * A_e) / (wavelength_m**2)
        return G

    @property
    def range_resolution_m(self) -> np.float64:
        """Calculate the range resolution in metres."""
        B = self.signal.bandwidth_hz
        a_wr = self.signal.broadening_factor_range
        delta_r = (self.physical_constants.c) / (2.0 * B) * a_wr  # c /(2 * BW) * a_wr
        return delta_r

    @property
    def atmospheric_loss_db(self) -> np.float64:
        """Calculate the atmospheric loss in dB scale."""
        return (
            self.environment_parameters.two_way_atmospheric_loss_db
            * self.satellite.slant_range_flat_earth_m
        )

    @property
    def atmospheric_loss_linear(self) -> np.float64:
        """Calculate the atmospheric loss in linear scale."""
        return 10.0 ** (self.atmospheric_loss_db / 10.0)

    @property
    def thermal_loss_linear(self) -> np.float64:
        """Calculate the thermal noise loss in linear scale."""
        T_k = self.environment_parameters.nominal_temperature_k
        k_b = self.physical_constants.boltzman
        F_N = self.satellite.receiver_noise_factor_linear
        L_thermal = k_b * T_k * F_N
        return L_thermal

    @property
    def system_loss_linear(self) -> np.float64:
        """Calculate the total system loss in linear scale."""
        L_radar = self.satellite.radar_loss_linear
        L_atmosphere = self.atmospheric_loss_linear
        L_range = self.signal.range_processing_loss_linear
        L_azimuth = self.signal.azimuth_processing_loss_linear
        L_total = L_radar * L_atmosphere * L_range * L_azimuth
        return L_total

    @property
    def system_loss_db(self) -> np.float64:
        """Calculate the total system loss in dB scale."""
        L_total_linear = self.system_loss_linear
        L_total_db = 10.0 * np.log10(L_total_linear)
        return L_total_db

    @property
    def nes0_linear(self) -> np.float64:
        """Calculate the Noise Equivalent Sigma Zero (NES0)."""
        lambda_m = self.signal.nominal_wavelength_m
        P_avg_w = self.phased_array.average_tx_power_w
        G = self.antenna_gain_linear
        L_sys = self.system_loss_linear
        R_m = self.satellite.slant_range_flat_earth_m
        delta_r_m = self.range_resolution_m
        a_wa = self.signal.broadening_factor_azimuth
        v_x = self.satellite.platform_velocity_mps
        kTF_N = self.thermal_loss_linear

        nes0 = (2 * (4 * np.pi) ** 3 * R_m**3 * v_x * kTF_N * L_sys) / (
            P_avg_w * G**2 * lambda_m**3 * delta_r_m * a_wa
        )
        return nes0

    @property
    def nes0_db(self) -> np.float64:
        """Calculate the Noise Equivalent Sigma Zero (NES0) in dB."""
        nes0_linear = self.nes0_linear
        nes0_db = 10.0 * np.log10(nes0_linear)
        return nes0_db

    @abstractmethod
    def azimuth_resolution_m(self) -> np.float64:
        """Calculate the azimuth resolution in metres."""
        pass
