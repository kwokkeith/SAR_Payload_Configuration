"""
Dataclass representing radar signal parameters.

Author: Kwok Keith
Date: 14 Jan 2026
"""

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Signal:
    centre_frequency_hz: np.float64  # f_c
    bandwidth_hz: np.float64  # B
    prf_hz: np.float64  # Pulse Repetition Frequency

    broadening_factor_azimuth: np.float64  # a_wa
    broadening_factor_range: np.float64  # a_wr

    range_processing_loss: np.float64  # L_r
    azimuth_processing_loss: np.float64  # L_a

    pulse_width_us: np.float64  # tau

    doppler_gain_constant: np.float64  # k_a

    @property
    def nominal_wavelength_m(self) -> np.float64:
        return np.float64(
            3e8 / self.centre_frequency_hz
        )  # Speed of light divided by frequency

    @property
    def pri_s(self) -> np.float64:
        return np.float64(1.0 / self.prf_hz)

    @property
    def radar_loss_linear(self) -> float:
        return 10.0 ** (self.radar_loss_db / 10.0)

    @property
    def receiver_noise_factor_linear(self) -> float:
        return 10.0 ** (self.receiver_noise_factor_db / 10.0)
