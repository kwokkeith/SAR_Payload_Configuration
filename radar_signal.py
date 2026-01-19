"""
Dataclass representing radar signal parameters.

Author: Kwok Keith
Date: 19 Jan 2026
"""

from dataclasses import dataclass
import numpy as np
from mission_environment import C


@dataclass
class Signal:
    centre_frequency_hz: np.float64  # f_c
    bandwidth_hz: np.float64  # B
    prf_hz: np.float64  # Pulse Repetition Frequency

    broadening_factor_azimuth: np.float64  # a_wa
    broadening_factor_range: np.float64  # a_wr

    range_processing_loss_db: np.float64  # L_r
    azimuth_processing_loss_db: np.float64  # L_a

    pulse_width_us: np.float64  # tau

    doppler_gain_constant: np.float64  # k_a

    @property
    def nominal_wavelength_m(self) -> np.float64:
        return np.float64(C / self.centre_frequency_hz)

    @property
    def pri_s(self) -> np.float64:
        return np.float64(1.0 / self.prf_hz)

    @property
    def radar_loss_linear(self) -> float:
        return 10.0 ** (self.radar_loss_db / 10.0)

    @property
    def receiver_noise_factor_linear(self) -> float:
        return 10.0 ** (self.receiver_noise_factor_db / 10.0)

    @property
    def range_processing_loss_linear(self) -> float:
        return 10.0 ** (self.range_processing_loss_db / 10.0)

    @property
    def azimuth_processing_loss_linear(self) -> float:
        return 10.0 ** (self.azimuth_processing_loss_db / 10.0)

    @property
    def tx_duty_cycle(self) -> np.float64:
        """Transmit duty cycle (fraction in [0, 1])."""
        return self.pulse_width_us * 1e-6 * self.prf_hz
