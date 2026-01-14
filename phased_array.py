"""
This module defines a data class representing a phased array in the system.

Author: Kwok Keith
Date: 14 Jan 2026
"""

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class PhasedArray:
    num_width_elements: int
    num_height_elements: int
    element_width_m: np.float64
    element_height_m: np.float64
    element_power_w: np.float64
    antenna_efficiency: np.float64
    tx_duty_cycle: np.float64  # fraction in [0, 1]

    @property
    def total_elements(self) -> int:
        return self.num_width_elements * self.num_height_elements

    @property
    def antenna_area_m2(self) -> np.float64:
        return (
            self.num_width_elements
            * self.element_width_m
            * self.num_height_elements
            * self.element_height_m
        )

    @property
    def effective_antenna_area_m2(self) -> np.float64:
        return self.antenna_efficiency * self.antenna_area_m2

    @property
    def total_peak_power_w(self) -> np.float64:
        return self.total_elements * self.element_power_w

    @property
    def average_tx_power_w(self) -> np.float64:
        return self.total_peak_power_w * self.tx_duty_cycle
