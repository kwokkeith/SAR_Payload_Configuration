"""
Module to define the PatchAntenna class, representing a patch antenna

Author: Kwok Keith
Date: 20 Jan 2026
"""

from dataclasses import dataclass
from antenna import Antenna


@dataclass(frozen=True)
class PatchAntenna(Antenna):
    length: float = 1.0
    width: float = 1.0
    tx_power_w: float = 1000.0

    @property
    def antenna_area_m2(self) -> float:
        """Calculate the physical antenna area in square metres."""
        return self.length * self.width

    @property
    def total_peak_power_w(self) -> float:
        """Total peak power of the antenna in watts."""
        return self.tx_power_w
