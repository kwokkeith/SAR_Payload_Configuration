from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Antenna:
    antenna_efficiency: np.float64  # η_a

    @property
    def antenna_area_m2(self) -> np.float64:
        """Calculate the physical antenna area in square metres."""
        raise NotImplementedError("Subclasses must implement antenna_area_m2 property.")

    @property
    def effective_antenna_area_m2(self) -> np.float64:
        return self.antenna_efficiency * self.antenna_area_m2

    @property
    def total_peak_power_w(self) -> np.float64:
        """Calculate the total peak power of the antenna in watts."""
        raise NotImplementedError(
            "Subclasses must implement total_peak_power_w property."
        )
