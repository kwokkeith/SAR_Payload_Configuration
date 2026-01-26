"""
Mission base class for radar satellite missions.

Author: Kwok Keith
Date: 26 Jan 2026
"""

from abc import abstractmethod
from dataclasses import dataclass
import numpy as np
from radar_signal import Signal
from satellite import Satellite
from phased_array import PhasedArray
from antenna import Antenna
from mission_environment import EnvironmentParameters
from mission_environment import C, BOLTZMAN, EARTH_RADIUS_M


@dataclass
class Mission:
    swath_range_m: np.float64  # Swath length in meters
    swath_azimuth_m: np.float64  # Swath width in meters
    signal: Signal
    satellite: Satellite
    antenna: Antenna
    environment_parameters: EnvironmentParameters

    @property
    def range_resolution_m(self) -> np.float64:
        """
        Calculate the range resolution in metres (this uses traditional formula).
        Actual range resolution depends on radar signal processing.
        """
        B = self.signal.bandwidth_hz
        a_wr = self.signal.broadening_factor_range
        delta_r = C / (2.0 * B) * a_wr  # c /(2 * BW) * a_wr
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
        k_b = BOLTZMAN
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

    def _calculate_nes0_linear(
        self,
        antenna_gain_linear: np.float64,
        slant_range_m: np.float64 | None = None,
        graze_angle_rad: np.float64 | None = None,
    ) -> np.float64:
        """Calculate the Noise Equivalent Sigma Zero (NES0) for a given antenna gain.

        Args:
            antenna_gain_linear: The antenna gain in linear units.
            slant_range_m: The slant range in meters. If None, uses center swath value.
            graze_angle_rad: The grazing angle in radians. If None, uses center swath value.

        Returns:
            The NES0 in linear units.
        """
        lambda_m = self.signal.nominal_wavelength_m
        P_avg_w = self.signal.average_tx_power_w
        G = antenna_gain_linear
        L_sys = self.system_loss_linear
        R_m = (
            slant_range_m
            if slant_range_m is not None
            else self.satellite.slant_range_flat_earth_m
        )
        delta_r_m = self.range_resolution_m
        a_wa = self.signal.broadening_factor_azimuth
        v_x = self.satellite.platform_velocity_mps
        kTF_N = self.thermal_loss_linear
        gz_ang_rad = (
            graze_angle_rad
            if graze_angle_rad is not None
            else self.satellite.graze_angle_rad
        )

        nes0 = (
            2 * (4 * np.pi) ** 3 * R_m**3 * v_x * np.cos(gz_ang_rad) * kTF_N * L_sys
        ) / (P_avg_w * G**2 * lambda_m**3 * delta_r_m * a_wa)
        return nes0

    @property
    def nes0_linear(self) -> np.float64:
        """Calculate the Noise Equivalent Sigma Zero (NES0)."""
        return self._calculate_nes0_linear(self.signal.peak_antenna_gain_linear)

    @property
    def nes0_db(self) -> np.float64:
        """Calculate the Noise Equivalent Sigma Zero (NES0) in dB."""
        nes0_linear = self.nes0_linear
        nes0_db = 10.0 * np.log10(nes0_linear)
        return nes0_db

    @property
    def elevation_yaw_angles_to_corners_rad(
        self,
    ) -> tuple[
        tuple[np.float64, np.float64, np.float64, np.float64],
        tuple[np.float64, np.float64, np.float64, np.float64],
        tuple[np.float64, np.float64, np.float64, np.float64],
        tuple[np.float64, np.float64, np.float64, np.float64],
    ]:
        """
        Calculate the off-boresight angle (theta), azimuthal angle (phi), slant range, and
        grazing angle for all four corners of the rectangular swath using ray-plane intersection geometry.

        Uses a 3D geometric approach:
        1. Define ground coordinate system with z=0 at ground, sensor at height H
        2. Boresight points at swath center with grazing angle γ and azimuth α
        3. For each corner, find the ray from sensor through corner
        4. Calculate theta and phi of that ray in antenna reference frame

        Returns:
            Tuple of 4 tuples, one for each corner: (near_left, near_right, far_left, far_right)
            Each corner tuple contains:
                - theta_rad: Off-boresight angle (0 = along boresight)
                - phi_rad: Azimuthal angle around boresight (0 = range direction)
                - slant_range_m: Slant range from sensor to corner
                - graze_angle_rad: Grazing angle at corner
        """
        # Sensor height and geometry
        H = self.satellite.orbit_altitude_m  # Height above ground
        gamma = self.satellite.graze_angle_rad  # Grazing angle
        alpha = (
            0.0  # Azimuth heading (0 = perpendicular to flight path for side-looking)
        )

        # Ground swath dimensions relative to center
        half_range_m = self.swath_range_m / 2.0  # Range (across-track)
        half_azimuth_m = self.swath_azimuth_m / 2.0  # Azimuth (along-track)

        # Ground position of swath center (where boresight points)
        # For side-looking SAR, alpha ≈ 0, so:
        # Center is at ground position (H*cot(gamma), 0, 0)
        range_center: np.float64 = H / np.tan(gamma)
        azimuth_center = 0.0

        # Boresight direction (pointing at swath center)
        # Sperical to cartesian coordinates
        boresight_direction = np.array(
            [
                np.cos(gamma) * np.cos(alpha),
                np.cos(gamma) * np.sin(alpha),
                -np.sin(gamma),
            ]
        )

        # Pre-calculate reference directions in the plane perpendicular to boresight
        # Range direction (across-track, perpendicular to velocity)
        range_ref = np.array([1.0, 0.0, 0.0])
        range_ref_perp = (
            range_ref - np.dot(range_ref, boresight_direction) * boresight_direction
        )  # Gets the component of range_ref perpendicular to boresight
        range_ref_perp_unit = range_ref_perp / np.linalg.norm(range_ref_perp)

        # Azimuth direction (along-track, parallel to velocity)
        azimuth_ref = np.array([0.0, 1.0, 0.0])
        azimuth_ref_perp = (
            azimuth_ref - np.dot(azimuth_ref, boresight_direction) * boresight_direction
        )  # Gets the component of azimuth_ref perpendicular to boresight
        azimuth_ref_perp_unit = azimuth_ref_perp / np.linalg.norm(azimuth_ref_perp)

        # Helper function to calculate theta, phi, slant_range, and graze_angle for a given corner
        def calculate_corner_angles(
            corner_range_offset: float, corner_azimuth_offset: float
        ) -> tuple[np.float64, np.float64, np.float64, np.float64]:
            # Ground position of corner
            corner_range = range_center + corner_range_offset
            corner_azimuth = azimuth_center + corner_azimuth_offset

            # Vector from sensor at (0, 0, H) to corner on ground
            sensor_to_corner = np.array([corner_range, corner_azimuth, -H])

            # Magnitude (slant range to corner)
            slant_range_corner = np.linalg.norm(sensor_to_corner)

            # Unit direction vector to corner
            direction_to_corner = sensor_to_corner / slant_range_corner

            # Calculate grazing angle at corner
            # look_angle_up_from_horizontal: angle from horizontal plane up to look vector
            # We find the angle between the -look vector and the vertical reference from the swath reference frame
            # We then subtract 90 degrees from it to find the graze angle
            grazing_angle = np.pi / 2.0 - np.arccos(-direction_to_corner[2])

            # Calculate separable plane angles (range/azimuth) relative to boresight
            boresight_comp = np.dot(boresight_direction, direction_to_corner)
            range_comp = np.dot(direction_to_corner, range_ref_perp_unit)
            azimuth_comp = np.dot(direction_to_corner, azimuth_ref_perp_unit)

            # Range angle (theta) in plane of boresight + range axis
            theta_rad = np.arctan2(range_comp, boresight_comp)

            # Azimuth angle (phi) in plane of boresight + azimuth axis
            phi_rad = np.arctan2(azimuth_comp, boresight_comp)

            return (
                theta_rad,
                phi_rad,
                slant_range_corner,
                grazing_angle,
            )

        # Calculate angles for all four corners
        # Near range = negative offset, Far range = positive offset
        # Left azimuth = negative offset, Right azimuth = positive offset
        near_left = calculate_corner_angles(-half_range_m, -half_azimuth_m)
        near_right = calculate_corner_angles(-half_range_m, +half_azimuth_m)
        far_left = calculate_corner_angles(+half_range_m, -half_azimuth_m)
        far_right = calculate_corner_angles(+half_range_m, +half_azimuth_m)

        return (near_left, near_right, far_left, far_right)

    @property
    def nes0_db_corner(self) -> np.float64:
        """
        Calculates the worst-case (maximum) nes0 in dB among the four corners of the swath,
        assuming boresight is at scene centre.

        Returns:
            Maximum nes0 in dB among all four corners (worst case).
        """
        # Get angles for all four corners
        corners = self.elevation_yaw_angles_to_corners_rad

        nes0_values = []
        for theta_rad, phi_rad, slant_range_m, graze_angle_rad in corners:
            # Get the absolute gain at this corner in dB
            corner_gain_db = self.signal.gain_pattern_db(theta_rad, phi_rad)

            # Convert gain from dB to linear
            corner_gain_linear = 10.0 ** (corner_gain_db / 10.0)

            # Calculate nes0 using the actual gain, slant range, and graze angle at the corner
            nes0_linear = self._calculate_nes0_linear(
                corner_gain_linear, slant_range_m, graze_angle_rad
            )
            nes0_db = 10.0 * np.log10(nes0_linear)
            nes0_values.append(nes0_db)

        # Return the worst case (maximum NESZ = highest value, least negative)
        return np.max(nes0_values)

    @property
    def nes0_db_all_corners(self) -> dict:
        """
        Calculates nes0 in dB at all four corners of the swath.

        Returns:
            Dictionary with keys: 'near_left', 'near_right', 'far_left', 'far_right'
            Each value is a tuple: (nes0_db, theta_rad, phi_rad, gain_db, slant_range_m, graze_angle_rad)
        """
        corners = self.elevation_yaw_angles_to_corners_rad
        corner_names = ["near_left", "near_right", "far_left", "far_right"]

        results = {}
        for name, (theta_rad, phi_rad, slant_range_m, graze_angle_rad) in zip(
            corner_names, corners
        ):
            corner_gain_db = self.signal.gain_pattern_db(theta_rad, phi_rad)
            corner_gain_linear = 10.0 ** (corner_gain_db / 10.0)
            nes0_linear = self._calculate_nes0_linear(
                corner_gain_linear, slant_range_m, graze_angle_rad
            )
            nes0_db = 10.0 * np.log10(nes0_linear)
            results[name] = (
                nes0_db,
                theta_rad,
                phi_rad,
                corner_gain_db,
                slant_range_m,
                graze_angle_rad,
            )

        return results

    def nes0_db_at_position(
        self, range_offset_m: float, azimuth_offset_m: float
    ) -> tuple[np.float64, dict]:
        """
        Calculate NES0 in dB at any position within the swath.

        Args:
            range_offset_m: Range offset from swath center in meters (negative = near, positive = far)
            azimuth_offset_m: Azimuth offset from swath center in meters (negative = left, positive = right)

        Returns:
            Tuple of (nes0_db, info_dict) where info_dict contains:
                - theta_rad: Off-boresight angle
                - phi_rad: Azimuthal angle
                - gain_db: Antenna gain at this position
                - slant_range_m: Slant range to this position
                - graze_angle_rad: Grazing angle at this position
        """
        # Sensor height and geometry (same setup as elevation_yaw_angles_to_corners_rad)
        H = self.satellite.orbit_altitude_m
        gamma = self.satellite.graze_angle_rad
        alpha = 0.0  # Side-looking SAR

        # Boresight direction
        boresight_direction = np.array(
            [
                np.cos(gamma) * np.cos(alpha),
                np.cos(gamma) * np.sin(alpha),
                -np.sin(gamma),
            ]
        )

        # Ground center position
        range_center: np.float64 = H / np.tan(gamma)
        azimuth_center = 0.0

        # Reference directions for phi calculation
        range_ref = np.array([1.0, 0.0, 0.0])
        range_ref_perp = (
            range_ref - np.dot(range_ref, boresight_direction) * boresight_direction
        )
        range_ref_perp_unit = range_ref_perp / np.linalg.norm(range_ref_perp)

        azimuth_ref = np.array([0.0, 1.0, 0.0])
        azimuth_ref_perp = (
            azimuth_ref - np.dot(azimuth_ref, boresight_direction) * boresight_direction
        )
        azimuth_ref_perp_unit = azimuth_ref_perp / np.linalg.norm(azimuth_ref_perp)

        # Calculate ground position
        position_range = range_center + range_offset_m
        position_azimuth = azimuth_center + azimuth_offset_m

        # Vector from sensor to position
        sensor_to_position = np.array([position_range, position_azimuth, -H])
        slant_range = np.linalg.norm(sensor_to_position)
        direction_to_position = sensor_to_position / slant_range

        # Calculate grazing angle
        look_angle_from_horizontal = np.arcsin(-direction_to_position[2])
        graze_angle = np.pi / 2.0 - look_angle_from_horizontal

        # Calculate separable plane angles (range/azimuth) relative to boresight
        boresight_comp = np.dot(boresight_direction, direction_to_position)
        range_comp = np.dot(direction_to_position, range_ref_perp_unit)
        azimuth_comp = np.dot(direction_to_position, azimuth_ref_perp_unit)

        # Range angle (theta) in plane of boresight + range axis
        theta_rad = np.arctan2(range_comp, boresight_comp)

        # Azimuth angle (phi) in plane of boresight + azimuth axis
        phi_rad = np.arctan2(azimuth_comp, boresight_comp)

        # Calculate antenna gain at this position
        gain_db = self.signal.gain_pattern_db(theta_rad, phi_rad)
        gain_linear = 10.0 ** (gain_db / 10.0)

        # Calculate NES0
        nes0_linear = self._calculate_nes0_linear(gain_linear, slant_range, graze_angle)
        nes0_db = 10.0 * np.log10(nes0_linear)

        # Package info
        info = {
            "theta_rad": theta_rad,
            "phi_rad": phi_rad,
            "gain_db": gain_db,
            "slant_range_m": slant_range,
            "graze_angle_rad": graze_angle,
        }

        return nes0_db, info

    @property
    def nes0_linear_corner(self) -> np.float64:
        """
        Calculates the nes0 in linear at the corner of swath assuming boresight
        is at scene centre.

        Returns:
            nes0 in linear at the corner of swath.
        """
        nes0_db_corner = self.nes0_db_corner
        nes0_linear_corner = 10.0 ** (nes0_db_corner / 10.0)
        return nes0_linear_corner

    @abstractmethod
    def azimuth_resolution_m(self) -> np.float64:
        """Calculate the azimuth resolution in metres."""
        pass
