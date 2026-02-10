"""
Module is used to optimise mission parameters such as look angle and swath dimensions

Author: Kwok Keith
Date: 10 Feb 2026
"""

from dataclasses import dataclass
import numpy as np
from mission import Mission
from mission_environment import C, EARTH_RADIUS_M

@dataclass
class MissionConfigurator:
    mission: Mission

    def max_look_angle_for_nesz(
        self, desired_nesz_db: float, tolerance_db: float = 0.1, max_iterations: int = 100
    ) -> dict:
        """
        Find the maximum look angle from nadir that achieves a desired worst-case NESZ.
        Given current swath size optimise boresight look angle.
        
        Args:
            desired_nesz_db: Target NESZ value in dB (e.g., -25 dB)
            tolerance_db: Acceptable difference from target in dB (default: 0.1 dB)
            max_iterations: Maximum number of iterations (default: 100)
            
        Returns:
            Dictionary Containing:
                - 'look_angle_deg': The calculated maximum look angle from nadir in degrees
                - 'achieved_nesz_db': The actual NESZ at this look angle in dB
                - 'iterations': Number of iterations used
                - 'converged': Whether the solution converged within tolerance
                - 'all_corners_nesz': NESZ values at all four corners
        """
        return self._calculate_max_look_angle_for_nesz(desired_nesz_db, tolerance_db, max_iterations)
    
    def max_swath_for_nesz(
        self,
        desired_nesz_db: float,
        aspect_ratio: float = 1.0,
        tolerance_db: float = 0.1,
        max_iterations: int = 100,
        initial_swath_m: float = 1000.0,
        max_swath_m: float = 500e3
    ) -> dict:
        """
        Find the maximum swath dimensions that achieve a desired worst-case NESZ. 
        Given current boresight angle optimise swath dimensions.
        
        Args:
            desired_nesz_db: Target maximum NESZ value in dB (e.g., -25 dB)
            aspect_ratio: Ratio of swath_range / swath_azimuth (default: 1.0 for square)
            tolerance_db: Acceptable difference from target in dB (default: 0.1 dB)
            max_iterations: Maximum number of iterations (default: 100)
            initial_swath_m: Starting swath size for search in meters (default: 1000 m)
            max_swath_m: Maximum swath size to consider in meters (default: 500 km)
            
        Returns:
            Dictionary Containing:
                - 'swath_range_m': Optimised range swath dimension in meters
                - 'swath_azimuth_m': Optimised azimuth swath dimension in meters
                - 'swath_area_km2': Total swath area in square kilometers
                - 'achieved_nesz_db': Actual worst-case NESZ at these dimensions in dB
                - 'iterations': Number of iterations used
                - 'converged': Whether the solution converged within tolerance
                - 'all_corners_nesz': NESZ values at all four corners
        """
        return self._calculate_max_swath_for_nesz(
            desired_nesz_db, aspect_ratio, tolerance_db, max_iterations, initial_swath_m, max_swath_m
        )

    def min_look_angle_for_resolution(
        self,
        desired_ground_range_resolution_m: float,
        tolerance_m: float = 0.01,
        max_iterations: int = 100
    ) -> dict:
        """
        Find the minimum look angle from nadir that achieves a desired ground range resolution.
        
        Ground range resolution improves (gets smaller) as look angle increases (larger grazing angle).
        This finds the minimum look angle (closest to nadir) that still meets the resolution requirement.
        
        Args:
            desired_ground_range_resolution_m: Target ground range resolution in meters
            tolerance_m: Acceptable difference from target in meters (default: 0.01 m)
            max_iterations: Maximum number of iterations (default: 100)
            
        Returns:
            Dictionary containing:
                - 'look_angle_deg': Minimum look angle from nadir in degrees
                - 'achieved_resolution_m': Actual ground range resolution at this angle in meters
                - 'graze_angle_deg': Corresponding grazing angle in degrees
                - 'iterations': Number of iterations used
                - 'converged': Whether the solution converged within tolerance
                - 'error_m': Difference between achieved and desired resolution in meters
        """
        return self._calculate_min_look_angle_for_resolution(
            desired_ground_range_resolution_m, tolerance_m, max_iterations
        )

    def boresight_for_maximum_swath(
        self,
        desired_nesz_db: float,
        desired_resolution_m: float | None = None,
        aspect_ratio: float = 1.0,
        min_boresight_deg: float = 20.0,
        max_boresight_deg: float = 70.0,
        boresight_samples: int = 20,
        tolerance_db: float = 0.1,
        max_iterations: int = 100
    ) -> dict:
        """
        Find the optimal boresight angle and swath dimensions that maximize swath area
        while meeting NESZ and optionally resolution requirements.
        
        This searches over different boresight angles and finds the configuration
        that gives the largest swath area meeting all constraints.
        
        Args:
            desired_nesz_db: Maximum acceptable NESZ in dB (e.g., -20 dB or 80 dB depending on convention)
            desired_resolution_m: Minimum ground range resolution in meters (optional)
            aspect_ratio: Ratio of swath_range / swath_azimuth (default: 1.0 for square)
            min_boresight_deg: Minimum boresight angle to search (default: 20°)
            max_boresight_deg: Maximum boresight angle to search (default: 70°)
            boresight_samples: Number of boresight angles to test (default: 20)
            tolerance_db: NESZ tolerance in dB (default: 0.1 dB)
            max_iterations: Maximum iterations for optimization (default: 100)
            
        Returns:
            Dictionary containing:
                - 'optimal_boresight_deg': Best boresight angle in degrees
                - 'swath_range_m': Optimal range swath dimension in meters
                - 'swath_azimuth_m': Optimal azimuth swath dimension in meters
                - 'swath_area_km2': Maximum achievable swath area in square kilometers
                - 'achieved_nesz_db': Actual NESZ at corners in dB
                - 'near_look_angle_deg': Look angle to near edge in degrees
                - 'far_look_angle_deg': Look angle to far corner in degrees
                - 'converged': Whether optimization converged
                - 'resolution_constraint_met': Whether resolution requirement was met (if specified)
        """
        # Store original configuration
        original_boresight = self.mission.satellite.look_angle_from_nadir_deg
        original_swath_range = self.mission.swath_range_m
        original_swath_azimuth = self.mission.swath_azimuth_m
        
        # Search grid of boresight angles
        boresight_angles = np.linspace(min_boresight_deg, max_boresight_deg, boresight_samples)
        
        best_area = 0.0
        best_config = None
        
        for boresight_deg in boresight_angles:
            # Set test boresight
            self.mission.satellite.look_angle_from_nadir_deg = boresight_deg
            
            # Check resolution constraint if specified
            if desired_resolution_m is not None:
                min_look_result = self.min_look_angle_for_resolution(desired_resolution_m)
                if not min_look_result['converged'] or min_look_result['look_angle_deg'] > boresight_deg:
                    # Resolution cannot be met at this boresight
                    continue
            
            # Find maximum swath at this boresight
            swath_result = self.max_swath_for_nesz(
                desired_nesz_db=desired_nesz_db,
                aspect_ratio=aspect_ratio,
                tolerance_db=tolerance_db,
                max_iterations=max_iterations
            )
            
            if not swath_result['converged']:
                continue
            
            # Check if this is the best so far
            swath_area = swath_result['swath_area_km2']
            if swath_area > best_area:
                best_area = swath_area
                
                # Set mission to this configuration to get look angles
                self.mission.swath_range_m = swath_result['swath_range_m']
                self.mission.swath_azimuth_m = swath_result['swath_azimuth_m']
                near_look, far_look = self.mission.look_angle_range_deg
                
                best_config = {
                    'optimal_boresight_deg': boresight_deg,
                    'swath_range_m': swath_result['swath_range_m'],
                    'swath_azimuth_m': swath_result['swath_azimuth_m'],
                    'swath_area_km2': swath_area,
                    'achieved_nesz_db': swath_result['achieved_nesz_db'],
                    'near_look_angle_deg': near_look,
                    'far_look_angle_deg': far_look,
                    'converged': True,
                    'resolution_constraint_met': True if desired_resolution_m is not None else None,
                }
        
        # Restore original configuration
        self.mission.satellite.look_angle_from_nadir_deg = original_boresight
        self.mission.swath_range_m = original_swath_range
        self.mission.swath_azimuth_m = original_swath_azimuth
        
        if best_config is None:
            # No valid configuration found
            return {
                'optimal_boresight_deg': np.nan,
                'swath_range_m': 0.0,
                'swath_azimuth_m': 0.0,
                'swath_area_km2': 0.0,
                'achieved_nesz_db': np.nan,
                'near_look_angle_deg': np.nan,
                'far_look_angle_deg': np.nan,
                'converged': False,
                'resolution_constraint_met': False,
            }
        
        return best_config

    def calculate_swath_from_look_angles(
        self,
        min_look_angle_deg: float,
        max_look_angle_deg: float,
    ) -> dict:
        """
        Calculate swath dimensions given minimum and maximum look angles.
        
        The minimum look angle corresponds to the near edge at the centerline (no azimuth offset).
        The maximum look angle corresponds to the far corner (with both range and azimuth offsets).
        
        Args:
            min_look_angle_deg: Minimum look angle from nadir in degrees (near edge)
            max_look_angle_deg: Maximum look angle from nadir in degrees (far corner)
            aspect_ratio: Ratio of swath_range / swath_azimuth (default: 1.0 for square)
            
        Returns:
            Dictionary containing:
                - 'swath_range_m': Range swath dimension in meters
                - 'swath_azimuth_m': Azimuth swath dimension in meters
                - 'swath_area_km2': Total swath area in square kilometers
                - 'boresight_look_angle_deg': Required boresight look angle in degrees
        """
        H = self.mission.satellite.orbit_altitude_m
        look_angle_center_rad = self.mission.satellite.look_angle_from_nadir_rad
        
        # Convert boresight look angle to ground range
        arc_angle_center = np.arcsin(
            (EARTH_RADIUS_M + H) * np.sin(look_angle_center_rad) / EARTH_RADIUS_M
        )
        ground_range_center = arc_angle_center * EARTH_RADIUS_M
        
        # Convert min look angle to ground range (near edge at centerline)
        min_look_angle_rad = np.radians(min_look_angle_deg)
        arc_angle_near = np.arcsin(
            (EARTH_RADIUS_M + H) * np.sin(min_look_angle_rad) / EARTH_RADIUS_M
        )
        ground_range_near = arc_angle_near * EARTH_RADIUS_M
        
        # Convert max look angle to ground range (far corner)
        max_look_angle_rad = np.radians(max_look_angle_deg)
        arc_angle_far = np.arcsin(
            (EARTH_RADIUS_M + H) * np.sin(max_look_angle_rad) / EARTH_RADIUS_M
        )
        ground_range_far_corner = arc_angle_far * EARTH_RADIUS_M
        
        # Range swath: distance from near edge to far edge (along centerline)
        # near_edge = center - swath_range/2
        # far_edge = center + swath_range/2
        swath_range_m = 2 * (ground_range_center - ground_range_near)
        
        # Far corner has both range and azimuth offset:
        # ground_range_far_corner = sqrt(far_range^2 + far_azimuth^2)
        # where far_range = center + swath_range/2
        #       far_azimuth = swath_azimuth/2
        far_range = ground_range_center + swath_range_m / 2.0
        
        # Solve for azimuth swath
        # ground_range_far_corner^2 = far_range^2 + (swath_azimuth/2)^2
        # (swath_azimuth/2)^2 = ground_range_far_corner^2 - far_range^2
        azimuth_half_squared = ground_range_far_corner**2 - far_range**2
        
        if azimuth_half_squared < 0:
            # Invalid geometry - far corner is closer than far edge
            raise ValueError(
                f"Invalid geometry: max look angle ({max_look_angle_deg:.2f}°) "
                f"does not reach beyond the far edge. "
                f"Required ground range: {ground_range_far_corner:.0f} m, "
                f"Far edge range: {far_range:.0f} m"
            )
        
        swath_azimuth_m = 2 * np.sqrt(azimuth_half_squared)
        
        # Calculate area
        swath_area_km2 = (swath_range_m * swath_azimuth_m) / 1e6
        
        return {
            'swath_range_m': swath_range_m,
            'swath_azimuth_m': swath_azimuth_m,
            'swath_area_km2': swath_area_km2,
            'boresight_look_angle_deg': self.mission.satellite.look_angle_from_nadir_deg,
            'computed_aspect_ratio': swath_range_m / swath_azimuth_m,
        }

    def _calculate_min_look_angle_for_resolution(
        self,
        desired_ground_range_resolution_m: float,
        tolerance_m: float = 0.01,
        max_iterations: int = 100
    ) -> dict:
        """
        Calculate the minimum look angle from nadir that achieves desired ground range resolution.
        Uses direct calculation from resolution to grazing angle to look angle.
        
        Ground range resolution = (C / (2 * B)) * a_wr / cos(graze_angle)
        Solving for graze_angle: graze_angle = arccos((C / (2 * B)) * a_wr / desired_resolution)
        """
        
        # Store original look angle to restore later
        original_look_angle = self.mission.satellite.look_angle_from_nadir_deg
        
        # Calculate required grazing angle for desired resolution
        a_wr = self.mission.signal.broadening_factor_range
        B = self.mission.signal.bandwidth_hz
        
        # From: ground_range_resolution = slant_range_resolution / cos(graze_angle)
        # where slant_range_resolution = (C / (2 * B)) * a_wr
        # Therefore: cos(graze_angle) = (C / (2 * B)) * a_wr / desired_resolution
        cos_graze = (C / (2.0 * B)) * a_wr / desired_ground_range_resolution_m
        
        # Check if physically feasible
        if cos_graze > 1.0:
            # Resolution requirement is too fine - cannot be achieved
            # Restore original look angle
            self.mission.satellite.look_angle_from_nadir_deg = original_look_angle
            return {
                'look_angle_deg': np.float64(90.0),
                'achieved_resolution_m': (C / (2.0 * B)) * a_wr,  # Best possible at 90°
                'graze_angle_deg': np.float64(0.0),
                'iterations': 0,
                'converged': False,
                'error_m': np.inf,
            }
        
        if cos_graze < 0.0:
            # Invalid - this shouldn't happen with positive resolution
            raise ValueError(f"Invalid resolution requirement: {desired_ground_range_resolution_m} m")
        
        # Calculate grazing angle
        graze_angle_rad = np.arccos(cos_graze)
        graze_angle_deg = np.degrees(graze_angle_rad)
        
        # Convert grazing angle to look angle from nadir using spherical Earth geometry
        H = self.mission.satellite.orbit_altitude_m
        
        # From grazing angle, calculate look angle
        # graze_angle is measured from horizontal
        # Incidence angle from vertical = π/2 - graze_angle
        incidence_angle = np.pi / 2.0 - graze_angle_rad
        
        # Using spherical Earth geometry:
        # sin(look_angle) = (R_E / (R_E + H)) * sin(incidence_angle)
        sin_look = (EARTH_RADIUS_M / (EARTH_RADIUS_M + H)) * np.sin(incidence_angle)
        
        # Clip to valid range to avoid domain errors
        sin_look = np.clip(sin_look, -1.0, 1.0)
        
        look_angle_rad = np.arcsin(sin_look)
        look_angle_deg = np.degrees(look_angle_rad)
        
        # Verify by calculating actual resolution at this look angle
        self.mission.satellite.look_angle_from_nadir_deg = look_angle_deg
        achieved_graze_angle_rad = self.mission.satellite.graze_angle_rad
        achieved_resolution_m = self.mission.ground_range_resolution_m(achieved_graze_angle_rad)
        
        # Check convergence
        converged = abs(achieved_resolution_m - desired_ground_range_resolution_m) <= tolerance_m
        
        # Restore original look angle
        self.mission.satellite.look_angle_from_nadir_deg = original_look_angle
        
        return {
            'look_angle_deg': np.float64(look_angle_deg),
            'achieved_resolution_m': achieved_resolution_m,
            'graze_angle_deg': graze_angle_deg,
            'iterations': 1,
            'converged': converged,
            'error_m': achieved_resolution_m - desired_ground_range_resolution_m,
        }

    def _calculate_max_look_angle_for_nesz(
        self, desired_nesz_db: float, tolerance_db: float = 0.1, max_iterations: int = 100
    ) -> dict:
        """
        Back-calculate the maximum look angle from nadir that achieves a desired worst-case NESZ.
        
        This function iteratively adjusts the satellite's look angle to find the angle where
        the worst-case corner NESZ equals the desired value. Uses binary search for efficiency.
        
        Args:
            desired_nesz_db: Target NESZ value in dB (e.g., -25 dB)
            tolerance_db: Acceptable difference from target in dB (default: 0.1 dB)
            max_iterations: Maximum number of iterations (default: 100)
            
        Returns:
            Dict:
                - 'look_angle_deg': The calculated maximum look angle from nadir in degrees
                - 'achieved_nesz_db': The actual NESZ at this look angle in dB
                - 'iterations': Number of iterations used
                - 'converged': Whether the solution converged within tolerance
                - 'all_corners_nesz': NESZ values at all four corners
        """
        # Store original look angle to restore later
        original_look_angle = self.mission.satellite.look_angle_from_nadir_deg
        
        # Initialise binary search bounds
        # Start with a reasonable range: 0° to 90° from nadir
        look_angle_min_deg = 0.0
        look_angle_max_deg = 90.0
        
        converged = False # Flag indicates if we converged
        iterations = 0    # Count number of iterations
        best_look_angle = original_look_angle
        best_nesz = self.mission.nes0_db_corner
        
        for _ in range(max_iterations):
            iterations += 1
            
            # Try midpoint of current range
            look_angle_test = (look_angle_min_deg + look_angle_max_deg) / 2.0
            
            # Temporarily update satellite look angle
            self.mission.satellite.look_angle_from_nadir_deg = look_angle_test
            
            # Calculate worst-case NESZ at this look angle
            current_nesz_db = self.mission.nes0_db_corner
            
            # Check if converged
            if abs(current_nesz_db - desired_nesz_db) <= tolerance_db:
                converged = True
                best_look_angle = look_angle_test
                best_nesz = current_nesz_db
                break
            
            # Update best if this is closer
            if abs(current_nesz_db - desired_nesz_db) < abs(best_nesz - desired_nesz_db):
                best_look_angle = look_angle_test
                best_nesz = current_nesz_db
            
            # Adjust search range based on result
            # Higher NESZ (less negative, worse) means we need smaller look angle (closer to nadir)
            # Lower NESZ (more negative, better) means we can use larger look angle
            if current_nesz_db > desired_nesz_db:
                # Current NESZ is worse than desired, reduce look angle
                look_angle_max_deg = look_angle_test
            else:
                # Current NESZ is better than desired, can increase look angle
                look_angle_min_deg = look_angle_test
            
            # Check if search range has become too narrow
            if abs(look_angle_max_deg - look_angle_min_deg) < 0.001:
                break
        
        # Set to best found angle and get final corner information
        self.mission.satellite.look_angle_from_nadir_deg = best_look_angle
        all_corners = self.mission.nes0_db_all_corners
        
        # Calculate actual look angles to the corners from nadir
        # Get the angular offsets to corners
        corners_angles = self.mission.elevation_yaw_angles_to_corners_rad
        
        # Find the far corner (maximum slant range)
        far_corner_idx = np.argmax([corners_angles[i][2] for i in range(4)])
        _,_,_, graze_far = corners_angles[far_corner_idx]
        
        # Calculate look angle to far corner from nadir using exact geometry
        # Satellite at height H, slant range to corner is slant_range_far
        # Look angle from nadir: angle between vertical and slant range vector
        H = self.mission.satellite.orbit_altitude_m
        
        # Using spherical Earth geometry:
        # From grazing angle at corner, we can get the incidence angle at corner
        # Then use spherical law of sines to get look angle from satellite
        incidence_angle_far = np.pi / 2.0 - graze_far
        
        # Spherical law of sines: sin(look_angle) / R_E = sin(incidence_angle) / (R_E + H)
        sin_look_far = (EARTH_RADIUS_M / (EARTH_RADIUS_M + H)) * np.sin(incidence_angle_far)
        sin_look_far = np.clip(sin_look_far, -1.0, 1.0)
        look_angle_far_rad = np.arcsin(sin_look_far)
        look_angle_far_deg = np.degrees(look_angle_far_rad)
        
        # Restore original look angle
        self.mission.satellite.look_angle_from_nadir_deg = original_look_angle
        
        return {
            'look_angle_deg': np.float64(look_angle_far_deg),  # Return far corner angle, not boresight
            'boresight_look_angle_deg': np.float64(best_look_angle),  # Also return boresight
            'achieved_nesz_db': best_nesz,
            'iterations': iterations,
            'converged': converged,
            'all_corners_nesz': {
                name: info[0] for name, info in all_corners.items()
            },
            'error_db': best_nesz - desired_nesz_db,
        }


    def _calculate_max_swath_for_nesz(
        self,
        desired_nesz_db: float,
        aspect_ratio: float = 1.0,
        tolerance_db: float = 0.1,
        max_iterations: int = 100,
        initial_swath_m: float = 1000.0,
        max_swath_m: float = 500e3
    ) -> dict:
        """
        Calculate the maximum swath dimensions that achieve a desired worst-case NESZ.
        
        This function optimizes swath dimensions (range and azimuth) to maximize coverage area
        while ensuring the worst-case corner NESZ meets the specified requirement. Uses binary
        search for efficiency.
        
        Args:
            desired_nesz_db: Target maximum NESZ value in dB (e.g., -25 dB)
            aspect_ratio: Ratio of swath_range / swath_azimuth (default: 1.0 for square)
            tolerance_db: Acceptable difference from target in dB (default: 0.1 dB)
            max_iterations: Maximum number of iterations (default: 100)
            initial_swath_m: Starting swath size for search in meters (default: 1000 m)
            max_swath_m: Maximum swath size to consider in meters (default: 500 km)
            
        Returns:
            Dict containing:
                - 'swath_range_m': Optimized range swath dimension in meters
                - 'swath_azimuth_m': Optimized azimuth swath dimension in meters
                - 'swath_area_km2': Total swath area in square kilometers
                - 'achieved_nesz_db': Actual worst-case NESZ at these dimensions in dB
                - 'iterations': Number of iterations used
                - 'converged': Whether the solution converged within tolerance
                - 'all_corners_nesz': NESZ values at all four corners
        """
        # Store original swath dimensions to restore later
        original_range = self.mission.swath_range_m
        original_azimuth = self.mission.swath_azimuth_m
        
        # Initialise binary search bounds
        swath_min = initial_swath_m
        swath_max = max_swath_m
        
        converged = False
        iterations = 0
        best_swath = swath_min
        best_nesz = -float('inf')  
        
        for _ in range(max_iterations):
            iterations += 1
            
            # Try midpoint of current range (this is the "characteristic swath size")
            swath_test = (swath_min + swath_max) / 2.0
            
            # Apply aspect ratio to get range and azimuth dimensions
            # If aspect_ratio = swath_range / swath_azimuth, then:
            # swath_range = aspect_ratio * swath_azimuth
            # For a characteristic size S: swath_range * swath_azimuth = S^2
            # So: aspect_ratio * swath_azimuth^2 = S^2
            # Therefore: swath_azimuth = S / sqrt(aspect_ratio)
            #            swath_range = S * sqrt(aspect_ratio)
            swath_azimuth_test = swath_test / np.sqrt(aspect_ratio)
            swath_range_test = swath_test * np.sqrt(aspect_ratio)
            
            # Temporarily update swath dimensions
            self.mission.swath_range_m = swath_range_test
            self.mission.swath_azimuth_m = swath_azimuth_test
            
            # Calculate worst-case NESZ at these dimensions
            current_nesz_db = self.mission.nes0_db_corner
            
            # Check if we have a valid result
            if not np.isfinite(current_nesz_db):
                # Invalid geometry, reduce swath size
                swath_max = swath_test
                continue
            
            # Check if converged
            if abs(current_nesz_db - desired_nesz_db) <= tolerance_db:
                converged = True
                best_swath = swath_test
                best_nesz = current_nesz_db
                break
            
            # Update best if this is closer to target (higher NESZ = closer to desired for meeting requirement)
            # We want the largest swath that still meets the NESZ requirement
            if current_nesz_db <= desired_nesz_db and current_nesz_db > best_nesz:
                best_swath = swath_test
                best_nesz = current_nesz_db
            
            # Adjust search range based on result
            # Higher NESZ (less negative, worse) means we need smaller swath
            # Lower NESZ (more negative, better) means we can use larger swath
            if current_nesz_db > desired_nesz_db:
                # Current NESZ is worse than desired, reduce swath size
                swath_max = swath_test
            else:
                # Current NESZ is better than desired, can increase swath size
                swath_min = swath_test
            
            # Check if search range has become too narrow
            if abs(swath_max - swath_min) < 1.0:  # 1 meter precision
                break
        
        # Set to best found dimensions and get final corner information
        swath_azimuth_best = best_swath / np.sqrt(aspect_ratio)
        swath_range_best = best_swath * np.sqrt(aspect_ratio)
        
        self.mission.swath_range_m = swath_range_best
        self.mission.swath_azimuth_m = swath_azimuth_best
        all_corners = self.mission.nes0_db_all_corners
        
        # Calculate total area
        swath_area_km2 = (swath_range_best * swath_azimuth_best) / 1e6
        
        # Restore original swath dimensions
        self.mission.swath_range_m = original_range
        self.mission.swath_azimuth_m = original_azimuth
        
        return {
            'swath_range_m': swath_range_best,
            'swath_azimuth_m': swath_azimuth_best,
            'swath_area_km2': swath_area_km2,
            'achieved_nesz_db': best_nesz,
            'iterations': iterations,
            'converged': converged,
            'all_corners_nesz': {
                name: info[0] for name, info in all_corners.items()
            },
            'error_db': best_nesz - desired_nesz_db,
        }