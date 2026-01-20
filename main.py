"""
Demonstration script for calculating NESZ for a Spot SAR mission.
Author: Kwok Keith
Date: 20 Jan 2026
"""

from spot_mission import SpotMission
from mission_environment import EnvironmentParameters
from radar_signal import Signal
from satellite import Satellite
from phased_array import PhasedArray


def main():
    satellite = Satellite(
        platform_velocity_mps=7000.0,
        orbit_altitude_m=580000.0,
        look_angle_from_nadir_deg=20.0,
        radar_loss_db=5.0,
        receiver_noise_factor_db=4.0,
    )
    phased_array = PhasedArray(
        num_width_elements=32,
        num_height_elements=32,
        element_width_m=0.1,
        element_height_m=0.053125,
        element_power_w=3.125,
        antenna_efficiency=0.95,
    )
    signal = Signal(
        centre_frequency_hz=10e9,
        bandwidth_hz=1.2e9,
        prf_hz=4000.0,
        broadening_factor_azimuth=1.2,
        broadening_factor_range=1.2,
        range_processing_loss_db=0,
        azimuth_processing_loss_db=0,
        pulse_width_us=37.5,
        doppler_gain_constant=1.5,
        antenna=phased_array,
    )
    environment_parameters = EnvironmentParameters(
        nominal_temperature_k=300.0,
        two_way_atmospheric_loss_db=3.24e-06,
    )

    mission = SpotMission(
        swath_m=5000.0,
        signal=signal,
        satellite=satellite,
        antenna=phased_array,
        environment_parameters=environment_parameters,
        integration_angle_deg=30.0,
    )

    print("--- Input Parameters ---")
    print(
        f"""
Swath Width: {mission.swath_m} m
Bandwidth: {mission.signal.bandwidth_hz / 1e6} MHz
Transmit Power: {mission.antenna.total_peak_power_w} W
TX Duty Cycle: {mission.signal.tx_duty_cycle * 100} %
PRF: {mission.signal.prf_hz} Hz
Pulse Width: {mission.signal.pulse_width_us} µs
Antenna Gain: {mission.signal.peak_antenna_gain_linear} W
Nominal Wavelength: {mission.signal.nominal_wavelength_m} m
Broadening Factor (Azimuth): {mission.signal.broadening_factor_azimuth}
Broadening Factor (Range): {mission.signal.broadening_factor_range}
Look Angle (deg): {mission.satellite.look_angle_from_nadir_deg} °
Satellite Height: {mission.satellite.orbit_altitude_m} m
Platform Velocity: {mission.satellite.platform_velocity_mps} m/s
Nominal Temperature: {mission.environment_parameters.nominal_temperature_k} K
Receiver Noise Factor: {mission.satellite.receiver_noise_factor_db} dB
Two-way Atmospheric Loss: {mission.environment_parameters.two_way_atmospheric_loss_db} dB
Atmospheric Loss over Swath: {mission.atmospheric_loss_db} dB
Range Processing Loss: {mission.signal.range_processing_loss_db} dB
Azimuth Processing Loss: {mission.signal.azimuth_processing_loss_db} dB
"""
    )
    print("--- Derived Parameters ---")
    print(
        f"""
Average Power: {mission.signal.average_tx_power_w} W
Range: {mission.satellite.slant_range_flat_earth_m} m
Thermal Loss: {mission.thermal_loss_linear} W
System Loss: {mission.system_loss_linear} W
Graze Angle: {mission.satellite.graze_angle_deg} °
Range Resolution: {mission.range_resolution_m} m
Azimuth Resolution: {mission.azimuth_resolution_m} m
Antenna Area: {mission.antenna.antenna_area_m2} m²
Effective Antenna Area: {mission.antenna.effective_antenna_area_m2} m²
Total Elements: {mission.antenna.total_elements} elements
Total transmit Power: {mission.antenna.total_peak_power_w} W
        """
    )
    print("--- NESZ Calculation ---")
    print("NESZ (Noise Equivalent Sigma Zero):", mission.nes0_db, "dB")


if __name__ == "__main__":
    main()
