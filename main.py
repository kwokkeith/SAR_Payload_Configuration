from spot_mission import SpotMission
from mission_environment import PhysicalConstants, EnvironmentParameters
from signal import Signal
from satellite import Satellite
from phased_array import PhasedArray


def main():
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
    )
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
        tx_duty_cycle=0.15,
    )
    physical_constants = PhysicalConstants()
    environment_parameters = EnvironmentParameters(
        nominal_temperature_k=300.0,
        two_way_atmospheric_loss_db=3.24e-06,
    )

    mission = SpotMission(
        swath_m=5000.0,
        signal=signal,
        satellite=satellite,
        phased_array=phased_array,
        physical_constants=physical_constants,
        environment_parameters=environment_parameters,
        integration_angle_deg=30.0,
    )

    print(
        "Effective Antenna Area (m^2):", mission.phased_array.effective_antenna_area_m2
    )
    print("Range (m):", mission.satellite.slant_range_flat_earth_m)
    print("Thermal Loss (linear):", mission.thermal_loss_linear)

    print("\n")
    print("L_radar (linear):", mission.satellite.radar_loss_linear)
    print("L_thermal (linear):", mission.thermal_loss_linear)
    print("L_range (linear):", mission.signal.range_processing_loss_linear)
    print("L_azimuth (linear):", mission.signal.azimuth_processing_loss_linear)
    print("L_atmosphere (db):", mission.atmospheric_loss_db)
    print("\n")

    print("System Loss (linear):", mission.system_loss_linear)
    print("System Loss (dB):", mission.system_loss_db)
    print("Average Power (W):", mission.phased_array.average_tx_power_w)
    print("Nominal wavelength (m):", mission.signal.nominal_wavelength_m)
    print("Antenna Gain (linear):", mission.antenna_gain_linear)
    print("Range Resolution (m):", mission.range_resolution_m)
    print("NESZ (Noise Equivalent Sigma Zero):", mission.nes0_db, "dB")


if __name__ == "__main__":
    main()
