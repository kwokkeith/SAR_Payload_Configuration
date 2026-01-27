# SAR Payload Configurator

This repository allows SAR payload designers to set mission parameters required for their SAR processing needs. 

It includes a way to check for a SAR system's NESZ, ground range resolution and is modularisable for other antenna and SAR processing types.

Reference Material:

Doerry, A. W. (2006). Performance limits for synthetic aperture radar (2nd ed., SAND2006-0821). Sandia National Laboratories. https://doi.org/10.2172/773988

## Core Module Overview

### Mission Planning & Optimization

#### `mission.py`
Base class for radar satellite missions. Provides core functionality for:
- NESZ (Noise Equivalent Sigma Zero) calculations at swath center and corners
- Slant range and ground range resolution calculations
- Look angle range calculations (minimum and maximum across swath)
- Geometric calculations for corner positions using spherical Earth geometry
- Antenna gain pattern integration across the swath

#### `spot_mission.py`
Spotlight SAR mission implementation extending the base `Mission` class. Calculates:
- Azimuth resolution for spotlight mode based on integration angle
- SAR processing parameters specific to spotlight imaging geometry

#### `mission_optimiser.py`
Optimisation tools for mission parameter tuning:
- `optimise_max_look_angle_for_nesz()`: Find maximum look angle achieving desired NESZ
- `optimise_swath_for_nesz()`: Calculate maximum swath dimensions for NESZ requirements
- `optimise_min_look_angle_for_resolution()`: Find minimum look angle for target ground range resolution
- `calculate_swath_from_look_angles()`: Determine swath dimensions from look angle constraints
- Uses binary search algorithms for efficient convergence

### Antenna Models

#### `antenna.py`
Abstract base class defining the antenna interface:
- Physical and effective antenna area calculations
- Total peak power properties
- Antenna efficiency parameters

#### `patch_antenna.py`
Rectangular patch antenna implementation:
- Physical dimensions (length × width)
- Transmit power configuration
- Simple area-based calculations for single patch elements

#### `phased_array.py`
Phased array antenna system implementation:
- Element configuration (width × height array dimensions)
- Per-element physical dimensions and power
- Total array area and power calculations
- Supports large-scale arrays with multiple elements

### Signal Processing

#### `radar_signal.py`
Base radar signal parameters and calculations:
- Center frequency and bandwidth
- Pulse Repetition Frequency (PRF) and Pulse Width
- Processing losses (range and azimuth)
- Doppler gain constants
- Broadening factors for resolution calculations
- Links to antenna for gain pattern integration

#### `patch_antenna_signal.py`
Signal characteristics specific to patch antennas:
- Separable 3D radiation pattern modeling
- Normalized radiation pattern using sinc functions
- Gain pattern calculations in dB
- Peak antenna gain computation
- Reference: Based on antenna theory for rectangular apertures

### Platform & Environment

#### `satellite.py`
Satellite platform parameters and derived properties:
- Orbit altitude and platform velocity
- Look angle from nadir configuration
- Grazing angle calculations (spherical Earth geometry)
- Slant range calculations
- Radar loss and receiver noise factor
- Supports both flat Earth and spherical Earth approximations

#### `mission_environment.py`
Physical constants and environmental parameters:
- Speed of light, Boltzmann constant, Earth radius
- Nominal temperature (Kelvin/Celsius)
- Two-way atmospheric loss modeling
- Provides fundamental constants used throughout calculations

### Example Scripts

#### `main.py`
Demonstration script showing:
- Mission configuration setup
- NESZ calculations for Spot SAR missions
- Example parameter configurations
- Integration of all components

## Usage Example

```python
from spot_mission import SpotMission
from mission_environment import EnvironmentParameters
from patch_antenna_signal import PatchAntennaSignal
from patch_antenna import PatchAntenna
from satellite import Satellite
from mission_optimiser import MissionOptimiser

# Configure satellite
satellite = Satellite(
    orbit_altitude_m=500_000.0,
    look_angle_from_nadir_deg=40.0,
    platform_velocity_mps=7000.0,
    receiver_noise_factor_db=4.0,
    radar_loss_db=5.0,
)

# Configure patch antenna
patch_antenna = PatchAntenna(
    length=0.02,
    width=0.02,
    tx_power_w=3200.0,
    antenna_efficiency=0.85,
)

# Configure signal
signal = PatchAntennaSignal(
    centre_frequency_hz=12e9,
    bandwidth_hz=1.2e9,
    prf_hz=4000.0,
    pulse_width_us=37.5,
    antenna=patch_antenna,
    broadening_factor_azimuth=1.2,
    broadening_factor_range=1.2,
    range_processing_loss_db=0,
    azimuth_processing_loss_db=0,
    doppler_gain_constant=1.5,
)

# Configure environment
environment = EnvironmentParameters(
    nominal_temperature_k=300.0,
)

# Create mission
mission = SpotMission(
    satellite=satellite,
    signal=signal,
    environment_parameters=environment,
    swath_range_m=20e3,
    swath_azimuth_m=20e3,
    antenna=patch_antenna,
    integration_angle_deg=20.0,
)

# Calculate NESZ
print(f"NESZ at boresight: {mission.nes0_db:.2f} dB")
print(f"Worst-case corner NESZ: {mission.nes0_db_corner:.2f} dB")

# Optimize mission parameters
optimiser = MissionOptimiser(mission)

# Find maximum look angle for desired NESZ
result = optimiser.optimise_max_look_angle_for_nesz(desired_nesz_db=-25.0)
print(f"Max look angle: {result['look_angle_deg']:.2f}°")

# Calculate swath from look angles
min_look, max_look = mission.look_angle_range_deg
print(f"Look angle range: {min_look:.2f}° to {max_look:.2f}°")
```

## Key Features

- **Modular Design**: Easily swap antenna types (patch, phased array) and mission modes
- **Spherical Earth Geometry**: Accurate calculations accounting for Earth's curvature
- **NESZ Analysis**: Center, corner, and arbitrary position calculations
- **Optimization Tools**: Binary search algorithms for parameter tuning
- **Resolution Calculations**: Both slant range and ground range resolutions
- **Extensible Architecture**: Abstract base classes enable custom implementations

## Requirements

- Python 3.x
- NumPy
- Matplotlib (for visualization notebooks)

## Author

Kwok Keith  
January 2026


