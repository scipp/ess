# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
# Temporary helper for mcstas loader.
# This will be moved to the odin package in the future.
from types import MappingProxyType
from typing import NewType, cast

import scipp as sc
import scipp.constants as scc
import scippnexus as snx
from scippneutron.chopper import DiskChopper
from scippneutron.conversion import graph

from ess.reduce.nexus.types import FilePath

# Prepare graph for transformation
PLAIN_GRAPH = {**graph.beamline.beamline(False), **graph.tof.kinematic("tof")}

_DataPath = NewType('_DataPath', str)
_DefaultDataPath = _DataPath(
    "entry1/data/transmission_event_signal_dat_list_p_t_x_y_z_vx_vy_vz/events"
)
_FileLock = NewType('_FileLock', bool)
"""Lock the file to prevent concurrent access."""
_DefaultFileLock = _FileLock(True)
OdinSimulationRawData = NewType('OdinSimulationRawData', sc.DataArray)
ProbabilityToCountsScaleFactor = NewType('ProbabilityToCountsScaleFactor', sc.Variable)
"""Translate the probability to counts."""
DefaultProbabilityToCountsScaleFactor = ProbabilityToCountsScaleFactor(
    sc.scalar(1_000, unit='dimensionless')
)
DetectorStartX = NewType('DetectorStartX', sc.Variable)
"""Start of the detector in x direction."""
DefaultDetectorStartX = DetectorStartX(sc.scalar(-0.03, unit='m'))
DetectorStartY = NewType('DetectorStartY', sc.Variable)
"""Start of the detector in y direction."""
DefaultDetectorStartY = DetectorStartY(sc.scalar(-0.03, unit='m'))

DetectorEndX = NewType('DetectorEndX', sc.Variable)
"""End of the detector in x direction."""
DefaultDetectorEndX = DetectorEndX(sc.scalar(0.03, unit='m'))
DetectorEndY = NewType('DetectorEndY', sc.Variable)
"""End of the detector in y direction."""
DefaultDetectorEndY = DetectorEndY(sc.scalar(0.03, unit='m'))

McStasManualResolution = NewType('McStasManualResolution', tuple)
"""Manual resolution for McStas data (how many pixels per axis x, y)"""
DefaultMcStasManualResolution = McStasManualResolution((1024, 1024))


def _nth_col_or_row_lookup(
    start: sc.Variable, stop: sc.Variable, resolution: int, dim: str
) -> sc.Lookup:
    """Lookup the nth column or row."""
    position = sc.linspace(
        dim, start=start, stop=stop, num=resolution + 1, unit=start.unit
    )
    nth_col_or_row = sc.arange(dim=dim, start=0, stop=resolution, unit='dimensionless')
    hist = sc.DataArray(data=nth_col_or_row, coords={dim: position})
    return sc.lookup(hist, dim)


def _position_to_pixel_id(
    *,
    x_pos: sc.Variable,
    y_pos: sc.Variable,
    detector_start_x: DetectorStartX = DefaultDetectorStartX,
    detector_start_y: DetectorStartY = DefaultDetectorStartY,
    detector_end_x: DetectorEndX = DefaultDetectorEndX,
    detector_end_y: DetectorEndY = DefaultDetectorEndY,
    resolution: McStasManualResolution = DefaultMcStasManualResolution,
) -> sc.Variable:
    """Hardcode pixel ids from positions."""
    x_position_lookup = _nth_col_or_row_lookup(
        detector_start_x, detector_end_x, resolution[0], 'x'
    )
    y_position_lookup = _nth_col_or_row_lookup(
        detector_start_y, detector_end_y, resolution[1], 'y'
    )
    n_cols = x_position_lookup[x_pos]
    n_rows = y_position_lookup[y_pos]
    return n_rows * resolution[0] + n_cols


McStasVelocities = NewType('McStasVelocities', sc.DataGroup)


def load_velocities(
    file_path: FilePath,
    _data_path: _DataPath = _DefaultDataPath,
    _file_lock: _FileLock = _DefaultFileLock,
) -> McStasVelocities:
    with snx.File(file_path, "r", locking=_file_lock) as f:
        data = f[_data_path][()].rename_dims({'dim_0': 'event'})
        velocities = data['dim_1', 5:8]
        vx = cast(sc.Variable, velocities['dim_1', 0].copy())
        vy = cast(sc.Variable, velocities['dim_1', 1].copy())
        vz = cast(sc.Variable, velocities['dim_1', 2].copy())
        for v_component in (vx, vy, vz):
            v_component.unit = 'm/s'
        # Add special tags if you want to use them as coordinates
        # for example, da.coords['vx_MC'] = vx
        # to distinguish them from the measurement
        return McStasVelocities(sc.DataGroup(vx=vx, vy=vy, vz=vz))


LoadTrueVelocities = NewType('LoadTrueVelocities', bool)
DefaultLoadTrueVelocities = LoadTrueVelocities(True)


def load_odin_simulation_data(
    file_path: FilePath,
    _data_path: _DataPath = _DefaultDataPath,
    _file_lock: _FileLock = _DefaultFileLock,
    detector_start_x: DetectorStartX = DefaultDetectorStartX,
    detector_start_y: DetectorStartY = DefaultDetectorStartY,
    detector_end_x: DetectorEndX = DefaultDetectorEndX,
    detector_end_y: DetectorEndY = DefaultDetectorEndY,
    resolution: McStasManualResolution = DefaultMcStasManualResolution,
    probability_scale_factor: ProbabilityToCountsScaleFactor = DefaultProbabilityToCountsScaleFactor,  # noqa: E501
    load_true_velocities: LoadTrueVelocities = DefaultLoadTrueVelocities,
) -> OdinSimulationRawData:
    with snx.File(file_path, "r", locking=_file_lock) as f:
        # The name p_t_x_y_z_vx_vy_vz represents
        # probability, time of arrival, position(x, y, z) and velocity(vx, vy, vz).
        # The name also represents the order of each field in the table.
        # For example, probability is the first field, so data['dim_1', 0] is the probability.  # noqa: E501
        data = f[_data_path][()].rename_dims({'dim_0': 'event'})
        probabilities = cast(sc.Variable, data['dim_1', 0].copy())
        probabilities.unit = 'dimensionless'
        time_of_arrival = cast(sc.Variable, data['dim_1', 1].copy())
        time_of_arrival.unit = 's'  # Hardcoded unit from the data.
        positions = data['dim_1', 2:5]
        counts = (probabilities / probabilities.max()) * probability_scale_factor
        counts.unit = 'counts'
        counts.variances = counts.values**2
        # Units are hardcoded from the data.
        x_pos = cast(sc.Variable, positions['dim_1', 0].copy())
        x_pos.unit = 'm'
        y_pos = cast(sc.Variable, positions['dim_1', 1].copy())
        y_pos.unit = 'm'
        pixel_id = _position_to_pixel_id(
            x_pos=x_pos,
            y_pos=y_pos,
            detector_start_x=detector_start_x,
            detector_start_y=detector_start_y,
            detector_end_x=detector_end_x,
            detector_end_y=detector_end_y,
            resolution=resolution,
        )
        da = sc.DataArray(
            data=counts.copy(),
            coords={
                'time_of_arrival': time_of_arrival.to(unit='us'),
                'sample_position': sc.vector([0.0, 0.0, 60.5], unit='m'),
                # Hardcoded from the data.
                'source_position': sc.vector([0.0, 0.0, 0.0], unit="m"),
                # Hardcoded from the data.
                'pixel_id': pixel_id,
            },
        )
        if load_true_velocities:
            velocities = load_velocities(file_path, _data_path, _file_lock)
            speeds = sc.norm(
                sc.vectors(
                    dims=['event'],
                    values=sc.transpose(
                        sc.concat(list(velocities.values()), 'speed')
                    ).values,
                    unit='m/s',
                )
            )
            da.coords['sim_wavelength'] = (scc.h / scc.neutron_mass / speeds).to(
                unit='angstrom'
            )

        return OdinSimulationRawData(da.to(dtype=float))


def pixel_ids_to_x(
    *,
    pixel_id: sc.Variable,
    resolution: McStasManualResolution = DefaultMcStasManualResolution,
    detector_start_x: DetectorStartX = DefaultDetectorStartX,
    detector_end_x: DetectorEndX = DefaultDetectorEndX,
) -> sc.Variable:
    n_col = pixel_id % resolution[0]
    x_interval = (detector_end_x - detector_start_x) / resolution[0]
    return (
        detector_start_x + n_col * x_interval
    ) + x_interval / 2  # Center of the pixel|


def pixel_ids_to_y(
    *,
    pixel_id: sc.Variable,
    resolution: McStasManualResolution = DefaultMcStasManualResolution,
    detector_start_y: DetectorStartY = DefaultDetectorStartY,
    detector_end_y: DetectorEndY = DefaultDetectorEndY,
) -> sc.Variable:
    n_row = pixel_id // resolution[0]
    y_interval = (detector_end_y - detector_start_y) / resolution[1]
    return (
        detector_start_y + n_row * y_interval
    ) + y_interval / 2  # Center of the pixel


def pixel_ids_to_position(
    *, x: sc.Variable, y: sc.Variable, z: sc.Variable
) -> sc.Variable:
    z = sc.zeros_like(x) + z
    var = (
        sc.concat([x, y, z], 'event')
        .fold('event', dims=['pos', 'event'], shape=[3, len(x)])
        .transpose(dims=['event', 'pos'])
        .values
    )
    return sc.vectors(dims=['event'], values=var, unit='m')


def to_nexus(da, *, resolution: McStasManualResolution, graph: dict = PLAIN_GRAPH):
    # Copy data and add new coordinates
    out = da.copy(deep=False)
    # Event time zero/offset
    unit = 'ns'
    period = (1.0 / sc.scalar(14.0, unit='Hz')).to(unit=unit)
    start = sc.datetime("2024-01-01T12:00:00.000000000")
    out.coords['event_time_zero'] = (
        period * (da.coords['time_of_arrival'].to(unit='ns', copy=False) // period)
    ).to(dtype=int) + start
    out.coords['event_time_offset'] = out.coords.pop('time_of_arrival') % period.to(
        unit=da.coords['time_of_arrival'].unit
    )
    # Position
    x = pixel_ids_to_x(pixel_id=out.coords['pixel_id'], resolution=resolution)
    y = pixel_ids_to_y(pixel_id=out.coords['pixel_id'], resolution=resolution)
    z = sc.scalar(60.5, unit='m')  # Hardcoded from the data.
    out.coords['position'] = pixel_ids_to_position(x=x, y=y, z=z)
    # Group by pixel_id to have pixel positions on the top level
    out = out.group('pixel_id')
    out.coords['position'] = out.bins.coords['position'].bins.mean()
    del out.bins.coords['position']
    return out.transform_coords("Ltotal", graph=graph, keep_intermediate=True)


# Choppers
Hz = sc.Unit("Hz")
deg = sc.Unit("deg")
meter = sc.Unit("m")

parameters = {
    "WFMC_1": {
        "frequency": 56.0,
        "phase": 93.244,
        "distance": 6.85,
        "open": [-1.9419, 49.5756, 98.9315, 146.2165, 191.5176, 234.9179],
        "close": [1.9419, 55.7157, 107.2332, 156.5891, 203.8741, 249.1752],
    },
    "WFMC_2": {
        "frequency": 56.0,
        "phase": 97.128,
        "distance": 7.15,
        "open": [-1.9419, 51.8318, 103.3493, 152.7052, 199.9903, 245.2914],
        "close": [1.9419, 57.9719, 111.6510, 163.0778, 212.3468, 259.5486],
    },
    "FOC_1": {
        "frequency": 42.0,
        "phase": 81.303297,
        "distance": 8.4,
        "open": [-5.1362, 42.5536, 88.2425, 132.0144, 173.9497, 216.7867],
        "close": [5.1362, 54.2095, 101.2237, 146.2653, 189.417, 230.7582],
    },
    "BP_1": {
        "frequency": 7.0,
        "phase": 31.080,
        "distance": 8.45,
        "open": [-23.6029],
        "close": [23.6029],
    },
    "FOC_2": {
        "frequency": 42.0,
        "phase": 107.013442,
        "distance": 12.2,
        "open": [-16.3227, 53.7401, 120.8633, 185.1701, 246.7787, 307.0165],
        "close": [16.3227, 86.8303, 154.3794, 218.7551, 280.7508, 340.3188],
    },
    "BP_2": {
        "frequency": 7.0,
        "phase": 44.224,
        "distance": 12.25,
        "open": [-34.4663],
        "close": [34.4663],
    },
    "T0_alpha": {
        "frequency": 14.0,
        "phase": 179.672,
        "distance": 13.5,
        "open": [-167.8986],
        "close": [167.8986],
    },
    "T0_beta": {
        "frequency": 14.0,
        "phase": 179.672,
        "distance": 13.7,
        "open": [-167.8986],
        "close": [167.8986],
    },
    "FOC_3": {
        "frequency": 28.0,
        "phase": 92.993,
        "distance": 17.0,
        "open": [-20.302, 45.247, 108.0457, 168.2095, 225.8489, 282.2199],
        "close": [20.302, 85.357, 147.6824, 207.3927, 264.5977, 319.4024],
    },
    "FOC_4": {
        "frequency": 14.0,
        "phase": 61.584,
        "distance": 23.69,
        "open": [-16.7157, 29.1882, 73.1661, 115.2988, 155.6636, 195.5254],
        "close": [16.7157, 61.8217, 105.0352, 146.4355, 186.0987, 224.0978],
    },
    "FOC_5": {
        "frequency": 14.0,
        "phase": 82.581,
        "distance": 33.0,
        "open": [-25.8514, 38.3239, 99.8064, 160.1254, 217.4321, 272.5426],
        "close": [25.8514, 88.4621, 147.4729, 204.0245, 257.7603, 313.7139],
    },
}

DISK_CHOPPERS = MappingProxyType(
    {
        key: DiskChopper(
            frequency=-ch["frequency"] * Hz,
            beam_position=sc.scalar(0.0, unit="deg"),
            phase=-ch["phase"] * deg,
            axle_position=sc.vector(value=[0, 0, ch["distance"]], unit="m"),
            slit_begin=sc.array(dims=["cutout"], values=ch["open"], unit="deg"),
            slit_end=sc.array(dims=["cutout"], values=ch["close"], unit="deg"),
        )
        for key, ch in parameters.items()
    }
)
"""Hard-coded DISK_CHOPPERS dictionary for ESS ODIN choppers."""
