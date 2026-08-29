# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
"""McStas input adapter for the SKADI workflow."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import scipp as sc
import scippnexus as snx
from scippneutron.conversion.graph import tof

from ..sans.conversions import ElasticCoordTransformGraph, sans_elastic
from ..sans.types import (
    CorrectForGravity,
    Filename,
    GravityVector,
    Position,
    RawDetector,
    RunType,
    WavelengthDetector,
)

# The McStas detector geometry description needs some adjustments.
# Specifically, the outermost pixels on the banks cover a slightly larger
# area than the rest. But that is not reflected in the McStas geometry data.
# These parameters are used to make the adjustments:
# Width and height assigned to low-resolution pixels.
LOW_RES_PIXEL_SIZE = sc.scalar(0.006, unit='m')
# Width and height assigned to high-resolution pixels.
HIGH_RES_PIXEL_SIZE = sc.scalar(0.003, unit='m')
# Added to the width or height of perimeter pixels; their centers move outward by half.
PERIMETER_PIXEL_EXTENSION = sc.scalar(0.00025, unit='m')
# Depth assigned to every pixel.
PIXEL_DEPTH = sc.scalar(0.001, unit='m')


@dataclass(frozen=True)
class _DetectorSpec:
    group_name: str
    component_name: str
    pixel_min: int
    x_limits: tuple[float, float]
    y_limits: tuple[float, float]
    shape: tuple[int, int]
    event_count: int

    @property
    def pixel_count(self) -> int:
        return self.shape[0] * self.shape[1]


_NUMBER = r"[-+]?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?"
_X_OPTIONS = re.compile(
    rf"x limits=\[\s*({_NUMBER})\s*,\s*({_NUMBER})\s*\]\s+bins=(\d+)"
)
_Y_OPTIONS = re.compile(
    rf"y limits=\[\s*({_NUMBER})\s*,\s*({_NUMBER})\s*\]\s+bins=(\d+)"
)
_PIXEL_MIN = re.compile(r"pixel min=(\d+)")


def _decode(value: str | bytes) -> str:
    return value.decode('utf-8') if isinstance(value, bytes) else value


def _mcstas_path(filename: str | Path) -> Path:
    path = Path(filename)
    if path.is_dir():
        path = path / 'mccode.h5'
    if not path.exists():
        raise FileNotFoundError(f"McStas file does not exist: {path}")
    return path


def _parse_detector_spec(name: str, group: h5py.Group) -> _DetectorSpec:
    options = _decode(group.attrs['options'])
    x_match = _X_OPTIONS.search(options)
    y_match = _Y_OPTIONS.search(options)
    pixel_match = _PIXEL_MIN.search(options)
    if x_match is None or y_match is None or pixel_match is None:
        raise ValueError(f"Cannot parse detector geometry options for {group.name!r}")
    return _DetectorSpec(
        group_name=name,
        component_name=_decode(group.attrs['component']),
        pixel_min=int(pixel_match.group(1)),
        x_limits=(float(x_match.group(1)), float(x_match.group(2))),
        y_limits=(float(y_match.group(1)), float(y_match.group(2))),
        shape=(int(y_match.group(3)), int(x_match.group(3))),
        event_count=group['events'].shape[0],
    )


def corrected_pixel_geometry(
    *,
    component_position: sc.Variable,
    component_rotation: np.ndarray,
    x_limits: tuple[float, float],
    y_limits: tuple[float, float],
    shape: tuple[int, int],
) -> tuple[sc.Variable, sc.Variable, sc.Variable]:
    """Build corrected global pixel positions, sizes, and surface normals.

    McStas records an even grid of pixel centers. The outermost pixels represent the
    remaining quarter-millimetre perimeter of each detector tile. Their centers are
    shifted outwards by half that amount and their corresponding size is enlarged.
    This is the correction used by the scripts supplied with the SKADI example data.

    Parameters
    ----------
    component_position:
        Global position of the McStas detector component in metres.
    component_rotation:
        McStas rotation matrix for the detector component.
    x_limits:
        Limits of the local pixel grid in metres.
    y_limits:
        Limits of the local pixel grid in metres.
    shape:
        Number of pixels as ``(y, x)``.

    Returns
    -------
    positions: scipp.Variable
        Corrected global pixel centers.
    sizes: scipp.Variable
        Corrected pixel sizes as ``(x, y, depth)``.
    normals: scipp.Variable
        Global detector surface normals.
    """
    ny, nx = shape
    if nx != ny or nx not in (8, 16):
        raise ValueError(
            f"Unsupported SKADI McStas detector shape {shape}; expected 8x8 or 16x16"
        )
    nominal_size = LOW_RES_PIXEL_SIZE if nx == 8 else HIGH_RES_PIXEL_SIZE

    x = sc.midpoints(sc.linspace('x', *x_limits, num=nx + 1, unit='m'))
    y = sc.midpoints(sc.linspace('y', *y_limits, num=ny + 1, unit='m'))
    shift = PERIMETER_PIXEL_EXTENSION / 2
    x['x', 0] -= shift
    x['x', -1] += shift
    y['y', 0] -= shift
    y['y', -1] += shift

    local = (
        sc.spatial.as_vectors(x=x, y=y, z=sc.scalar(0.0, unit='m'))
        .transpose(['y', 'x'])
        .flatten(to='detector_number')
    )
    # McStas applies the stored matrix to row vectors. Scipp transformations act on
    # column vectors, so the matrix must be transposed.
    rotation = sc.spatial.linear_transform(value=component_rotation.T)
    positions = component_position + rotation * local

    width = sc.broadcast(nominal_size, sizes={'y': ny, 'x': nx}).copy()
    height = width.copy()
    width['x', 0] += PERIMETER_PIXEL_EXTENSION
    width['x', -1] += PERIMETER_PIXEL_EXTENSION
    height['y', 0] += PERIMETER_PIXEL_EXTENSION
    height['y', -1] += PERIMETER_PIXEL_EXTENSION
    sizes = sc.spatial.as_vectors(
        width,
        height,
        sc.broadcast(PIXEL_DEPTH, sizes=width.sizes),
    ).flatten(to='detector_number')

    normal = rotation * sc.vector([0.0, 0.0, 1.0])
    normals = sc.broadcast(normal, sizes=positions.sizes)
    return positions, sizes, normals


def _component_map(components: h5py.Group) -> dict[str, h5py.Group]:
    return {
        name.split('_', maxsplit=1)[-1]: group
        for name, group in components.items()
        if isinstance(group, h5py.Group)
    }


def _component_position(
    components: dict[str, h5py.Group], component_name: str
) -> sc.Variable:
    component = components.get(component_name)
    if component is None or 'Position' not in component:
        raise ValueError(f"No instrument component found for {component_name!r}")
    return sc.vector(np.asarray(component['Position'][()], dtype=np.float64), unit='m')


def load_skadi_mcstas(
    filename: str | Path,
    *,
    source_name: str = 'sourceESS',
    sample_name: str = 'sample_position',
) -> sc.DataArray:
    """Load SKADI McStas detector events and apply the geometry correction.

    The McStas event probability is retained as the event value, with its square as
    the variance. Events are grouped by the global McStas pixel ID.

    Parameters
    ----------
    filename:
        McStas ``mccode.h5`` file or its containing directory.
    source_name:
        Component name identifying the source position.
    sample_name:
        Component name identifying the sample position.

    Returns
    -------
    :
        Calibrated, event-mode detector data suitable for the SKADI workflow.
    """
    path = _mcstas_path(filename)
    with h5py.File(path, 'r') as file:
        data_groups = file['entry1/data']
        specs = sorted(
            (
                _parse_detector_spec(name, group)
                for name, group in data_groups.items()
                if isinstance(group, h5py.Group) and 'events' in group
            ),
            key=lambda spec: spec.pixel_min,
        )
        if not specs:
            raise ValueError(f"No McStas detector event groups found in {path}")

        total_pixels = 0
        for spec in specs:
            if spec.pixel_min != total_pixels:
                raise ValueError(
                    "McStas detector groups do not cover a contiguous pixel-ID "
                    f"range; expected ID {total_pixels}, got {spec.pixel_min}"
                )
            total_pixels += spec.pixel_count

        total_events = sum(spec.event_count for spec in specs)
        positions = []
        pixel_sizes = []
        detector_normals = []
        weights = np.empty(total_events, dtype=np.float64)
        event_time_offset = np.empty(total_events, dtype=np.float64)
        pixel_ids = np.empty(total_events, dtype=np.int64)

        components = file['entry1/instrument/components']
        component_by_name = _component_map(components)
        event_cursor = 0
        for spec in specs:
            pixel_stop = spec.pixel_min + spec.pixel_count
            component = component_by_name.get(spec.component_name)
            if component is None:
                raise ValueError(
                    f"No instrument component found for {spec.component_name!r}"
                )
            geometry = corrected_pixel_geometry(
                component_position=sc.vector(component['Position'][()], unit='m'),
                component_rotation=np.asarray(component['Rotation'][()]),
                x_limits=spec.x_limits,
                y_limits=spec.y_limits,
                shape=spec.shape,
            )
            positions.append(geometry[0])
            pixel_sizes.append(geometry[1])
            detector_normals.append(geometry[2])

            group = data_groups[spec.group_name]
            columns = _decode(group.attrs.get('variables', 'p x y n id t')).split()
            column = {name: i for i, name in enumerate(columns)}
            events = group['events'][()]
            ids = events[:, column['id']].astype(np.int64)
            if np.any((ids < spec.pixel_min) | (ids >= pixel_stop)):
                raise ValueError(f"Out-of-range pixel ID in {group.name!r}")

            event_stop = event_cursor + spec.event_count
            weights[event_cursor:event_stop] = events[:, column['p']]
            event_time_offset[event_cursor:event_stop] = events[:, column['t']]
            pixel_ids[event_cursor:event_stop] = ids
            event_cursor = event_stop

        source_position = _component_position(component_by_name, source_name)
        sample_position = _component_position(component_by_name, sample_name)

    events = sc.DataArray(
        sc.array(
            dims=['event'],
            values=weights,
            variances=np.square(weights),
        ),
        coords={
            'event_time_offset': sc.array(
                dims=['event'], values=event_time_offset, unit='s'
            ),
            'detector_number': sc.array(dims=['event'], values=pixel_ids, unit=None),
        },
    )
    detector_dim = 'detector_number'
    detector_numbers = sc.arange(detector_dim, total_pixels, unit=None)
    return events.group(detector_numbers).assign_coords(
        position=sc.concat(positions, detector_dim),
        pixel_size=sc.concat(pixel_sizes, detector_dim),
        detector_normal=sc.concat(detector_normals, detector_dim),
        source_position=source_position,
        sample_position=sample_position,
    )


def load_skadi_mcstas_provider(
    filename: Filename[RunType],
) -> RawDetector[RunType]:
    """Load a McStas file for a run in the SKADI workflow."""
    return RawDetector[RunType](load_skadi_mcstas(filename))


def source_position_from_mcstas(
    detector: RawDetector[RunType],
) -> Position[snx.NXsource, RunType]:
    """Extract the source position attached by the McStas loader."""
    return Position[snx.NXsource, RunType](detector.coords['source_position'])


def sample_position_from_mcstas(
    detector: RawDetector[RunType],
) -> Position[snx.NXsample, RunType]:
    """Extract the sample position attached by the McStas loader."""
    return Position[snx.NXsample, RunType](detector.coords['sample_position'])


def mcstas_detector_coord_transform_graph(
    correct_for_gravity: CorrectForGravity,
    *,
    sample_position: Position[snx.NXsample, RunType],
    source_position: Position[snx.NXsource, RunType],
    gravity: GravityVector,
) -> ElasticCoordTransformGraph[RunType]:
    """Build the coordinate graph for McStas, whose event time is already TOF."""
    graph = sans_elastic(
        correct_for_gravity=correct_for_gravity,
        sample_position=sample_position,
        source_position=source_position,
        gravity=gravity,
    )
    return ElasticCoordTransformGraph[RunType](
        {**graph, **tof.elastic_wavelength('tof')}
    )


def mcstas_data_to_wavelength(
    detector: RawDetector[RunType],
    graph: ElasticCoordTransformGraph[RunType],
) -> WavelengthDetector[RunType]:
    """Convert McStas time-of-flight events to wavelength."""
    event_time_offset = detector.bins.coords['event_time_offset']
    detector = detector.bins.drop_coords('event_time_offset')
    detector.bins.coords['tof'] = event_time_offset
    return WavelengthDetector[RunType](
        detector.transform_coords(
            'wavelength', graph=graph, keep_intermediate=False, rename_dims=False
        )
    )


mcstas_providers = (
    load_skadi_mcstas_provider,
    source_position_from_mcstas,
    sample_position_from_mcstas,
    mcstas_detector_coord_transform_graph,
    mcstas_data_to_wavelength,
)
