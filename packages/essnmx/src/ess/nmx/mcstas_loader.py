# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from dataclasses import dataclass
from types import MappingProxyType
from typing import Iterable, NamedTuple, NewType, Optional, Protocol, Tuple

import numpy as np
import scipp as sc
import scippnexus as snx
from typing_extensions import Self

PixelIDs = NewType("PixelIDs", sc.Variable)
InputFilepath = NewType("InputFilepath", str)
NMXData = NewType("NMXData", sc.DataGroup)

# McStas Configurations
MaximumProbability = NewType("MaximumProbability", int)
DefaultMaximumProbability = MaximumProbability(100_000)


AXIS_TO_VECTOR = MappingProxyType(
    {
        'x': sc.vector([1.0, 0.0, 0.0]),
        'y': sc.vector([0.0, 1.0, 0.0]),
        'z': sc.vector([0.0, 0.0, 1.0]),
    }
)


class _XML(Protocol):
    """XML element type. Temporarily used for type hinting.

    Builtin XML type is blocked by bandit security check."""

    tag: str
    attrib: dict[str, str]

    def find(self, name: str) -> Optional[Self]:
        ...

    def __iter__(self) -> Self:
        ...

    def __next__(self) -> Self:
        ...


class Position3D(NamedTuple):
    """3D vector of location."""

    x: float
    y: float
    z: float


class RotationAxisAngle(NamedTuple):
    """Rotation in axis-angle representation."""

    theta: float
    x: float
    y: float
    z: float


@dataclass
class DetectorDesc:
    """Combined information of detector and detector type in McStas."""

    component_type: str  # 'type'
    name: str
    id_start: int  # 'idstart'
    fast_axis_name: str  # 'idfillbyfirst'
    position: sc.Variable  # <location> 'x', 'y', 'z'
    rotation: RotationAxisAngle
    num_x: int  # 'xpixels'
    num_y: int  # 'ypixels'
    step_x: float  # 'xstep'
    step_y: float  # 'ystep'
    # Calculated fields
    _rotation_matrix: Optional[sc.Variable] = None
    _fast_axis: Optional[sc.Variable] = None
    _slow_axis: Optional[sc.Variable] = None

    @property
    def total_pixels(self) -> int:
        return self.num_x * self.num_y

    @property
    def slow_axis_name(self) -> str:
        if self.fast_axis_name not in 'xy':
            raise ValueError(
                f"Invalid slow axis {self.fast_axis_name}.Should be 'x' or 'y'."
            )

        return 'xy'.replace(self.fast_axis_name, '')

    @property
    def rotation_matrix(self) -> sc.Variable:
        if self._rotation_matrix is None:
            from .rotation import axis_angle_to_quaternion, quaternion_to_matrix

            theta, x, y, z = self.rotation
            q = axis_angle_to_quaternion(x, y, z, sc.scalar(-theta, unit='deg'))
            self._rotation_matrix = quaternion_to_matrix(*q)

        return self._rotation_matrix

    def _rotate_axis(self, axis: sc.Variable) -> sc.Variable:
        return sc.vector(np.round((self.rotation_matrix * axis).values, 2))

    @property
    def fast_axis(self) -> sc.Variable:
        if self._fast_axis is None:
            self._fast_axis = self._rotate_axis(AXIS_TO_VECTOR[self.fast_axis_name])

        return self._fast_axis

    @property
    def slow_axis(self) -> sc.Variable:
        if self._slow_axis is None:
            self._slow_axis = self._rotate_axis(AXIS_TO_VECTOR[self.slow_axis_name])

        return self._slow_axis


def _retrieve_event_list_name(keys: Iterable[str]) -> str:
    prefix = "bank01_events_dat_list"

    # (weight, x, y, n, pixel id, time of arrival)
    mandatory_fields = 'p_x_y_n_id_t'

    for key in keys:
        if key.startswith(prefix) and mandatory_fields in key:
            return key

    raise ValueError("Can not find event list name.")


def _copy_partial_var(
    var: sc.Variable, idx: int, unit: Optional[str] = None, dtype: Optional[str] = None
) -> sc.Variable:
    """Retrieve a property from a variable."""
    var = var['dim_1', idx].astype(dtype or var.dtype, copy=True)
    if unit:
        var.unit = sc.Unit(unit)
    return var


def _get_mcstas_pixel_ids(detector_descs: Tuple[DetectorDesc, ...]) -> PixelIDs:
    """pixel IDs for each detector"""
    intervals = [
        (desc.id_start, desc.id_start + desc.total_pixels) for desc in detector_descs
    ]
    ids = [sc.arange('id', start, stop, unit=None) for start, stop in intervals]
    return PixelIDs(sc.concat(ids, 'id'))


def _pixel_positions(detector: DetectorDesc, sample_position: sc.Variable):
    """pixel IDs for each detector"""
    pixel_idx = sc.arange('id', detector.total_pixels)
    n_rows = sc.scalar(
        detector.num_x if detector.fast_axis_name == 'x' else detector.num_y
    )
    steps = {
        'x': sc.scalar(detector.step_x, unit='m'),
        'y': sc.scalar(detector.step_y, unit='m'),
    }

    pixel_n_slow = pixel_idx // n_rows
    pixel_n_fast = pixel_idx % n_rows

    fast_axis_steps = detector.fast_axis * steps[detector.fast_axis_name]
    slow_axis_steps = detector.slow_axis * steps[detector.slow_axis_name]

    return (
        (pixel_n_slow * slow_axis_steps)
        + (pixel_n_fast * fast_axis_steps)
        + (detector.position - sample_position)
    )


def _get_mcstas_pixel_positions(
    detector_descs: Tuple[DetectorDesc, ...], sample_position
):
    """pixel IDs for each detector"""
    positions = [
        _pixel_positions(detector, sample_position) for detector in detector_descs
    ]
    return sc.concat(positions, 'panel')


def _read_mcstas_geometry_xml(file_path: InputFilepath) -> bytes:
    """Retrieve geometry parameters from mcstas file"""
    import h5py

    instrument_xml_path = 'entry1/instrument/instrument_xml/data'
    with h5py.File(file_path) as file:
        return file[instrument_xml_path][...][0]


def _select_by_type_prefix(components: list[_XML], prefix: str) -> list[_XML]:
    """Select components by type prefix."""
    return [comp for comp in components if comp.attrib['type'].startswith(prefix)]


def _check_and_unpack_if_only_one(xml_items: list[_XML], name: str) -> _XML:
    """Check if there is only one element with ``name``."""
    if len(xml_items) > 1:
        raise ValueError(f"Multiple {name}s found.")
    elif len(xml_items) == 0:
        raise ValueError(f"No {name} found.")

    return xml_items.pop()


def _retrieve_attribs(component: _XML, *args: str) -> list[float]:
    """Retrieve ``args`` from xml."""

    return [float(component.attrib[key]) for key in args]


def find_location(component: _XML) -> _XML:
    """Retrieve ``location`` from xml component."""
    location = component.find('location')
    if location is None:
        raise ValueError("No location found in component ", component.find('name'))

    return location


def _retrieve_3d_position(component: _XML) -> sc.Variable:
    """Retrieve x, y, z position from xml."""
    location = find_location(component)

    return sc.vector(_retrieve_attribs(location, 'x', 'y', 'z'), unit='m')


def _retrieve_detector_descriptions(tree: _XML) -> Tuple[DetectorDesc, ...]:
    """Retrieve detector geometry descriptions from mcstas file."""

    def _retrieve_rotation_axis_angle(component: _XML) -> RotationAxisAngle:
        """Retrieve rotation angle(theta), x, y, z axes from location."""
        location = find_location(component)
        return RotationAxisAngle(
            *_retrieve_attribs(location, 'rot', 'axis-x', 'axis-y', 'axis-z')
        )

    def _find_type_desc(det: _XML, types: list[_XML]) -> _XML:
        for type_ in types:
            if type_.attrib['name'] == det.attrib['type']:
                return type_

        raise ValueError(
            f"Cannot find type {det.attrib['type']} for {det.attrib['name']}."
        )

    components = [branch for branch in tree if branch.tag == 'component']
    detectors = [
        comp for comp in components if comp.attrib['type'].startswith('MonNDtype')
    ]
    type_list = [branch for branch in tree if branch.tag == 'type']

    detector_components = []
    for det in detectors:
        det_type = _find_type_desc(det, type_list)

        detector_components.append(
            DetectorDesc(
                component_type=det_type.attrib['name'],
                name=det.attrib['name'],
                id_start=int(det.attrib['idstart']),
                fast_axis_name=det.attrib['idfillbyfirst'],
                position=_retrieve_3d_position(det),
                rotation=RotationAxisAngle(*_retrieve_rotation_axis_angle(det)),
                num_x=int(det_type.attrib['xpixels']),
                num_y=int(det_type.attrib['ypixels']),
                step_x=float(det_type.attrib['xstep']),
                step_y=float(det_type.attrib['ystep']),
            )
        )

    return tuple(sorted(detector_components, key=lambda x: x.id_start))


def load_mcstas_nexus(
    file_path: InputFilepath,
    max_probability: Optional[MaximumProbability] = None,
) -> NMXData:
    """Load McStas simulation result from h5(nexus) file.

    Parameters
    ----------
    file_path:
        File name to load.

    max_probability:
        The maximum probability to scale the weights.

    """
    from defusedxml.ElementTree import fromstring

    tree = fromstring(_read_mcstas_geometry_xml(file_path))
    detector_descs = _retrieve_detector_descriptions(tree)
    components = [branch for branch in tree if branch.tag == 'component']
    sources = _select_by_type_prefix(components, 'sourceMantid-type')
    samples = _select_by_type_prefix(components, 'sampleMantid-type')
    source = _check_and_unpack_if_only_one(sources, 'source')
    sample = _check_and_unpack_if_only_one(samples, 'sample')
    sample_position = _retrieve_3d_position(sample)

    slow_axes = [det.slow_axis for det in detector_descs]
    fast_axes = [det.fast_axis for det in detector_descs]
    origins = [det.position - sample_position for det in detector_descs]

    probability = max_probability or DefaultMaximumProbability

    with snx.File(file_path) as file:
        bank_name = _retrieve_event_list_name(file["entry1/data"].keys())
        var: sc.Variable
        var = file["entry1/data/" + bank_name]["events"][()].rename_dims(
            {'dim_0': 'event'}
        )

        weights = _copy_partial_var(var, idx=0, unit='counts')  # p
        id_list = _copy_partial_var(var, idx=4, dtype='int64')  # id
        t_list = _copy_partial_var(var, idx=5, unit='s')  # t

        weights = (probability / weights.max()) * weights

        loaded = sc.DataArray(data=weights, coords={'t': t_list, 'id': id_list})
        grouped = loaded.group(_get_mcstas_pixel_ids(detector_descs))
        da = grouped.fold(dim='id', sizes={'panel': len(detector_descs), 'id': -1})
        da.coords['fast_axis'] = sc.concat(fast_axes, 'panel')
        da.coords['slow_axis'] = sc.concat(slow_axes, 'panel')
        da.coords['origin_position'] = sc.concat(origins, 'panel')
        da.coords['position'] = _get_mcstas_pixel_positions(
            detector_descs, sample_position
        )
        da.coords['sample_position'] = sample_position - sample_position
        da.coords['source_position'] = _retrieve_3d_position(source) - sample_position

        return NMXData(da)
