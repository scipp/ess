# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
# Mantid IDF related functions.
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from types import MappingProxyType
from typing import Protocol

import h5py
import scipp as sc
from defusedxml.ElementTree import fromstring

from ess.nmx.rotation import axis_angle_to_quaternion, quaternion_to_matrix
from ess.reduce.nexus.types import FilePath

_AXISNAME_TO_UNIT_VECTOR = MappingProxyType(
    {
        'x': sc.vector([1.0, 0.0, 0.0]),
        'y': sc.vector([0.0, 1.0, 0.0]),
        'z': sc.vector([0.0, 0.0, 1.0]),
    }
)


class _XML(Protocol):
    """XML element or tree type.

    Temporarily used for type hinting.
    Builtin XML type is blocked by bandit security check."""

    tag: str
    attrib: dict[str, str]

    def find(self, name: str) -> '_XML | None': ...

    def findall(self, tag: str) -> 'Iterable[_XML]': ...

    def get(self, name: str, default: str | None = None) -> str: ...

    def __iter__(self) -> '_XML': ...

    def __next__(self) -> '_XML': ...


@dataclass
class DetectorDesc:
    """Detector information extracted from McStas instrument xml description."""

    # Name defined in the location.
    name: str
    id_start: int  # 'idstart'
    fast_axis_name: str  # 'idfillbyfirst'
    # From <type name="MonNDtype-n" ...>
    num_x: int  # 'xpixels'
    num_y: int  # 'ypixels'
    step_x: sc.Variable  # 'xstep'
    step_y: sc.Variable  # 'ystep'
    start_x: float  # 'xstart'
    start_y: float  # 'ystart'
    # From <location> under <component type="MonNDtype-n" ...>
    position: sc.Variable  # <location> 'x', 'y', 'z'
    # Calculated fields
    rotation_matrix: sc.Variable
    slow_axis_name: str
    fast_axis: sc.Variable
    slow_axis: sc.Variable

    @property
    def total_pixels(self) -> int:
        return self.num_x * self.num_y

    @property
    def slow_step(self) -> sc.Variable:
        return self.step_y if self.fast_axis_name == 'x' else self.step_x

    @property
    def fast_step(self) -> sc.Variable:
        return self.step_x if self.fast_axis_name == 'x' else self.step_y

    @property
    def num_fast_pixels_per_row(self) -> int:
        """Number of pixels in each row of the detector along the fast axis."""
        return self.num_x if self.fast_axis_name == 'x' else self.num_y

    @property
    def detector_shape(self) -> tuple:
        """Shape of the detector panel. (num_x, num_y)"""
        return (self.num_x, self.num_y)

    @property
    def pixel_ids(self) -> sc.Variable:
        start, stop = (
            self.id_start,
            self.id_start + self.total_pixels,
        )
        return sc.arange('event_id', start, stop, unit=None)

    @property
    def pixel_positions(self) -> sc.Variable:
        # Assuming sample is always at 0,0,0
        pixel_idx = sc.arange('event_id', self.total_pixels)
        n_col = sc.scalar(self.num_fast_pixels_per_row)

        pixel_n_slow = pixel_idx // n_col
        pixel_n_fast = pixel_idx % n_col

        fast_axis_steps = self.fast_axis * self.fast_step
        slow_axis_steps = self.slow_axis * self.slow_step

        return self.position + (
            (pixel_n_slow * slow_axis_steps)
            + (pixel_n_fast * fast_axis_steps)
            + self.rotation_matrix
            * sc.vector(
                [self.start_x, self.start_y, 0.0], unit=self.position.unit
            )  # Detector pixel offset should also be rotated first.
        )

    def fold(self, da: sc.DataArray) -> sc.DataArray:
        sizes = {'x': self.num_x, 'y': self.num_y}
        axis_names = (self.fast_axis_name, self.slow_axis_name)
        sizes = {f'{i}_pixel_offset': sizes[i] for i in axis_names}
        return da.fold(dim='event_id', sizes=sizes)


@dataclass
class SampleDesc:
    """Sample description extracted from McStas instrument xml description."""

    name: str
    position: sc.Variable

    def position_from_sample(self, other: sc.Variable) -> sc.Variable:
        """Position of ``other`` relative to the sample.

        All positions and distance are stored relative to the sample position.

        Parameters
        ----------
        other:
            Position of the other object in 3D vector.

        """
        return other - self.position


@dataclass
class SourceDesc:
    """Source description extracted from IDF."""

    # From <type is="Source" name="...">
    name: str
    # From <location> under <component type="Source" ...>
    position: sc.Variable


@dataclass
class MonitorDesc:
    """Monitor description extracted from IDF."""

    name: str
    position: sc.Variable


@dataclass
class MandiInstrument:
    detectors: tuple[DetectorDesc, ...]
    monitors: tuple[MonitorDesc, ...]
    source: SourceDesc
    sample: SampleDesc


@dataclass(frozen=True)
class ReferenceFrame:
    along_beam_axis: str
    pointing_up_axis: str
    handedness: str


@dataclass(frozen=True)
class DefaultSettings:
    length_unit: str
    angle_unit: str
    reference_frame: ReferenceFrame
    default_view: str


def _retrieve_default_settings(tree: _XML, find: Callable) -> DefaultSettings:
    default_settings_xml = find(tree, "defaults")
    reference_frame_xml = find(default_settings_xml, "reference-frame")
    reference_frame = ReferenceFrame(
        along_beam_axis=find(reference_frame_xml, "along-beam").get("axis"),
        pointing_up_axis=find(reference_frame_xml, "pointing-up").get("axis"),
        handedness=find(reference_frame_xml, "handedness").get("val"),
    )
    default_settings = DefaultSettings(
        length_unit=find(default_settings_xml, "length").get("unit"),
        angle_unit=find(default_settings_xml, "angle").get("unit"),
        default_view=find(default_settings_xml, "default-view").get("view"),
        reference_frame=reference_frame,
    )
    return default_settings


def _retrieve_location(tree: _XML, find: Callable, length_unit: str) -> sc.Variable:
    # Sometimes one component contains multiple locations
    location_xml = find(tree, "location") if not tree.tag.endswith("location") else tree
    xyz = [float(location_xml.get(i, 0)) for i in "xyz"]
    return sc.vector(xyz, unit=length_unit)


def _retrieve_source(
    *,
    all_types: list[_XML],
    all_components: list[_XML],
    find: Callable,
    length_unit: str,
) -> SourceDesc:
    # Use the first one assuming there is single source.
    source_type = next(
        type_xml for type_xml in all_types if type_xml.get('is') == "Source"
    )
    source_type_name = source_type.get("name") or ""
    source_component = next(
        comp for comp in all_components if comp.get("type") == source_type_name
    )
    location = _retrieve_location(source_component, find=find, length_unit=length_unit)

    return SourceDesc(name=source_type_name, position=location)


def _retrieve_sample(
    *,
    all_types: list[_XML],
    all_components: list[_XML],
    find: Callable,
    length_unit: str,
) -> SampleDesc:
    # Use the first one assuming there is single sample.
    sample_type = next(
        type_xml for type_xml in all_types if type_xml.get('is') == "SamplePos"
    )
    sample_type_name = sample_type.get("name") or ""
    sample_component = next(
        comp for comp in all_components if comp.get("type") == sample_type_name
    )
    location = _retrieve_location(sample_component, find=find, length_unit=length_unit)

    return SampleDesc(name=sample_type_name, position=location)


def _retrieve_monitors(
    *,
    all_types: list[_XML],
    find: Callable,
    findall: Callable,
    length_unit: str,
) -> list[MonitorDesc]:
    # All monitors are defined under single "monitors" type.
    monitors = next(
        type_xml for type_xml in all_types if type_xml.get("name") == "monitors"
    )
    monitor = find(monitors, "component")
    locations = findall(monitor, "location")
    return [
        MonitorDesc(
            name=loc.get("name"),
            position=_retrieve_location(loc, find=find, length_unit=length_unit),
        )
        for loc in locations
    ]


def _resolve_rotation_chain(
    loc_tree: _XML,
    *,
    find: Callable,
    angle_unit: str,
    cur_matrix: sc.Variable | None = None,
    handedness: str,
) -> sc.Variable:
    """Resolve nested rotation chain.

    In the IDF, rotations can be nested.
    MANDI does not have any translation/rotation chains
    so this helper only resolves nested rotations.

    Returns
    -------
    :
        Rotation matrix.

    """
    try:
        rot = find(loc_tree, "rot")
    except KeyError:
        return cur_matrix

    theta = sc.scalar(float(rot.get("val", 0)), unit=angle_unit)
    if handedness == "right":
        theta = -theta

    x, y, z, w = axis_angle_to_quaternion(
        x=float(rot.get("axis-x", 0)),
        y=float(rot.get("axis-y", 0)),
        z=float(rot.get("axis-z", 1)),
        theta=theta,
    )
    new_matrix = quaternion_to_matrix(x=x, y=y, z=z, w=w)

    if cur_matrix is not None:
        cur_matrix = cur_matrix * new_matrix
    else:
        cur_matrix = new_matrix

    return _resolve_rotation_chain(
        loc_tree=rot,
        find=find,
        angle_unit=angle_unit,
        cur_matrix=cur_matrix,
        handedness=handedness,
    )


def _retrieve_detectors(
    *,
    all_types: list[_XML],
    all_components: list[_XML],
    find: Callable,
    length_unit: str,
    angle_unit: str,
    handedness: str,
) -> list[DetectorDesc]:
    detector_types = {
        type_xml.get("name"): type_xml
        for type_xml in all_types
        if type_xml.get('is') == "rectangular_detector"
    }
    detector_type_names = set(detector_types.keys())
    detector_components = [
        comp for comp in all_components if comp.get("type") in detector_type_names
    ]
    detectors = []
    for comp in detector_components:
        type_def = detector_types[comp.get("type")]
        location_xml = find(comp, "location")
        fast_axis_name = comp.get("idfillbyfirst")
        slow_axis_name = 'x' if fast_axis_name == 'y' else 'y'
        step_x = sc.scalar(float(type_def.get("xstep")), unit=length_unit)
        step_y = sc.scalar(float(type_def.get("ystep")), unit=length_unit)
        start_x = float(type_def.get("xstart"))
        start_y = float(type_def.get("ystart"))

        position = _retrieve_location(location_xml, find=find, length_unit=length_unit)
        rotation_matrix = _resolve_rotation_chain(
            location_xml, find=find, angle_unit=angle_unit, handedness=handedness
        )

        cur_det = DetectorDesc(
            name=location_xml.get("name"),
            id_start=int(comp.get("idstart")),
            fast_axis_name=fast_axis_name,
            slow_axis_name=slow_axis_name,
            num_x=int(type_def.get("xpixels")),
            num_y=int(type_def.get("ypixels")),
            step_x=step_x,
            step_y=step_y,
            start_x=start_x,
            start_y=start_y,
            position=position,
            rotation_matrix=rotation_matrix,
            fast_axis=rotation_matrix * _AXISNAME_TO_UNIT_VECTOR[fast_axis_name],
            slow_axis=rotation_matrix * _AXISNAME_TO_UNIT_VECTOR[slow_axis_name],
        )
        detectors.append(cur_det)

    return detectors


def read_mandi_geometry_xml(file_path: FilePath) -> MandiInstrument:
    """Retrieve geometry parameters from Mandi file."""
    instrument_xml_path = 'entry/instrument/instrument_xml/data'
    with h5py.File(file_path) as file:
        tree = fromstring(file[instrument_xml_path][...][0])

    # Probably better way to retrieve the namespace...
    namespace = tree.tag.removesuffix("instrument")

    def find(tree: _XML, tag) -> _XML:
        elem = tree.find(f"{namespace}{tag}")
        if elem is None:
            raise KeyError(f"{tag=} not found in {elem=}")
        return elem

    def findall(tree: _XML, tag) -> Iterable[_XML]:
        return tree.findall(f"{namespace}{tag}")

    default_settings = _retrieve_default_settings(tree, find)
    all_types = list(findall(tree, "type"))
    all_components = list(findall(tree, "component"))

    source = _retrieve_source(
        all_types=all_types,
        all_components=all_components,
        find=find,
        length_unit=default_settings.length_unit,
    )
    sample = _retrieve_sample(
        all_types=all_types,
        all_components=all_components,
        find=find,
        length_unit=default_settings.length_unit,
    )
    monitors = _retrieve_monitors(
        all_types=all_types,
        find=find,
        findall=findall,
        length_unit=default_settings.length_unit,
    )
    detectors = _retrieve_detectors(
        all_types=all_types,
        all_components=all_components,
        find=find,
        length_unit=default_settings.length_unit,
        angle_unit=default_settings.angle_unit,
        handedness=default_settings.reference_frame.handedness,
    )

    return MandiInstrument(
        detectors=tuple(detectors),
        monitors=tuple(monitors),
        sample=sample,
        source=source,
    )
