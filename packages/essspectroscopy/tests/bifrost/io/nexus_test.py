# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import h5py
import numpy as np
import pytest
import scipp as sc
import scippnexus as snx
from ess.bifrost.io.nexus import extract_instrument_angle, load_analyzer_for_detector
from ess.spectroscopy.types import ElasticMonitor, SampleRun

from ess.reduce.nexus.types import NeXusComponentLocationSpec, NeXusTransformationChain


@pytest.fixture
def path(tmp_path: Path) -> Path:
    return tmp_path / 'bifrost.nxs'


def _make_group(parent: h5py.Group, name: str, nx_class: str) -> h5py.Group:
    group = parent.create_group(name)
    group.attrs['NX_class'] = nx_class
    return group


@contextmanager
def _write_instrument(path: Path) -> Iterator[h5py.Group]:
    with h5py.File(path, 'w') as f:
        entry = _make_group(f, 'entry', 'NXentry')
        yield _make_group(entry, 'instrument', 'NXinstrument')


def _add_crystal(instrument: h5py.Group, name: str, d_spacing: float) -> None:
    crystal = _make_group(instrument, name, 'NXcrystal')
    crystal.create_dataset('d_spacing', data=d_spacing).attrs['units'] = 'angstrom'


def _load_analyzer_d_spacing(path: Path) -> sc.Variable:
    analyzer = load_analyzer_for_detector(
        NeXusComponentLocationSpec[snx.NXdetector, SampleRun](
            filename=path, component_name='triplet'
        )
    )
    return analyzer['d_spacing']


def test_load_analyzer_inputs_as_scalar_dataset(path: Path) -> None:
    with _write_instrument(path) as instrument:
        _add_crystal(instrument, 'other_crystal', 1.0)
        _add_crystal(instrument, 'analyzer', 3.355)
        detector = _make_group(instrument, 'triplet', 'NXdetector')
        detector['inputs'] = '/entry/instrument/analyzer'

    assert sc.identical(
        _load_analyzer_d_spacing(path), sc.scalar(3.355, unit='angstrom')
    )


def test_load_analyzer_inputs_as_1d_dataset(path: Path) -> None:
    with _write_instrument(path) as instrument:
        _make_group(instrument, 'collimator', 'NXcollimator')
        _add_crystal(instrument, 'analyzer', 3.355)
        detector = _make_group(instrument, 'triplet', 'NXdetector')
        detector['inputs'] = np.array(['collimator', 'analyzer'], dtype=object)

    assert sc.identical(
        _load_analyzer_d_spacing(path), sc.scalar(3.355, unit='angstrom')
    )


def test_load_analyzer_inputs_as_attribute_through_intermediate_group(
    path: Path,
) -> None:
    with _write_instrument(path) as instrument:
        _add_crystal(instrument, 'analyzer', 3.355)
        angle = _make_group(instrument, 'detector_angle', 'NXcollection')
        angle.attrs['inputs'] = 'analyzer'
        detector = _make_group(instrument, 'triplet', 'NXdetector')
        detector.attrs['inputs'] = 'detector_angle'

    assert sc.identical(
        _load_analyzer_d_spacing(path), sc.scalar(3.355, unit='angstrom')
    )


def _add_transformation(
    component: h5py.Group,
    name: str,
    *,
    transformation_type: str,
    value: float,
    depends_on: str,
    time_dependent: bool = False,
) -> None:
    transformations = component.require_group('transformations')
    transformations.attrs['NX_class'] = 'NXtransformations'
    if time_dependent:
        transformation = _make_group(transformations, name, 'NXlog')
        transformation.create_dataset('value', data=[value]).attrs['units'] = 'deg'
        transformation.create_dataset('time', data=[0]).attrs['units'] = 'ns'
    else:
        transformation = transformations.create_dataset(name, data=value)
        transformation.attrs['units'] = (
            'deg' if transformation_type == 'rotation' else 'm'
        )
    transformation.attrs['transformation_type'] = transformation_type
    transformation.attrs['vector'] = [0.0, 1.0, 0.0]
    transformation.attrs['depends_on'] = depends_on


def _add_elastic_monitor(instrument: h5py.Group, chain_continues_at: str) -> None:
    """Add a monitor whose own transformations continue at ``chain_continues_at``."""
    monitor = _make_group(instrument, 'elastic_monitor', 'NXmonitor')
    monitor['data'] = [0, 0]
    monitor['depends_on'] = 'transformations/elastic_monitor_r0'
    _add_transformation(
        monitor,
        'elastic_monitor_r0',
        transformation_type='rotation',
        value=90.0,
        depends_on='elastic_monitor_t0_z',
    )
    _add_transformation(
        monitor,
        'elastic_monitor_t0_z',
        transformation_type='translation',
        value=0.1,
        depends_on='elastic_monitor_t0_x',
    )
    _add_transformation(
        monitor,
        'elastic_monitor_t0_x',
        transformation_type='translation',
        value=0.2,
        depends_on=chain_continues_at,
    )


def _load_elastic_monitor_chain(
    path: Path,
) -> NeXusTransformationChain[ElasticMonitor, SampleRun]:
    with snx.File(path) as f:
        monitor = f['entry/instrument/elastic_monitor'][()]
    return NeXusTransformationChain[ElasticMonitor, SampleRun](monitor['depends_on'])


@pytest.mark.parametrize(
    ('tank', 'tank_angle', 'sample', 'sample_transformation'),
    [
        pytest.param(
            'detector_tank_angle',
            'detector_tank_angle_r0',
            'sample_origin',
            'sample_origin_t0',
            id='detector_tank_angle',
        ),
        pytest.param(
            'detector_tank', 'tank_angle', 'sample_position', 'beam_axis', id='tank'
        ),
    ],
)
@pytest.mark.parametrize('time_dependent', [True, False])
def test_extract_instrument_angle_finds_first_transformation_outside_monitor(
    path: Path,
    tank: str,
    tank_angle: str,
    sample: str,
    sample_transformation: str,
    time_dependent: bool,
) -> None:
    with _write_instrument(path) as instrument:
        _add_elastic_monitor(
            instrument,
            chain_continues_at=f'/entry/instrument/{tank}/transformations/{tank_angle}',
        )
        _add_transformation(
            _make_group(instrument, tank, 'NXcollection'),
            tank_angle,
            transformation_type='rotation',
            value=31.0,
            depends_on=f'/entry/instrument/{sample}/transformations/{sample_transformation}',
            time_dependent=time_dependent,
        )
        _add_transformation(
            _make_group(instrument, sample, 'NXcollection'),
            sample_transformation,
            transformation_type='rotation',
            value=-90.0,
            depends_on='.',
        )

    angle = extract_instrument_angle(_load_elastic_monitor_chain(path))

    if time_dependent:
        angle = angle.data['time', 0]
    assert sc.identical(angle, sc.scalar(31.0, unit='deg'))


def test_extract_instrument_angle_requires_transformation_outside_monitor(
    path: Path,
) -> None:
    with _write_instrument(path) as instrument:
        _add_elastic_monitor(instrument, chain_continues_at='.')

    with pytest.raises(ValueError, match='Could not find the detector tank angle'):
        extract_instrument_angle(_load_elastic_monitor_chain(path))


def test_extract_instrument_angle_requires_rotation(path: Path) -> None:
    with _write_instrument(path) as instrument:
        _add_elastic_monitor(
            instrument,
            chain_continues_at='/entry/instrument/tank/transformations/offset',
        )
        _add_transformation(
            _make_group(instrument, 'tank', 'NXcollection'),
            'offset',
            transformation_type='translation',
            value=1.0,
            depends_on='.',
        )

    with pytest.raises(ValueError, match='to be a rotation'):
        extract_instrument_angle(_load_elastic_monitor_chain(path))
