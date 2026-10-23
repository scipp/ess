# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)

import h5py
import numpy as np
import pytest
import scipp as sc
from ess.magic import MagicMcStasWorkflow
from ess.magic.mcstas import mcstas_detector_geometry
from ess.magic.types import DetectorBank, EmptyDetector, Filename, SampleRun
from scipp.testing import assert_identical

# calc_xyz_by_id from magic-notebooks/mantid_worfkflow/dependencies/voxelization.py,
# commit da5844563df1a51442171c8178241879b52a06b0. Counter pairs share one plane.
# Landmarks: first pixel, next cassette, next anode, next cathode, last pixel.
_REFERENCE = {
    DetectorBank.detector_a: (
        [1, 3, 121, 15361, 491520],
        [
            [-0.0013098892101839482, 0.25157728161621096, 1.0081769977265218],
            [0.017165105059185757, 0.25157728161621096, 1.0080317126573515],
            [-0.0013098892101839482, 0.24389078234863285, 1.0081769977265218],
            [-0.003929667630551845, 0.2566290167236328, 1.0245309931795656],
            [1.2983717665556587, -1.066644226196289, 0.7853170846571818],
        ],
    ),
    DetectorBank.detector_b: (
        [491521, 491523, 491777, 495617, 622592],
        [
            [0.001438023971304267, 0.0482211181640625, 1.0081554392046324],
            [-0.015101321462914348, 0.0482211181640625, 1.0080433560124786],
            [0.001438023971304267, 0.0417916357421875, 1.0081554392046324],
            [0.0043140719139128005, 0.0490083544921875, 1.024466317613897],
            [-1.3635425231286782, -0.0726254443359375, -0.6637225184556543],
        ],
    ),
}


def _write_geometry(
    tmp_path, *, parameters=None, position=(0.0, 0.0, 0.0), rotation=None
):
    values = {
        'A_r1': 1.0,
        'A_section_length': 0.53,
        'A_casette_gamma': 1.05,
        'A_casette_omega': -9.101,
        'B_casette_omega': 10.0,
    } | (parameters or {})
    filename = tmp_path / 'geometry.h5'
    with h5py.File(filename, 'w') as file:
        file.create_group('entry1/data')
        simulation = file.create_group('entry1/simulation')
        simulation.attrs['program'] = np.bytes_('McStas 3.5.32')
        params = simulation.create_group('Param')
        for name, value in values.items():
            params[name] = [np.bytes_(str(value))]
        for index, bank in enumerate(('A', 'B'), start=1):
            component = file.create_group(
                f'entry1/instrument/components/{index:04d}_{bank}_abs_logger'
            )
            component['Position'] = position
            component['Rotation'] = np.eye(3) if rotation is None else rotation
            # No output or event datasets: geometry must be independent of events.
    return filename


@pytest.mark.parametrize('bank', list(DetectorBank))
def test_geometry_matches_mantid_voxel_centres_without_reading_events(tmp_path, bank):
    filename = _write_geometry(tmp_path)
    other = '0002_B' if bank is DetectorBank.detector_a else '0001_A'
    with h5py.File(filename, 'r+') as file:
        del file[f'entry1/instrument/components/{other}_abs_logger']
    detector = mcstas_detector_geometry(filename, bank)
    ids, expected = _REFERENCE[bank]
    first, last = ids[0], ids[-1]
    assert detector.sizes == {'pixel_id': last - first + 1}
    assert set(detector.coords) == {'pixel_id', 'position'}
    assert detector.coords['position'].unit == 'm'
    assert_identical(detector.data, detector.coords['pixel_id'])
    assert_identical(
        detector.coords['pixel_id'],
        sc.arange('pixel_id', first, last + 1, dtype='int32', unit=None),
    )
    positions = detector.coords['position'].values
    np.testing.assert_allclose(
        positions[np.array(ids) - first], expected, atol=1e-14, rtol=0
    )
    np.testing.assert_array_equal(positions[::2], positions[1::2])


@pytest.mark.parametrize('bank', list(DetectorBank))
def test_geometry_uses_recorded_component_placement(tmp_path, bank):
    filename = _write_geometry(
        tmp_path,
        position=[1.0, 2.0, 3.0],
        rotation=[[0, 0, -1], [0, 1, 0], [1, 0, 0]],
    )
    detector = mcstas_detector_geometry(filename, bank)
    ids, local = _REFERENCE[bank]
    x, y, z = np.array(local).T
    expected = np.column_stack((z + 1, y + 2, 3 - x))
    np.testing.assert_allclose(
        detector.coords['position'].values[np.array(ids) - ids[0]],
        expected,
        atol=1e-14,
        rtol=0,
    )


def test_geometry_uses_recorded_a_dimensions_and_spacing(tmp_path):
    filename = _write_geometry(
        tmp_path,
        parameters={
            'A_r1': 2.0,
            'A_section_length': 0.64,
            'A_casette_gamma': 1.0,
            'A_casette_omega': 0.0,
        },
    )
    # Accept numeric scalar parameters as well as the usual one-element strings.
    with h5py.File(filename, 'r+') as file:
        params = file['entry1/simulation/Param']
        del params['A_r1']
        params['A_r1'] = 2.0
    positions = (
        mcstas_detector_geometry(filename, DetectorBank.detector_a)
        .coords['position']
        .values
    )
    np.testing.assert_allclose(
        positions[[0, 2]][:, [0, 2]],
        [[0.0, 2.01], [0.03507933693893986, 2.0096938672643465]],
        atol=1e-14,
        rtol=0,
    )


@pytest.mark.parametrize('bank', list(DetectorBank))
def test_geometry_requires_recorded_cassette_tilt(tmp_path, bank):
    filename = _write_geometry(tmp_path)
    name = 'A_casette_omega' if bank is DetectorBank.detector_a else 'B_casette_omega'
    with h5py.File(filename, 'r+') as file:
        del file[f'entry1/simulation/Param/{name}']
    with pytest.raises(KeyError, match=name):
        mcstas_detector_geometry(filename, bank)


@pytest.mark.parametrize(
    ('name', 'value'),
    [
        ('A_r1', 0.0),
        ('A_section_length', -1.0),
        ('A_casette_gamma', 0.0),
        ('A_casette_omega', 90.0),
        ('A_casette_omega', np.nan),
        ('A_r1', np.inf),
    ],
)
def test_geometry_rejects_invalid_parameters(tmp_path, name, value):
    filename = _write_geometry(tmp_path, parameters={name: value})
    with pytest.raises(ValueError, match=f'{name}|Detector geometry'):
        mcstas_detector_geometry(filename, DetectorBank.detector_a)


@pytest.mark.parametrize(
    'placement',
    [{'position': [0.0, np.nan, 0.0]}, {'rotation': np.diag([-1.0, 1.0, 1.0])}],
)
def test_geometry_rejects_invalid_placement(tmp_path, placement):
    filename = _write_geometry(tmp_path, **placement)
    with pytest.raises(ValueError, match='finite position and proper rotation'):
        mcstas_detector_geometry(filename, DetectorBank.detector_a)


@pytest.mark.parametrize('bank', [None, DetectorBank.detector_b])
def test_workflow_computes_geometry_without_events(tmp_path, bank):
    filename = _write_geometry(tmp_path)
    workflow = MagicMcStasWorkflow()
    workflow[Filename[SampleRun]] = filename
    if bank is not None:
        workflow[DetectorBank] = bank
    actual = workflow.compute(EmptyDetector[SampleRun])
    expected = mcstas_detector_geometry(filename, bank or DetectorBank.detector_a)
    assert_identical(actual, expected)
