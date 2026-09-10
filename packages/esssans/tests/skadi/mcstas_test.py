# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)

from pathlib import Path

import h5py
import numpy as np
import pytest
import scipp as sc
import scippnexus as snx
from ess.sans.conversions import ElasticCoordTransformGraph
from ess.sans.types import (
    DetectorMasks,
    Filename,
    IntensityQ,
    Position,
    QBins,
    RawDetector,
    SampleRun,
    WavelengthBins,
    WavelengthDetector,
)
from ess.skadi import SkadiMcStasWorkflow, load_skadi_mcstas
from scipp.testing import assert_allclose, assert_identical


def _component(
    components: h5py.Group,
    name: str,
    position: list[float],
    rotation: np.ndarray | None = None,
) -> None:
    group = components.create_group(name)
    group.create_dataset('Position', data=position)
    group.create_dataset('Rotation', data=np.eye(3) if rotation is None else rotation)


def _small_mcstas_file(
    path: Path,
    *,
    detector_rotation: np.ndarray | None = None,
    sample_rotation: np.ndarray | None = None,
) -> Path:
    with h5py.File(path, 'w') as file:
        entry = file.create_group('entry1')
        data = entry.create_group('data')
        detector = data.create_group('detector_events')
        detector.attrs['component'] = 'detector_0'
        detector.attrs['variables'] = 'p x y n id t '
        detector.attrs['options'] = (
            'mantid square x limits=[-0.024,0.024] bins=8 '
            'y limits=[-0.024,0.024] bins=8, neutron pixel min=0 t, '
            'list all neutrons'
        )
        detector.create_dataset(
            'events',
            data=np.array(
                [
                    [2.0, 0.0, 0.0, 1.0, 1.0, 0.014],
                    [1.0, 0.0, 0.0, 2.0, 0.0, 0.012],
                    [3.0, 0.0, 0.0, 3.0, 1.0, 0.016],
                ]
            ),
        )

        instrument = entry.create_group('instrument')
        components = instrument.create_group('components')
        _component(components, '0001_sourceESS', [0.0, 0.0, 0.0])
        _component(
            components,
            '0002_sample_position',
            [0.0, 0.0, 10.0],
            rotation=sample_rotation,
        )
        _component(
            components,
            '0003_detector_0',
            [0.0, 0.0, 12.0],
            rotation=detector_rotation,
        )
    return path


def test_mcstas_loader_groups_events_by_pixel_id(tmp_path: Path) -> None:
    detector = load_skadi_mcstas(_small_mcstas_file(tmp_path / 'mccode.h5'))
    events_per_pixel = detector.bins.size()

    assert events_per_pixel['detector_number', 0].value == 1
    assert events_per_pixel['detector_number', 1].value == 2
    assert events_per_pixel['detector_number', 2:].sum().value == 0


def test_mcstas_loader_does_not_attach_source_or_sample_position(
    tmp_path: Path,
) -> None:
    detector = load_skadi_mcstas(_small_mcstas_file(tmp_path / 'mccode.h5'))

    assert 'source_position' not in detector.coords
    assert 'sample_position' not in detector.coords


@pytest.mark.parametrize('use_directory', [False, True])
def test_mcstas_positions_do_not_require_detector_data_S4(
    tmp_path: Path, use_directory: bool
) -> None:
    # The sample frame is rotated so the outgoing collimation axis is global -x.
    rotation = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
    filename = _small_mcstas_file(tmp_path / 'mccode.h5', sample_rotation=rotation)
    with h5py.File(filename, 'r+') as file:
        del file['entry1/data']
    workflow = SkadiMcStasWorkflow()
    workflow[Filename[SampleRun]] = tmp_path if use_directory else filename

    sample = workflow.compute(Position[snx.NXsample, SampleRun])
    source = workflow.compute(Position[snx.NXsource, SampleRun])

    assert_identical(sample, sc.vector([0.0, 0.0, 10.0], unit='m'))
    # Preserve the instrument's 38.42 m flight path, not the 10 m straight-line
    # separation between moderator and sample in this synthetic file.
    assert_allclose(source, sc.vector([38.42, 0.0, 10.0], unit='m'))


def test_mcstas_effective_source_gives_zero_q_along_incident_beam(
    tmp_path: Path,
) -> None:
    rotation = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
    filename = _small_mcstas_file(tmp_path / 'mccode.h5', sample_rotation=rotation)
    workflow = SkadiMcStasWorkflow()
    workflow[Filename[SampleRun]] = filename
    graph = workflow.compute(ElasticCoordTransformGraph[SampleRun])
    detector = sc.DataArray(
        sc.scalar(1.0),
        coords={
            'position': sc.vector([-2.0, 0.0, 10.0], unit='m'),
            'tof': sc.scalar(0.012, unit='s'),
        },
    )

    converted = detector.transform_coords(['Q', 'L1'], graph=graph)

    assert_allclose(converted.coords['L1'].to(unit='m'), sc.scalar(38.42, unit='m'))
    assert_allclose(
        converted.coords['Q'],
        sc.scalar(0.0, unit='1/angstrom'),
        atol=sc.scalar(1e-14, unit='1/angstrom'),
    )


@pytest.mark.parametrize(
    ('component', 'position'),
    [(snx.NXsource, [0.0, 0.0, 5.0]), (snx.NXsample, [1.0, 0.0, 10.0])],
)
def test_mcstas_workflow_uses_position_overrides_for_wavelength(
    tmp_path: Path, component: type, position: list[float]
) -> None:
    filename = _small_mcstas_file(tmp_path / 'mccode.h5')
    workflow = SkadiMcStasWorkflow()
    workflow[Filename[SampleRun]] = filename
    source = sc.vector([0.0, 0.0, 0.0], unit='m')
    sample = sc.vector([0.0, 0.0, 10.0], unit='m')
    workflow[Position[snx.NXsource, SampleRun]] = source
    workflow[Position[snx.NXsample, SampleRun]] = sample
    original = workflow.compute(WavelengthDetector[SampleRun])
    updated = sc.vector(position, unit='m')
    workflow[Position[component, SampleRun]] = updated
    if component is snx.NXsource:
        source = updated
    else:
        sample = updated

    detector = workflow.compute(WavelengthDetector[SampleRun])

    distance = sc.norm(sample - source) + sc.norm(
        detector.coords['position'][0] - sample
    )
    expected = (
        sc.constants.h / sc.constants.m_n * sc.scalar(0.012, unit='s') / distance
    ).to(unit='angstrom')
    assert_allclose(
        detector.bins.constituents['data'].coords['wavelength'][0], expected
    )
    assert not sc.allclose(
        detector.bins.constituents['data'].coords['wavelength'],
        original.bins.constituents['data'].coords['wavelength'],
    )


def test_mcstas_loader_uses_mcstas_rotation_convention(tmp_path: Path) -> None:
    # McStas stores matrices for multiplication from the left by row vectors.
    rotation = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
    unrotated = load_skadi_mcstas(_small_mcstas_file(tmp_path / 'unrotated.h5'))
    rotated = load_skadi_mcstas(
        _small_mcstas_file(tmp_path / 'rotated.h5', detector_rotation=rotation)
    )
    transform = sc.spatial.linear_transform(value=rotation.T)

    assert_allclose(
        rotated.coords['position'][1] - rotated.coords['position'][0],
        transform * (unrotated.coords['position'][1] - unrotated.coords['position'][0]),
    )
    assert_allclose(
        rotated.coords['detector_normal'][0],
        transform * unrotated.coords['detector_normal'][0],
    )


def test_mcstas_workflow_converts_event_time_to_wavelength(tmp_path: Path) -> None:
    filename = _small_mcstas_file(tmp_path / 'mccode.h5')
    workflow = SkadiMcStasWorkflow()
    workflow[Filename[SampleRun]] = filename
    workflow[Position[snx.NXsource, SampleRun]] = sc.vector([0.0, 0.0, 0.0], unit='m')

    detector = workflow.compute(WavelengthDetector[SampleRun])
    events = detector.bins.constituents['data']
    source_to_sample = sc.scalar(10.0, unit='m')
    sample_to_pixel = sc.norm(
        detector.coords['position'][0] - sc.vector([0.0, 0.0, 10.0], unit='m')
    )
    expected = (
        sc.constants.h
        / sc.constants.m_n
        * sc.scalar(0.012, unit='s')
        / (source_to_sample + sample_to_pixel)
    ).to(unit='angstrom')

    assert sc.allclose(events.coords['wavelength'][0], expected)


def test_mcstas_workflow_computes_intensity_q(tmp_path: Path) -> None:
    filename = _small_mcstas_file(tmp_path / 'mccode.h5')
    workflow = SkadiMcStasWorkflow()
    workflow[Filename[SampleRun]] = filename
    workflow[Position[snx.NXsource, SampleRun]] = sc.vector([0.0, 0.0, 0.0], unit='m')
    workflow[WavelengthBins] = sc.linspace(
        'wavelength', start=2.0, stop=8.0, num=31, unit='angstrom'
    )
    workflow[QBins] = sc.linspace('Q', start=0.0, stop=0.1, num=51, unit='1/angstrom')

    intensity = workflow.compute(IntensityQ[SampleRun])

    assert intensity.dims == ('Q',)
    assert intensity.sizes == {'Q': 50}
    assert sc.isfinite(intensity.data).any().value


def test_mcstas_reduction_preserves_masks_attached_to_raw_detector(
    tmp_path: Path,
) -> None:
    filename = _small_mcstas_file(tmp_path / 'mccode.h5')
    workflow = SkadiMcStasWorkflow()
    workflow[Filename[SampleRun]] = filename
    workflow[Position[snx.NXsource, SampleRun]] = sc.vector([0.0, 0.0, 0.0], unit='m')
    workflow[WavelengthBins] = sc.linspace('wavelength', 2.0, 8.0, 31, unit='angstrom')
    workflow[QBins] = sc.linspace('Q', 0.0, 1.0, 2, unit='1/angstrom')
    raw = workflow.compute(RawDetector[SampleRun])
    mask = raw.coords['detector_number'] > sc.scalar(0, unit=None)
    workflow[DetectorMasks] = {'pixel_mask': mask}
    expected = workflow.compute(IntensityQ[SampleRun])
    workflow[DetectorMasks] = {}
    workflow[RawDetector[SampleRun]] = raw.assign_masks(pixel_mask=mask)

    actual = workflow.compute(IntensityQ[SampleRun])

    assert sc.isfinite(expected.data).all().value
    assert (expected.data > sc.scalar(0.0)).all().value
    assert_allclose(actual, expected)
