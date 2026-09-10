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
    SolidAngle,
    WavelengthBins,
    WavelengthDetector,
)
from ess.skadi import SkadiMcStasWorkflow, load_skadi_mcstas
from ess.skadi.mcstas import flat_monitor_term
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
    pixels_per_side: int = 8,
) -> Path:
    with h5py.File(path, 'w') as file:
        entry = file.create_group('entry1')
        data = entry.create_group('data')
        detector = data.create_group('detector_events')
        detector.attrs['component'] = 'detector_0'
        detector.attrs['variables'] = 'p x y n id t '
        detector.attrs['options'] = (
            f'mantid square x limits=[-0.024,0.024] bins={pixels_per_side} '
            f'y limits=[-0.024,0.024] bins={pixels_per_side}, neutron pixel min=0 t, '
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


def test_mcstas_loader_groups_weighted_events_across_tiles(tmp_path: Path) -> None:
    filename = _small_mcstas_file(tmp_path / 'mccode.h5')
    with h5py.File(filename, 'r+') as file:
        data = file['entry1/data']
        # Group names deliberately sort in the opposite order to pixel IDs.
        for name, start, count in [('b_second', 64, 3), ('a_empty', 128, 0)]:
            file.copy(data['detector_events'], data, name=name)
            group = data[name]
            group.attrs['options'] = group.attrs['options'].replace(
                'pixel min=0', f'pixel min={start}'
            )
            group.attrs['component'] = f'detector_{start}'
            _component(
                file['entry1/instrument/components'],
                f'0004_detector_{start}',
                [float(start), 0.0, 12.0],
            )
            events = group['events'][()]
            events[:, 4] += start
            # Exercise column metadata rather than assuming a fixed column order.
            group.attrs['variables'] = np.bytes_('t id p x y n')
            del group['events']
            group.create_dataset('events', data=events[:count, [5, 4, 0, 1, 2, 3]])

    detector = load_skadi_mcstas(filename)
    events_per_pixel = detector.bins.size()

    assert detector.sizes == {'detector_number': 192}
    for start in (0, 64):
        assert events_per_pixel['detector_number', start].value == 1
        assert events_per_pixel['detector_number', start + 1].value == 2
        assert (
            events_per_pixel['detector_number', start + 2 : start + 64].sum().value == 0
        )
        events = detector['detector_number', start + 1].value
        assert_identical(
            events.data,
            sc.array(dims=['event'], values=[2.0, 3.0], variances=[4.0, 9.0]),
        )
        assert_identical(
            events.coords['event_time_offset'],
            sc.array(dims=['event'], values=[0.014, 0.016], unit='s'),
        )
    assert events_per_pixel['detector_number', 128:].sum().value == 0
    for start in (64, 128):
        assert_allclose(
            detector.coords['position'][start] - detector.coords['position'][0],
            sc.vector([float(start), 0.0, 0.0], unit='m'),
        )


@pytest.mark.parametrize('pixel_id', [-1.0, -0.5, 1.9, 64.0, np.nan, np.inf])
def test_mcstas_loader_rejects_invalid_pixel_ids(
    tmp_path: Path, pixel_id: float
) -> None:
    filename = _small_mcstas_file(tmp_path / 'mccode.h5')
    with h5py.File(filename, 'r+') as file:
        file['entry1/data/detector_events/events'][0, 4] = pixel_id

    with pytest.raises(ValueError, match='Invalid pixel ID'):
        load_skadi_mcstas(filename)


@pytest.mark.parametrize('use_directory', [False, True])
def test_mcstas_positions_do_not_require_detector_data(
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


@pytest.mark.parametrize(
    ('pixels_per_side', 'pixel_width', 'corner'),
    [(8, 0.006, 0.021125), (16, 0.003, 0.022625)],
)
def test_mcstas_loader_corrects_and_rotates_pixel_geometry(
    tmp_path: Path, pixels_per_side: int, pixel_width: float, corner: float
) -> None:
    # McStas stores matrices for multiplication from the left by row vectors.
    rotation = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
    detector = load_skadi_mcstas(
        _small_mcstas_file(
            tmp_path / 'mccode.h5',
            detector_rotation=rotation,
            pixels_per_side=pixels_per_side,
        )
    )

    assert_allclose(
        detector.coords['position'][0],
        sc.vector([0.0, -corner, 12.0 - corner], unit='m'),
    )
    assert_allclose(
        detector.coords['position'][-1],
        sc.vector([0.0, corner, 12.0 + corner], unit='m'),
    )
    assert_allclose(
        detector.coords['position'][1] - detector.coords['position'][0],
        sc.vector([0.0, 0.0, pixel_width + 0.000125], unit='m'),
    )
    assert_allclose(
        detector.coords['position'][pixels_per_side] - detector.coords['position'][0],
        sc.vector([0.0, pixel_width + 0.000125, 0.0], unit='m'),
    )
    assert_identical(detector.coords['detector_normal'][0], sc.vector([-1.0, 0.0, 0.0]))
    for index in (0, pixels_per_side**2 - 1):
        assert_allclose(
            detector.coords['pixel_size'][index],
            sc.vector([pixel_width + 0.00025, pixel_width + 0.00025, 0.001], unit='m'),
        )
    assert_allclose(
        detector.coords['pixel_size'][pixels_per_side + 1],
        sc.vector([pixel_width, pixel_width, 0.001], unit='m'),
    )


@pytest.mark.parametrize(
    'edges',
    [np.linspace(2.0, 8.0, 16), np.linspace(2.0, 8.0, 31), [2.0, 3.0, 5.0, 8.0]],
    ids=['coarse', 'fine', 'uneven'],
)
def test_mcstas_intensity_normalization_is_independent_of_wavelength_binning(
    tmp_path: Path, edges: list[float] | np.ndarray
) -> None:
    filename = _small_mcstas_file(tmp_path / 'mccode.h5')
    workflow = SkadiMcStasWorkflow()
    workflow[Filename[SampleRun]] = filename
    workflow[Position[snx.NXsource, SampleRun]] = sc.vector([0.0, 0.0, 0.0], unit='m')
    workflow[WavelengthBins] = sc.array(
        dims=['wavelength'], values=edges, unit='angstrom'
    )
    workflow[QBins] = sc.linspace('Q', 0.0, 1.0, 2, unit='1/angstrom')

    intensity = workflow.compute(IntensityQ[SampleRun])

    # All events and pixels contribute to one Q bin over a six-angstrom band.
    solid_angle = workflow.compute(SolidAngle[SampleRun]).sum().data
    expected = sc.scalar(6.0, variance=14.0) / (6.0 * solid_angle)
    assert intensity.sizes == {'Q': 1}
    assert_allclose(intensity.data[0], expected)


def test_flat_monitor_term_accounts_for_wavelength_units() -> None:
    edges = sc.array(dims=['wavelength'], values=[0.2, 0.3, 0.5, 0.8], unit='nm')

    monitor = flat_monitor_term(edges)

    assert_allclose(
        monitor,
        sc.DataArray(
            sc.array(dims=['wavelength'], values=[1.0, 2.0, 3.0]),
            coords={
                'wavelength': sc.array(
                    dims=['wavelength'], values=[0.25, 0.4, 0.65], unit='nm'
                )
            },
        ),
    )


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
