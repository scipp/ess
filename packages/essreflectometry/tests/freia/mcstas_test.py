# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)

import math

import h5py
import numpy as np
import pytest
import scipp as sc
from ess.reduce.nexus.types import (
    DiskChoppers,
    Filename,
    NeXusDetectorName,
    RawDetector,
)
from ess.reduce.unwrap import (
    DetectorLtotal,
    FrameUnwrapBackend,
    LookupTable,
    PulsePeriod,
    SourceBounds,
    WavelengthDetector,
)
from scipp.testing import assert_allclose, assert_identical
from scippneutron.chopper import DiskChopper
from scippnexus import NXdetector

from ess.freia import FreiaMcStasWorkflow
from ess.freia.mcstas import load_mcstas
from ess.reflectometry.types import ReferenceRun, SampleRun


def _component(components, name, position):
    group = components.create_group(f'{len(components):04d}_{name}')
    group['Position'] = position
    group['Rotation'] = np.eye(3)
    return group


@pytest.fixture
def mcstas_file(tmp_path):
    """Small on-disk McStas file read by the real mcstastox library."""
    filename = tmp_path / 'freia.h5'
    with h5py.File(filename, 'w') as f:
        entry = f.create_group('entry1')
        entry.create_group('data')
        simulation = entry.create_group('simulation')
        simulation.attrs['program'] = np.bytes_('3.7.18, git')
        simulation.create_group('Param')
        instrument = entry.create_group('instrument')
        components = instrument.create_group('components')
        _component(components, 'Source', [0.0, 0.0, 0.0])
        _component(components, 'Arm_Sample', [0.0, 0.0, 20.0])
        # A debug monitor must never be included in the detector event sum.
        debug = _component(components, 'Slit_event', [0.0, 0.0, 19.0])
        debug_output = debug.create_group('output/events')
        debug_output.attrs['variables'] = np.bytes_('p t id')
        debug_output['events'] = [[999.0, 0.001, 0.0]]
        detector = _component(components, 'Multiblade', [0.0, 0.0, 20.0])
        # Rotate the banana into the vertical scattering plane.
        detector['Rotation'][...] = [[0, 1, 0], [-1, 0, 0], [0, 0, 1]]
        geometry = detector.create_group('Geometry')
        geometry.attrs['Shape identifier'] = np.bytes_('4')
        geometry.attrs['radius'] = np.bytes_('3')
        bins = detector.create_group('output/BINS')
        for key, value in {
            'xvar': 'th',
            'yvar': 'y',
            'xlabel': 'theta',
            'ylabel': 'height',
        }.items():
            bins.attrs[key] = np.bytes_(value)
        bins['theta'] = [1.0, 2.0]
        bins['height'] = [-0.001, 0.001]
        # IDs need not be contiguous or sorted in geometry order.
        bins['pixels'] = [[12, 10], [99, 101]]
        output = detector.create_group('output/detector_events')
        output.attrs['variables'] = np.bytes_('p t id')
        output['events'] = [
            [2.0, 0.025, 10.0],
            [3.0, 0.025 + 1 / 14, 10.0],
            [0.0, 0.025, 99.0],
            [4.0, 0.025, 12.0],
        ]
    return filename


def test_loader_reads_only_final_detector_and_retains_empty_pixels(mcstas_file):
    detector = load_mcstas(mcstas_file)
    # Check pixel associations without requiring the loader to return a given order.
    image = sc.sort(detector.bins.sum(), 'pixel_id')
    np.testing.assert_array_equal(image.coords['pixel_id'].values, [10, 12, 99, 101])
    assert_identical(
        image.data,
        sc.array(
            dims=['pixel_id'],
            values=[5.0, 4.0, 0.0, 0.0],
            variances=[13.0, 16.0, 0.0, 0.0],
            unit='counts',
        ),
    )
    offsets = detector.bins.constituents['data'].coords['event_time_offset']
    assert_allclose(
        offsets.to(unit='s'),
        sc.full(sizes=offsets.sizes, value=0.025, unit='s'),
    )
    # Pixel 12 is the first banana pixel, rotated into the global frame.
    assert_allclose(
        image.coords['position'][1],
        sc.vector(
            [0.001, 3 * math.sin(math.pi / 180), 20 + 3 * math.cos(math.pi / 180)],
            unit='m',
        ),
    )


def test_loader_rejects_missing_detector_without_using_debug_events(mcstas_file):
    with h5py.File(mcstas_file, 'r+') as f:
        del f['entry1/instrument/components/0003_Multiblade']
    with pytest.raises(ValueError, match='No Mantid detector events'):
        load_mcstas(mcstas_file)


@pytest.mark.parametrize('pixel_id', [999, 10.5])
def test_loader_rejects_events_missing_from_pixel_map(mcstas_file, pixel_id):
    with h5py.File(mcstas_file, 'r+') as f:
        events = f[
            'entry1/instrument/components/0003_Multiblade/output/detector_events/events'
        ]
        events[0, 2] = pixel_id
    with pytest.raises(ValueError, match='pixel IDs absent from the pixel map'):
        load_mcstas(mcstas_file)


def test_lookup_table_uses_histogram_geometry_without_loading_events(mcstas_file):
    with h5py.File(mcstas_file, 'r+') as f:
        components = f['entry1/instrument/components']
        components.move('0003_Multiblade', '0003_Detector')
        detector = components['0003_Detector']
        detector['Position'][...] = [0.0, -0.25, 20.0]
        del detector['output']
        histogram = detector.create_group('output/histogram')
        histogram.attrs['xlabel'] = np.bytes_('theta [deg]')
        histogram.attrs['ylabel'] = np.bytes_('Height [cm]')
        histogram['theta__deg_'] = [0.0, 15.0]
        histogram['Height__cm_'] = [-25, 25]
        # No intensities or event arrays are needed anywhere in the file.
        del components['0002_Slit_event']

    workflow = FreiaMcStasWorkflow()
    workflow[Filename[SampleRun]] = str(mcstas_file)
    workflow[NeXusDetectorName] = 'Detector'
    results = workflow.compute(
        (DetectorLtotal[SampleRun], LookupTable[SampleRun, NXdetector])
    )
    # The detector's vertical offset shortens the distance at larger angles.
    expected = [
        20 + math.sqrt(9.125 - 1.5 * math.sin(angle)) for angle in (0, math.pi / 12)
    ]
    assert_allclose(
        results[DetectorLtotal[SampleRun]],
        sc.array(dims=['pixel_id'], values=expected * 2, unit='m'),
    )
    table = results[LookupTable[SampleRun, NXdetector]].array
    assert sc.isfinite(table.data).any().value
    # The histogram geometry must not make it possible to load fake events.
    with pytest.raises(ValueError, match='No Mantid detector events'):
        workflow.compute(RawDetector[SampleRun])


@pytest.mark.parametrize('run', [SampleRun, ReferenceRun])
def test_workflow_loads_and_unwraps_detector_with_generic_providers(mcstas_file, run):
    workflow = FreiaMcStasWorkflow()
    workflow[Filename[run]] = str(mcstas_file)
    workflow[FrameUnwrapBackend] = FrameUnwrapBackend.scipy
    # Override WFM with a single disk for an independently calculable wavelength.
    workflow[DiskChoppers[run]] = {
        'test_chopper': DiskChopper(
            frequency=sc.scalar(14.0, unit='Hz'),
            beam_position=sc.scalar(0.0, unit='deg'),
            phase=sc.scalar(0.0, unit='deg'),
            axle_position=sc.vector([0.0, 0.0, 5.0], unit='m'),
            slit_begin=sc.array(dims=['cutout'], values=[279.68], unit='deg'),
            slit_end=sc.array(dims=['cutout'], values=[359.68], unit='deg'),
        ),
    }
    workflow[SourceBounds] = SourceBounds(
        time=(sc.scalar(0.9, unit='ms'), sc.scalar(1.1, unit='ms')),
        wavelength=(sc.scalar(0.5, unit='angstrom'), sc.scalar(12.0, unit='angstrom')),
    )
    result = workflow.compute((RawDetector[run], WavelengthDetector[run]))
    raw = result[RawDetector[run]]
    unwrapped = result[WavelengthDetector[run]]
    assert_allclose(raw.bins.sum().data, unwrapped.bins.sum().data)
    wavelength = unwrapped.bins.constituents['data'].coords['wavelength']
    # Arrival time 25 ms minus emission time 1 ms, flight path about 23 m.
    expected = (
        sc.constants.h
        / sc.constants.m_n
        * sc.scalar(24.0, unit='ms')
        / sc.scalar(23.0, unit='m')
    ).to(unit='angstrom')
    assert_allclose(
        wavelength,
        sc.full(sizes=wavelength.sizes, value=expected.value, unit=expected.unit),
        rtol=sc.scalar(0.002),
    )


def test_loader_uses_workflow_pulse_period(mcstas_file):
    workflow = FreiaMcStasWorkflow()
    workflow[Filename[SampleRun]] = str(mcstas_file)
    workflow[PulsePeriod] = sc.scalar(50.0, unit='ms')
    events = workflow.compute(RawDetector[SampleRun]).bins.constituents['data']
    events = sc.sort(events, 'event_time_zero')
    # The event at 25 ms + 1/14 s falls in the second 50 ms pulse period.
    assert_identical(
        events.coords['event_time_zero'].to(unit='ns'),
        sc.datetimes(dims=['event'], values=[0, 0, 50_000_000], unit='ns'),
    )
    assert_allclose(
        events.coords['event_time_offset'].to(unit='s'),
        sc.array(
            dims=['event'], values=[0.025, 0.025, 0.025 + 1 / 14 - 0.05], unit='s'
        ),
    )
