# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
from pathlib import Path

import h5py
import numpy as np
import pytest
import scipp as sc
from ess.reduce.nexus.types import Filename, NeXusName, RawDetector
from scipp.testing import assert_allclose, assert_identical

from ess.freia import FreiaMcStasWorkflow
from ess.freia.mcstas import mcstas_detector_geometry
from ess.freia.types import DetectorRegionOfInterest, IncidentMonitor, WavelengthMonitor
from ess.reflectometry.types import CorrectedDetector, SampleRun, WavelengthBins


def _component(components, name, position):
    group = components.create_group(f'{len(components):04d}_{name}')
    group['Position'] = position
    group['Rotation'] = np.eye(3)
    return group


@pytest.fixture
def mcstas_file(tmp_path: Path) -> Path:
    """Minimal file for the McStasToX reader; FREIA has no published event fixture."""
    filename = tmp_path / 'freia.h5'
    with h5py.File(filename, 'w') as f:
        entry = f.create_group('entry1')
        entry.create_group('data')
        simulation = entry.create_group('simulation')
        simulation.attrs['program'] = np.bytes_('3.7.18, git')
        simulation.create_group('Param')
        components = entry.create_group('instrument/components')
        _component(components, 'Source', [0.0, 0.0, 0.0])
        _component(components, 'Arm_Sample', [0.0, 0.0, 20.0])
        detector = _component(components, 'Multiblade', [0.0, 0.0, 20.0])
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


def test_load_detector_and_compute_q(mcstas_file):
    workflow = FreiaMcStasWorkflow()
    workflow[Filename[SampleRun]] = mcstas_file
    workflow[DetectorRegionOfInterest[SampleRun]] = {}
    workflow[WavelengthBins] = sc.array(
        dims=['wavelength'], values=[1.0, 12.0], unit='angstrom'
    )

    result = workflow.compute((RawDetector[SampleRun], CorrectedDetector[SampleRun]))

    raw = result[RawDetector[SampleRun]]
    image = sc.sort(raw.bins.sum(), 'pixel_id')
    assert_identical(
        image.coords['pixel_id'],
        sc.array(dims=['pixel_id'], values=[10, 12, 99, 101], unit=None),
    )
    assert_identical(
        image.data,
        sc.array(
            dims=['pixel_id'],
            values=[5.0, 4.0, 0.0, 0.0],
            variances=[13.0, 16.0, 0.0, 0.0],
            unit='counts',
        ),
    )
    events = raw.bins.constituents['data']
    assert_allclose(
        events.coords['event_time_offset'],
        sc.full(sizes=events.sizes, value=0.025, unit='s'),
    )
    q = result[CorrectedDetector[SampleRun]].bins.constituents['data'].coords['Q']
    assert sc.isfinite(q).all().value


def test_load_histogram_geometry(mcstas_file):
    with h5py.File(mcstas_file, 'r+') as f:
        detector = f['entry1/instrument/components/0002_Multiblade']
        del detector['output']
        histogram = detector.create_group('output/histogram')
        histogram.attrs['xlabel'] = np.bytes_('theta [deg]')
        histogram.attrs['ylabel'] = np.bytes_('Height [cm]')
        histogram['theta__deg_'] = [0.0, 90.0]
        histogram['Height__cm_'] = [-25, 25]

    detector = mcstas_detector_geometry(mcstas_file, 'Multiblade')

    assert_allclose(
        detector.coords['position'],
        sc.vectors(
            dims=['pixel_id'],
            values=[[0.25, 0, 23], [0.25, 3, 20], [-0.25, 0, 23], [-0.25, 3, 20]],
            unit='m',
        ),
    )


def test_load_monitor(mcstas_file):
    with h5py.File(mcstas_file, 'r+') as f:
        monitor = _component(
            f['entry1/instrument/components'], 'IncidentLambda', [0, 0, 19]
        )
        histogram = monitor.create_group('output/spectrum')
        histogram.attrs['xvar'] = np.bytes_('L')
        histogram.attrs['xlimits'] = np.bytes_('1 5')
        histogram['data'] = [10.0, 20.0]
        histogram['errors'] = [3.0, 4.0]
    workflow = FreiaMcStasWorkflow()
    workflow[Filename[SampleRun]] = mcstas_file
    workflow[NeXusName[IncidentMonitor]] = 'IncidentLambda'

    result = workflow.compute(WavelengthMonitor[SampleRun])

    assert_identical(
        result,
        sc.DataArray(
            sc.array(
                dims=['wavelength'],
                values=[10.0, 20.0],
                variances=[9.0, 16.0],
                unit='counts',
            ),
            coords={
                'wavelength': sc.array(
                    dims=['wavelength'], values=[1.0, 3.0, 5.0], unit='angstrom'
                )
            },
        ),
    )
