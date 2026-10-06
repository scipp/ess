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
from ess.freia.data import freia_mcstas_reference_run, freia_mcstas_sample_run
from ess.freia.mcstas import mcstas_detector_geometry
from ess.freia.types import DetectorRegionOfInterest, IncidentMonitor, WavelengthMonitor
from ess.reflectometry.types import CorrectedDetector, SampleRun, WavelengthBins


def _component(components, name, position):
    group = components.create_group(f'{len(components):04d}_{name}')
    group['Position'] = position
    group['Rotation'] = np.eye(3)
    return group


def test_load_detector_and_compute_q():
    workflow = FreiaMcStasWorkflow()
    workflow[Filename[SampleRun]] = freia_mcstas_sample_run()
    workflow[DetectorRegionOfInterest[SampleRun]] = {}
    workflow[WavelengthBins] = sc.array(
        dims=['wavelength'], values=[1.0, 12.0], unit='angstrom'
    )

    result = workflow.compute((RawDetector[SampleRun], CorrectedDetector[SampleRun]))

    raw = result[RawDetector[SampleRun]]
    assert raw.sizes == {'pixel_id': 2048 * 64}
    events = raw.bins.constituents['data']
    assert events.sizes == {'event': 555874}
    assert_allclose(
        events.sum().data,
        sc.scalar(744336.6691526351, variance=679997917.939667, unit='counts'),
    )
    assert events.coords['wavelength_from_mcstas'].unit == 'angstrom'
    offsets = events.coords['event_time_offset']
    assert offsets.min() >= sc.scalar(0.0, unit='s')
    assert offsets.max() < sc.scalar(1 / 14, unit='s')
    q = result[CorrectedDetector[SampleRun]].bins.constituents['data'].coords['Q']
    assert sc.isfinite(q).any().value


def test_load_histogram_geometry(tmp_path: Path):
    filename = tmp_path / 'histogram.h5'
    with h5py.File(filename, 'w') as f:
        entry = f.create_group('entry1')
        entry.create_group('data')
        simulation = entry.create_group('simulation')
        simulation.attrs['program'] = np.bytes_('3.7.18, git')
        simulation.create_group('Param')
        components = entry.create_group('instrument/components')
        detector = _component(components, 'Multiblade', [0.0, 0.0, 20.0])
        detector['Rotation'][...] = [[0, 1, 0], [-1, 0, 0], [0, 0, 1]]
        geometry = detector.create_group('Geometry')
        geometry.attrs['Shape identifier'] = np.bytes_('4')
        geometry.attrs['radius'] = np.bytes_('3')
        histogram = detector.create_group('output/histogram')
        histogram.attrs['xlabel'] = np.bytes_('theta [deg]')
        histogram.attrs['ylabel'] = np.bytes_('Height [cm]')
        histogram['theta__deg_'] = [0.0, 90.0]
        histogram['Height__cm_'] = [-25, 25]

    detector = mcstas_detector_geometry(filename, 'Multiblade')

    assert_allclose(
        detector.coords['position'],
        sc.vectors(
            dims=['pixel_id'],
            values=[[0.25, 0, 23], [0.25, 3, 20], [-0.25, 0, 23], [-0.25, 3, 20]],
            unit='m',
        ),
    )


@pytest.mark.parametrize(
    'filename', [freia_mcstas_sample_run, freia_mcstas_reference_run]
)
def test_load_monitor(filename):
    path = filename()
    workflow = FreiaMcStasWorkflow()
    workflow[Filename[SampleRun]] = path
    workflow[NeXusName[IncidentMonitor]] = 'SampleLambda'

    result = workflow.compute(WavelengthMonitor[SampleRun])

    assert_identical(
        result.coords['wavelength'],
        sc.linspace('wavelength', 0.0, 25.0, 101, unit='angstrom'),
    )
    with h5py.File(path, 'r') as f:
        monitor = f['entry1/data/SampleLambda_dat']
        assert_identical(
            result.data,
            sc.array(
                dims=['wavelength'],
                values=monitor['data'][:],
                variances=monitor['errors'][:] ** 2,
                unit='counts',
            ),
        )
