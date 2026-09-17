# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
import numpy as np
import scipp as sc
from ess.reduce.nexus.types import GravityVector, Position
from scipp.testing import assert_allclose, assert_identical
from scippnexus import NXsample, NXsource

from ess.freia import FreiaWorkflow
from ess.freia.corrections import RunNormalization
from ess.freia.types import (
    DetectorRegionOfInterest,
    SampleIlluminatedFraction,
    SampleSurfaceNormal,
    WavelengthMonitor,
)
from ess.reflectometry.corrections import footprint_on_sample
from ess.reflectometry.types import (
    BeamSize,
    QBins,
    ReducibleData,
    ReferenceRun,
    ReflectivityOverQ,
    SampleRun,
    SampleSize,
    WavelengthBins,
    WavelengthDetector,
)


def _make_workflow(run_norm):
    wf = FreiaWorkflow(run_norm=run_norm)
    wf[GravityVector] = sc.vector([0.0, 0.0, 0.0], unit='m/s^2')
    for run, sign, weights in [
        (SampleRun, 1, [20.0, 40.0, 999.0]),
        (ReferenceRun, -1, [100.0, 100.0, 999.0]),
    ]:
        events = sc.DataArray(
            sc.array(dims=['event'], values=weights, variances=weights, unit='counts'),
            coords={
                'wavelength': sc.array(
                    dims=['event'], values=[2.0, 4.0, 2.0], unit='angstrom'
                ),
                'pixel_id': sc.array(dims=['event'], values=[0, 0, 1], unit=None),
            },
        ).group('pixel_id')
        # Sample rotation is 10 degrees. The matching beams make angles of
        # 30 degrees with the surface; the second pixel is outside the ROI.
        angles = np.deg2rad([10.0 + sign * 30.0, 10.0 + sign * 60.0])
        events.coords['position'] = sc.vectors(
            dims=['pixel_id'],
            values=[[0, np.sin(angle), np.cos(angle)] for angle in angles],
            unit='m',
        )
        wf[WavelengthDetector[run]] = events
        wf[Position[NXsample, run]] = sc.vector([0.0, 0.0, 0.0], unit='m')
        # Source position must not set the angular reference frame.
        wf[Position[NXsource, run]] = sc.vector([3.0, 2.0, -20.0], unit='m')
        wf[SampleSurfaceNormal[run]] = sc.vector(
            [0.0, np.cos(np.deg2rad(10.0)), -np.sin(np.deg2rad(10.0))]
        )
        low, high = sorted([10.0 + sign * 20.0, 10.0 + sign * 40.0])
        wf[DetectorRegionOfInterest[run]] = {
            'scattering_angle': (
                sc.scalar(low, unit='deg'),
                sc.scalar(high, unit='deg'),
            )
        }
    wf[WavelengthBins] = sc.array(
        dims=['wavelength'], values=[1.0, 3.0, 5.0], unit='angstrom'
    )
    wf[QBins] = sc.array(dims=['Q'], values=[1.0, 2.0, 4.0, 6.0], unit='1/angstrom')
    return wf


def test_reduce_reflectivity_with_monitor_and_footprint():
    wf = _make_workflow(RunNormalization.monitor_histogram)
    sample_size = sc.scalar(10.0, unit='mm')
    beam_size = sc.scalar(5.0, unit='mm')
    wf[SampleSize[SampleRun]] = sample_size
    wf[BeamSize[SampleRun]] = beam_size
    for run, values in [(SampleRun, [4.0, 8.0]), (ReferenceRun, [2.0, 2.0])]:
        wf[WavelengthMonitor[run]] = sc.DataArray(
            sc.array(dims=['wavelength'], values=values, unit='counts'),
            coords={'wavelength': wf.compute(WavelengthBins)},
        )

    results = wf.compute((ReflectivityOverQ, ReducibleData[ReferenceRun]))
    result = results[ReflectivityOverQ]
    direct_beam = results[ReducibleData[ReferenceRun]]
    assert 'theta' not in direct_beam.bins.coords
    assert 'Q' not in direct_beam.bins.coords

    fraction = footprint_on_sample(sc.scalar(30.0, unit='deg'), beam_size, sample_size)
    # Q bins contain wavelengths 4 and 2 angstrom, respectively.
    expected = (
        sc.array(dims=['Q'], values=[0.1, 0.1], variances=[0.00035, 0.0006]) / fraction
    )
    assert_allclose(result['Q', :2].data, expected)
    assert_identical(
        result.masks['direct_beam'],
        sc.array(dims=['Q'], values=[False, False, True]),
    )


def test_rebinning_integrates_before_dividing():
    wf = _make_workflow(RunNormalization.none)
    wf[SampleIlluminatedFraction] = sc.scalar(1.0)
    wf[QBins] = sc.array(dims=['Q'], values=[1.0, 4.0], unit='1/angstrom')

    result = wf.compute(ReflectivityOverQ)

    assert_allclose(
        result.data,
        sc.array(
            dims=['Q'], values=[60.0 / 200.0], variances=[0.3**2 * (1 / 60 + 1 / 200)]
        ),
    )
