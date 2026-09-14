# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
# Small known intensities exercise the complete reduction graph without a LUT.
import numpy as np
import pytest
import scipp as sc
from ess.reduce import workflow as reduce_workflow
from ess.reduce.nexus.types import Position
from scipp.testing import assert_allclose, assert_identical
from scippnexus import NXsample, NXsource

from ess import freia
from ess.freia.corrections import RunNormalization
from ess.freia.types import (
    DetectorRegionOfInterest,
    QDetector,
    SampleIlluminatedFraction,
    SampleSurfaceNormal,
    WavelengthMonitor,
)
from ess.reflectometry.types import (
    BeamSize,
    QBins,
    Reference,
    ReferenceRun,
    ReflectivityOverQ,
    RunUnnormalizedData,
    SampleRun,
    SampleSize,
    WavelengthBins,
    WavelengthDetector,
)


def test_freia_workflow_registers_run_normalization_variants():
    for wf in (
        freia.FreiaMcStasUnnormalizedWorkflow,
        freia.FreiaMcStasMonitorHistogramWorkflow,
        freia.FreiaMcStasMonitorIntegratedWorkflow,
        freia.FreiaMcStasProtonChargeWorkflow,
        freia.FreiaUnnormalizedWorkflow,
        freia.FreiaMonitorHistogramWorkflow,
        freia.FreiaMonitorIntegratedWorkflow,
        freia.FreiaProtonChargeWorkflow,
    ):
        assert wf in reduce_workflow.workflow_registry


def _detector(angles, weights):
    position = np.array([1.0, 2.0, 3.0]) + 3.0 * np.column_stack(
        [np.zeros(len(angles)), np.sin(np.deg2rad(angles)), np.cos(np.deg2rad(angles))]
    )
    return (
        sc.DataArray(
            sc.array(dims=['event'], values=weights, variances=weights, unit='counts'),
            coords={
                'wavelength': sc.array(
                    dims=['event'], values=[2.0, 4.0, 2.0], unit='angstrom'
                ),
                'pixel_id': sc.array(dims=['event'], values=[0, 0, 1], unit=None),
            },
        )
        .group('pixel_id')
        .assign_coords(
            position=sc.vectors(dims=['pixel_id'], values=position[[0, 2]], unit='m')
        )
    )


def _workflow(run_norm=RunNormalization.none):
    wf = freia.FreiaWorkflow(run_norm=run_norm)
    for run, angles, weights in [
        (SampleRun, [1.0, 1.0, 2.0], [20.0, 40.0, 999.0]),
        (ReferenceRun, [-1.0, -1.0, -2.0], [100.0, 100.0, 999.0]),
    ]:
        wf[WavelengthDetector[run]] = _detector(angles, weights)
        wf[Position[NXsample, run]] = sc.vector([1.0, 2.0, 3.0], unit='m')
        wf[Position[NXsource, run]] = sc.vector([1.0, 2.0, -17.0], unit='m')
        wf[SampleSurfaceNormal[run]] = sc.vector([0.0, 1.0, 0.0])
    wf[DetectorRegionOfInterest[SampleRun]] = {
        'theta': (sc.scalar(0.5, unit='deg'), sc.scalar(1.5, unit='deg'))
    }
    wf[DetectorRegionOfInterest[ReferenceRun]] = {
        'theta': (sc.scalar(-1.5, unit='deg'), sc.scalar(-0.5, unit='deg'))
    }
    wf[WavelengthBins] = sc.array(
        dims=['wavelength'], values=[1.0, 3.0, 5.0], unit='angstrom'
    )
    wf[QBins] = sc.array(dims=['Q'], values=[0.04, 0.08, 0.13, 0.3], unit='1/angstrom')
    return wf


def test_q_uses_sample_plane_and_translated_pixels_without_reduction_inputs():
    wf = _workflow()
    result = wf.compute(QDetector[SampleRun])
    q = result.bins.constituents['data'].coords['Q']
    expected = 4 * np.pi * np.sin(_expected_exit_angles()) / [2.0, 4.0, 2.0]
    assert_allclose(q, sc.array(dims=['event'], values=expected, unit='1/angstrom'))
    assert_allclose(
        result.coords['L2'], sc.full(sizes=result.sizes, value=3.0, unit='m')
    )


def test_q_follows_tilted_sample_surface():
    wf = _workflow()
    tilt = np.deg2rad(0.5)
    wf[SampleSurfaceNormal[SampleRun]] = sc.vector([0.0, np.cos(tilt), -np.sin(tilt)])
    result = wf.compute(QDetector[SampleRun])
    assert_allclose(
        result.bins.constituents['data'].coords['theta'],
        sc.array(dims=['event'], values=_expected_exit_angles() - tilt, unit='rad'),
    )


def _expected_exit_angles(sign=1.0):
    """Independent calculation from displacement = velocity*time + gravity*time²/2."""
    angle = sign * np.deg2rad([1.0, 1.0, 2.0])
    speed = (
        (
            sc.constants.h
            / sc.constants.m_n
            / sc.array(dims=['event'], values=[2.0, 4.0, 2.0], unit='angstrom')
        )
        .to(unit='m/s')
        .values
    )
    time = 3.0 / speed
    y = 3.0 * np.sin(angle) + 0.5 * sc.constants.g.value * time**2
    return np.arctan2(y, 3.0 * np.cos(angle))


def test_theta_roi_distinguishes_wavelengths_in_the_same_pixel():
    wf = _workflow()
    wf[DetectorRegionOfInterest[SampleRun]] = {
        'theta': (sc.scalar(1.0005, unit='deg'), sc.scalar(1.001, unit='deg')),
    }
    selected = wf.compute(RunUnnormalizedData[SampleRun])
    assert selected.bins.sum().sum().value == 40.0


def test_direct_beam_gravity_is_corrected_before_specular_mapping():
    wf = _workflow()
    reference = wf.compute(Reference)
    assert_allclose(
        reference.bins.constituents['data'].coords['theta'],
        sc.array(dims=['event'], values=-_expected_exit_angles(sign=-1.0), unit='rad'),
    )


@pytest.mark.parametrize(
    ('run_norm', 'expected'),
    [
        (RunNormalization.none, [0.8, 0.4]),
        (RunNormalization.monitor_histogram, [0.2, 0.2]),
        (RunNormalization.monitor_integrated, [0.8 / 3, 0.4 / 3]),
    ],
)
def test_direct_beam_reduction_integrates_separate_rois(run_norm, expected):
    wf = _workflow(run_norm)
    wf[SampleIlluminatedFraction] = sc.scalar(0.5)
    for run, values in [(SampleRun, [4.0, 8.0]), (ReferenceRun, [2.0, 2.0])]:
        wf[WavelengthMonitor[run]] = sc.DataArray(
            sc.array(dims=['wavelength'], values=values, unit='counts'),
            coords={'wavelength': wf.compute(WavelengthBins)},
        )
    result = wf.compute(ReflectivityOverQ)
    assert result.dims == ('Q',)
    assert result.unit == sc.units.dimensionless
    np.testing.assert_allclose(result.values[:2], expected)
    assert_identical(
        result.masks['direct_beam'],
        sc.array(dims=['Q'], values=[False, False, True]),
    )
    assert np.isnan(result.values[2])
    # The independent sample and direct-beam counting uncertainties both contribute.
    np.testing.assert_allclose(
        result.variances[:2],
        np.array(expected) ** 2 * (1 / np.array([40, 20]) + 1 / 100),
    )


def test_reduction_is_ratio_of_integrals_when_q_bins_are_merged():
    wf = _workflow()
    wf[SampleIlluminatedFraction] = sc.scalar(1.0)
    wf[QBins] = sc.array(dims=['Q'], values=[0.04, 0.13], unit='1/angstrom')
    result = wf.compute(ReflectivityOverQ)
    np.testing.assert_allclose(result.values, [60.0 / 200.0])


def test_zero_monitor_intensity_masks_corresponding_reflectivity_bin():
    wf = _workflow(RunNormalization.monitor_histogram)
    wf[SampleIlluminatedFraction] = sc.scalar(1.0)
    for run, values in [(SampleRun, [2.0, 2.0]), (ReferenceRun, [0.0, 2.0])]:
        wf[WavelengthMonitor[run]] = sc.DataArray(
            sc.array(dims=['wavelength'], values=values, unit='counts'),
            coords={'wavelength': wf.compute(WavelengthBins)},
        )
    result = wf.compute(ReflectivityOverQ)
    assert_identical(
        result.masks['direct_beam'],
        sc.array(dims=['Q'], values=[False, True, True]),
    )


def test_footprint_uses_amor_model_and_requires_only_sample_sizes():
    wf = _workflow()
    wf[SampleSize[SampleRun]] = sc.scalar(10.0, unit='mm')
    wf[BeamSize[SampleRun]] = sc.scalar(10.0 * np.sin(np.deg2rad(1.0)), unit='mm')
    result = wf.compute(ReflectivityOverQ)
    # The two wavelengths in the same pixel have different corrected footprints.
    projected_size_over_beam = np.sin(_expected_exit_angles()[:2]) / np.sin(
        np.deg2rad(1.0)
    )
    fraction = sc.erf(
        sc.array(
            dims=['event'], values=projected_size_over_beam / np.sqrt(8.0 * np.log(2.0))
        )
    ).values
    np.testing.assert_allclose(result.values[:2], np.array([0.4, 0.2]) / fraction[::-1])
