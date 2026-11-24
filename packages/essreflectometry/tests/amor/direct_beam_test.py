# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
"""Analytic rays for direct-beam geometry and footprint normalization."""

import numpy as np
import pytest
import scipp as sc
from scipp.testing import assert_allclose, assert_identical

from ess import amor
from ess.amor import conversions
from ess.amor.types import (
    ChopperDistance,
    ChopperFrequency,
    ChopperPhaseOffset,
    ChopperSeparation,
    GravityToggle,
)
from ess.reflectometry.corrections import footprint_on_sample
from ess.reflectometry.types import (
    BeamDivergenceLimits,
    BeamSize,
    DetectorRotation,
    ProtonCharge,
    QBins,
    RawDetector,
    ReducedReference,
    ReducibleData,
    Reference,
    ReferenceRun,
    ReflectivityOverQ,
    SampleRotation,
    SampleRun,
    SampleSize,
    WavelengthBins,
    YIndexLimits,
    ZIndexLimits,
)


def _rays(weights, pixel_angles, wavelength=None):
    """Two detector rows, one event per row, with known intensities."""
    if wavelength is None:
        wavelength = sc.scalar(8.0, unit='angstrom')
    flight_time = (
        wavelength * sc.scalar(18.842, unit='m') * sc.constants.m_n / sc.constants.h
    ).to(unit='ns')
    # Zero trigger phase. Include the pixel opening time in the recorded arrival.
    offsets = flight_time + pixel_angles / sc.scalar(180.0, unit='deg') * sc.scalar(
        1e9 / 15, unit='ns'
    )
    events = sc.DataArray(
        sc.array(dims=['event'], values=weights, variances=weights, unit='counts'),
        coords={
            'event_time_offset': offsets.rename_dims({'wire': 'event'}),
            'event_time_zero': sc.datetimes(
                dims=['event'], values=['2026-01-01', '2026-01-01'], unit='ns'
            ),
        },
    )
    begin = sc.array(dims=['blade', 'wire', 'strip'], values=[[[0], [1]]], unit=None)
    return sc.DataArray(
        sc.bins(
            data=events, dim='event', begin=begin, end=begin + sc.scalar(1, unit=None)
        ),
        coords={
            'blade': sc.arange('blade', 1),
            'wire': sc.arange('wire', 2),
            'strip': sc.arange('strip', 1),
            'z_index': sc.arange('wire', 2),
            'pixel_divergence_angle': pixel_angles.broadcast(
                sizes={'blade': 1, 'wire': 2}
            ).copy(),
            'distance_in_detector': sc.scalar(0.0, unit='mm'),
        },
    )


def _workflow(mode, sample_angle, reference_angle, reference_size=50.0):
    """Simulate a sample with R=0.3, including the known footprint losses."""
    wf = amor.AmorWorkflow(normalized_by=mode)
    wf[GravityToggle] = False
    delta = sc.array(dims=['wire'], values=[-0.25, 0.25], unit='deg')
    beam_size = sc.scalar(2.0, unit='mm')
    sample_size = sc.scalar(50.0, unit='mm')
    ref_size = sc.scalar(reference_size, unit='mm')
    sample_fraction = footprint_on_sample(
        sc.scalar(sample_angle, unit='deg') + delta, beam_size, sample_size
    )
    reference_fraction = footprint_on_sample(
        sc.scalar(reference_angle, unit='deg') - delta, beam_size, ref_size
    )
    ref_weights = (
        reference_fraction.values if mode == 'direct_beam_with_aperture' else np.ones(2)
    )
    for run, mu, nu, weights in (
        (SampleRun, sample_angle, 2 * sample_angle, 0.3 * sample_fraction.values),
        (ReferenceRun, reference_angle, 0.0, ref_weights),
    ):
        wf[RawDetector[run]] = _rays(1e6 * weights, delta)
        wf[SampleRotation[run]] = sc.scalar(mu, unit='deg')
        wf[DetectorRotation[run]] = sc.scalar(nu, unit='deg')
        wf[BeamSize[run]] = beam_size
        wf[ChopperFrequency[run]] = sc.scalar(7.5, unit='Hz')
        wf[ChopperPhaseOffset[run]] = sc.scalar(0.0, unit='deg')
        wf[ChopperDistance[run]] = sc.scalar(-14842.0, unit='mm')
        wf[ChopperSeparation[run]] = sc.scalar(1000.0, unit='mm')
        wf[ProtonCharge[run]] = sc.DataArray(
            sc.array(dims=['time'], values=[1.0], unit='uC'),
            coords={
                'time': sc.datetimes(dims=['time'], values=['2026-01-01'], unit='ns')
            },
        )
    wf[SampleSize[SampleRun]] = sample_size
    if reference_size != 50.0:
        wf[SampleSize[ReferenceRun]] = ref_size
    wf[WavelengthBins] = sc.array(
        dims=['wavelength'], values=[7.5, 8.5], unit='angstrom'
    )
    wf[QBins] = sc.linspace('Q', 0.001, 0.1, 51, unit='1/angstrom')
    wf[YIndexLimits] = sc.scalar(0), sc.scalar(0)
    wf[ZIndexLimits] = sc.scalar(0), sc.scalar(1)
    return wf


@pytest.mark.parametrize('mode', ['direct_beam', 'direct_beam_with_aperture'])
@pytest.mark.parametrize(('sample_angle', 'reference_angle'), [(1.0, 1.0), (0.5, 1.5)])
@pytest.mark.parametrize('reference_size', [25.0, 50.0])
def test_recovers_reflectivity_with_footprint_losses(
    mode, sample_angle, reference_angle, reference_size
):
    wf = _workflow(mode, sample_angle, reference_angle, reference_size)
    result = wf.compute(ReflectivityOverQ).hist()
    covered = result[sc.isfinite(result.data) & (result.data > sc.scalar(0.0))]
    assert covered.sizes['Q'] == 2
    np.testing.assert_allclose(covered.values, 0.3, rtol=1e-12)
    # Q and its resolution must be evaluated with reflected sample geometry,
    # even though the reference events used transmitted-ray incidence angles.
    reference = wf.compute(Reference)
    expected_theta = sc.array(
        dims=['wire'], values=[sample_angle - 0.25, sample_angle + 0.25], unit='deg'
    ).to(unit='rad')
    assert_allclose(reference.coords['theta']['blade', 0], expected_theta)
    assert sc.all(reference.coords['Q'] > sc.scalar(0.0, unit='1/angstrom'))
    assert sc.all(sc.isfinite(reference.coords['Q_resolution']))


@pytest.mark.parametrize('mode', ['direct_beam', 'direct_beam_with_aperture'])
def test_asymmetric_divergence_limits_select_matching_rays(mode):
    wf = _workflow(mode, 0.5, 1.5)
    wf[BeamDivergenceLimits] = sc.scalar(-0.3, unit='deg'), sc.scalar(-0.2, unit='deg')
    prepared = wf.compute((ReducibleData[SampleRun], ReducibleData[ReferenceRun]))
    for run, expected in ((SampleRun, [False, True]), (ReferenceRun, [True, False])):
        mask = prepared[ReducibleData[run]].bins.masks['divergence_too_large']
        np.testing.assert_array_equal(mask.bins.constituents['data'].values, expected)
    reduced = wf.compute(ReducedReference)
    # The masked reference event must stay excluded through histogramming and mirroring.
    assert reduced['wire', 1].sum().value == 0
    result = wf.compute(ReflectivityOverQ).hist()
    covered = result[sc.isfinite(result.data) & (result.data > sc.scalar(0.0))]
    assert covered.sizes['Q'] == 1
    np.testing.assert_allclose(covered.values, 0.3, rtol=1e-12)


@pytest.mark.parametrize('gravity', [False, True])
def test_reference_divergence_mask_uses_each_events_wavelength(gravity):
    wf = _workflow('direct_beam', 1.0, 1.0)
    wf[GravityToggle] = gravity
    # Two wavelengths at the same geometric detector angle. Gravity puts only
    # the long-wavelength ray beyond the -0.255 degree divergence boundary.
    delta = sc.array(dims=['wire'], values=[0.25, 0.25], unit='deg')
    wf[RawDetector[ReferenceRun]] = _rays(
        [1.0, 1.0],
        delta,
        sc.array(dims=['wire'], values=[3.0, 12.0], unit='angstrom'),
    )
    wf[WavelengthBins] = sc.linspace('wavelength', 2.0, 13.0, 12, unit='angstrom')
    wf[BeamDivergenceLimits] = (
        sc.scalar(-1.0, unit='deg'),
        sc.scalar(-0.255, unit='deg'),
    )
    reference = wf.compute(ReducibleData[ReferenceRun])
    mask = reference.bins.masks['divergence_too_large'].bins.constituents['data']
    np.testing.assert_array_equal(mask.values, [True, not gravity])


@pytest.mark.parametrize('gravity', [False, True])
def test_transmitted_incidence_angle_and_divergence_include_gravity(gravity):
    # Start from the actual ray tangent at the sample, and propagate to the detector.
    gamma = sc.scalar(0.4, unit='deg').to(unit='rad')
    mu = sc.scalar(1.0, unit='deg')
    nu = sc.scalar(0.1, unit='deg')
    length = sc.scalar(4.0, unit='m')
    wavelength = sc.scalar(12.0, unit='angstrom')
    speed = sc.constants.h / (sc.constants.m_n * wavelength)
    drop = (sc.constants.g * length / (2 * speed**2)).to(unit='dimensionless')
    geometric_angle = sc.asin(sc.sin(gamma) - (drop if gravity else sc.scalar(0.0)))
    kwargs = {
        'wavelength': wavelength,
        'pixel_divergence_angle': geometric_angle - nu.to(unit='rad'),
        'sample_rotation': mu,
        'detector_rotation': nu,
    }
    if gravity:
        angle = conversions.direct_beam_theta(**kwargs, L2=length)
    else:
        angle = conversions.direct_beam_theta_no_gravity(**kwargs)
    assert_allclose(angle, sc.scalar(0.6, unit='deg').to(unit='rad'))
    divergence = conversions.direct_beam_divergence_angle(angle, mu, nu)
    assert_allclose(divergence, sc.scalar(-0.3, unit='deg').to(unit='rad'))


@pytest.mark.parametrize('binned', [False, True])
@pytest.mark.parametrize(
    ('unit', 'dtype'), [('ns', 'int64'), ('ns', 'float64'), ('us', 'float64')]
)
def test_wavelength_conversion_preserves_input_and_folds_both_pulses(
    binned, unit, dtype
):
    # 10 Hz gives a 50 ms pulse period. These arrivals differ by exactly one pulse.
    times = sc.array(dims=['event'], values=[10_000_000, 60_000_000], unit='ns').to(
        unit=unit, dtype=dtype
    )
    if binned:
        times = sc.bins(
            data=times,
            dim='event',
            begin=sc.array(dims=['pixel'], values=[0], unit=None),
            end=sc.array(dims=['pixel'], values=[2], unit=None),
        )
    before = times.copy()
    result = conversions.wavelength(
        event_time_offset=times,
        pixel_divergence_angle=sc.scalar(0.0, unit='deg'),
        L1=sc.scalar(16.0, unit='m'),
        L2=sc.scalar(4.0, unit='m'),
        chopper_frequency=sc.scalar(10.0, unit='Hz'),
        chopper_phase_offset=sc.scalar(0.0, unit='deg'),
    )
    expected = (
        sc.constants.h / sc.constants.m_n * sc.scalar(0.01 / 20.0, unit='s/m')
    ).to(unit='angstrom')
    actual = result.bins.constituents['data'] if binned else result
    np.testing.assert_allclose(actual.values, expected.value, rtol=1e-12)
    assert_identical(times, before)
