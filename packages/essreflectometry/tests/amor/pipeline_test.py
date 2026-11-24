# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
from pathlib import Path

import pytest
import sciline
import scipp as sc
from orsopy import fileio
from scipp.testing import assert_allclose, assert_identical

from ess import amor
from ess.amor.conversions import wavelength
from ess.amor.types import (
    ChopperDistance,
    ChopperFrequency,
    ChopperSeparation,
)
from ess.reflectometry import orso, supermirror
from ess.reflectometry.types import (
    BeamDivergenceLimits,
    BeamSize,
    DetectorRotation,
    Filename,
    ProtonCharge,
    QBins,
    RawSampleRotation,
    ReducedReference,
    ReducibleData,
    ReferenceRun,
    ReflectivityOverQ,
    SampleRotation,
    SampleRotationOffset,
    SampleRun,
    SampleSize,
    WavelengthBins,
    YIndexLimits,
    ZIndexLimits,
)
from ess.reflectometry.workflow import with_filenames

# Proton current in the simulated measurements, see conftest.py
PROTON_CURRENT_UA = 1500.0

# In the simulated measurements the direct beam falls on the lower half of the
# detector and the reflected beam on the upper half. The sample reflects half
# of the neutrons, independent of wavelength and angle.
REFLECTIVITY = 0.5
LOWER_BLADES = (0, 7)
UPPER_BLADES = (7, 14)


@pytest.fixture(scope='module')
def direct_beam_file(make_amor_file) -> Path:
    return make_amor_file(
        mu=0.0, nu=0.0, blades=LOWER_BLADES, n_events=400_000, n_pulses=100, seed=1
    )


@pytest.fixture(scope='module')
def sample_file(make_amor_file) -> Path:
    # Longer than the reference measurement to test the normalization
    return make_amor_file(
        mu=0.5, nu=1.5, blades=UPPER_BLADES, n_events=300_000, n_pulses=150, seed=2
    )


def _set_parameters(workflow: sciline.Pipeline) -> sciline.Pipeline:
    workflow[SampleSize[SampleRun]] = sc.scalar(50.0, unit='mm')
    # A beam much narrower than the sample: the footprint correction is one
    workflow[BeamSize[SampleRun]] = sc.scalar(1.0, unit='nm')
    workflow[WavelengthBins] = sc.geomspace(
        'wavelength', 3.0, 12.0, 31, unit='angstrom'
    )
    workflow[YIndexLimits] = sc.scalar(11), sc.scalar(41)
    workflow[ZIndexLimits] = sc.scalar(0), sc.scalar(447)
    # These files populate rows uniformly, without simulating ray transport.
    # Exercise divergence selection separately with analytic incident rays.
    workflow[BeamDivergenceLimits] = (
        sc.scalar(-2.0, unit='deg'),
        sc.scalar(2.0, unit='deg'),
    )
    workflow[QBins] = sc.geomspace('Q', 0.005, 0.2, 21, unit='1/angstrom')
    workflow[orso.OrsoCreator] = orso.OrsoCreator(
        fileio.base.Person(
            name='Max Mustermann',
            affiliation='European Spallation Source ERIC',
            contact='max.mustermann@ess.eu',
        )
    )
    return workflow


@pytest.fixture
def direct_beam_workflow(direct_beam_file: Path, sample_file: Path) -> sciline.Pipeline:
    workflow = _set_parameters(amor.AmorWorkflow(normalized_by='direct_beam'))
    workflow[Filename[SampleRun]] = sample_file
    workflow[Filename[ReferenceRun]] = direct_beam_file
    return workflow


@pytest.fixture
def supermirror_workflow(direct_beam_file: Path, sample_file: Path) -> sciline.Pipeline:
    workflow = _set_parameters(amor.AmorWorkflow())
    workflow[SampleSize[ReferenceRun]] = sc.scalar(50.0, unit='mm')
    workflow[BeamSize[ReferenceRun]] = sc.scalar(1.0, unit='nm')
    workflow[Filename[SampleRun]] = sample_file
    # A reflection from the sample plays the role of an ideal supermirror
    workflow[Filename[ReferenceRun]] = sample_file
    workflow[supermirror.CriticalEdge] = sc.scalar(float('inf'), unit='1/angstrom')
    return workflow


def _mean_reflectivity(reflectivity: sc.DataArray) -> sc.Variable:
    r = reflectivity.hist()
    covered = sc.isfinite(r.data) & (r.data > sc.scalar(0.0, unit=r.unit))
    return sc.values(r[covered].mean().data)


def test_normalized_by_must_be_a_known_kind_of_reference():
    with pytest.raises(ValueError, match='normalized_by'):
        amor.AmorWorkflow(normalized_by='monitor')


def test_default_reference_is_a_supermirror(supermirror_workflow: sciline.Pipeline):
    corrections = supermirror_workflow.compute(orso.OrsoCorrectionList)
    assert 'supermirror calibration' in corrections


def test_loads_instrument_settings(direct_beam_workflow: sciline.Pipeline):
    results = direct_beam_workflow.compute(
        (
            RawSampleRotation[SampleRun],
            DetectorRotation[SampleRun],
            DetectorRotation[ReferenceRun],
            ChopperFrequency[SampleRun],
            ChopperDistance[SampleRun],
            ChopperSeparation[SampleRun],
        )
    )
    assert_identical(results[RawSampleRotation[SampleRun]], sc.scalar(0.5, unit='deg'))
    assert_identical(results[DetectorRotation[SampleRun]], sc.scalar(1.5, unit='deg'))
    assert_identical(
        results[DetectorRotation[ReferenceRun]], sc.scalar(0.0, unit='deg')
    )
    assert_allclose(results[ChopperFrequency[SampleRun]], sc.scalar(7.5, unit='Hz'))
    assert_allclose(results[ChopperDistance[SampleRun]], sc.scalar(-14842.0, unit='mm'))
    assert_allclose(
        results[ChopperSeparation[SampleRun]],
        sc.scalar(1000.0, unit='mm', dtype='float32'),
    )


def test_sample_rotation_offset(direct_beam_workflow: sciline.Pipeline):
    direct_beam_workflow[SampleRotationOffset[SampleRun]] = sc.scalar(0.1, unit='deg')
    assert_allclose(
        direct_beam_workflow.compute(SampleRotation[SampleRun]),
        sc.scalar(0.6, unit='deg'),
    )


@pytest.mark.parametrize('pulse', [0, 1])
def test_wavelength_is_recovered_from_event_time(pulse: int):
    chopper_frequency = sc.scalar(7.5, unit='Hz')
    chopper_phase_offset = sc.scalar(-5.0, unit='deg')
    pixel_divergence_angle = sc.array(dims=['pixel'], values=[-0.01, 0.0, 0.01, 0.0])
    pixel_divergence_angle.unit = 'rad'
    L1 = sc.scalar(14842.0, unit='mm')
    L2 = sc.array(dims=['pixel'], values=[4000.0, 4010.0, 4100.0, 4124.0], unit='mm')
    # The frame is slightly shorter than 14 angstrom
    true_wavelength = sc.array(
        dims=['pixel'], values=[0.01, 3.0, 9.0, 13.5], unit='angstrom'
    )

    half_period = (1 / (2 * chopper_frequency)).to(unit='ns')
    time_of_flight = (
        true_wavelength * (L1 + L2) * sc.constants.m_n / sc.constants.h
    ).to(unit='ns')
    # Time between the trigger and the opening of the chopper
    delay = half_period * (
        (2 * chopper_phase_offset - pixel_divergence_angle.to(unit='deg'))
        / sc.scalar(180.0, unit='deg')
    )
    # Events are recorded relative to the trigger, once per revolution
    event_time_offset = (time_of_flight - delay + pulse * half_period) % (
        2 * half_period
    )

    result = wavelength(
        event_time_offset=event_time_offset,
        pixel_divergence_angle=pixel_divergence_angle,
        L1=L1,
        L2=L2,
        chopper_frequency=chopper_frequency,
        chopper_phase_offset=chopper_phase_offset,
    )
    assert_allclose(result, true_wavelength)


@pytest.mark.parametrize('run_type', [SampleRun, ReferenceRun])
def test_wavelength_is_independent_of_histogram_edges(
    direct_beam_workflow: sciline.Pipeline,
    run_type: type,
):
    events = direct_beam_workflow.compute(ReducibleData[run_type])
    wavelengths = events.bins.coords['wavelength'].bins.constituents['data']
    direct_beam_workflow[WavelengthBins] = sc.linspace(
        'wavelength', 4.0, 12.5, 20, unit='angstrom'
    )
    rebinned = direct_beam_workflow.compute(ReducibleData[run_type])
    assert_identical(
        rebinned.bins.coords['wavelength'].bins.constituents['data'], wavelengths
    )
    assert wavelengths.min() >= sc.scalar(0.0, unit='angstrom')
    assert wavelengths.max() < sc.scalar(14.0, unit='angstrom')


@pytest.mark.parametrize(('low', 'high'), [(-1.0, 12.0), (3.0, 14.0)])
def test_wavelength_range_must_fit_first_frame(direct_beam_workflow, low, high):
    direct_beam_workflow[WavelengthBins] = sc.linspace(
        'wavelength', low, high, 20, unit='angstrom'
    )
    with pytest.raises(ValueError, match='first chopper frame'):
        direct_beam_workflow.compute(ReducibleData[SampleRun])


def test_wavelength_range_is_checked_at_run_frequency(direct_beam_workflow):
    direct_beam_workflow[ChopperFrequency[SampleRun]] = sc.scalar(15.0, unit='Hz')
    with pytest.raises(ValueError, match='first chopper frame'):
        direct_beam_workflow.compute(ReducibleData[SampleRun])


def test_proton_charge_is_current_times_duration_of_run(
    direct_beam_workflow: sciline.Pipeline,
):
    charge = direct_beam_workflow.compute(ProtonCharge[SampleRun])
    assert charge.sizes == {'time': 150}
    assert_allclose(
        charge.sum().data, sc.scalar(150 * PROTON_CURRENT_UA / 7.5, unit='uC')
    )


def test_revolutions_with_low_proton_current_are_discarded(
    direct_beam_workflow: sciline.Pipeline, make_amor_file
):
    direct_beam_workflow[Filename[SampleRun]] = make_amor_file(
        mu=0.5,
        nu=1.5,
        blades=UPPER_BLADES,
        n_events=300_000,
        n_pulses=150,
        low_current_pulses=50,
        seed=3,
    )
    results = direct_beam_workflow.compute(
        (ProtonCharge[SampleRun], ReducibleData[SampleRun], ReflectivityOverQ)
    )
    charge = results[ProtonCharge[SampleRun]]
    assert charge.masks['proton_current_too_low'].sum().value == 50
    assert_allclose(
        charge.sum().data, sc.scalar(100 * PROTON_CURRENT_UA / 7.5, unit='uC')
    )

    events = results[ReducibleData[SampleRun]]
    discarded = events.bins.masks['proton_current_too_low'].bins.constituents['data']
    assert discarded.sum().value / discarded.sizes['event'] == pytest.approx(
        50 / 150, rel=0.02
    )
    # The remaining revolutions give the same reflectivity
    assert_allclose(
        _mean_reflectivity(results[ReflectivityOverQ]),
        sc.scalar(REFLECTIVITY),
        rtol=sc.scalar(0.03),
    )


def test_direct_beam_is_mirrored(direct_beam_workflow: sciline.Pipeline):
    events = direct_beam_workflow.compute(ReducibleData[ReferenceRun])
    counts = events.bins.size().sum(('wire', 'strip')).data
    assert counts['blade', :7].min().value > 0
    assert counts['blade', 7:].max().value == 0

    reference = direct_beam_workflow.compute(ReducedReference)
    intensity = reference.sum(('wire', 'wavelength')).data
    assert intensity['blade', :7].max().value == 0
    assert intensity['blade', 7:].min().value > 0
    assert_identical(reference.coords['blade'], events.coords['blade'])


def test_direct_beam_normalization_gives_simulated_reflectivity(
    direct_beam_workflow: sciline.Pipeline,
):
    reflectivity = direct_beam_workflow.compute(ReflectivityOverQ)
    assert 'Q_resolution' in reflectivity.coords
    assert_allclose(
        _mean_reflectivity(reflectivity), sc.scalar(REFLECTIVITY), rtol=sc.scalar(0.03)
    )


def test_footprint_correction_is_applied_to_sample_only(
    direct_beam_workflow: sciline.Pipeline,
):
    narrow_beam = _mean_reflectivity(direct_beam_workflow.compute(ReflectivityOverQ))
    # At about 1 degree the footprint of a 2 mm beam is about 115 mm, the sample 50 mm
    direct_beam_workflow[BeamSize[SampleRun]] = sc.scalar(2.0, unit='mm')
    wide_beam = _mean_reflectivity(direct_beam_workflow.compute(ReflectivityOverQ))
    assert wide_beam.value > 1.5 * narrow_beam.value


def test_resolution_is_not_affected_by_switching_off_footprint_correction(
    direct_beam_workflow: sciline.Pipeline,
):
    reflectivity = direct_beam_workflow.compute(ReflectivityOverQ)
    relative = reflectivity.coords['Q_resolution'] / sc.midpoints(
        reflectivity.coords['Q']
    )
    covered = relative[sc.isfinite(relative)]
    assert covered.sizes['Q'] > 5
    assert covered.max() < sc.scalar(0.1)


def test_supermirror_normalization_of_run_by_itself_gives_unity(
    supermirror_workflow: sciline.Pipeline,
):
    reflectivity = supermirror_workflow.compute(ReflectivityOverQ)
    assert 'Q_resolution' in reflectivity.coords
    assert_allclose(
        _mean_reflectivity(reflectivity), sc.scalar(1.0), rtol=sc.scalar(0.03)
    )


@pytest.mark.parametrize('run_type', [SampleRun, ReferenceRun])
@pytest.mark.parametrize('mode', ['direct_beam', 'direct_beam_with_aperture'])
def test_reflectivity_is_unchanged_when_a_run_is_added_to_itself(
    direct_beam_file: Path, sample_file: Path, run_type: type, mode: str
):
    pipeline = _set_parameters(amor.AmorWorkflow(normalized_by=mode))
    pipeline[Filename[SampleRun]] = sample_file
    pipeline[Filename[ReferenceRun]] = direct_beam_file
    # Model an aperture that accepts the whole narrow synthetic beam.
    pipeline[SampleRotation[ReferenceRun]] = sc.scalar(1.5, unit='deg')
    pipeline[BeamSize[ReferenceRun]] = sc.scalar(1.0, unit='nm')
    expected = pipeline.compute(ReflectivityOverQ).hist()
    filename = pipeline.compute(Filename[run_type])
    workflow = with_filenames(pipeline, run_type, [filename, filename])
    result = workflow.compute(ReflectivityOverQ).hist()
    assert_allclose(
        sc.values(result.data), sc.values(expected.data), rtol=sc.scalar(1e-9)
    )


def test_orso_dataset_with_direct_beam(direct_beam_workflow: sciline.Pipeline):
    dataset = direct_beam_workflow.compute(orso.OrsoIofQDataset)
    assert dataset.info.data_source.experiment.instrument == 'Amor'
    assert dataset.info.data_source.sample.name == 'synthetic'
    assert dataset.info.reduction.corrections == [
        'chopper ToF correction',
        'footprint correction',
        'direct beam normalization',
    ]
    assert dataset.data.shape[1] == 4
