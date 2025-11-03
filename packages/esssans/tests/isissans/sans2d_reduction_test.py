# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from functools import lru_cache

import pytest
import sciline
import scipp as sc

import ess.isissans.data  # noqa: F401
from ess import isissans as isis
from ess import sans
from ess.isissans import MonitorOffset, SampleOffset, sans2d
from ess.sans.types import (
    BackgroundRun,
    BackgroundSubtractedIofQ,
    BeamCenter,
    CorrectForGravity,
    DimsToKeep,
    DirectBeam,
    DirectBeamFilename,
    EmptyBeamRun,
    EmptyDetector,
    Filename,
    Incident,
    IntensityQ,
    MaskedData,
    NeXusMonitorName,
    NonBackgroundWavelengthRange,
    QBins,
    RawDetector,
    ReturnEvents,
    SampleRun,
    SolidAngle,
    Transmission,
    UncertaintyBroadcastMode,
    WavelengthBands,
    WavelengthBins,
    WavelengthMask,
)


def make_params() -> dict:
    params = isis.default_parameters()
    params[WavelengthBins] = sc.linspace(
        'wavelength', start=2.0, stop=16.0, num=141, unit='angstrom'
    )
    params[WavelengthMask] = sc.DataArray(
        data=sc.array(dims=['wavelength'], values=[True]),
        coords={
            'wavelength': sc.array(
                dims=['wavelength'], values=[2.21, 2.59], unit='angstrom'
            )
        },
    )
    params[sans2d.LowCountThreshold] = sc.scalar(100.0, unit='counts')

    params[QBins] = sc.linspace(
        dim='Q', start=0.01, stop=0.55, num=141, unit='1/angstrom'
    )
    params[DirectBeamFilename] = isis.data.sans2d_tutorial_direct_beam()
    params[Filename[SampleRun]] = isis.data.sans2d_tutorial_sample_run()
    params[Filename[BackgroundRun]] = isis.data.sans2d_tutorial_background_run()
    params[Filename[EmptyBeamRun]] = isis.data.sans2d_tutorial_empty_beam_run()

    params[NeXusMonitorName[Incident]] = 'monitor2'
    params[NeXusMonitorName[Transmission]] = 'monitor4'
    params[SampleOffset] = sc.vector([0.0, 0.0, 0.053], unit='m')
    params[MonitorOffset[Transmission]] = sc.vector([0.0, 0.0, -6.719], unit='m')

    params[NonBackgroundWavelengthRange] = sc.array(
        dims=['wavelength'], values=[0.7, 17.1], unit='angstrom'
    )
    params[CorrectForGravity] = True
    params[UncertaintyBroadcastMode] = UncertaintyBroadcastMode.upper_bound
    params[ReturnEvents] = False
    params[DimsToKeep] = ()
    params[BeamCenter] = sc.vector([0, 0, 0], unit='m')

    return params


# Especially the I(Q) beam-cetner finder calls this many times, so we cache it for
# faster test runtime.
cached_load = lru_cache(maxsize=None)(isis.io.load_tutorial_run)


# Wrapper adding type-hints back to the cached function
def cached_load_tutorial_run(
    filename: Filename[SampleRun],
) -> isis.io.LoadedFileContents[SampleRun]:
    return cached_load(filename)


@pytest.fixture
def pipeline():
    wf = isis.sans2d.Sans2dTutorialWorkflow()
    wf.insert(isis.io.transmission_from_background_run)
    wf.insert(isis.io.transmission_from_sample_run)
    wf.insert(cached_load_tutorial_run)
    for key, param in make_params().items():
        wf[key] = param
    return wf


def test_can_create_pipeline(pipeline):
    pipeline.get(IntensityQ[SampleRun])


@pytest.mark.parametrize(
    'uncertainties',
    [UncertaintyBroadcastMode.drop, UncertaintyBroadcastMode.upper_bound],
)
def test_pipeline_can_compute_background_subtracted_IofQ(pipeline, uncertainties):
    pipeline[UncertaintyBroadcastMode] = uncertainties
    result = pipeline.compute(BackgroundSubtractedIofQ)
    assert result.dims == ('Q',)


def test_pipeline_can_compute_background_subtracted_IofQ_in_wavelength_bands(pipeline):
    pipeline[WavelengthBands] = sc.linspace(
        'wavelength', start=2.0, stop=16.0, num=11, unit='angstrom'
    )
    result = pipeline.compute(BackgroundSubtractedIofQ)
    assert result.dims == ('band', 'Q')
    assert result.sizes['band'] == 10


def test_pipeline_wavelength_bands_is_optional(pipeline):
    pipeline[BeamCenter] = sans.beam_center_from_center_of_mass(pipeline)
    noband = pipeline.compute(BackgroundSubtractedIofQ)
    assert pipeline.compute(WavelengthBands) is None
    band = sc.linspace('wavelength', 2.0, 16.0, num=2, unit='angstrom')
    pipeline[WavelengthBands] = band
    assert sc.identical(band, pipeline.compute(WavelengthBands))
    withband = pipeline.compute(BackgroundSubtractedIofQ)
    assert sc.identical(noband, withband)


def test_workflow_is_deterministic(pipeline):
    pipeline[UncertaintyBroadcastMode] = UncertaintyBroadcastMode.drop
    pipeline[BeamCenter] = sans.beam_center_from_center_of_mass(pipeline)
    # This is Sciline's default scheduler, but we want to be explicit here
    scheduler = sciline.scheduler.DaskScheduler()
    graph = pipeline.get(IntensityQ[SampleRun], scheduler=scheduler)
    reference = graph.compute().data
    result = graph.compute().data
    assert sc.identical(sc.values(result), sc.values(reference))


def test_pipeline_raises_VariancesError_if_normalization_errors_not_dropped(pipeline):
    pipeline[NonBackgroundWavelengthRange] = (
        None  # Make sure we raise in iofq_denominator
    )
    pipeline[UncertaintyBroadcastMode] = UncertaintyBroadcastMode.fail
    with pytest.raises(sc.VariancesError):
        pipeline.compute(BackgroundSubtractedIofQ)


def test_uncertainty_broadcast_mode_drop_yields_smaller_variances(pipeline):
    # Errors with the full range have some NaNs or infs
    pipeline[QBins] = sc.linspace(
        dim='Q', start=0.01, stop=0.5, num=141, unit='1/angstrom'
    )
    pipeline[UncertaintyBroadcastMode] = UncertaintyBroadcastMode.drop
    drop = pipeline.compute(IntensityQ[SampleRun]).data
    pipeline[UncertaintyBroadcastMode] = UncertaintyBroadcastMode.upper_bound
    upper_bound = pipeline.compute(IntensityQ[SampleRun]).data
    assert sc.all(sc.variances(drop) < sc.variances(upper_bound)).value


def test_pipeline_can_visualize_background_subtracted_IofQ(pipeline):
    pipeline.visualize(BackgroundSubtractedIofQ)


def test_pipeline_can_compute_intermediate_results(pipeline):
    result = pipeline.compute(SolidAngle[SampleRun])
    assert result.dims == ('spectrum',)


def pixel_dependent_direct_beam(
    filename: DirectBeamFilename, shape: EmptyDetector[SampleRun]
) -> DirectBeam:
    direct_beam = isis.io.load_tutorial_direct_beam(filename)
    sizes = {'spectrum': shape.sizes['spectrum'], **direct_beam.sizes}
    return DirectBeam(direct_beam.broadcast(sizes=sizes).copy())


@pytest.mark.parametrize(
    'uncertainties',
    [UncertaintyBroadcastMode.drop, UncertaintyBroadcastMode.upper_bound],
)
def test_pixel_dependent_direct_beam_is_supported(pipeline, uncertainties):
    pipeline[UncertaintyBroadcastMode] = uncertainties
    pipeline.insert(pixel_dependent_direct_beam)
    pipeline[BeamCenter] = sc.vector([0, 0, 0], unit='m')
    result = pipeline.compute(BackgroundSubtractedIofQ)
    assert result.dims == ('Q',)


MANTID_BEAM_CENTER = sc.vector([0.09288, -0.08195, 0], unit='m')


def test_beam_center_from_center_of_mass_is_close_to_verified_result(pipeline):
    center = sans.beam_center_from_center_of_mass(pipeline)
    # This is the result obtained from Mantid, using the full IofQ
    # calculation. The difference is about 3 mm in X or Y, probably due to a bias
    # introduced by the sample holder, which the center-of-mass approach cannot ignore.
    assert sc.allclose(center, MANTID_BEAM_CENTER, atol=sc.scalar(3e-3, unit='m'))


def test_beam_center_from_center_of_mass_independent_of_set_beam_center(pipeline):
    pipeline[BeamCenter] = sc.vector([0.1, -0.1, 0], unit='m')
    center = sans.beam_center_from_center_of_mass(pipeline)
    assert sc.allclose(center, MANTID_BEAM_CENTER, atol=sc.scalar(3e-3, unit='m'))


def test_beam_center_finder_without_direct_beam_reproduces_verified_result(pipeline):
    pipeline[DirectBeam] = None
    center = sans.beam_center_finder.beam_center_from_iofq(
        workflow=pipeline, q_bins=sc.linspace('Q', 0.02, 0.3, 71, unit='1/angstrom')
    )
    assert sc.allclose(center, MANTID_BEAM_CENTER, atol=sc.scalar(2e-3, unit='m'))


def test_beam_center_can_get_closer_to_verified_result_with_low_counts_mask(pipeline):
    def low_counts_mask(
        sample: RawDetector[SampleRun],
        low_counts_threshold: sans2d.LowCountThreshold,
    ) -> sans2d.SampleHolderMask:
        return sans2d.SampleHolderMask(sample.hist().data < low_counts_threshold)

    pipeline[sans2d.LowCountThreshold] = sc.scalar(80.0, unit='counts')
    pipeline.insert(low_counts_mask)  # replaces sans2d.sample_holder_mask
    pipeline[DirectBeam] = None
    q_bins = sc.linspace('Q', 0.02, 0.3, 71, unit='1/angstrom')
    center = sans.beam_center_finder.beam_center_from_iofq(
        workflow=pipeline, q_bins=q_bins
    )
    assert sc.allclose(center, MANTID_BEAM_CENTER, atol=sc.scalar(5e-4, unit='m'))


def test_beam_center_finder_works_with_direct_beam(pipeline):
    q_bins = sc.linspace('Q', 0.02, 0.3, 71, unit='1/angstrom')
    center_with_direct_beam = sans.beam_center_finder.beam_center_from_iofq(
        workflow=pipeline, q_bins=q_bins
    )
    assert sc.allclose(
        center_with_direct_beam, MANTID_BEAM_CENTER, atol=sc.scalar(2e-3, unit='m')
    )


def test_beam_center_finder_independent_of_set_beam_center(pipeline):
    pipeline[BeamCenter] = sc.vector([0.1, -0.1, 0], unit='m')
    q_bins = sc.linspace('Q', 0.02, 0.3, 71, unit='1/angstrom')
    center_with_direct_beam = sans.beam_center_finder.beam_center_from_iofq(
        workflow=pipeline, q_bins=q_bins
    )
    assert sc.allclose(
        center_with_direct_beam, MANTID_BEAM_CENTER, atol=sc.scalar(2e-3, unit='m')
    )


def test_beam_center_finder_works_with_pixel_dependent_direct_beam(pipeline):
    q_bins = sc.linspace('Q', 0.02, 0.3, 71, unit='1/angstrom')
    center_pixel_independent_direct_beam = (
        sans.beam_center_finder.beam_center_from_iofq(workflow=pipeline, q_bins=q_bins)
    )

    direct_beam = pipeline.compute(DirectBeam)
    pixel_dependent_direct_beam = direct_beam.broadcast(
        sizes={
            'spectrum': pipeline.compute(MaskedData[SampleRun]).sizes['spectrum'],
            'wavelength': direct_beam.sizes['wavelength'],
        }
    ).copy()

    pipeline[DirectBeam] = pixel_dependent_direct_beam

    center = sans.beam_center_finder.beam_center_from_iofq(
        workflow=pipeline, q_bins=q_bins
    )
    assert sc.identical(center, center_pixel_independent_direct_beam)


def test_workflow_runs_without_gravity_if_beam_center_is_provided(pipeline):
    pipeline[CorrectForGravity] = False
    da = pipeline.compute(RawDetector[SampleRun])
    del da.coords['gravity']
    pipeline[RawDetector[SampleRun]] = da
    pipeline[BeamCenter] = MANTID_BEAM_CENTER
    result = pipeline.compute(BackgroundSubtractedIofQ)
    assert result.dims == ('Q',)
