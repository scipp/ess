# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
import sys
from pathlib import Path

import pytest
import sciline
import scipp as sc

from ess import sans
from ess.sans.conversions import ElasticCoordTransformGraph
from ess.sans.types import (
    BackgroundRun,
    BackgroundSubtractedIofQ,
    BeamCenter,
    CalibratedMaskedData,
    CleanWavelengthMasked,
    CorrectForGravity,
    Denominator,
    DimsToKeep,
    Filename,
    FilenameType,
    FilePath,
    FinalSummedQ,
    IofQ,
    NeXusDetectorName,
    Numerator,
    PixelMaskFilename,
    QBins,
    QxyBins,
    ReturnEvents,
    SampleRun,
    UncertaintyBroadcastMode,
    WavelengthBands,
    WavelengthBins,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    loki_providers,
    loki_providers_no_beam_center_finder,
    make_params,
)


@pytest.mark.parametrize('qxy', [False, True])
def test_can_create_pipeline(qxy: bool):
    pipeline = sciline.Pipeline(loki_providers(), params=make_params(qxy=qxy))
    pipeline.set_param_series(PixelMaskFilename, ['mask_new_July2022.xml'])
    pipeline.get(BackgroundSubtractedIofQ)


@pytest.mark.parametrize(
    'uncertainties',
    [UncertaintyBroadcastMode.drop, UncertaintyBroadcastMode.upper_bound],
)
@pytest.mark.parametrize('qxy', [False, True])
def test_pipeline_can_compute_IofQ(uncertainties, qxy: bool):
    params = make_params(qxy=qxy)
    params[UncertaintyBroadcastMode] = uncertainties
    pipeline = sciline.Pipeline(loki_providers(), params=params)
    pipeline.set_param_series(PixelMaskFilename, ['mask_new_July2022.xml'])
    result = pipeline.compute(BackgroundSubtractedIofQ)
    assert result.dims == ('Qy', 'Qx') if qxy else ('Q',)
    if qxy:
        assert sc.identical(result.coords['Qx'], params[QxyBins]['Qx'])
        assert sc.identical(result.coords['Qy'], params[QxyBins]['Qy'])
        assert result.sizes['Qx'] == 90
        assert result.sizes['Qy'] == 77
    else:
        assert sc.identical(result.coords['Q'], params[QBins])
        assert result.sizes['Q'] == 100


@pytest.mark.parametrize(
    'uncertainties',
    [UncertaintyBroadcastMode.drop, UncertaintyBroadcastMode.upper_bound],
)
@pytest.mark.parametrize('target', [IofQ[SampleRun], BackgroundSubtractedIofQ])
@pytest.mark.parametrize('qxy', [False, True])
def test_pipeline_can_compute_IofQ_in_event_mode(uncertainties, target, qxy: bool):
    params = make_params(qxy=qxy)
    params[UncertaintyBroadcastMode] = uncertainties
    pipeline = sciline.Pipeline(loki_providers(), params=params)
    pipeline.set_param_series(PixelMaskFilename, ['mask_new_July2022.xml'])
    reference = pipeline.compute(target)
    pipeline[ReturnEvents] = True
    result = pipeline.compute(target)
    assert result.bins is not None
    assert result.dims == ('Qy', 'Qx') if qxy else ('Q',)
    assert sc.allclose(
        sc.values(reference.data),
        sc.values(result.hist().data),
        # Could be 1e-11, but currently the workflow defaults to float32 data, as
        # returned by ScippNexus.
        rtol=sc.scalar(1e-7),
        atol=sc.scalar(1e-7),
    )
    if uncertainties == UncertaintyBroadcastMode.drop:
        # Could both be 1e-14 if using float64
        tol = sc.scalar(1e-7 if qxy else 1e-10)
    else:
        # Could be 1e-8 if using float64
        tol = sc.scalar(1e-7 if qxy else 1e-9)
    assert sc.allclose(
        sc.variances(reference).data,
        sc.variances(result.hist()).data,
        rtol=tol,
        atol=tol,
    )


@pytest.mark.parametrize('qxy', [False, True])
def test_pipeline_can_compute_IofQ_in_wavelength_bands(qxy: bool):
    params = make_params(qxy=qxy)
    params[WavelengthBands] = sc.linspace(
        'wavelength',
        params[WavelengthBins].min(),
        params[WavelengthBins].max(),
        11,
    )
    pipeline = sciline.Pipeline(loki_providers(), params=params)
    pipeline.set_param_series(PixelMaskFilename, ['mask_new_July2022.xml'])
    result = pipeline.compute(BackgroundSubtractedIofQ)
    assert result.dims == ('band', 'Qy', 'Qx') if qxy else ('band', 'Q')
    assert result.sizes['band'] == 10


@pytest.mark.parametrize('qxy', [False, True])
def test_pipeline_can_compute_IofQ_in_overlapping_wavelength_bands(qxy: bool):
    params = make_params(qxy=qxy)
    # Bands have double the width
    edges = sc.linspace(
        'band', params[WavelengthBins].min(), params[WavelengthBins].max(), 12
    )
    params[WavelengthBands] = sc.concat(
        [edges[:-2], edges[2::]], dim='wavelength'
    ).transpose()
    pipeline = sciline.Pipeline(loki_providers(), params=params)
    pipeline.set_param_series(PixelMaskFilename, ['mask_new_July2022.xml'])
    result = pipeline.compute(BackgroundSubtractedIofQ)
    assert result.dims == ('band', 'Qy', 'Qx') if qxy else ('band', 'Q')
    assert result.sizes['band'] == 10


@pytest.mark.parametrize('qxy', [False, True])
def test_pipeline_can_compute_IofQ_in_layers(qxy: bool):
    params = make_params(qxy=qxy)
    params[DimsToKeep] = ['layer']
    pipeline = sciline.Pipeline(loki_providers(), params=params)
    pipeline.set_param_series(PixelMaskFilename, ['mask_new_July2022.xml'])
    result = pipeline.compute(BackgroundSubtractedIofQ)
    assert result.dims == ('layer', 'Qy', 'Qx') if qxy else ('layer', 'Q')
    assert result.sizes['layer'] == 4


def _compute_beam_center():
    pipeline = sciline.Pipeline(loki_providers(), params=make_params())
    pipeline.set_param_series(PixelMaskFilename, ['mask_new_July2022.xml'])
    center = pipeline.compute(BeamCenter)
    return center


def test_pipeline_can_compute_IofQ_merging_events_from_multiple_runs():
    params = make_params()
    del params[Filename[SampleRun]]
    del params[Filename[BackgroundRun]]

    sample_runs = ['60250-2022-02-28_2215.nxs', '60339-2022-02-28_2215.nxs']
    background_runs = ['60248-2022-02-28_2215.nxs', '60393-2022-02-28_2215.nxs']
    pipeline = sciline.Pipeline(loki_providers_no_beam_center_finder(), params=params)
    pipeline[BeamCenter] = _compute_beam_center()
    pipeline.set_param_series(PixelMaskFilename, ['mask_new_July2022.xml'])

    # Set parameter series for file names
    pipeline.set_param_series(Filename[SampleRun], sample_runs)
    pipeline.set_param_series(Filename[BackgroundRun], background_runs)

    pipeline.insert(sans.merge_runs)
    result = pipeline.compute(BackgroundSubtractedIofQ)
    assert result.dims == ('Q',)


def test_pipeline_can_compute_IofQ_merging_events_from_banks():
    params = make_params()
    del params[NeXusDetectorName]

    pipeline = sciline.Pipeline(loki_providers_no_beam_center_finder(), params=params)
    pipeline[BeamCenter] = _compute_beam_center()
    pipeline.set_param_series(PixelMaskFilename, ['mask_new_July2022.xml'])
    pipeline.set_param_series(NeXusDetectorName, ['larmor_detector'])

    pipeline.insert(sans.merge_banks)
    result = pipeline.compute(BackgroundSubtractedIofQ)
    assert result.dims == ('Q',)


def test_pipeline_can_compute_IofQ_merging_events_from_multiple_runs_and_banks():
    params = make_params()
    del params[Filename[SampleRun]]
    del params[Filename[BackgroundRun]]

    sample_runs = ['60250-2022-02-28_2215.nxs', '60339-2022-02-28_2215.nxs']
    background_runs = ['60248-2022-02-28_2215.nxs', '60393-2022-02-28_2215.nxs']
    pipeline = sciline.Pipeline(loki_providers_no_beam_center_finder(), params=params)
    pipeline[BeamCenter] = _compute_beam_center()
    pipeline.set_param_series(PixelMaskFilename, ['mask_new_July2022.xml'])

    pipeline.insert(sans.merge_runs)
    pipeline.set_param_series(Filename[SampleRun], sample_runs)
    pipeline.set_param_series(Filename[BackgroundRun], background_runs)
    reference = pipeline.compute(BackgroundSubtractedIofQ)

    pipeline.insert(sans.merge_banks)
    del params[NeXusDetectorName]
    pipeline.set_param_series(NeXusDetectorName, ['larmor_detector'])
    result = pipeline.compute(BackgroundSubtractedIofQ)

    assert sc.identical(result, reference)


def test_pipeline_IofQ_merging_events_yields_consistent_results():
    N = 3
    params = make_params()
    center = _compute_beam_center()
    pipeline_single = sciline.Pipeline(
        loki_providers_no_beam_center_finder(), params=params
    )
    pipeline_single[BeamCenter] = center
    pipeline_single.set_param_series(PixelMaskFilename, ['mask_new_July2022.xml'])

    del params[Filename[SampleRun]]
    del params[Filename[BackgroundRun]]
    pipeline_triple = sciline.Pipeline(
        loki_providers_no_beam_center_finder(), params=params
    )
    pipeline_triple[BeamCenter] = center
    pipeline_triple.set_param_series(PixelMaskFilename, ['mask_new_July2022.xml'])

    # `set_param_series` does not allow multiple identical values, so we need to
    # map the file names to different ones.
    def get_mapped_path(filename: FilenameType) -> FilePath[FilenameType]:
        """Mapping file paths to allow loading same run multiple times."""
        from ess.loki.data import get_path

        mapping = {
            'sample_0.nxs': '60339-2022-02-28_2215.nxs',
            'sample_1.nxs': '60339-2022-02-28_2215.nxs',
            'sample_2.nxs': '60339-2022-02-28_2215.nxs',
            'background_0.nxs': '60393-2022-02-28_2215.nxs',
            'background_1.nxs': '60393-2022-02-28_2215.nxs',
            'background_2.nxs': '60393-2022-02-28_2215.nxs',
        }
        filename = mapping.get(filename, filename)
        return FilePath[FilenameType](get_path(filename))

    pipeline_triple.insert(get_mapped_path)

    pipeline_triple.set_param_series(
        Filename[SampleRun], [f'sample_{i}.nxs' for i in range(N)]
    )
    pipeline_triple.set_param_series(
        Filename[BackgroundRun], [f'background_{i}.nxs' for i in range(N)]
    )
    # Add event merging provider
    pipeline_triple.insert(sans.merge_runs)

    iofq1 = pipeline_single.compute(BackgroundSubtractedIofQ)
    iofq3 = pipeline_triple.compute(BackgroundSubtractedIofQ)
    assert sc.allclose(sc.values(iofq1.data), sc.values(iofq3.data))
    assert sc.identical(iofq1.coords['Q'], iofq3.coords['Q'])
    assert all(sc.variances(iofq1.data) > sc.variances(iofq3.data))
    assert sc.allclose(
        sc.values(
            pipeline_single.compute(FinalSummedQ[SampleRun, Numerator]).hist().data
        )
        * N,
        sc.values(
            pipeline_triple.compute(FinalSummedQ[SampleRun, Numerator]).hist().data
        ),
    )
    assert sc.allclose(
        sc.values(pipeline_single.compute(FinalSummedQ[SampleRun, Denominator]).data)
        * N,
        sc.values(pipeline_triple.compute(FinalSummedQ[SampleRun, Denominator]).data),
    )


def test_beam_center_from_center_of_mass_is_close_to_verified_result():
    params = make_params()
    pipeline = sciline.Pipeline(loki_providers(), params=params)
    pipeline.set_param_series(PixelMaskFilename, ['mask_new_July2022.xml'])
    center = pipeline.compute(BeamCenter)
    reference = sc.vector([-0.0291487, -0.0181614, 0], unit='m')
    assert sc.allclose(center, reference)


def test_phi_with_gravity():
    params = make_params()
    pipeline = sciline.Pipeline(loki_providers(), params=params)
    pipeline.set_param_series(PixelMaskFilename, ['mask_new_July2022.xml'])
    pipeline[CorrectForGravity] = False
    data_no_grav = pipeline.compute(
        CleanWavelengthMasked[SampleRun, Numerator]
    ).flatten(to='pixel')
    graph_no_grav = pipeline.compute(ElasticCoordTransformGraph)
    pipeline[CorrectForGravity] = True
    data_with_grav = (
        pipeline.compute(CleanWavelengthMasked[SampleRun, Numerator])
        .flatten(to='pixel')
        .hist(wavelength=sc.linspace('wavelength', 1.0, 12.0, 101, unit='angstrom'))
    )
    graph_with_grav = pipeline.compute(ElasticCoordTransformGraph)

    no_grav = data_no_grav.transform_coords(('two_theta', 'phi'), graph_no_grav)
    phi_no_grav = no_grav.coords['phi']
    with_grav = data_with_grav.transform_coords(('two_theta', 'phi'), graph_with_grav)
    phi_with_grav = with_grav.coords['phi'].mean('wavelength')

    assert not sc.identical(phi_no_grav, phi_with_grav)

    # Exclude pixels near y=0, since phi with gravity could drop below y=0 and give a
    # difference of almost 2*pi.
    y = sc.abs(
        pipeline.compute(CalibratedMaskedData[SampleRun])
        .coords['position']
        .fields.y.flatten(to='pixel')
    )
    not_near_origin = y > sc.scalar(0.05, unit='m')
    phi_no_grav = phi_no_grav[not_near_origin]
    phi_with_grav = phi_with_grav[not_near_origin]
    assert sc.all(
        sc.isclose(phi_no_grav, phi_with_grav, atol=sc.scalar(5.0e-3, unit='rad'))
    )

    # Phi is in [-pi, pi], measured from the X axis.
    pos_x = sc.abs(phi_no_grav) < sc.scalar(90.0, unit='deg').to(unit='rad')
    # Phi is larger with gravity, since it gives the position where it would have
    # been detected without gravity. That is, with gravity all points are pulled
    # "up" in the XY plane, so the angle is larger for positive X and smaller for
    # negative X.
    assert sc.all(phi_no_grav[pos_x] < phi_with_grav[pos_x])
    assert sc.all(phi_no_grav[~pos_x] > phi_with_grav[~pos_x])
