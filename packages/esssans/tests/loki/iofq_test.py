# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
import os
import sys
from pathlib import Path

import pytest
import sciline
import scipp as sc
from scipp.testing import assert_identical

import ess.loki.data  # noqa: F401
from ess import loki, sans
from ess.sans.conversions import ElasticCoordTransformGraph
from ess.sans.types import (
    BackgroundRun,
    BackgroundSubtractedIofQ,
    BackgroundSubtractedIofQxy,
    BeamCenter,
    CorrectedDetector,
    CorrectForGravity,
    Denominator,
    DimsToKeep,
    Filename,
    IntensityQ,
    IntensityQxQy,
    NormalizedQ,
    Numerator,
    QBins,
    QxBins,
    QyBins,
    ReturnEvents,
    SampleRun,
    UncertaintyBroadcastMode,
    WavelengthBands,
    WavelengthBins,
    WavelengthDetector,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import make_workflow


def test_can_create_pipeline():
    pipeline = make_workflow()
    pipeline[BeamCenter] = sc.vector([0, 0, 0], unit='m')
    pipeline.get(BackgroundSubtractedIofQ)


def test_can_create_pipeline_with_pixel_masks():
    pipeline = make_workflow(no_masks=False)
    pipeline = sans.with_pixel_mask_filenames(
        pipeline, loki.data.loki_tutorial_mask_filenames()
    )
    pipeline[BeamCenter] = sc.vector([0, 0, 0], unit='m')
    pipeline.get(BackgroundSubtractedIofQ)


@pytest.mark.parametrize(
    'uncertainties',
    [UncertaintyBroadcastMode.drop, UncertaintyBroadcastMode.upper_bound],
)
@pytest.mark.parametrize('qxy', [False, True])
def test_pipeline_can_compute_IofQ(uncertainties, qxy: bool):
    pipeline = make_workflow(no_masks=False)
    pipeline[UncertaintyBroadcastMode] = uncertainties
    pipeline = sans.with_pixel_mask_filenames(
        pipeline, loki.data.loki_tutorial_mask_filenames()
    )
    pipeline[BeamCenter] = sans.beam_center_from_center_of_mass(pipeline)
    if qxy:
        result = pipeline.compute(BackgroundSubtractedIofQxy)
        assert result.dims == ('Qy', 'Qx')
        assert sc.identical(result.coords['Qx'], pipeline.compute(QxBins))
        assert sc.identical(result.coords['Qy'], pipeline.compute(QyBins))
        assert result.sizes['Qx'] == 90
        assert result.sizes['Qy'] == 77
    else:
        result = pipeline.compute(BackgroundSubtractedIofQ)
        assert result.dims == ('Q',)
        assert sc.identical(result.coords['Q'], pipeline.compute(QBins))
        assert result.sizes['Q'] == 100
    if uncertainties == UncertaintyBroadcastMode.drop:
        test_dir = os.path.dirname(os.path.abspath(__file__))
        name = Path(f'reference_IofQ{"xy" if qxy else ""}_{uncertainties}.hdf5')
        reference = sc.io.load_hdf5(test_dir / name)
        assert_identical(result, reference)


@pytest.mark.parametrize(
    'uncertainties',
    [UncertaintyBroadcastMode.drop, UncertaintyBroadcastMode.upper_bound],
)
@pytest.mark.parametrize(
    'target',
    [
        IntensityQ[SampleRun],
        IntensityQxQy[SampleRun],
        BackgroundSubtractedIofQ,
        BackgroundSubtractedIofQxy,
    ],
)
def test_pipeline_can_compute_IofQ_in_event_mode(uncertainties, target):
    pipeline = make_workflow()
    pipeline[UncertaintyBroadcastMode] = uncertainties
    pipeline[BeamCenter] = sans.beam_center_from_center_of_mass(pipeline)
    reference = pipeline.compute(target)
    pipeline[ReturnEvents] = True
    result = pipeline.compute(target)
    qxy = target in (IntensityQxQy[SampleRun], BackgroundSubtractedIofQxy)
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
    pipeline = make_workflow()
    pipeline[WavelengthBands] = sc.linspace(
        'wavelength',
        pipeline.compute(WavelengthBins).min(),
        pipeline.compute(WavelengthBins).max(),
        11,
    )
    pipeline[BeamCenter] = _compute_beam_center()
    result = pipeline.compute(
        BackgroundSubtractedIofQxy if qxy else BackgroundSubtractedIofQ
    )
    assert result.dims == ('band', 'Qy', 'Qx') if qxy else ('band', 'Q')
    assert result.sizes['band'] == 10


@pytest.mark.parametrize('qxy', [False, True])
def test_pipeline_can_compute_IofQ_in_overlapping_wavelength_bands(qxy: bool):
    pipeline = make_workflow()
    # Bands have double the width
    edges = pipeline.compute(WavelengthBins)
    edges = sc.linspace('band', edges.min(), edges.max(), 12)
    pipeline[WavelengthBands] = sc.concat(
        [edges[:-2], edges[2::]], dim='wavelength'
    ).transpose()
    pipeline[BeamCenter] = _compute_beam_center()
    result = pipeline.compute(
        BackgroundSubtractedIofQxy if qxy else BackgroundSubtractedIofQ
    )
    assert result.dims == ('band', 'Qy', 'Qx') if qxy else ('band', 'Q')
    assert result.sizes['band'] == 10


@pytest.mark.parametrize('qxy', [False, True])
def test_pipeline_can_compute_IofQ_in_layers(qxy: bool):
    pipeline = make_workflow()
    pipeline[DimsToKeep] = ['layer']
    pipeline[BeamCenter] = _compute_beam_center()
    result = pipeline.compute(
        BackgroundSubtractedIofQxy if qxy else BackgroundSubtractedIofQ
    )
    assert result.dims == ('layer', 'Qy', 'Qx') if qxy else ('layer', 'Q')
    assert result.sizes['layer'] == 4


def _compute_beam_center():
    return sans.beam_center_from_center_of_mass(make_workflow())


def test_pipeline_can_compute_IofQ_merging_events_from_multiple_runs():
    sample_runs = [
        loki.data.loki_tutorial_sample_run_60250(),
        loki.data.loki_tutorial_sample_run_60339(),
    ]
    background_runs = [
        loki.data.loki_tutorial_background_run_60248(),
        loki.data.loki_tutorial_background_run_60393(),
    ]
    pipeline = make_workflow()
    pipeline[BeamCenter] = _compute_beam_center()

    # Remove previously set runs so we can be sure that below we use the mapped ones
    pipeline[Filename[SampleRun]] = None
    pipeline[Filename[BackgroundRun]] = None
    pipeline = sans.with_sample_runs(pipeline, runs=sample_runs)
    pipeline = sans.with_background_runs(pipeline, runs=background_runs)

    result = pipeline.compute(BackgroundSubtractedIofQ)
    assert result.dims == ('Q',)
    result = pipeline.compute(BackgroundSubtractedIofQxy)
    assert result.dims == ('Qy', 'Qx')


def test_pipeline_can_compute_IofQ_by_bank():
    pipeline = make_workflow()
    pipeline[BeamCenter] = _compute_beam_center()
    pipeline = sans.with_banks(pipeline, banks=['larmor_detector'])

    results = sciline.compute_mapped(pipeline, BackgroundSubtractedIofQ)
    assert results['larmor_detector'].dims == ('Q',)


def test_pipeline_can_compute_IofQ_merging_events_from_multiple_runs_by_bank():
    sample_runs = [
        loki.data.loki_tutorial_sample_run_60250(),
        loki.data.loki_tutorial_sample_run_60339(),
    ]
    background_runs = [
        loki.data.loki_tutorial_background_run_60248(),
        loki.data.loki_tutorial_background_run_60393(),
    ]
    pipeline = make_workflow()
    pipeline[BeamCenter] = _compute_beam_center()

    pipeline = sans.with_sample_runs(pipeline, runs=sample_runs)
    pipeline = sans.with_background_runs(pipeline, runs=background_runs)
    key = BackgroundSubtractedIofQ
    reference = pipeline.compute(key)

    pipeline = sans.with_banks(
        pipeline, banks=['larmor_detector', 'larmor_detector'], index=['bank0', 'bank1']
    )
    results = sciline.compute_mapped(pipeline, key)

    assert_identical(sc.values(results['bank0']), sc.values(reference))
    assert_identical(sc.values(results['bank1']), sc.values(reference))


def test_pipeline_IofQ_merging_events_yields_consistent_results():
    N = 3
    center = _compute_beam_center()
    pipeline_single = make_workflow()
    pipeline_single[BeamCenter] = center

    sample_runs = [loki.data.loki_tutorial_sample_run_60339()] * N
    background_runs = [loki.data.loki_tutorial_background_run_60393()] * N
    pipeline_triple = sans.with_sample_runs(pipeline_single, runs=sample_runs)
    pipeline_triple = sans.with_background_runs(pipeline_triple, runs=background_runs)

    iofq1 = pipeline_single.compute(BackgroundSubtractedIofQ)
    iofq3 = pipeline_triple.compute(BackgroundSubtractedIofQ)
    assert sc.allclose(sc.values(iofq1.data), sc.values(iofq3.data))
    assert sc.identical(iofq1.coords['Q'], iofq3.coords['Q'])
    assert all(sc.variances(iofq1.data) > sc.variances(iofq3.data))
    assert sc.allclose(
        sc.values(
            pipeline_single.compute(NormalizedQ[SampleRun, Numerator]).hist().data
        )
        * N,
        sc.values(
            pipeline_triple.compute(NormalizedQ[SampleRun, Numerator]).hist().data
        ),
    )
    assert sc.allclose(
        sc.values(pipeline_single.compute(NormalizedQ[SampleRun, Denominator]).data)
        * N,
        sc.values(pipeline_triple.compute(NormalizedQ[SampleRun, Denominator]).data),
    )


def test_beam_center_from_center_of_mass_is_close_to_verified_result():
    pipeline = make_workflow(no_masks=False)
    pipeline = sans.with_pixel_mask_filenames(
        pipeline, loki.data.loki_tutorial_mask_filenames()
    )
    center = sans.beam_center_from_center_of_mass(pipeline)
    reference = sc.vector([-0.0291487, -0.0181614, 0], unit='m')
    assert sc.allclose(center, reference)


def test_phi_with_gravity():
    pipeline = make_workflow()
    pipeline[BeamCenter] = _compute_beam_center()
    pipeline[CorrectForGravity] = False
    data_no_grav = pipeline.compute(WavelengthDetector[SampleRun, Numerator]).flatten(
        to='pixel'
    )
    graph_no_grav = pipeline.compute(ElasticCoordTransformGraph[SampleRun])
    pipeline[CorrectForGravity] = True
    data_with_grav = (
        pipeline.compute(WavelengthDetector[SampleRun, Numerator])
        .flatten(to='pixel')
        .hist(wavelength=sc.linspace('wavelength', 1.0, 12.0, 101, unit='angstrom'))
    )
    graph_with_grav = pipeline.compute(ElasticCoordTransformGraph[SampleRun])

    no_grav = data_no_grav.transform_coords(('two_theta', 'phi'), graph_no_grav)
    phi_no_grav = no_grav.coords['phi']
    with_grav = data_with_grav.transform_coords(('two_theta', 'phi'), graph_with_grav)
    phi_with_grav = with_grav.coords['phi'].mean('wavelength')

    assert not sc.identical(phi_no_grav, phi_with_grav)

    # Exclude pixels near y=0, since phi with gravity could drop below y=0 and give a
    # difference of almost 2*pi.
    y = sc.abs(
        pipeline.compute(CorrectedDetector[SampleRun, Numerator])
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
