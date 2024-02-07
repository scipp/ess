# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
import sys
from pathlib import Path
from typing import Callable, List

import pytest
import sciline
import scipp as sc

import esssans as sans
from esssans.conversions import ElasticCoordTransformGraph
from esssans.types import (
    BackgroundSubtractedIofQ,
    BeamCenter,
    CleanWavelengthMasked,
    CorrectForGravity,
    DimsToKeep,
    Numerator,
    QBins,
    QxyBins,
    ReturnEvents,
    SampleRun,
    UncertaintyBroadcastMode,
    WavelengthBands,
    WavelengthBins,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import make_params  # noqa: E402


def loki_providers() -> List[Callable]:
    return list(sans.providers + sans.loki.providers)


@pytest.mark.parametrize('qxy', [False, True])
def test_can_create_pipeline(qxy: bool):
    pipeline = sciline.Pipeline(loki_providers(), params=make_params(qxy=qxy))
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
@pytest.mark.parametrize(
    'target', [sans.IofQ[sans.SampleRun], sans.BackgroundSubtractedIofQ]
)
@pytest.mark.parametrize('qxy', [False, True])
def test_pipeline_can_compute_IofQ_in_event_mode(uncertainties, target, qxy: bool):
    params = make_params(qxy=qxy)
    params[UncertaintyBroadcastMode] = uncertainties
    pipeline = sciline.Pipeline(loki_providers(), params=params)
    reference = pipeline.compute(target)
    pipeline[ReturnEvents] = True
    result = pipeline.compute(target)
    assert result.bins is not None
    assert result.dims == ('Qy', 'Qx') if qxy else ('Q',)
    assert sc.allclose(
        sc.values(reference.data),
        sc.values(result.hist().data),
        rtol=sc.scalar(1e-11),
        atol=sc.scalar(1e-11),
    )
    if uncertainties == UncertaintyBroadcastMode.drop:
        tol = sc.scalar(1e-14)
    else:
        tol = sc.scalar(1e-8 if qxy else 1e-9)
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
    result = pipeline.compute(BackgroundSubtractedIofQ)
    assert result.dims == ('band', 'Qy', 'Qx') if qxy else ('band', 'Q')
    assert result.sizes['band'] == 10


@pytest.mark.parametrize('qxy', [False, True])
def test_pipeline_can_compute_IofQ_in_layers(qxy: bool):
    params = make_params(qxy=qxy)
    params[DimsToKeep] = ['layer']
    pipeline = sciline.Pipeline(loki_providers(), params=params)
    result = pipeline.compute(BackgroundSubtractedIofQ)
    assert result.dims == ('layer', 'Qy', 'Qx') if qxy else ('layer', 'Q')
    assert result.sizes['layer'] == 4


def test_pipeline_can_compute_IofQ_merging_events_from_multiple_runs():
    params = make_params(
        sample_runs=['60250-2022-02-28_2215.nxs', '60339-2022-02-28_2215.nxs'],
        background_runs=['60248-2022-02-28_2215.nxs', '60393-2022-02-28_2215.nxs'],
    )
    pipeline = sciline.Pipeline(loki_providers(), params=params)
    result = pipeline.compute(BackgroundSubtractedIofQ)
    assert result.dims == ('Q',)


def test_beam_center_from_center_of_mass_is_close_to_verified_result():
    params = make_params()
    pipeline = sciline.Pipeline(loki_providers(), params=params)
    center = pipeline.compute(BeamCenter)
    reference = sc.vector([-0.0309889, -0.0168854, 0], unit='m')
    assert sc.allclose(center, reference)


def test_phi_with_gravity():
    params = make_params()
    pipeline = sciline.Pipeline(loki_providers(), params=params)
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
    two_theta_no_grav = no_grav.coords['two_theta']
    phi_no_grav = no_grav.coords['phi']
    with_grav = data_with_grav.transform_coords(('two_theta', 'phi'), graph_with_grav)
    phi_with_grav = with_grav.coords['phi'].mean('wavelength')

    assert not sc.identical(phi_no_grav, phi_with_grav)

    # Exclude pixels near the origin, since phi will vary a lot there.
    not_near_origin = two_theta_no_grav > sc.scalar(0.1, unit='deg').to(unit='rad')
    assert sc.all(
        sc.isclose(
            phi_no_grav[not_near_origin],
            phi_with_grav[not_near_origin],
            atol=sc.scalar(3.0, unit='deg').to(unit='rad'),
        )
    )

    # Phi is in [-pi, pi], measured from the X axis.
    pos_x = sc.abs(phi_no_grav) < sc.scalar(90.0, unit='deg').to(unit='rad')
    # Phi is larger with gravity, since it gives the position where it would have
    # been detected without gravity. That is, with gravity all points are pulled
    # "up" in the XY plane, so the angle is larger for positive X and smaller for
    # negative X.
    assert sc.all(phi_no_grav[pos_x] < phi_with_grav[pos_x])
    assert sc.all(phi_no_grav[~pos_x] > phi_with_grav[~pos_x])
