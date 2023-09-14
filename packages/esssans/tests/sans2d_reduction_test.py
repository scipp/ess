# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from typing import Callable, List

import pytest
import sciline
import scipp as sc

import esssans as sans
from esssans.types import (
    BackgroundRun,
    BackgroundSubtractedIofQ,
    BeamCenter,
    CorrectForGravity,
    DirectBeam,
    DirectBeamFilename,
    DirectRun,
    Filename,
    Incident,
    IofQ,
    NeXusMonitorName,
    NonBackgroundWavelengthRange,
    QBins,
    RawData,
    SampleRun,
    SolidAngle,
    Transmission,
    UncertaintyBroadcastMode,
    WavelengthBands,
    WavelengthBins,
    WavelengthMask,
)


def make_params() -> dict:
    params = {}
    params[NeXusMonitorName[Incident]] = 'monitor2'
    params[NeXusMonitorName[Transmission]] = 'monitor4'
    band = sc.linspace('wavelength', 2.0, 16.0, num=2, unit='angstrom')
    params[WavelengthBands] = band
    params[WavelengthBins] = sc.linspace(
        'wavelength', start=band[0], stop=band[-1], num=141
    )
    params[WavelengthMask] = sc.DataArray(
        data=sc.array(dims=['wavelength'], values=[True]),
        coords={
            'wavelength': sc.array(
                dims=['wavelength'], values=[2.21, 2.59], unit='angstrom'
            )
        },
    )

    params[QBins] = sc.linspace(
        dim='Q', start=0.01, stop=0.55, num=141, unit='1/angstrom'
    )
    params[Filename[BackgroundRun]] = 'SANS2D00063159.hdf5'
    params[Filename[SampleRun]] = 'SANS2D00063114.hdf5'
    params[Filename[DirectRun]] = 'SANS2D00063091.hdf5'
    params[DirectBeamFilename] = 'DIRECT_SANS2D_REAR_34327_4m_8mm_16Feb16.hdf5'
    params[BeamCenter] = sc.vector(value=[0.0945643, -0.082074, 0.0], unit='m')
    params[NonBackgroundWavelengthRange] = sc.array(
        dims=['wavelength'], values=[0.7, 17.1], unit='angstrom'
    )
    params[CorrectForGravity] = True
    params[UncertaintyBroadcastMode] = UncertaintyBroadcastMode.upper_bound
    return params


def sans2d_providers():
    return sans.providers + sans.sans2d.providers


def test_can_create_pipeline():
    sciline.Pipeline(sans2d_providers(), params=make_params())


@pytest.mark.parametrize(
    'uncertainties',
    [UncertaintyBroadcastMode.drop, UncertaintyBroadcastMode.upper_bound],
)
def test_pipeline_can_compute_background_subtracted_IofQ(uncertainties):
    params = make_params()
    params[UncertaintyBroadcastMode] = uncertainties
    pipeline = sciline.Pipeline(sans2d_providers(), params=params)
    result = pipeline.compute(BackgroundSubtractedIofQ)
    assert result.dims == ('Q',)


def test_uncertainty_broadcast_mode_drop_yields_smaller_variances():
    params = make_params()
    params[UncertaintyBroadcastMode] = UncertaintyBroadcastMode.drop
    pipeline = sciline.Pipeline(sans2d_providers(), params=params)
    drop = pipeline.compute(IofQ[SampleRun]).hist().data
    params[UncertaintyBroadcastMode] = UncertaintyBroadcastMode.upper_bound
    pipeline = sciline.Pipeline(sans2d_providers(), params=params)
    upper_bound = pipeline.compute(IofQ[SampleRun]).data
    assert sc.all(sc.variances(drop) < sc.variances(upper_bound)).value


def test_pipeline_can_visualize_background_subtracted_IofQ():
    pipeline = sciline.Pipeline(sans2d_providers(), params=make_params())
    pipeline.visualize(BackgroundSubtractedIofQ)


def test_pipeline_can_compute_intermediate_results():
    pipeline = sciline.Pipeline(sans2d_providers(), params=make_params())
    result = pipeline.compute(SolidAngle[SampleRun])
    assert result.dims == ('spectrum',)


# TODO See scipp/sciline#57 for plans on a builtin way to do this
def as_dict(funcs: List[Callable[..., type]]) -> dict:
    from typing import get_type_hints

    return dict(zip([get_type_hints(func)['return'] for func in funcs], funcs))


def pixel_dependent_direct_beam(
    filename: DirectBeamFilename, shape: RawData[SampleRun]
) -> DirectBeam:
    direct_beam = sans.sans2d.pooch_load_direct_beam(filename)
    sizes = {'spectrum': shape.sizes['spectrum'], **direct_beam.sizes}
    return DirectBeam(direct_beam.broadcast(sizes=sizes).copy())


@pytest.mark.parametrize(
    'uncertainties',
    [UncertaintyBroadcastMode.drop, UncertaintyBroadcastMode.upper_bound],
)
def test_pixel_dependent_direct_beam_is_supported(uncertainties):
    params = make_params()
    params[UncertaintyBroadcastMode] = uncertainties
    providers = as_dict(sans2d_providers())
    providers[DirectBeam] = pixel_dependent_direct_beam
    pipeline = sciline.Pipeline(providers.values(), params=params)
    result = pipeline.compute(BackgroundSubtractedIofQ)
    assert result.dims == ('Q',)
