# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
import sciline
import scipp as sc
from ess import isissans as isis
from ess import sans
from ess.sans.types import (
    BeamCenter,
    CorrectForGravity,
    Filename,
    Incident,
    IofQ,
    IofQxy,
    NeXusMonitorName,
    NonBackgroundWavelengthRange,
    QBins,
    QxBins,
    QyBins,
    SampleRun,
    Transmission,
    UncertaintyBroadcastMode,
    WavelengthBins,
)


def make_params() -> dict:
    params = {
        **isis.default_parameters(),
        sans.types.DirectBeamFilename: isis.data.zoom_tutorial_direct_beam(),
        isis.CalibrationFilename: isis.data.zoom_tutorial_calibration(),
        Filename[sans.types.SampleRun]: isis.data.zoom_tutorial_sample_run(),
        Filename[sans.types.EmptyBeamRun]: isis.data.zoom_tutorial_empty_beam_run(),
        isis.SampleOffset: sc.vector([0.0, 0.0, 0.11], unit='m'),
        isis.DetectorBankOffset: sc.vector([0.0, 0.0, 0.5], unit='m'),
    }

    params[NeXusMonitorName[Incident]] = 'monitor3'
    params[NeXusMonitorName[Transmission]] = 'monitor5'

    params[WavelengthBins] = sc.geomspace(
        'wavelength', start=1.75, stop=16.5, num=141, unit='angstrom'
    )

    params[QBins] = sc.geomspace(
        dim='Q', start=0.004, stop=0.8, num=141, unit='1/angstrom'
    )

    params[NonBackgroundWavelengthRange] = sc.array(
        dims=['wavelength'], values=[0.7, 17.1], unit='angstrom'
    )
    params[CorrectForGravity] = True
    params[UncertaintyBroadcastMode] = UncertaintyBroadcastMode.upper_bound
    params[sans.ReturnEvents] = False
    return params


def zoom_providers():
    return list(
        sans.providers
        + isis.providers
        + isis.mantidio.providers
        + (
            isis.io.load_tutorial_direct_beam,
            isis.io.load_tutorial_run,
            isis.io.transmission_from_background_run,
            isis.io.transmission_from_sample_run,
        )
    )


def test_can_create_pipeline():
    pipeline = sciline.Pipeline(zoom_providers(), params=make_params())
    pipeline[BeamCenter] = sc.vector([0, 0, 0], unit='m')
    pipeline = sans.with_pixel_mask_filenames(
        pipeline, isis.data.zoom_tutorial_mask_filenames()
    )
    pipeline.get(IofQ[SampleRun])


def test_pipeline_can_compute_IofQ():
    pipeline = sciline.Pipeline(zoom_providers(), params=make_params())
    pipeline[BeamCenter] = sc.vector([0, 0, 0], unit='m')
    pipeline = sans.with_pixel_mask_filenames(
        pipeline, isis.data.zoom_tutorial_mask_filenames()
    )
    result = pipeline.compute(IofQ[SampleRun])
    assert result.dims == ('Q',)


def test_pipeline_can_compute_IofQxQy():
    pipeline = sciline.Pipeline(zoom_providers(), params=make_params())
    pipeline[BeamCenter] = sc.vector([0, 0, 0], unit='m')
    pipeline = sans.with_pixel_mask_filenames(
        pipeline, isis.data.zoom_tutorial_mask_filenames()
    )
    pipeline[QxBins] = sc.linspace(
        dim='Qx', start=-0.5, stop=0.5, num=101, unit='1/angstrom'
    )
    pipeline[QyBins] = sc.linspace(
        dim='Qy', start=-0.8, stop=0.8, num=101, unit='1/angstrom'
    )

    result = pipeline.compute(IofQxy[SampleRun])
    assert result.dims == ('Qy', 'Qx')
