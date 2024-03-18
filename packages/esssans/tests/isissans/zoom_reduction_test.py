# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
import sciline
import scipp as sc

from ess import isissans as isis
from ess import sans
from ess.sans.types import (
    CorrectForGravity,
    Filename,
    Incident,
    IofQ,
    NeXusMonitorName,
    NonBackgroundWavelengthRange,
    PixelMaskFilename,
    QBins,
    QxyBins,
    SampleRun,
    Transmission,
    UncertaintyBroadcastMode,
    WavelengthBins,
)


def make_params() -> dict:
    params = {
        sans.types.DirectBeamFilename: 'Direct_Zoom_4m_8mm_100522.txt',
        isis.CalibrationFilename: '192tubeCalibration_11-02-2019_r5_10lines.nxs',
        Filename[sans.types.SampleRun]: 'ZOOM00034786.nxs',
        Filename[sans.types.EmptyBeamRun]: 'ZOOM00034787.nxs',
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


def make_masks_table() -> sciline.ParamTable:
    masks = [
        'andru_test.xml',
        'left_beg_18_2.xml',
        'right_beg_18_2.xml',
        'small_bs_232.xml',
        'small_BS_31032023.xml',
        'tube_1120_bottom.xml',
        'tubes_beg_18_2.xml',
    ]
    return sciline.ParamTable(PixelMaskFilename, columns={}, index=masks)


def zoom_providers():
    return list(
        sans.providers
        + isis.providers
        + isis.data.providers
        + (
            isis.data.transmission_from_background_run,
            isis.data.transmission_from_sample_run,
            sans.beam_center_finder.beam_center_from_center_of_mass,
        )
    )


def test_can_create_pipeline():
    pipeline = sciline.Pipeline(zoom_providers(), params=make_params())
    pipeline.set_param_table(make_masks_table())
    pipeline.get(IofQ[SampleRun])


def test_pipeline_can_compute_IofQ():
    pipeline = sciline.Pipeline(zoom_providers(), params=make_params())
    pipeline.set_param_table(make_masks_table())
    result = pipeline.compute(IofQ[SampleRun])
    assert result.dims == ('Q',)


def test_pipeline_can_compute_IofQxQy():
    pipeline = sciline.Pipeline(zoom_providers(), params=make_params())
    pipeline.set_param_table(make_masks_table())
    pipeline[QxyBins] = {
        'Qx': sc.linspace(dim='Qx', start=-0.5, stop=0.5, num=101, unit='1/angstrom'),
        'Qy': sc.linspace(dim='Qy', start=-0.8, stop=0.8, num=101, unit='1/angstrom'),
    }

    result = pipeline.compute(IofQ[SampleRun])
    assert result.dims == ('Qy', 'Qx')
