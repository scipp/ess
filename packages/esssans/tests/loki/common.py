# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from typing import Callable, List

import scipp as sc

import esssans as sans
from esssans.loki import default_parameters
from esssans.types import (
    BackgroundRun,
    CorrectForGravity,
    EmptyBeamRun,
    Filename,
    QBins,
    QxyBins,
    ReturnEvents,
    SampleRun,
    TransmissionRun,
    UncertaintyBroadcastMode,
    WavelengthBins,
)


def make_params(qxy: bool = False) -> dict:
    params = default_parameters.copy()

    params[Filename[SampleRun]] = '60339-2022-02-28_2215.nxs'
    params[Filename[BackgroundRun]] = '60393-2022-02-28_2215.nxs'
    params[Filename[TransmissionRun[SampleRun]]] = '60394-2022-02-28_2215.nxs'
    params[Filename[TransmissionRun[BackgroundRun]]] = '60392-2022-02-28_2215.nxs'
    params[Filename[EmptyBeamRun]] = '60392-2022-02-28_2215.nxs'

    params[WavelengthBins] = sc.linspace(
        'wavelength', start=1.0, stop=13.0, num=51, unit='angstrom'
    )
    params[CorrectForGravity] = True
    params[UncertaintyBroadcastMode] = UncertaintyBroadcastMode.upper_bound
    params[ReturnEvents] = False

    if qxy:
        params[QxyBins] = {
            'Qx': sc.linspace('Qx', -0.3, 0.3, 91, unit='1/angstrom'),
            'Qy': sc.linspace('Qy', -0.2, 0.3, 78, unit='1/angstrom'),
        }
    else:
        params[QBins] = sc.linspace(
            dim='Q', start=0.01, stop=0.3, num=101, unit='1/angstrom'
        )

    return params


def loki_providers() -> List[Callable]:
    from esssans.isis.io import read_xml_detector_masking

    return list(
        sans.providers
        + sans.loki.providers
        + sans.loki.data.providers
        + (read_xml_detector_masking,)
    )
