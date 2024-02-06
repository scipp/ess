# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from typing import List, Optional

import scipp as sc

from esssans.loki import default_parameters
from esssans.types import (
    BackgroundRun,
    BeamStopPosition,
    BeamStopRadius,
    CorrectForGravity,
    EmptyBeamRun,
    FileList,
    QBins,
    QxyBins,
    ReturnEvents,
    SampleRun,
    TransmissionRun,
    UncertaintyBroadcastMode,
    WavelengthBins,
)


def make_params(
    sample_runs: Optional[List[str]] = None,
    background_runs: Optional[List[str]] = None,
    qxy: bool = False,
) -> dict:
    params = default_parameters.copy()

    if sample_runs is None:
        sample_runs = ['60339-2022-02-28_2215.nxs']
    if background_runs is None:
        background_runs = ['60393-2022-02-28_2215.nxs']

    # List of files
    params[FileList[SampleRun]] = sample_runs
    params[FileList[BackgroundRun]] = background_runs
    params[FileList[TransmissionRun[SampleRun]]] = ['60394-2022-02-28_2215.nxs']
    params[FileList[TransmissionRun[BackgroundRun]]] = ['60392-2022-02-28_2215.nxs']
    params[FileList[EmptyBeamRun]] = ['60392-2022-02-28_2215.nxs']

    params[WavelengthBins] = sc.linspace(
        'wavelength', start=1.0, stop=13.0, num=51, unit='angstrom'
    )
    params[BeamStopPosition] = sc.vector([-0.026, -0.022, 0.0], unit='m')
    params[BeamStopRadius] = sc.scalar(0.042, unit='m')
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
