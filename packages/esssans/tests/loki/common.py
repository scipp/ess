# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from typing import Callable, List

import scipp as sc

from ess import loki, sans
from ess.sans.types import (
    BackgroundRun,
    CorrectForGravity,
    DetectorMasks,
    DirectBeam,
    EmptyBeamRun,
    Filename,
    NeXusDetectorName,
    QBins,
    QxBins,
    QyBins,
    ReturnEvents,
    SampleRun,
    TransmissionRun,
    UncertaintyBroadcastMode,
    WavelengthBins,
)


def make_params() -> dict:
    params = loki.default_parameters()

    params[NeXusDetectorName] = 'larmor_detector'
    params[Filename[SampleRun]] = loki.data.loki_tutorial_sample_run_60339()
    params[Filename[BackgroundRun]] = loki.data.loki_tutorial_background_run_60393()
    params[
        Filename[TransmissionRun[SampleRun]]
    ] = loki.data.loki_tutorial_sample_transmission_run()
    params[
        Filename[TransmissionRun[BackgroundRun]]
    ] = loki.data.loki_tutorial_run_60392()
    params[Filename[EmptyBeamRun]] = loki.data.loki_tutorial_run_60392()

    params[WavelengthBins] = sc.linspace(
        'wavelength', start=1.0, stop=13.0, num=51, unit='angstrom'
    )
    params[CorrectForGravity] = True
    params[UncertaintyBroadcastMode] = UncertaintyBroadcastMode.upper_bound
    params[ReturnEvents] = False

    params[QxBins] = sc.linspace('Qx', start=-0.3, stop=0.3, num=91, unit='1/angstrom')
    params[QyBins] = sc.linspace('Qy', start=-0.2, stop=0.3, num=78, unit='1/angstrom')
    params[QBins] = sc.linspace('Q', start=0.01, stop=0.3, num=101, unit='1/angstrom')
    # We have no direct-beam file for Loki currently
    params[DirectBeam] = None
    params[DetectorMasks] = {}

    return params


def loki_providers_no_beam_center_finder() -> List[Callable]:
    from ess.isissans.io import read_xml_detector_masking

    return list(
        sans.providers
        + loki.providers
        + (
            read_xml_detector_masking,
            loki.io.dummy_load_sample,
        )
    )


def loki_providers() -> List[Callable]:
    return loki_providers_no_beam_center_finder() + [
        sans.beam_center_finder.beam_center_from_center_of_mass
    ]
