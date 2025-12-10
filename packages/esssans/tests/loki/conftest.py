# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
import pytest
import sciline
import scipp as sc

import ess.loki.data  # noqa: F401
from ess import loki
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


@pytest.fixture
def larmor_workflow() -> sciline.Pipeline:
    def make_workflow(no_masks: bool = True) -> sciline.Pipeline:
        wf = loki.LokiAtLarmorWorkflow()

        wf[NeXusDetectorName] = 'larmor_detector'
        wf[Filename[SampleRun]] = loki.data.loki_tutorial_sample_run_60339()
        wf[Filename[BackgroundRun]] = loki.data.loki_tutorial_background_run_60393()
        wf[Filename[TransmissionRun[SampleRun]]] = (
            loki.data.loki_tutorial_sample_transmission_run()
        )
        wf[Filename[TransmissionRun[BackgroundRun]]] = (
            loki.data.loki_tutorial_run_60392()
        )
        wf[Filename[EmptyBeamRun]] = loki.data.loki_tutorial_run_60392()

        wf[WavelengthBins] = sc.linspace(
            'wavelength', start=1.0, stop=13.0, num=51, unit='angstrom'
        )
        wf[CorrectForGravity] = True
        wf[UncertaintyBroadcastMode] = UncertaintyBroadcastMode.upper_bound
        wf[ReturnEvents] = False

        wf[QxBins] = sc.linspace('Qx', start=-0.3, stop=0.3, num=91, unit='1/angstrom')
        wf[QyBins] = sc.linspace('Qy', start=-0.2, stop=0.3, num=78, unit='1/angstrom')
        wf[QBins] = sc.linspace('Q', start=0.01, stop=0.3, num=101, unit='1/angstrom')
        # We have no direct-beam file for Loki currently
        wf[DirectBeam] = None
        if no_masks:
            wf[DetectorMasks] = {}

        return wf

    return make_workflow


@pytest.fixture
def loki_workflow() -> sciline.Pipeline:
    def make_workflow() -> sciline.Pipeline:
        wf = loki.LokiWorkflow()

        # For now, use a dummy file for all runs. This produces a result with all NaNs,
        # but is sufficient to test that the workflow runs end-to-end.
        # TODO: Replace with real test data when available.
        file = loki.data.loki_coda_file_one_event()
        wf[Filename[SampleRun]] = file
        wf[Filename[BackgroundRun]] = file
        wf[Filename[TransmissionRun[SampleRun]]] = file
        wf[Filename[TransmissionRun[BackgroundRun]]] = file
        wf[Filename[EmptyBeamRun]] = file

        wf[WavelengthBins] = sc.linspace(
            'wavelength', start=1.0, stop=13.0, num=51, unit='angstrom'
        )
        wf[CorrectForGravity] = True
        wf[UncertaintyBroadcastMode] = UncertaintyBroadcastMode.upper_bound
        wf[ReturnEvents] = False

        wf[QxBins] = sc.linspace('Qx', start=-0.3, stop=0.3, num=91, unit='1/angstrom')
        wf[QyBins] = sc.linspace('Qy', start=-0.2, stop=0.3, num=78, unit='1/angstrom')
        wf[QBins] = sc.linspace('Q', start=0.01, stop=0.3, num=101, unit='1/angstrom')
        # We have no direct-beam file for Loki currently
        wf[DirectBeam] = None
        wf[DetectorMasks] = {}

        return wf

    return make_workflow
