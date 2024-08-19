# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
"""
File loading functions for ISIS data, NOT using Mantid.
"""

from typing import NewType

import sciline
import scipp as sc

from ess.sans.types import (
    BackgroundRun,
    DirectBeam,
    DirectBeamFilename,
    Filename,
    RunType,
    SampleRun,
    TransmissionRun,
)

CalibrationFilename = NewType('CalibrationFilename', str)


class LoadedFileContents(sciline.Scope[RunType, sc.DataGroup], sc.DataGroup):
    """Contents of a loaded file."""


def load_tutorial_run(filename: Filename[RunType]) -> LoadedFileContents[RunType]:
    return LoadedFileContents[RunType](sc.io.load_hdf5(filename))


def load_tutorial_direct_beam(filename: DirectBeamFilename) -> DirectBeam:
    return DirectBeam(sc.io.load_hdf5(filename))


def transmission_from_sample_run(
    data: LoadedFileContents[SampleRun],
) -> LoadedFileContents[TransmissionRun[SampleRun]]:
    """
    Use transmission from a sample run, instead of dedicated run.
    """
    return LoadedFileContents[TransmissionRun[SampleRun]](data)


def transmission_from_background_run(
    data: LoadedFileContents[BackgroundRun],
) -> LoadedFileContents[TransmissionRun[BackgroundRun]]:
    """
    Use transmission from a background run, instead of dedicated run.
    """
    return LoadedFileContents[TransmissionRun[BackgroundRun]](data)


providers = (read_xml_detector_masking,)
