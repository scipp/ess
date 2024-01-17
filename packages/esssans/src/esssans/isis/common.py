# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
"""
"""

from ..types import BackgroundRun, LoadedFileContents, SampleRun, TransmissionRun


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
