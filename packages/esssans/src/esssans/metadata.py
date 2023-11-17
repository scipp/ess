# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

from .types import RawData, RunNumber, RunTitle, SampleRun


def run_number(raw_data: RawData[SampleRun]) -> RunNumber:
    """Get the run number from the raw sample data."""
    return RunNumber(int(raw_data['run_number']))


def run_title(raw_data: RawData[SampleRun]) -> RunTitle:
    """Get the run title from the raw sample data."""
    return RunTitle(raw_data['run_title'].value)


providers = (
    run_number,
    run_title,
)
