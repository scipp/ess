# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from typing import Iterable

import pandas as pd
import sciline
import scipp as sc

from .types import (
    BackgroundRun,
    CleanSummedQ,
    Denominator,
    DetectorMasks,
    NeXusDetectorName,
    Numerator,
    PixelMaskFilename,
    SampleRun,
)


def _merge(*dicts: dict) -> dict:
    return {key: value for d in dicts for key, value in d.items()}


def _merge_contributions(*data: sc.DataArray) -> sc.DataArray:
    if len(data) == 1:
        return data[0]
    reducer = sc.reduce(data)
    return reducer.bins.concat() if data[0].bins is not None else reducer.sum()


def make_workflow(
    pipeline: sciline.Pipeline,
    bank: str | Iterable[str],
    masks: Iterable[str] = (),
    sample_run: str | Iterable[str] = (),
    background_run: str | Iterable[str] = (),
) -> sciline.Pipeline:
    pipeline = pipeline.copy()
    by_mask = pipeline.map(
        pd.DataFrame({PixelMaskFilename: masks}).rename_axis('pixel_mask')
    )
    for run in (SampleRun, BackgroundRun):
        pipeline[DetectorMasks[run]] = by_mask[DetectorMasks[run]].reduce(func=_merge)
    banks = pd.DataFrame(
        {NeXusDetectorName: [bank] if isinstance(bank, str) else bank}
    ).rename_axis('bank')
    # sample_runs = ([sample_run] if isinstance(sample_run, str) else sample_run,)
    # background_runs = (
    #     [background_run] if isinstance(background_run, str) else background_run,
    # )
    by_bank_and_run = (
        pipeline.map(banks)
        # .map({Filename[SampleRun]: sample_runs})
        # .map({Filename[BackgroundRun]: background_runs})
    )
    # run_dim = {SampleRun: 'sample_run', BackgroundRun: 'background_run'}
    for run in (SampleRun, BackgroundRun):
        for part in (Numerator, Denominator):
            pipeline[CleanSummedQ[run, part]] = (
                by_bank_and_run[CleanSummedQ[run, part]]
                # .reduce(index=run_dim[run], func=_merge_contributions)
                .reduce(index='bank', func=_merge_contributions)
            )
    return pipeline
