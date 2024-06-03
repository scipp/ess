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
    Filename,
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


def set_pixel_mask_filenames(
    pipeline: sciline.Pipeline, masks: Iterable[str]
) -> sciline.Pipeline:
    pipeline = pipeline.copy()
    pipeline[DetectorMasks] = pipeline.map(
        pd.DataFrame({PixelMaskFilename: masks}).rename_axis('mask')
    )[DetectorMasks].reduce(func=_merge)
    return pipeline


def set_banks(pipeline: sciline.Pipeline, banks: Iterable[str]) -> sciline.Pipeline:
    pipeline = pipeline.copy()
    banks = pd.DataFrame({NeXusDetectorName: banks}).rename_axis('bank')
    by_bank = pipeline.map(banks)
    for run in (SampleRun, BackgroundRun):
        for part in (Numerator, Denominator):
            pipeline[CleanSummedQ[run, part]] = by_bank[CleanSummedQ[run, part]].reduce(
                index='bank', func=_merge_contributions
            )
    return pipeline


def set_sample_runs(
    pipeline: sciline.Pipeline, sample_runs: Iterable[str]
) -> sciline.Pipeline:
    by_sample_run = pipeline.map(
        pd.DataFrame({Filename[SampleRun]: sample_runs}).rename_axis('sample_run')
    )
    for part in (Numerator, Denominator):
        pipeline[CleanSummedQ[SampleRun, part]] = by_sample_run[
            CleanSummedQ[SampleRun, part]
        ].reduce(index='sample_run', func=_merge_contributions)
    return pipeline


def set_background_runs(
    pipeline: sciline.Pipeline, sample_runs: Iterable[str]
) -> sciline.Pipeline:
    by_sample_run = pipeline.map(
        pd.DataFrame({Filename[BackgroundRun]: sample_runs}).rename_axis(
            'background_run'
        )
    )
    for part in (Numerator, Denominator):
        pipeline[CleanSummedQ[BackgroundRun, part]] = by_sample_run[
            CleanSummedQ[BackgroundRun, part]
        ].reduce(index='background_run', func=_merge_contributions)
    return pipeline
