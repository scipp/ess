# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from typing import Hashable, Iterable

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
    for run in (SampleRun, BackgroundRun):
        for part in (Numerator, Denominator):
            pipeline[CleanSummedQ[run, part]] = (
                pipeline[CleanSummedQ[run, part]]
                .map(banks)
                .reduce(index='bank', func=_merge_contributions)
            )
    return pipeline


def _set_runs(
    pipeline: sciline.Pipeline, runs: Iterable[str], key: Hashable, axis_name: str
) -> sciline.Pipeline:
    pipeline = pipeline.copy()
    runs = pd.DataFrame({Filename[key]: runs}).rename_axis(axis_name)
    for part in (Numerator, Denominator):
        pipeline[CleanSummedQ[key, part]] = (
            pipeline[CleanSummedQ[key, part]]
            .map(runs)
            .reduce(index=axis_name, func=_merge_contributions)
        )
    return pipeline


def set_sample_runs(
    pipeline: sciline.Pipeline, runs: Iterable[str]
) -> sciline.Pipeline:
    return _set_runs(pipeline, runs, SampleRun, 'sample_run')


def set_background_runs(
    pipeline: sciline.Pipeline, runs: Iterable[str]
) -> sciline.Pipeline:
    return _set_runs(pipeline, runs, BackgroundRun, 'background_run')
