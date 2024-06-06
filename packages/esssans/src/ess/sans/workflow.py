# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from typing import Iterable, Hashable

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


def _maybe_mapped(
    pipeline: sciline.Pipeline, data: Iterable[str] | None, key: Hashable, dim: str
) -> sciline.Pipeline:
    if data is not None:
        return pipeline.map(pd.DataFrame({key: data}).rename_axis(dim))
    return pipeline


def set_sample_runsx(
    pipeline: sciline.Pipeline,
    sample_runs: Iterable[str] | None = None,
    background_runs: Iterable[str] | None = None,
    banks: Iterable[str] | None = None,
) -> sciline.Pipeline:
    result = pipeline.copy()
    pipeline = _maybe_mapped(pipeline, banks, NeXusDetectorName, 'bank')
    pipeline = _maybe_mapped(pipeline, sample_runs, Filename[SampleRun], 'sample_run')
    pipeline = _maybe_mapped(
        pipeline, background_runs, Filename[BackgroundRun], 'background_run'
    )

    # are we happy with how this works in cyclebane?
    # TODO keep cyclebane magic key replacement or not?

    # Filename
    #  / \
    # N   D

    # Filename(run,bank)
    #   /         \
    # N(run,bank)  D(run,bank)
    #  |           |
    # N(bank)    D(bank)
    #  |           |
    #  N           D

    # Options:
    # 1) fix logic to allow merging graphs with identical mapped values (here: Filename)
    # 2) get branches after mapping with explicit MappedNode name (more complicated than 1)
    # 3) support reduce on multiple sink nodes (here: Numerator, Denominator)

    # fixing 1) and removing key magic appears to be conceptually most consistent
    for part in (Numerator, Denominator):
        sample = pipeline[CleanSummedQ[SampleRun, part]]
        background = pipeline[CleanSummedQ[BackgroundRun, part]]

        if sample_runs is not None:
            sample = sample.reduce(index='sample_run', func=_merge_contributions)
        if background_runs is not None:
            background = background.reduce(
                index='background_run', func=_merge_contributions
            )
        if banks is not None:
            sample = sample.reduce(index='bank', func=_merge_contributions)
            background = background.reduce(index='bank', func=_merge_contributions)
        result[CleanSummedQ[SampleRun, part]] = sample
        result[CleanSummedQ[BackgroundRun, part]] = background
    return result


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
