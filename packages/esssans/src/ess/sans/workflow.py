# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
from collections.abc import Hashable, Iterable

import pandas as pd
import sciline
import scipp as sc

from ess.reduce.parameter import parameter_mappers
from ess.reduce.time_of_flight import GenericTofWorkflow

from . import common, conversions, i_of_q, masking, normalization
from .types import (
    BackgroundRun,
    CorrectForGravity,
    Denominator,
    DetectorBankSizes,
    DetectorMasks,
    DimsToKeep,
    EmptyBeamRun,
    Filename,
    Incident,
    NeXusDetectorName,
    NormalizedQ,
    NormalizedQxQy,
    Numerator,
    PixelMaskFilename,
    SampleRun,
    TransformationPath,
    Transmission,
    TransmissionRun,
    WavelengthBands,
    WavelengthMask,
)


def _merge(*dicts: dict) -> dict:
    return {key: value for d in dicts for key, value in d.items()}


def merge_contributions(*data: sc.DataArray) -> sc.DataArray:
    if len(data) == 1:
        return data[0]
    reducer = sc.reduce(data)
    return reducer.bins.concat() if data[0].bins is not None else reducer.sum()


def with_pixel_mask_filenames(
    workflow: sciline.Pipeline, masks: Iterable[str]
) -> sciline.Pipeline:
    """
    Return modified workflow with pixel mask filenames set.

    Parameters
    ----------
    workflow:
        Workflow to modify.
    masks:
        List or tuple of pixel mask filenames to set.
    """
    workflow = workflow.copy()
    workflow[DetectorMasks] = (
        workflow[DetectorMasks]
        .map(pd.DataFrame({PixelMaskFilename: masks}).rename_axis('mask'))
        .reduce(index='mask', func=_merge)
    )
    return workflow


def with_banks(
    workflow: sciline.Pipeline,
    banks: Iterable[str],
    index: Iterable[Hashable] | None = None,
) -> sciline.Pipeline:
    """
    Return modified workflow with bank names set.

    Since banks typically have different Q-resolution the I(Q) of banks are not merged.
    That is, the resulting workflow will have separate outputs for each bank. Use
    :py:func:`sciline.compute_mapped` to compute results for all banks.

    Parameters
    ----------
    workflow:
        Workflow to modify.
    banks:
        List or tuple of bank names to set.
    index:
        Index to use for the DataFrame. If not provided, the bank names are used.
    """
    index = index or banks
    return workflow.map(
        pd.DataFrame({NeXusDetectorName: banks}, index=index).rename_axis('bank')
    )


def _set_runs(
    pipeline: sciline.Pipeline, runs: Iterable[str], key: Hashable, axis_name: str
) -> sciline.Pipeline:
    pipeline = pipeline.copy()
    runs = pd.DataFrame({Filename[key]: runs}).rename_axis(axis_name)
    for part in (Numerator, Denominator):
        for qtype in (NormalizedQ, NormalizedQxQy):
            pipeline[qtype[key, part]] = (
                pipeline[qtype[key, part]]
                .map(runs)
                .reduce(index=axis_name, func=merge_contributions)
            )
    return pipeline


def with_sample_runs(
    workflow: sciline.Pipeline, runs: Iterable[str]
) -> sciline.Pipeline:
    """
    Return modified workflow with sample run filenames set.

    Parameters
    ----------
    workflow:
        Workflow to modify.
    runs:
        List or tuple of sample run filenames to set.
    """
    return _set_runs(workflow, runs, SampleRun, 'sample_run')


def with_background_runs(
    workflow: sciline.Pipeline, runs: Iterable[str]
) -> sciline.Pipeline:
    """
    Return modified workflow with background run filenames set.

    Parameters
    ----------
    workflow:
        Workflow to modify.
    runs:
        List or tuple of background run filenames to set.
    """
    return _set_runs(workflow, runs, BackgroundRun, 'background_run')


parameter_mappers[PixelMaskFilename] = with_pixel_mask_filenames
# TODO: for now, we leave the mapping over detector banks out, because we do not have a
# method to merge the I(Q) of different banks, and we thus cannot compute a single
# result from the workflow. So only a single detector bank can be processed at a time.
# parameter_mappers[NeXusDetectorName] = with_banks
parameter_mappers[Filename[SampleRun]] = with_sample_runs
parameter_mappers[Filename[BackgroundRun]] = with_background_runs


providers = (
    *conversions.providers,
    *i_of_q.providers,
    *masking.providers,
    *normalization.providers,
    common.beam_center_to_detector_position_offset,
)
"""
List of providers for setting up a Sciline pipeline.

This provides a default workflow, including a beam-center estimation based on a
center-of-mass approach. Providers for loadings files are not included. Combine with
the providers for a specific instrument, such as :py:data:`esssans.sans2d.providers`
to setup a complete workflow.
"""


def SansWorkflow() -> sciline.Pipeline:
    """
    Common base for SANS workflows.

    Returns
    -------
    :
        SANS workflow as a sciline.Pipeline
    """
    workflow = GenericTofWorkflow(
        run_types=(
            SampleRun,
            EmptyBeamRun,
            BackgroundRun,
            TransmissionRun[SampleRun],
            TransmissionRun[BackgroundRun],
        ),
        monitor_types=(Incident, Transmission),
    )
    for provider in providers:
        workflow.insert(provider)
    workflow[CorrectForGravity] = CorrectForGravity(False)
    workflow[DetectorBankSizes] = DetectorBankSizes({})
    workflow[DimsToKeep] = DimsToKeep(())
    workflow[TransformationPath] = TransformationPath('transform')
    workflow[WavelengthBands] = WavelengthBands(None)
    workflow[WavelengthMask] = WavelengthMask(None)
    return workflow
