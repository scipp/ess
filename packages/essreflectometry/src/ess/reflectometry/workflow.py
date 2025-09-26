# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from collections.abc import Hashable, Sequence
from itertools import chain

import pandas as pd
import sciline
import scipp as sc

from ess.reflectometry.orso import (
    OrsoExperiment,
    OrsoOwner,
    OrsoSample,
    OrsoSampleFilenames,
)
from ess.reflectometry.types import (
    DetectorRotation,
    Filename,
    # RawChopper,
    ReducibleData,
    RunType,
    SampleRotation,
    SampleRun,
)


def _concatenate_event_lists(*das):
    da = sc.reduce(das).bins.concat()
    missing_coords = set(das[0].coords) - set(da.coords)
    return da.assign_coords({coord: das[0].coords[coord] for coord in missing_coords})


def _any_value(x, *_):
    return x


def _concatenate_lists(*x):
    return list(chain(*x))


def with_filenames(
    workflow, runtype: Hashable, runs: Sequence[Filename[RunType]]
) -> sciline.Pipeline:
    '''Sets a number of :code:`Filename[runtype]` simultaneously.
    The events from all listed files are concatenated in the workflow.

    Arguments
    ----------
    workflow:
        the workflow to copy and add the filenames to
    runtype:
        the kind of runtype to add the files as.
        Example: :code:`SampleRun` or :code:`ReferenceRun`.
    runs:
        the list of filenames to map over

    Returns
    ---------
        A copy of the original workflow mapping over the provided files.
    '''
    axis_name = f'{str(runtype).lower()}_runs'
    df = pd.DataFrame({Filename[runtype]: runs}).rename_axis(axis_name)
    wf = workflow.copy()

    mapped = wf.map(df)

    reduce_functions = {
        ReducibleData[runtype]: _concatenate_event_lists,
        SampleRotation[runtype]: _any_value,
        DetectorRotation[runtype]: _any_value,
        # RawChopper[runtype]: _any_value,
    }
    if runtype is SampleRun:
        reduce_functions.update(
            {
                OrsoSample: _any_value,
                OrsoExperiment: _any_value,
                OrsoOwner: _any_value,
                OrsoSampleFilenames: _concatenate_lists,
            }
        )

    for tp, func in reduce_functions.items():
        try:
            wf[tp] = mapped[tp].reduce(index=axis_name, func=func)
        except (ValueError, KeyError):
            # ValueError: tp[runtype] is independent of Filename[runtype]
            # KeyError: tp[runtype] not in workflow
            pass

    return wf
