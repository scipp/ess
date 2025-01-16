from collections.abc import Hashable, Sequence
from itertools import chain

import pandas as pd
import sciline
import scipp as sc

from ess.amor.types import RawChopper
from ess.reflectometry.orso import (
    OrsoExperiment,
    OrsoOwner,
    OrsoSample,
    OrsoSampleFilenames,
)
from ess.reflectometry.types import (
    Filename,
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

    wf[ReducibleData[runtype]] = mapped[ReducibleData[runtype]].reduce(
        index=axis_name, func=_concatenate_event_lists
    )
    wf[RawChopper[runtype]] = mapped[RawChopper[runtype]].reduce(
        index=axis_name, func=_any_value
    )
    wf[SampleRotation[runtype]] = mapped[SampleRotation[runtype]].reduce(
        index=axis_name, func=_any_value
    )

    if runtype is SampleRun:
        wf[OrsoSample] = mapped[OrsoSample].reduce(index=axis_name, func=_any_value)
        wf[OrsoExperiment] = mapped[OrsoExperiment].reduce(
            index=axis_name, func=_any_value
        )
        wf[OrsoOwner] = mapped[OrsoOwner].reduce(index=axis_name, func=lambda x, *_: x)
        wf[OrsoSampleFilenames] = mapped[OrsoSampleFilenames].reduce(
            # When we don't map over filenames
            # each OrsoSampleFilenames is a list with a single entry.
            index=axis_name,
            func=_concatenate_lists,
        )
    return wf
