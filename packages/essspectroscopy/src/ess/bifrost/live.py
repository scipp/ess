# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

"""Live data reduction workflows for BIFROST."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import NewType

import sciline
import scipp as sc
import scippnexus as snx

from ess.reduce.streaming import EternalAccumulator, StreamProcessor
from ess.spectroscopy.types import (
    EnergyData,
    InstrumentAngles,
    NeXusData,
    NeXusDetectorName,
    RunType,
    SampleRun,
)

from .workflow import BifrostWorkflow


@dataclass(frozen=True, kw_only=True, slots=True)
class CutAxis:
    """Axis and binds for cutting 4D Q-E data."""

    output: str
    """Name of the output coordinate."""
    fn: Callable[[...], sc.Variable]
    """Function to perform the cut.

    Used in :func:`scipp.transform_coords` and so should request input coordinates
    by name.
    """
    bins: sc.Variable
    """Bin edges for the cut."""

    @classmethod
    def from_q_vector(cls, output: str, vec: sc.Variable, bins: sc.Variable):
        """Construct from an arbitrary direction in Q."""
        vec = vec / sc.norm(vec)
        return cls(
            output=output,
            fn=lambda sample_table_momentum_transfer: sc.dot(
                vec, sample_table_momentum_transfer
            ),
            bins=bins,
        )


CutAxis1 = NewType('CutAxis1', CutAxis)
CutAxis2 = NewType('CutAxis2', CutAxis)


class CutData(sciline.Scope[RunType, sc.DataArray], sc.DataArray):
    """Data that was cut along CutAxis1 and CutAxis2."""


def cut(
    data: EnergyData[RunType], *, axis_1: CutAxis1, axis_2: CutAxis2
) -> CutData[RunType]:
    """Cut data along two axes."""
    new_coords = {axis_1.output, axis_2.output}
    projected = data.bins.concat().transform_coords(
        new_coords,
        graph={axis_1.output: axis_1.fn, axis_2.output: axis_2.fn},
        keep_inputs=False,
    )
    projected = projected.drop_coords(list(set(projected.coords.keys()) - new_coords))
    return CutData[RunType](
        projected.hist({axis_2.output: axis_2.bins, axis_1.output: axis_1.bins})
    )


def BIFROSTQCutWorkflow(detector_names: list[NeXusDetectorName]) -> sciline.Pipeline:
    """Workflow for BIFROST to compute cuts in Q-E-space."""
    workflow = BifrostWorkflow(detector_names)
    workflow.insert(cut)
    return workflow


def BIFROSTQCutStreamProcessor(workflow: sciline.Pipeline) -> StreamProcessor:
    return StreamProcessor(
        workflow,
        dynamic_keys=(NeXusData[snx.NXdetector, SampleRun],),
        context_keys=(InstrumentAngles[SampleRun],),
        target_keys=(CutData[SampleRun],),
        accumulators=(CutData[SampleRun],),
    )
