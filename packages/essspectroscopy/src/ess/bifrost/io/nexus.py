# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)

"""NeXus input/output for BIFROST."""

import sciline
import scipp as sc
import scippnexus as snx

from ess.reduce.nexus.workflow import GenericNeXusWorkflow
from ess.spectroscopy.types import (
    InstrumentAngles,
    NeXusClass,
    NeXusFileSpec,
    RunType,
    SampleRun,
)

from ..types import (
    FrameMonitor0,
    FrameMonitor1,
    FrameMonitor2,
    FrameMonitor3,
)


# See https://github.com/scipp/essreduce/issues/98
def moderator_class_for_source() -> NeXusClass[snx.NXsource]:
    """Select NXmoderator as the source."""
    return NeXusClass[snx.NXsource](snx.NXmoderator)


def load_instrument_angles(
    file_spec: NeXusFileSpec[RunType],
) -> InstrumentAngles[RunType]:
    # TODO need mechanism in ESSreduce to load specific components of non-unique
    #  class by name
    from ess.reduce.nexus._nexus_loader import _open_nexus_file, _unique_child_group

    with _open_nexus_file(file_spec.value) as file:
        parameters = _unique_child_group(
            _unique_child_group(file, snx.NXentry, name=None),
            snx.NXparameters,
            name=None,
        )
        return InstrumentAngles[RunType](
            sc.DataGroup[sc.DataArray](
                {name: parameters[name][()]['value'] for name in ('a3', 'a4')}
            )
        )


def LoadNeXusWorkflow() -> sciline.Pipeline:
    """Workflow for loading BIFROST NeXus files."""
    workflow = GenericNeXusWorkflow(
        run_types=(SampleRun,),
        monitor_types=(
            FrameMonitor0,
            FrameMonitor1,
            FrameMonitor2,
            FrameMonitor3,
        ),
    )
    workflow.insert(moderator_class_for_source)
    workflow.insert(load_instrument_angles)
    return workflow
