# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)

"""NeXus input/output for BIFROST."""

import sciline
import scippnexus as snx

from ess.reduce.nexus import load_component
from ess.reduce.nexus.workflow import GenericNeXusWorkflow
from ess.spectroscopy.types import (
    NeXusComponent,
    NeXusComponentLocationSpec,
    RunType,
)


# TODO can override nx_class_for_source instead
# See https://github.com/scipp/essreduce/issues/98
def load_nexus_source_from_moderator(
    location: NeXusComponentLocationSpec[snx.NXsource, RunType],
) -> NeXusComponent[snx.NXsource, RunType]:
    """Load a NeXus moderator as a source."""
    return NeXusComponent[snx.NXsource, RunType](
        load_component(location, nx_class=snx.NXmoderator)
    )


def LoadNeXusWorkflow() -> sciline.Pipeline:
    """Workflow for loading BIFROST NeXus files."""
    workflow = GenericNeXusWorkflow()
    workflow.insert(load_nexus_source_from_moderator)
    return workflow
