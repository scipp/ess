# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

"""NeXus input/output for BIFROST."""

from collections.abc import Iterable

import sciline
import scipp as sc
import scippnexus as snx

from ess.reduce.nexus.workflow import GenericNeXusWorkflow
from ess.spectroscopy.types import (
    Analyzer,
    InstrumentAngles,
    NeXusClass,
    NeXusComponentLocationSpec,
    NeXusDetectorName,
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
    from ess.reduce.nexus._nexus_loader import _unique_child_group, open_nexus_file

    with open_nexus_file(file_spec.value) as file:
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


def _analyzer_name_for_detector_name(
    detector_name: NeXusDetectorName, all_names: Iterable[str]
) -> str:
    detector_index = int(detector_name.split('_', 1)[0])
    analyzer_index = str(detector_index - 2)
    for name in all_names:
        if name.startswith(analyzer_index):
            return name
    raise RuntimeError(f"No analyzer found for detector {detector_name}")


def load_analyzer_for_detector(
    detector_location: NeXusComponentLocationSpec[snx.NXdetector, RunType],
) -> Analyzer[RunType]:
    """Find and load the right analyzer for a detector triplet.

    Note
    ----
    Depends heavily on the names of components being preceded by an in-instrument index,
    and the analyzer and detector components being separated in index by 2.
    If either condition changes this function will need to be modified.

    Parameters
    ----------
    detector_location:
        The location of an NXdetector in the NeXus file.
        The analyzer is identified based on this location.

    Returns
    -------
    :
        The loaded analyzer for the given detector triplet.
        Only a subset of fields is returned.
    """
    from ess.reduce.nexus._nexus_loader import _open_component_parent

    with _open_component_parent(detector_location, nx_class=snx.NXcrystal) as parent:
        analyzer_name = _analyzer_name_for_detector_name(
            detector_location.component_name, parent.keys()
        )
        analyzer = snx.compute_positions(
            parent[analyzer_name][()], store_transform='transform'
        )
    return Analyzer[RunType](
        sc.DataGroup(
            dspacing=analyzer['d_spacing'],
            position=analyzer['position'],
            transform=analyzer['transform'],
        )
    )


_PROVIDERS = (
    load_analyzer_for_detector,
    load_instrument_angles,
    moderator_class_for_source,
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
    for provider in _PROVIDERS:
        workflow.insert(provider)
    return workflow
