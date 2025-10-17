# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

"""NeXus input/output for BIFROST."""

import scipp as sc
import scippnexus as snx

from ess.reduce.nexus import load_all_components, open_component_group
from ess.reduce.nexus.types import NeXusAllLocationSpec, NeXusLocationSpec
from ess.spectroscopy.types import (
    Analyzer,
    Analyzers,
    InstrumentAngle,
    NeXusClass,
    NeXusComponentLocationSpec,
    NeXusFileSpec,
    RunType,
    SampleAngle,
)


# See https://github.com/scipp/essreduce/issues/98
def moderator_class_for_source() -> NeXusClass[snx.NXsource]:
    """Select NXmoderator as the source."""
    return NeXusClass[snx.NXsource](snx.NXmoderator)


def load_sample_angle(
    file_spec: NeXusFileSpec[RunType],
) -> SampleAngle[RunType]:
    return SampleAngle[RunType](_load_experiment_parameter(file_spec, "a3"))


def load_instrument_angle(
    file_spec: NeXusFileSpec[RunType],
) -> InstrumentAngle[RunType]:
    return InstrumentAngle[RunType](_load_experiment_parameter(file_spec, "a4"))


def _load_experiment_parameter(
    file_spec: NeXusFileSpec[RunType], param_name: str
) -> sc.DataArray:
    with open_component_group(
        NeXusLocationSpec(filename=file_spec.value),
        nx_class=snx.NXparameters,
        parent_class=snx.NXentry,
    ) as group:
        return group[param_name][()]['value']


def load_analyzers(file_spec: NeXusFileSpec[RunType]) -> Analyzers[RunType]:
    """Load all analyzers in a NeXus file."""
    return Analyzers[RunType](
        load_all_components(
            NeXusAllLocationSpec(filename=file_spec.value),
            nx_class=snx.NXcrystal,
        )
    )


def _get_analyzer_for_detector_name(
    detector_name: str, analyzers: Analyzers[RunType]
) -> Analyzers[RunType]:
    detector_index = int(detector_name.split('_', 1)[0])
    analyzer_index = str(detector_index - 2)
    for name, analyzer in analyzers.items():
        if name.startswith(analyzer_index):
            return analyzer
    raise RuntimeError(f"No analyzer found for detector {detector_name}")


def analyzer_for_detector(
    analyzers: Analyzers[RunType],
    detector_location: NeXusComponentLocationSpec[snx.NXdetector, RunType],
) -> Analyzer[RunType]:
    """Extract the analyzer for a given detector.

    Note
    ----
    Depends heavily on the names of components being preceded by an instrument index,
    and the analyzer and detector components being separated in index by 2.
    If either condition changes, this function will need to be modified.

    Parameters
    ----------
    analyzers:
        Data group of loaded analyzers.
    detector_location:
        The location of an NXdetector in the NeXus file.
        The analyzer is identified based on this location.

    Returns
    -------
    :
        The analyzer for the given detector triplet.
        Only a subset of fields is returned.
    """
    if detector_location.component_name is None:
        raise ValueError("Detector component name is None")
    analyzer = snx.compute_positions(
        _get_analyzer_for_detector_name(detector_location.component_name, analyzers),
        store_transform='transform',
    )
    return Analyzer[RunType](
        sc.DataGroup(
            dspacing=analyzer['d_spacing'],
            position=analyzer['position'],
            transform=analyzer['transform'],
        )
    )


providers = (
    analyzer_for_detector,
    load_analyzers,
    load_instrument_angle,
    load_sample_angle,
    moderator_class_for_source,
)
