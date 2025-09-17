# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

"""NeXus input/output for BIFROST."""

from collections.abc import Iterable

import scipp as sc
import scippnexus as snx

from ess.spectroscopy.types import (
    Analyzer,
    InstrumentAngle,
    NeXusClass,
    NeXusComponentLocationSpec,
    NeXusDetectorName,
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
    # TODO need mechanism in ESSreduce to load specific components of non-unique
    #  class by name
    from ess.reduce.nexus._nexus_loader import _unique_child_group, open_nexus_file

    with open_nexus_file(file_spec.value) as file:
        parameters = _unique_child_group(
            _unique_child_group(file, snx.NXentry, name=None),
            snx.NXparameters,
            name=None,
        )
        return parameters[param_name][()]['value']


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


providers = (
    load_analyzer_for_detector,
    load_instrument_angle,
    load_sample_angle,
    moderator_class_for_source,
)
