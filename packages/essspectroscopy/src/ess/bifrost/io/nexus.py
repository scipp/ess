# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

"""NeXus input/output for BIFROST."""

import scipp as sc
import scippnexus as snx

from ess.bifrost.types import DetectorAnalyzerMap
from ess.reduce.nexus import load_all_components, open_component_group, open_nexus_file
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


def _do_breadth_first_search(group, targets, obj_deque, obj_next):
    """
    Look for a unique element of targets by following the 'next' for object in a queue

    Parameters
    ----------
    group: HDF5 Group
        The group that contains all possible next named groups
    targets:
        A structure with named targets that supports `name in targets`
    obj_deque:
        A queue.deque of HDF5 Groups to be checked
    obj_next:
        A function that extracts a list of named groups to check from a given group
    """
    while len(obj_deque) > 0:
        check = obj_next(obj_deque.popleft())
        matches = [element for element in check if element in targets]
        if len(matches) > 1:
            raise ValueError("Non-unique elmeent match")
        if len(matches) == 1:
            return matches[0]
        for element in check:
            obj_deque.append(group[element])
    raise ValueError("No unique element found")


def analyzer_search(hdf5_instrument_group, analyzers, hdf5_detector_group):
    """
    Use a NeXus Group's @inputs attribute to find an analyzer given a detector group

    Parameters
    ----------
    hdf5_instrument_group: hdf5.Group
        works if inside of a context group
        ```
        scippnexus.File(filename, 'r') as f:
            hdf5_instrument_group = f['/entry/instrument']
        ```
    analyzers: Anything with __contains__(str), e.g. dict[str, hdf5.Group]
        Something to identify whether we've found a valid analyzer (by name)
    hdf5_detector_group: hdf5.Group
        any of f['/entry/detector'][scippnexus.NXdectector].values()
    """
    from queue import deque

    from h5py import Group

    def obj_inputs(obj: Group) -> list[str]:
        """Return the specified preceding component(s) list"""
        if 'inputs' not in obj.attrs:
            raise ValueError('@inputs attribute required for this search to work')
        val = obj.attrs['inputs']
        # Deal with nexusformat (Python module) or kafka-to-nexus (filewriter)
        # silently converting a len(list[str]) == 1 attribute to a str attribute:
        return [val] if isinstance(val, str) else val

    return _do_breadth_first_search(
        hdf5_instrument_group, analyzers, deque([hdf5_detector_group]), obj_inputs
    )


def get_detector_analyzer_map(
    file_spec: NeXusFileSpec[RunType],
) -> DetectorAnalyzerMap[RunType]:
    """Probably not the right sciline way to do this."""

    from scippnexus import NXcrystal, NXdetector

    filename = file_spec.value
    with open_nexus_file(filename) as file:
        inst = file['entry/instrument']
        analyzers = inst[NXcrystal]
        detectors = inst[NXdetector]
        return {k: analyzer_search(inst, analyzers, v) for k, v in detectors.items()}


def analyzer_for_detector(
    analyzers: Analyzers[RunType],
    detector_location: NeXusComponentLocationSpec[snx.NXdetector, RunType],
    detector_analyzer_map: DetectorAnalyzerMap[RunType],
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
    if (
        analyzer_name := detector_analyzer_map.get(detector_location.component_name)
    ) is None:
        raise RuntimeError(
            f"No analyzer found for detector {detector_location.component_name}"
        )
    analyzer = snx.compute_positions(
        analyzers[analyzer_name],
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
    get_detector_analyzer_map,
)
