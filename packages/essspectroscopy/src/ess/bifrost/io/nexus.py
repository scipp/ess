# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

"""NeXus input/output for BIFROST."""

import warnings

import scipp as sc
import scippnexus as snx

from ess.reduce.nexus import open_component_group
from ess.reduce.nexus.types import NeXusLocationSpec
from ess.spectroscopy.types import (
    Analyzer,
    InstrumentAngle,
    NeXusClass,
    NeXusComponent,
    NeXusComponentLocationSpec,
    NeXusFileSpec,
    NeXusTransformation,
    Position,
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


def load_analyzer_for_detector(
    detector_location: NeXusComponentLocationSpec[snx.NXdetector, RunType],
) -> NeXusComponent[snx.NXcrystal, RunType]:
    """Load the analyzer component for the given detector.

    This function searches for an ``NXcrystal`` in the inputs (via the
    'input' attribute) of the detector and loads the first NeXus group it finds.
    """
    with open_component_group(detector_location, nx_class=snx.NXdetector) as det_group:
        analyzer_group = _find_class_in_inputs(
            group=det_group.parent, target=snx.NXcrystal, start=det_group
        )
        return analyzer_group[()]


def _find_class_in_inputs(
    group: snx.Group, target: type, start: snx.Group
) -> snx.Group:
    """Search for a NeXus class in a group's inputs.

    This function uses a breadth-first search through ``'input'`` attributes.
    It begins at ``start`` and walks along chains of inputs until a group with the
    given class is found, the chain ends, or the chain leads outside ``group``.

    Parameters
    ----------
    group: HDF5 Group
        The group that contains all possible next named groups
    target:
        The NeXus class to look for.

    Returns
    -------
    :
        The group with the target NeXus class found within ``group``.
    """
    pending = [start]
    while pending:
        element = pending.pop(0)
        if element.nx_class == target:
            return element
        for name in _get_inputs(element):
            try:
                pending.append(group[name])
            except KeyError:
                warnings.warn(f"No '{name}' in NeXus group {group.name}", stacklevel=2)
                continue
    raise ValueError(f"No {target} found in the inputs of {start.name}")


def _get_inputs(group: snx.Group) -> list[str]:
    try:
        inputs = group.attrs['inputs']
    except KeyError:
        return []
    # Deal with nexusformat (Python module) or kafka-to-nexus (filewriter)
    # silently converting a len(list[str]) == 1 attribute to a str attribute:
    return [inputs] if isinstance(inputs, str) else inputs


# This function is separate from load_analyzer_for_detector so we get the default
# behavior for resolving NXtransformations.
def get_calibrated_analyzer(
    analyzer_component: NeXusComponent[snx.NXcrystal, RunType],
    analyzer_transform: NeXusTransformation[snx.NXcrystal, RunType],
    analyzer_position: Position[snx.NXcrystal, RunType],
) -> Analyzer[RunType]:
    """Collect the data for a single analyzer.

    Parameters
    ----------
    analyzer_component:
        Data group of loaded analyzers.
    analyzer_transform:
        Transformation matrix of the analyzer.
    analyzer_position:
        The computed position vector of the analyzer.

    Returns
    -------
    :
        A given analyzer.
        Only a subset of fields is returned.
    """

    return Analyzer[RunType](
        sc.DataGroup(
            dspacing=analyzer_component['d_spacing'],
            position=analyzer_position,
            transform=analyzer_transform,
        )
    )


providers = (
    get_calibrated_analyzer,
    load_analyzer_for_detector,
    load_instrument_angle,
    load_sample_angle,
    moderator_class_for_source,
)
