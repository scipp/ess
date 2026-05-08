# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

"""NeXus input/output for BIFROST."""

import warnings

import numpy as np
import scipp as sc
import scippnexus as snx

from ess.reduce.nexus import open_component_group
from ess.reduce.nexus.types import NeXusLocationSpec, TransformationTimeFilter
from ess.spectroscopy.types import (
    Analyzer,
    DynamicPosition,
    InstrumentAngle,
    NeXusClass,
    NeXusComponent,
    NeXusComponentLocationSpec,
    NeXusFileSpec,
    NeXusTransformation,
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

    See Also
    --------
    get_calibrated_analyzer:
        A provider that combines loaded analyzer data into an ``Analyzer`` object.
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


def get_calibrated_analyzer(
    analyzer_component: NeXusComponent[snx.NXcrystal, RunType],
    analyzer_transform: NeXusTransformation[snx.NXcrystal, RunType],
    analyzer_position: DynamicPosition[snx.NXcrystal, RunType],
) -> Analyzer[RunType]:
    """Collect the data for a single analyzer.

    This provider works together with :func:`load_analyzer_for_detector` and the
    generic NeXus workflow from ESSreduce.
    ``load_analyzer_for_detector`` loads a raw analyzer component.
    Then the default providers from ESSreduce extract a transform and position like
    for any other component.
    Finally, this provider combines the data into a single Analyzer object.

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


def _collapse_runs(transform: sc.DataArray, dim: str) -> sc.DataArray:
    """Collapse runs of equal values into a single value."""
    # Find indices where the data changes
    different_from_previous = np.hstack(
        [True, ~np.isclose(transform.values[:-1], transform.values[1:])]
    )
    change_indices = np.flatnonzero(different_from_previous)
    if change_indices.shape == transform.shape:
        return transform  # Return early to avoid expensive indexing
    # Get unique values
    unique_values = transform[change_indices]

    # Make bin-edges and extend range to include the whole measurement
    last = unique_values.coords[dim][-1]
    unique_values.coords[dim] = sc.concat(
        [
            # bin-edges are left-inclusive, so we can start with coord[0] as first edge
            unique_values.coords[dim],
            # Surely, no experiment will last more than 10 years...
            last + sc.scalar(10, unit='Y').to(unit=last.unit),
        ],
        dim=dim,
    )

    return unique_values


def stepwise_transformation_time_filter(transform: sc.DataArray) -> sc.DataArray:
    """Collapse runs of equal values into a single value.

    This can be used as a time filter for NeXus transformations when the component
    mostly stays at a position and only rarely moves.
    For example, a stepwise scan across detector rotations.
    """
    collapsed = _collapse_runs(transform, 'time')
    if collapsed.sizes['time'] == 1:
        return collapsed.squeeze('time')
    return collapsed


providers = (
    get_calibrated_analyzer,
    load_analyzer_for_detector,
    load_instrument_angle,
    load_sample_angle,
    moderator_class_for_source,
)

parameters = {
    TransformationTimeFilter: stepwise_transformation_time_filter,
}
