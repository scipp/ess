# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

"""NeXus input/output for BIFROST."""

import warnings

import numpy as np
import scipp as sc
import scippnexus as snx
from ess.spectroscopy.types import (
    Analyzer,
    Analyzers,
    ElasticMonitor,
    InstrumentAngle,
    NeXusClass,
    NeXusComponent,
    NeXusComponentLocationSpec,
    NeXusData,
    NeXusLocationSpec,
    NeXusMultiComponent,
    NeXusMultiComponentLocationSpec,
    NeXusMultiData,
    NeXusMultiTransformation,
    NeXusTransformation,
    NeXusTransformationChain,
    RunType,
    SampleAngle,
    TransformationTimeFilter,
)

from ess.reduce import nexus

from ..types import InstrumentAnglePath, SampleAngleNeXusName


def load_sample_angle(
    chain: NeXusTransformationChain[snx.NXsample, RunType], name: SampleAngleNeXusName
) -> SampleAngle[RunType]:
    """Load the rotation angle of a sample from a BIFROST NeXus file."""
    path, transform = _find_sample_rotation_in_chain(chain.transformations, name)
    if transform.transformation_type != "rotation":
        raise ValueError(
            f"Expected the sample angle at {path} to be a rotation,"
            f" got '{transform.transformation_type}' instead.'"
        )
    return SampleAngle[RunType](transform.value)  # type: ignore[return-value]


def _find_sample_rotation_in_chain(
    transformations: sc.DataGroup[snx.nxtransformations.Transform],
    name: SampleAngleNeXusName,
) -> tuple[str, snx.nxtransformations.Transform]:
    for path, transform in transformations.items():
        if name in path:
            return path, transform
    raise ValueError(
        "No sample rotation found in the NXtransformation chain of the sample. "
        f"Got entries {list(transformations.keys())}"
    )


def extract_instrument_angle(
    chain: NeXusTransformationChain[ElasticMonitor, RunType], path: InstrumentAnglePath
) -> InstrumentAngle[RunType]:
    """Extract the rotation angle for the BIFROST detector."""
    # This function extracts the angle from a transformation chain instead of loading it
    # directly from a file to avoid loading the tank angle multiple times
    # as that would cause problems in livedata.
    # We could use any detector or the elastic monitor to get a chain that contains
    # the angle. Using the monitor is simplest because there is only one, and we
    # do not have to deal with multiple detectors.
    try:
        transformation = chain.transformations[path]
    except KeyError:
        raise ValueError(
            f'The transformation chain of {chain.parent} does not contain the '
            f'detector tank angle {path}. The chain contains '
            f'{list(chain.transformations)}'
        ) from None
    if transformation.transformation_type != 'rotation':
        raise ValueError(
            f"Expected the instrument angle at {path} to be a rotation, "
            f"got '{transformation.transformation_type}' instead.'"
        )
    angle = transformation.value
    if isinstance(angle, sc.Variable):
        # A fixed angle or a value from TransformationOverrides, not an NXlog.
        angle = sc.DataArray(angle)
    return InstrumentAngle[RunType](angle)  # type: ignore[return-value]


# See https://github.com/scipp/ess/issues/120
def moderator_class_for_source() -> NeXusClass[snx.NXsource]:
    """Select NXmoderator as the source."""
    return NeXusClass[snx.NXsource](snx.NXmoderator)


def load_bifrost_analyzers(
    detector_locations: NeXusMultiComponentLocationSpec[snx.NXdetector, RunType],
    detector_components: NeXusMultiComponent[snx.NXdetector, RunType],
) -> NeXusMultiComponent[snx.NXcrystal, RunType]:
    """Load analyzer components for the given detectors.

    This function searches for an ``NXcrystal`` in the inputs (via the
    ``inputs`` dataset or attribute) of each detector and loads the first NeXus
    group it finds.

    Parameters
    ----------
    detector_locations:
        Location spec for the detectors.
        Used to locate the instrument group to be searched for analyzers.
    detector_components:
        The loaded detector components to load analyzers for.

    Returns
    -------
    :
        The loaded analyzer components.

    See Also
    --------
    .get_calibrated_analyzers:
        A provider that combines loaded analyzer data into an ``Analyzer`` object.
    """
    instrument_location = NeXusLocationSpec(
        filename=detector_locations.filename,
        entry_name=detector_locations.entry_name,
        component_name=None,
        selection=detector_locations.selection,
    )
    out = {}
    with nexus.open_component_group(
        instrument_location, nx_class=snx.NXinstrument
    ) as instrument_group:
        for detector in detector_components.values():
            detector_name = detector['nexus_component_name']
            analyzer = _find_class_in_inputs(
                instrument_group, snx.NXcrystal, instrument_group[detector_name]
            )[()]
            analyzer['detector_name'] = detector_name
            out[detector_name] = analyzer
    return NeXusMultiComponent[snx.NXcrystal, RunType](sc.DataGroup(out))


# This function is useful in the Bragg Peak monitor workflow.
def load_bifrost_analyzer_for_detector(
    detector_location: NeXusComponentLocationSpec[snx.NXdetector, RunType],
) -> NeXusComponent[snx.NXcrystal, RunType]:
    """Load the analyzer component for the given detector.

    This function searches for an ``NXcrystal`` in the inputs (via the
    ``inputs`` dataset or attribute) of the detector and loads the first NeXus
    group it finds.

    See Also
    --------
    get_calibrated_analyzer:
        A provider that combines loaded analyzer data into an ``Analyzer`` object.
    """
    with nexus.open_component_group(
        detector_location, nx_class=snx.NXdetector
    ) as det_group:
        analyzer = _find_class_in_inputs(
            group=det_group.parent, target=snx.NXcrystal, start=det_group
        )[()]
        analyzer['detector_name'] = det_group.name.rsplit('/', 1)[-1]
        return analyzer


def _find_class_in_inputs(
    group: snx.Group, target: type, start: snx.Group
) -> snx.Group:
    """Search for a NeXus class in a group's inputs.

    This function uses a breadth-first search through ``'inputs'`` datasets.
    It begins at ``start`` and walks along chains of inputs until a group with the
    given class is found, the chain ends, or the chain leads outside ``group``.

    Parameters
    ----------
    group: HDF5 Group
        The group that contains all possible next named groups
    target:
        The NeXus class to look for.
    start:
        The group to start the search from.

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
            except KeyError:  # noqa: PERF203
                warnings.warn(f"No '{name}' in NeXus group {group.name}", stacklevel=2)
                continue
    raise ValueError(f"No {target} found in the inputs of {start.name}")


def _get_inputs(group: snx.Group) -> list[str]:
    try:
        inputs = group['inputs'][()]
    except KeyError:
        try:
            # Older ESS encoding. The NeXus standard requires 'inputs' to be a dataset
            # but ESS used an attribute for BIFROST files for a while.
            inputs = group.attrs['inputs']
        except KeyError:
            return []
    if isinstance(inputs, sc.Variable):  # 1-d dataset
        inputs = list(inputs.values)
    # A scalar dataset loads as a str.
    # Also deal with nexusformat (Python module) or kafka-to-nexus (filewriter)
    # silently converting a len(list[str]) == 1 attribute to a str attribute:
    return [inputs] if isinstance(inputs, str) else inputs


def get_calibrated_analyzers(
    analyzer_components: NeXusMultiComponent[snx.NXcrystal, RunType],
    transforms: NeXusMultiTransformation[snx.NXcrystal, RunType],
) -> Analyzers[RunType]:
    """Collect the data for all given analyzer.

    Parameters
    ----------
    analyzer_components:
        Data groups of loaded analyzers.
    transforms:
        NeXus transformations for the detectors.

    Returns
    -------
    :
        Processed analyzers as a ``DataGroup``.
    """
    return Analyzers[RunType](
        sc.DataGroup(
            {
                analyzer_name: get_calibrated_analyzer(
                    component, transforms.value[analyzer_name]
                )
                for analyzer_name, component in analyzer_components.items()
            }
        )
    )


def get_calibrated_analyzer(
    analyzer_component: NeXusComponent[snx.NXcrystal, RunType],
    analyzer_transform: NeXusTransformation[snx.NXcrystal, RunType],
) -> Analyzer[RunType]:
    """Collect the data for a single analyzer.

    Parameters
    ----------
    analyzer_component:
        Data group of loaded analyzer.
    analyzer_transform:
        Transformation matrix of the analyzer.

    Returns
    -------
    :
        A given analyzer.
        Only a subset of fields is returned.
    """
    return Analyzer[RunType](
        sc.DataGroup(
            dspacing=analyzer_component['d_spacing'],
            position=nexus.workflow.compute_dynamic_position(analyzer_transform),
            transform=analyzer_transform,
            detector_name=analyzer_component['detector_name'],
        )
    )


def concat_detector_events(
    events: NeXusMultiData[snx.NXdetector, RunType],
) -> NeXusData[snx.NXdetector, RunType]:
    """Concatenate events from multiple detectors into a single data array."""
    return NeXusData[snx.NXdetector, RunType](
        sc.concat(list(events.values()), dim='event_time_zero')
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
    mostly stays at one position and only rarely moves.
    For example, a stepwise scan across detector rotations.

    Repeated values are identified using :func:`numpy.isclose` with default tolerances
    applied to the individual transformation components.
    I.e., for the BIFROST detector, the detector angle (currently in degrees)
    is checked for approximate equality between consecutive values.

    Note
    ----
    This approach is meant to handle noisy NXlogs if they are written
    from readback values or repeated setpoint values.
    We currently do not know enough about how ESS NeXus files will be written for
    real measurements, so we may need to revisit this approach.
    """
    collapsed = _collapse_runs(transform, 'time')
    if collapsed.sizes['time'] == 1:
        return collapsed.squeeze('time')
    return collapsed


providers = (
    concat_detector_events,
    get_calibrated_analyzer,
    get_calibrated_analyzers,
    extract_instrument_angle,
    load_bifrost_analyzers,
    load_bifrost_analyzer_for_detector,
    load_sample_angle,
    moderator_class_for_source,
)

parameters = {
    InstrumentAnglePath: '/entry/instrument/detector_tank/transformations/tank_angle',
    SampleAngleNeXusName: 'sample_rotation',
    TransformationTimeFilter: stepwise_transformation_time_filter,
}
