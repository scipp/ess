# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

"""McStas input for BIFROST."""

import scipp as sc
import scippnexus as snx
from ess.spectroscopy.types import (
    EmptyDetector,
    InstrumentAngle,
    NeXusData,
    NeXusFileSpec,
    PulsePeriod,
    RawDetector,
    RunType,
    SampleAngle,
)

from ess.reduce.nexus import open_component_group
from ess.reduce.nexus.types import NeXusLocationSpec

from ..types import McStasRawDetector


def load_sample_angle(
    file_spec: NeXusFileSpec[RunType],
) -> SampleAngle[RunType]:
    """Load the rotation angle of a sample from a McStas BIFROST NeXus file."""
    return SampleAngle[RunType](_load_experiment_parameter(file_spec, "a3"))


def load_instrument_angle(
    file_spec: NeXusFileSpec[RunType],
) -> InstrumentAngle[RunType]:
    """Load the rotation angle for the BIFROST detector from a McStas NeXus file."""
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


def assemble_detector_data(
    detector: EmptyDetector[RunType],
    event_data: NeXusData[snx.NXdetector, RunType],
) -> McStasRawDetector[RunType]:
    """Custom assemble function that returns McStasDetectorData."""
    from ess.reduce.nexus.workflow import assemble_detector_data as reduce_assemble

    return McStasRawDetector[RunType](reduce_assemble(detector, event_data))


def convert_simulated_time_to_event_time_offset(
    mcstas_data: McStasRawDetector[RunType],
    pulse_period: PulsePeriod,
) -> RawDetector[RunType]:
    """Helper to make McStas simulated event data look more like real data

    McStas has the ability to track the time-of-flight from source to detector for
    every probabilistic neutron ray. This is very helpful, but unfortunately real
    instrument at ESS are not able to record the same information due to how the
    timing and data collection systems work.

    Real neutron events will record their event_time_zero most-recent-pulse reference
    time, and their event_time_offset detection time relative to that reference time.
    These two values added together give a real wall time; and information about the
    primary spectrometer is necessary to find any time-of-flight

    This function takes event data with per-event coordinate event_time_offset
    (actually McStas time-of-arrival) and converts the coordinate to be
    the time-of-arrival modulo the source repetition period.

    Notes
    -----
    If the input data has realistic event_time_offset values, this function should
    be a noop.

    Returns
    -------
    :
        A copy of the data with realistic per-event coordinate event_time_offset.
    """

    def wrap_event_time_offset(event_time_offset: sc.Variable) -> sc.Variable:
        return event_time_offset % pulse_period.to(unit=event_time_offset.unit)

    res = mcstas_data.transform_coords(
        frame_time=wrap_event_time_offset,
        rename_dims=False,
        keep_intermediate=False,
        keep_inputs=False,
    )
    return RawDetector[RunType](
        res.transform_coords(event_time_offset='frame_time', keep_inputs=False)
    )


providers = (
    assemble_detector_data,
    convert_simulated_time_to_event_time_offset,
    load_instrument_angle,
    load_sample_angle,
)
