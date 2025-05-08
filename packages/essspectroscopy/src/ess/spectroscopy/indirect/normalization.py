# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

"""Normalization routines."""

import sciline

from ess.reduce import nexus, time_of_flight

from ..types import (
    MonitorCoordTransformGraph,
    MonitorData,
    MonitorTofData,
    MonitorType,
    RunType,
    SampleRun,
    TimeOfFlightLookupTable,
)


def unwrap_monitor(
    monitor: MonitorData[RunType, MonitorType],
    table: TimeOfFlightLookupTable,
    coord_transform_graph: MonitorCoordTransformGraph,
) -> MonitorTofData[RunType, MonitorType]:
    path_length = monitor.transform_coords(
        'Ltotal',
        graph=coord_transform_graph,
        keep_intermediate=False,
        keep_aliases=False,
        rename_dims=False,
    ).coords['Ltotal']

    tof_wf = sciline.Pipeline(
        (
            *time_of_flight.providers(),
            time_of_flight.resample_monitor_time_of_flight_data,
        ),
        params={
            **time_of_flight.default_parameters(),
            time_of_flight.TimeOfFlightLookupTable: table,
            time_of_flight.MonitorLtotal[SampleRun, nexus.types.Monitor1]: path_length,
            nexus.types.MonitorData[SampleRun, nexus.types.Monitor1]: monitor.rename(
                t='tof'
            ),
        },
    )
    unwrapped = tof_wf.compute(
        time_of_flight.ResampledMonitorTofData[SampleRun, nexus.types.Monitor1]
    )
    return MonitorTofData[RunType, MonitorType](unwrapped)


providers = (unwrap_monitor,)
