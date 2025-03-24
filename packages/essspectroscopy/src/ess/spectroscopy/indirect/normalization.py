# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

import sciline

from ess.reduce import time_of_flight

from ..types import (
    MonitorCoordTransformGraph,
    MonitorData,
    MonitorType,
    RunType,
    TimeOfFlightLookupTable,
    TofMonitor,
)


def unwrap_monitor(
    monitor: MonitorData[RunType, MonitorType],
    table: TimeOfFlightLookupTable,
    coord_transform_graph: MonitorCoordTransformGraph,
) -> TofMonitor[RunType, MonitorType]:
    path_length = monitor.transform_coords(
        'Ltotal',
        graph=coord_transform_graph,
        keep_intermediate=False,
        keep_aliases=False,
        rename_dims=False,
    ).coords['Ltotal']

    tof_wf = sciline.Pipeline(
        (*time_of_flight.providers(), time_of_flight.resample_tof_data),
        params={
            **time_of_flight.default_parameters(),
            time_of_flight.TimeOfFlightLookupTable: table,
            time_of_flight.Ltotal: path_length,
            time_of_flight.RawData: monitor.rename(t='tof'),
        },
    )
    unwrapped = tof_wf.compute(time_of_flight.ResampledTofData)
    return TofMonitor[RunType, MonitorType](unwrapped)


providers = (unwrap_monitor,)
