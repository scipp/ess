# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

import sciline
import scipp as sc

from ess.reduce import time_of_flight

from ..types import (
    Filename,
    MonitorCoordTransformGraph,
    MonitorData,
    MonitorName,
    MonitorPosition,
    MonitorType,
    NormWavelengthEvents,
    RunType,
    SourceMonitorPathLength,
    SourcePosition,
    TimeOfFlightLookupTable,
    TofMonitor,
    WavelengthBins,
    WavelengthEvents,
    WavelengthMonitor,
)


def monitor_coordinate_transformation_graph() -> MonitorCoordTransformGraph:
    from scippneutron.conversion.graph import beamline, tof

    return MonitorCoordTransformGraph(
        {
            **beamline.beamline(scatter=False),
            **tof.elastic_wavelength(start='tof'),
        }
    )


def monitor_position(file: Filename, monitor: MonitorName) -> MonitorPosition:
    """Extract the position of the named monitor from a NeXus file"""
    from scippnexus import File, compute_positions

    with File(file) as data:
        return compute_positions(data['entry/instrument'][monitor][...])['position']


def source_monitor_path_length(
    file: Filename, source: SourcePosition, monitor: MonitorPosition
) -> SourceMonitorPathLength:
    """Compute the primary spectrometer path length from source to monitor positions

    Note:
        This *requires* that the instrument group *is sorted* along the beam path.
        HDF5 group entries are sorted alphabetically, so you should ensure that
        the NeXus file was constructed with this in mind.
    """
    from scippnexus import File, NXguide, compute_positions

    with File(file) as data:
        positions = [
            compute_positions(v[...])['position']
            for v in data['entry/instrument'][NXguide].values()
        ]

    # Find the closest guide to the monitor position, ignoring the possibility that
    # a guide could be _beyond_ the monitor _and_ closest :(
    closest = 0
    distance = sc.norm(source - monitor)
    for i, position in enumerate(positions):
        d = sc.norm(position - monitor)
        if d < distance:
            distance = d
            closest = i

    positions = sc.concat((source, *positions[:closest], monitor), dim='path')
    diff = positions['path', 1:] - positions['path', :-1]
    return sc.sum(sc.norm(diff))


def normalise(
    events: WavelengthEvents,
    monitor: WavelengthMonitor,
    edges: WavelengthBins,
) -> NormWavelengthEvents:
    """Ensure the WavelengthEvents are binned according to the WavelengthBins, then use
    the WavelengthMonitor as a lookup table to record the per-bin normalization.

    Parameters
    ----------
    events
        Event data which includes a per-event sloth coordinate
    monitor
        (Probably) 1-D histogram data with a sloth coordinate
    edges
        1-D bin boundaries for the event data

    Returns
    -------
    :
        Event data binned by sloth, with a coordinate that is the per-bin normalization
    """
    from scipp import lookup

    dim = 'incident_wavelength'
    centres = (edges[:-1] + edges[1:]) / 2
    variances = None
    if monitor.variances is not None:
        monitor = monitor.copy()
        variances = monitor.copy()
        variances.values = monitor.variances
        monitor.variances = None
        variances.variances = None
    counts = lookup(monitor, dim)[centres]
    if variances is not None:
        # Bad form, maybe. But two events with the same sloth bin are normalized
        # by the same monitor counts -- which has a known uncertainty -- and scipp
        # refuses to allow lookup on data which has variances defined.
        counts.variances = (lookup(variances, dim)[centres]).values

    binned = events.bin(**{dim: edges})
    binned.coords['monitor'] = counts
    return binned


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


providers = (
    monitor_position,
    source_monitor_path_length,
    normalise,
    unwrap_monitor,
)
