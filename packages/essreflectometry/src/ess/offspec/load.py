# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import scipp as sc

from ..reflectometry.types import Filename, RawDetectorData, ReferenceRun, RunType
from .types import CoordTransformationGraph, MonitorData


def load_offspec_events(
    filename: Filename[RunType],
) -> RawDetectorData[RunType]:
    full = sc.io.load_hdf5(filename)
    da = full['data']
    da.coords['theta'] = full.pop('Theta')[-1].data
    da = da.bins.concat('tof')
    return da


def load_offspec_monitor(
    filename: Filename[RunType],
    graph: CoordTransformationGraph[ReferenceRun],
) -> MonitorData[RunType]:
    full = sc.io.load_hdf5(filename)
    mon = full["monitors"]["monitor2"]["data"].transform_coords(
        "wavelength", graph=graph
    )
    return mon


providers = (
    load_offspec_events,
    load_offspec_monitor,
)
