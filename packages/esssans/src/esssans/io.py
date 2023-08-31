# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
import scipp as sc
from .types import Filename, RunType, RawData, MonitorType, RawMonitor, NeXusMonitorName


def load(filename: Filename[RunType]) -> RawData[RunType]:
    return RawData[RunType](sc.io.load_hdf5(filename=filename))


def get_monitor(
    da: RawData[RunType], nexus_name: NeXusMonitorName[MonitorType]
) -> RawMonitor[RunType, MonitorType]:
    return RawMonitor(da.attrs[nexus_name].value)
    # TODO We get an exception about __init__ when using this with a DataArray:
    return IncidentMonitor[RunType](da.attrs['monitor2'].value)


providers = [load, get_monitor]
