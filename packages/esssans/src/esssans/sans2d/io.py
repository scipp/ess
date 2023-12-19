# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
"""
"""

import scipp as sc

from ..common import gravity_vector
from ..types import (
    DirectBeam,
    DirectBeamFilename,
    Filename,
    LoadedDetectorContents,
    MonitorType,
    NeXusMonitorName,
    RunType,
    TofData,
    TofMonitor,
)


def pooch_load(filename: Filename[RunType]) -> LoadedDetectorContents[RunType]:
    from .data import get_path

    dg = sc.io.load_hdf5(filename=get_path(filename))
    data = dg['data']
    if 'gravity' not in data.coords:
        data.coords["gravity"] = gravity_vector()

    # Some fixes specific for these Sans2d runs
    sample_pos_z_offset = 0.053 * sc.units.m
    # There is some uncertainty here
    monitor4_pos_z_offset = -6.719 * sc.units.m

    data.coords['sample_position'].fields.z += sample_pos_z_offset
    # Results are actually slightly better at high-Q if we do not apply a bench offset
    # bench_pos_y_offset = 0.001 * sc.units.m
    # data.coords['position'].fields.y += bench_pos_y_offset
    dg['monitors']['monitor4']['data'].coords[
        'position'
    ].fields.z += monitor4_pos_z_offset
    return LoadedDetectorContents[RunType](dg)


def pooch_load_direct_beam(filename: DirectBeamFilename) -> DirectBeam:
    from .data import get_path

    return DirectBeam(sc.io.load_hdf5(filename=get_path(filename)))


def get_detector_data(
    dg: LoadedDetectorContents[RunType],
) -> TofData[RunType]:
    return TofData[RunType](dg['data'])


def get_monitor(
    dg: LoadedDetectorContents[RunType], nexus_name: NeXusMonitorName[MonitorType]
) -> TofMonitor[RunType, MonitorType]:
    # See https://github.com/scipp/sciline/issues/52 why copy needed
    mon = dg['monitors'][nexus_name]['data'].copy()
    return TofMonitor[RunType, MonitorType](mon)


providers = (
    pooch_load_direct_beam,
    pooch_load,
    get_detector_data,
    get_monitor,
)
"""
"""
