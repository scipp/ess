# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

# from __future__ import annotations

import sciline
from scippneutron.metadata import Software

from ess.powder.types import (
    BunkerMonitor,
    CaveMonitor,
    EmptyCanRun,
    ReducerSoftware,
    SampleRun,
    TimeOfFlightLookupTableFilename,
    VanadiumRun,
)
from ess.reduce.nexus.types import DetectorBankSizes, NeXusName
from ess.reduce.time_of_flight import GenericTofWorkflow

from ..beamline import InstrumentConfiguration

DETECTOR_BANK_SIZES = {
    "endcap_backward_detector": {
        "strip": 16,
        "wire": 16,
        "module": 11,
        "segment": 28,
        "counter": 2,
    },
    "endcap_forward_detector": {
        "strip": 16,
        "wire": 16,
        "module": 5,
        "segment": 28,
        "counter": 2,
    },
    "mantle_detector": {
        "wire": 32,
        "module": 5,
        "segment": 6,
        "strip": 256,
        "counter": 2,
    },
    "high_resolution_detector": {"strip": 32, "other": -1},
    "sans_detector": {"strip": 32, "other": -1},
}


def _get_lookup_table_filename_from_configuration(
    configuration: InstrumentConfiguration,
) -> TimeOfFlightLookupTableFilename:
    from ..data import tof_lookup_table_high_flux

    match configuration:
        case InstrumentConfiguration.high_flux_BC215:
            out = tof_lookup_table_high_flux(bc=215)
        case InstrumentConfiguration.high_flux_BC240:
            out = tof_lookup_table_high_flux(bc=240)
        case InstrumentConfiguration.high_resolution:
            raise NotImplementedError("High resolution configuration not yet supported")

    return TimeOfFlightLookupTableFilename(out)


def _collect_reducer_software() -> ReducerSoftware:
    return ReducerSoftware(
        [
            Software.from_package_metadata('essdiffraction'),
            Software.from_package_metadata('scippneutron'),
            Software.from_package_metadata('scipp'),
        ]
    )


def DreamGenericWorkflow() -> sciline.Pipeline:
    """
    Dream generic workflow with default parameters.
    The workflow is based on the GenericTofWorkflow.
    """
    wf = GenericTofWorkflow(
        run_types=[SampleRun, VanadiumRun, EmptyCanRun],
        monitor_types=[BunkerMonitor, CaveMonitor],
    )
    wf[DetectorBankSizes] = DETECTOR_BANK_SIZES
    wf[NeXusName[BunkerMonitor]] = "monitor_bunker"
    wf[NeXusName[CaveMonitor]] = "monitor_cave"
    wf.insert(_get_lookup_table_filename_from_configuration)
    wf[ReducerSoftware] = _collect_reducer_software()
    return wf


__all__ = ['DreamGenericWorkflow']
