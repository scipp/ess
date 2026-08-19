# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)

import pytest
import scipp as sc
import scippnexus as snx
from ess.beer.mcstas.beamline import (
    ModulationMode,
    PulseShapingMode,
    simulation_choppers,
)
from ess.beer.types import SampleRun

from ess.reduce import unwrap
from ess.reduce.nexus.types import Position
from ess.reduce.unwrap import lut


@pytest.mark.parametrize("mode", [*PulseShapingMode, *ModulationMode])
def test_can_make_analytical_lookup_table_from_beer_choppers(
    mode: PulseShapingMode | ModulationMode,
) -> None:
    wf = unwrap.GenericUnwrapWorkflow(
        run_types=[SampleRun], monitor_types=[], wavelength_from="analytical"
    )
    source_position = sc.vector([0.0, 0.0, 0.0], unit="m")
    wf[Position[snx.NXsource, SampleRun]] = source_position
    wf[unwrap.DiskChoppers[SampleRun]] = simulation_choppers(mode, source_position)
    wf[lut.LtotalRange[SampleRun, snx.NXdetector]] = (
        sc.scalar(150.0, unit="m"),
        sc.scalar(151.0, unit="m"),
    )
    wf[lut.DistanceResolution] = sc.scalar(1.0, unit="m")
    wf[lut.TimeResolution] = sc.scalar(1000.0, unit="us")

    lookup = wf.compute(unwrap.LookupTable[SampleRun, snx.NXdetector])

    assert lookup.array.sizes["distance"] > 0
    assert lookup.array.sizes["event_time_offset"] > 0
    assert sc.any(sc.isfinite(lookup.array.data)).value
