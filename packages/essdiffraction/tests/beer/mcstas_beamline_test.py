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
from ess.beer.mcstas.load import chopper_mode_from_mcstas_mode
from ess.beer.types import SampleRun
from scipp.testing import assert_allclose

from ess.reduce import unwrap
from ess.reduce.nexus.types import Position
from ess.reduce.unwrap import lut


def _assert_chopper_parameters(
    mode: PulseShapingMode | ModulationMode,
    name: str,
    frequency: float,
    phase: float,
    distance: float,
    slit_begin: list[float],
    slit_end: list[float],
) -> None:
    source_position = sc.vector(value=[1.0, 2.0, 3.0], unit="m")

    chopper = simulation_choppers(mode, source_position)[name]

    assert_allclose(chopper.frequency, sc.scalar(frequency, unit="Hz"))
    assert_allclose(chopper.phase, sc.scalar(phase, unit="deg"))
    assert_allclose(
        chopper.axle_position,
        sc.vector(value=[1.0, 2.0, 3.0 + distance], unit="m"),
    )
    assert_allclose(
        chopper.slit_begin,
        sc.array(dims=["cutout"], values=slit_begin, unit="deg"),
    )
    assert_allclose(
        chopper.slit_end, sc.array(dims=["cutout"], values=slit_end, unit="deg")
    )


@pytest.mark.parametrize(
    ("mode", "name", "frequency", "phase", "distance", "slit_end"),
    [
        (PulseShapingMode.ps0, "PSC1", 168.0, 318.6929881679336, 6.450, 144.0),
        (PulseShapingMode.ps0, "PSC3", -168.0, -318.6929881679336, 7.375, 144.0),
        (PulseShapingMode.ps0, "FC1A", -28.0, -18.44878787209148, 8.283, 72.0),
        (PulseShapingMode.ps0, "FC2A", -14.0, -134.52965314925247, 79.975, 175.0),
        (PulseShapingMode.ps1, "PSC1", 168.0, 318.6929881679336, 6.450, 144.0),
        (PulseShapingMode.ps1, "PSC3", -168.0, -318.6929881679336, 7.375, 144.0),
        (PulseShapingMode.ps1, "FC1A", -28.0, -18.44878787209148, 8.283, 72.0),
        (PulseShapingMode.ps1, "FC2A", -14.0, -134.52965314925247, 79.975, 175.0),
        (PulseShapingMode.ps2, "PSC1", 168.0, 310.265456971683, 6.450, 144.0),
        (PulseShapingMode.ps2, "PSC2", -168.0, -310.265456971683, 6.850, 144.0),
        (PulseShapingMode.ps2, "FC1A", -28.0, -18.44878787209148, 8.283, 72.0),
        (PulseShapingMode.ps2, "FC2A", -14.0, -134.52965314925247, 79.975, 175.0),
        (PulseShapingMode.ps3, "PSC1", 168.0, 307.05496889692084, 6.450, 144.0),
        (PulseShapingMode.ps3, "PSC2", -168.0, -307.05496889692084, 6.650, 144.0),
        (PulseShapingMode.ps3, "FC1A", -28.0, -18.44878787209148, 8.283, 72.0),
        (PulseShapingMode.ps3, "FC2A", -14.0, -134.52965314925247, 79.975, 175.0),
        (PulseShapingMode.ds1, "PSC1", 168.0, 318.6929881679336, 6.450, 144.0),
        (PulseShapingMode.ds1, "PSC3", -168.0, -318.6929881679336, 7.375, 144.0),
        (PulseShapingMode.ds1, "FC1A", -14.0, -3.22439393604574, 8.283, 72.0),
        (PulseShapingMode.ds1, "FC1B", -63.0, -46.41910994173803, 8.317, 180.0),
        (PulseShapingMode.ds1, "FC2B", -7.0, -68.58171174285046, 80.025, 85.0),
    ],
)
def test_pulse_shaping_chopper_parameters(
    mode: PulseShapingMode,
    name: str,
    frequency: float,
    phase: float,
    distance: float,
    slit_end: float,
) -> None:
    _assert_chopper_parameters(
        mode, name, frequency, phase, distance, [0.0], [slit_end]
    )


@pytest.mark.parametrize(
    ("mode", "frequency", "phase"),
    [
        (ModulationMode.m0, -70.0, -162.22641289703336),
        (ModulationMode.m1, -70.0, -162.22641289703336),
        (ModulationMode.m2, -140.0, -326.9528257940667),
        (ModulationMode.m3, -280.0, -656.4056515881334),
    ],
)
def test_m0_to_m3_chopper_parameters(
    mode: ModulationMode, frequency: float, phase: float
) -> None:
    _assert_chopper_parameters(
        mode, "FC1A", -28.0, -18.44878787209148, 8.283, [0.0], [72.0]
    )
    _assert_chopper_parameters(
        mode,
        "MCA",
        frequency,
        phase,
        9.300,
        [float(angle) for angle in range(0, 360, 45)],
        [float(angle) for angle in range(5, 360, 45)],
    )
    _assert_chopper_parameters(
        mode, "FC2A", -14.0, -134.52965314925247, 79.975, [0.0], [175.0]
    )


def test_m4_chopper_parameters() -> None:
    mode = ModulationMode.m4
    _assert_chopper_parameters(
        mode, "FC1A", -28.0, -18.44878787209148, 8.283, [0.0], [72.0]
    )
    _assert_chopper_parameters(
        mode,
        "MCA",
        -140.0,
        -326.9528257940667,
        9.300,
        [float(angle) for angle in range(0, 360, 45)],
        [float(angle) for angle in range(5, 360, 45)],
    )
    _assert_chopper_parameters(
        mode,
        "MCB",
        -280.0,
        -659.0810583171018,
        9.350,
        [angle * 22.5 for angle in range(16)],
        [5.0 + angle * 22.5 for angle in range(16)],
    )
    _assert_chopper_parameters(
        mode, "FC2A", -14.0, -134.52965314925247, 79.975, [0.0], [175.0]
    )


def test_mcstas_mode_16_is_m4() -> None:
    assert chopper_mode_from_mcstas_mode('16') == ModulationMode.m4


def test_ds0_chopper_parameters() -> None:
    mode = ModulationMode.ds0
    _assert_chopper_parameters(
        mode, "FC1A", -7.0, 22.86286291805168, 8.283, [0.0], [72.0]
    )
    _assert_chopper_parameters(
        mode,
        "MCC",
        -70.0,
        -109.43563284346213,
        9.875,
        [0.0, 22.5, 45.0, 67.5, 90.0, 112.5, 135.0, 160.0],
        [5.0, 27.5, 50.0, 72.5, 95.0, 117.5, 140.0, 340.0],
    )
    _assert_chopper_parameters(
        mode, "FC2A", -7.0, -120.30881252309756, 79.975, [0.0], [175.0]
    )


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
