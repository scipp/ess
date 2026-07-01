# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)

import enum
from types import MappingProxyType
from typing import TypedDict

import scipp as sc
from scippneutron.chopper import DiskChopper


class PulseShapingMode(enum.Enum):
    """Pulse-shaping chopper modes for BEER."""

    ps0 = "PS0"
    ps1 = "PS1"
    ps2 = "PS2"
    ps3 = "PS3"
    ds1 = "DS1"


class _ChopperParameters(TypedDict):
    frequency: float
    phase: float
    distance: float
    open: list[float]
    close: list[float]


Hz = sc.Unit("Hz")
deg = sc.Unit("deg")
meter = sc.Unit("m")

MCSTAS_T_OFFSET = sc.scalar(1.6, unit="ms")
"""Time offset applied by the BEER McStas simulation source model."""

_HIGH_FLUX: dict[str, _ChopperParameters] = {
    "PSC1": {
        "frequency": 168.0,
        "phase": 318.6929881679336,
        "distance": 6.450,
        "open": [0.0],
        "close": [144.0],
    },
    "PSC3": {
        "frequency": -168.0,
        "phase": -318.6929881679336,
        "distance": 7.375,
        "open": [0.0],
        "close": [144.0],
    },
    "FC1A": {
        "frequency": -28.0,
        "phase": -18.44878787209148,
        "distance": 8.283,
        "open": [0.0],
        "close": [72.0],
    },
    "FC2A": {
        "frequency": -14.0,
        "phase": -134.52965314925247,
        "distance": 79.975,
        "open": [0.0],
        "close": [175.0],
    },
}

_parameters: dict[PulseShapingMode, dict[str, _ChopperParameters]] = {
    PulseShapingMode.ps0: _HIGH_FLUX,
    PulseShapingMode.ps1: _HIGH_FLUX,
    PulseShapingMode.ps2: {
        "PSC1": {
            "frequency": 168.0,
            "phase": 310.265456971683,
            "distance": 6.450,
            "open": [0.0],
            "close": [144.0],
        },
        "PSC2": {
            "frequency": -168.0,
            "phase": -310.265456971683,
            "distance": 6.850,
            "open": [0.0],
            "close": [144.0],
        },
        "FC1A": {
            "frequency": -28.0,
            "phase": -18.44878787209148,
            "distance": 8.283,
            "open": [0.0],
            "close": [72.0],
        },
        "FC2A": {
            "frequency": -14.0,
            "phase": -134.52965314925247,
            "distance": 79.975,
            "open": [0.0],
            "close": [175.0],
        },
    },
    PulseShapingMode.ps3: {
        "PSC1": {
            "frequency": 168.0,
            "phase": 307.05496889692084,
            "distance": 6.450,
            "open": [0.0],
            "close": [144.0],
        },
        "PSC2": {
            "frequency": -168.0,
            "phase": -307.05496889692084,
            "distance": 6.650,
            "open": [0.0],
            "close": [144.0],
        },
        "FC1A": {
            "frequency": -28.0,
            "phase": -18.44878787209148,
            "distance": 8.283,
            "open": [0.0],
            "close": [72.0],
        },
        "FC2A": {
            "frequency": -14.0,
            "phase": -134.52965314925247,
            "distance": 79.975,
            "open": [0.0],
            "close": [175.0],
        },
    },
    PulseShapingMode.ds1: {
        "PSC1": {
            "frequency": 168.0,
            "phase": 318.6929881679336,
            "distance": 6.450,
            "open": [0.0],
            "close": [144.0],
        },
        "PSC3": {
            "frequency": -168.0,
            "phase": -318.6929881679336,
            "distance": 7.375,
            "open": [0.0],
            "close": [144.0],
        },
        "FC1A": {
            "frequency": -14.0,
            "phase": -3.22439393604574,
            "distance": 8.283,
            "open": [0.0],
            "close": [72.0],
        },
        "FC1B": {
            "frequency": -63.0,
            "phase": -46.41910994173803,
            "distance": 8.317,
            "open": [0.0],
            "close": [180.0],
        },
        "FC2B": {
            "frequency": -7.0,
            "phase": -68.58171174285046,
            "distance": 80.025,
            "open": [0.0],
            "close": [85.0],
        },
    },
}


def simulation_choppers(
    mode: PulseShapingMode, source_position: sc.Variable
) -> MappingProxyType[str, DiskChopper]:
    """
    Dict of ESS BEER McStas choppers for the selected pulse-shaping mode.

    We make the chopper information available in this way as loading it directly from
    the NeXus files is currently not available for these simulated BEER data.

    Parameters
    ----------
    mode:
        BEER pulse-shaping chopper mode.
    source_position:
        Position of the source in the coordinate system of the choppers.
        The raw chopper positions are defined relative to the source position.
    """
    return MappingProxyType(
        {
            key: DiskChopper(
                frequency=ch["frequency"] * Hz,
                beam_position=sc.scalar(0.0, unit="deg"),
                phase=ch["phase"] * deg,
                axle_position=sc.vector(value=[0, 0, ch["distance"]], unit="m")
                + source_position,
                slit_begin=sc.array(dims=["cutout"], values=ch["open"], unit="deg"),
                slit_end=sc.array(dims=["cutout"], values=ch["close"], unit="deg"),
            )
            for key, ch in _parameters[mode].items()
        }
    )
