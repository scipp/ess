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


class ModulationMode(enum.Enum):
    """Modulation chopper modes for BEER."""

    m0 = "M0"
    m1 = "M1"
    m2 = "M2"
    m3 = "M3"
    m4 = "M4"
    ds0 = "DS0"


class _ChopperParameters(TypedDict):
    frequency: float
    phase: float
    distance: float
    open: list[float]
    close: list[float]


Hz = sc.Unit("Hz")
deg = sc.Unit("deg")

MCSTAS_T_OFFSET = sc.scalar(1.6, unit="ms")
"""Time offset applied by the BEER McStas simulation source model."""

_PULSE_SHAPING_HIGH_FLUX: dict[str, _ChopperParameters] = {
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

# McStas modes 7--10 use the eight-opening MCA configurations.
_MODULATION_HIGH_FLUX: dict[str, _ChopperParameters] = {
    "FC1A": {
        "frequency": -28.0,
        "phase": -18.44878787209148,
        "distance": 8.283,
        "open": [0.0],
        "close": [72.0],
    },
    "MCA": {
        "frequency": -70.0,
        "phase": -162.22641289703336,
        "distance": 9.300,
        "open": [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0],
        "close": [5.0, 50.0, 95.0, 140.0, 185.0, 230.0, 275.0, 320.0],
    },
    "FC2A": {
        "frequency": -14.0,
        "phase": -134.52965314925247,
        "distance": 79.975,
        "open": [0.0],
        "close": [175.0],
    },
}

_parameters: dict[PulseShapingMode | ModulationMode, dict[str, _ChopperParameters]] = {
    PulseShapingMode.ps0: _PULSE_SHAPING_HIGH_FLUX,
    PulseShapingMode.ps1: _PULSE_SHAPING_HIGH_FLUX,
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
    ModulationMode.m0: _MODULATION_HIGH_FLUX,
    ModulationMode.m1: _MODULATION_HIGH_FLUX,
    ModulationMode.m2: {
        "FC1A": {
            "frequency": -28.0,
            "phase": -18.44878787209148,
            "distance": 8.283,
            "open": [0.0],
            "close": [72.0],
        },
        "MCA": {
            "frequency": -140.0,
            "phase": -326.9528257940667,
            "distance": 9.300,
            "open": [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0],
            "close": [5.0, 50.0, 95.0, 140.0, 185.0, 230.0, 275.0, 320.0],
        },
        "FC2A": {
            "frequency": -14.0,
            "phase": -134.52965314925247,
            "distance": 79.975,
            "open": [0.0],
            "close": [175.0],
        },
    },
    ModulationMode.m3: {
        "FC1A": {
            "frequency": -28.0,
            "phase": -18.44878787209148,
            "distance": 8.283,
            "open": [0.0],
            "close": [72.0],
        },
        "MCA": {
            "frequency": -280.0,
            "phase": -656.4056515881334,
            "distance": 9.300,
            "open": [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0],
            "close": [5.0, 50.0, 95.0, 140.0, 185.0, 230.0, 275.0, 320.0],
        },
        "FC2A": {
            "frequency": -14.0,
            "phase": -134.52965314925247,
            "distance": 79.975,
            "open": [0.0],
            "close": [175.0],
        },
    },
    ModulationMode.m4: {
        "FC1A": {
            "frequency": -28.0,
            "phase": -18.44878787209148,
            "distance": 8.283,
            "open": [0.0],
            "close": [72.0],
        },
        "MCA": {
            "frequency": -140.0,
            "phase": -326.9528257940667,
            "distance": 9.300,
            "open": [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0],
            "close": [5.0, 50.0, 95.0, 140.0, 185.0, 230.0, 275.0, 320.0],
        },
        "MCB": {
            "frequency": -280.0,
            "phase": -659.0810583171018,
            "distance": 9.350,
            "open": [
                0.0,
                22.5,
                45.0,
                67.5,
                90.0,
                112.5,
                135.0,
                157.5,
                180.0,
                202.5,
                225.0,
                247.5,
                270.0,
                292.5,
                315.0,
                337.5,
            ],
            "close": [
                5.0,
                27.5,
                50.0,
                72.5,
                95.0,
                117.5,
                140.0,
                162.5,
                185.0,
                207.5,
                230.0,
                252.5,
                275.0,
                297.5,
                320.0,
                342.5,
            ],
        },
        "FC2A": {
            "frequency": -14.0,
            "phase": -134.52965314925247,
            "distance": 79.975,
            "open": [0.0],
            "close": [175.0],
        },
    },
    ModulationMode.ds0: {
        "FC1A": {
            "frequency": -7.0,
            "phase": 22.86286291805168,
            "distance": 8.283,
            "open": [0.0],
            "close": [72.0],
        },
        "MCC": {
            "frequency": -70.0,
            "phase": -109.43563284346213,
            "distance": 9.875,
            "open": [0.0, 22.5, 45.0, 67.5, 90.0, 112.5, 135.0, 160.0],
            "close": [5.0, 27.5, 50.0, 72.5, 95.0, 117.5, 140.0, 340.0],
        },
        "FC2A": {
            "frequency": -7.0,
            "phase": -120.30881252309756,
            "distance": 79.975,
            "open": [0.0],
            "close": [175.0],
        },
    },
}


def simulation_choppers(
    mode: PulseShapingMode | ModulationMode, source_position: sc.Variable
) -> MappingProxyType[str, DiskChopper]:
    """
    Dict of ESS BEER McStas choppers for the selected chopper mode.

    We make the chopper information available in this way as loading it directly from
    the NeXus files is currently not available for these simulated BEER data.

    Parameters
    ----------
    mode:
        BEER pulse-shaping or modulation chopper mode.
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
