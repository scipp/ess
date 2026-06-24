# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)

"""BEER McStas simulation helpers."""

from .beamline import MCSTAS_T_OFFSET, PulseShapingMode, simulation_choppers
from .load import (
    MCSTAS_PULSE_SHAPING_MODES,
    load_beer_mcstas,
    load_beer_mcstas_monitor,
    load_beer_mcstas_monitor_provider,
    load_beer_mcstas_provider,
    mcstas_chopper_delay_from_mode,
    mcstas_chopper_delay_from_mode_new_simulations,
    mcstas_detector_ltotal,
    mcstas_modulation_period_from_mode,
    mcstas_providers,
    mcstas_pulse_shaping_choppers,
    mcstas_source_position,
    pulse_shaping_mcstas_providers,
    pulse_shaping_mode_from_mcstas_mode,
)

__all__ = [
    'MCSTAS_PULSE_SHAPING_MODES',
    'MCSTAS_T_OFFSET',
    'PulseShapingMode',
    'load_beer_mcstas',
    'load_beer_mcstas_monitor',
    'load_beer_mcstas_monitor_provider',
    'load_beer_mcstas_provider',
    'mcstas_chopper_delay_from_mode',
    'mcstas_chopper_delay_from_mode_new_simulations',
    'mcstas_detector_ltotal',
    'mcstas_modulation_period_from_mode',
    'mcstas_providers',
    'mcstas_pulse_shaping_choppers',
    'mcstas_source_position',
    'pulse_shaping_mcstas_providers',
    'pulse_shaping_mode_from_mcstas_mode',
    'simulation_choppers',
]
