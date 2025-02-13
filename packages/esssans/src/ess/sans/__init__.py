# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
# ruff: noqa: E402, F401, I

import importlib.metadata

from . import (
    beam_center_finder,
    common,
    conversions,
    i_of_q,
    io,
    masking,
    normalization,
)
from .beam_center_finder import beam_center_from_center_of_mass
from .direct_beam import direct_beam
from .types import BackgroundSubtractedIofQ, IofQ, ReturnEvents, SampleRun
from .workflow import (
    SansWorkflow,
    providers,
    with_background_runs,
    with_banks,
    with_pixel_mask_filenames,
    with_sample_runs,
)

try:
    __version__ = importlib.metadata.version("esssans")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"

del importlib

__all__ = [
    'BackgroundSubtractedIofQ',
    'IofQ',
    'ReturnEvents',
    'SampleRun',
    'SansWorkflow',
    'beam_center_finder',
    'beam_center_from_center_of_mass',
    'common',
    'conversions',
    'direct_beam',
    'i_of_q',
    'io',
    'masking',
    'normalization',
    'providers',
    'with_background_runs',
    'with_banks',
    'with_pixel_mask_filenames',
    'with_sample_runs',
]
