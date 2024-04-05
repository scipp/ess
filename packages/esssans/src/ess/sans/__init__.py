# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)

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
from .i_of_q import merge_banks, merge_runs, no_bank_merge, no_run_merge
from .types import BackgroundSubtractedIofQ, IofQ, ReturnEvents, SampleRun

try:
    __version__ = importlib.metadata.version(__package__ or __name__)
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"

providers = (
    *conversions.providers,
    *i_of_q.providers,
    *masking.providers,
    *normalization.providers,
)
"""
List of providers for setting up a Sciline pipeline.

This provides a default workflow, including a beam-center estimation based on a
center-of-mass approach. Providers for loadings files are not included. Combine with
the providers for a specific instrument, such as :py:data:`esssans.sans2d.providers`
to setup a complete workflow.
"""

del importlib

__all__ = [
    'BackgroundSubtractedIofQ',
    'IofQ',
    'SampleRun',
    'ReturnEvents',
    'beam_center_finder',
    'beam_center_from_center_of_mass',
    'common',
    'conversions',
    'direct_beam',
    'i_of_q',
    'io',
    'masking',
    'merge_banks',
    'merge_runs',
    'no_bank_merge',
    'no_run_merge',
    'normalization',
    'providers',
]
