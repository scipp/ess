# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

# flake8: noqa
import importlib.metadata

try:
    __version__ = importlib.metadata.version(__package__ or __name__)
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"

from . import (
    beam_center_finder,
    common,
    conversions,
    i_of_q,
    io,
    loki,
    normalization,
    sans2d,
)
from .common import transmission_from_background_run, transmission_from_sample_run
from .direct_beam import direct_beam
from .types import BackgroundSubtractedIofQ, IofQ, ReturnEvents, SampleRun

providers = (
    *conversions.providers,
    *i_of_q.providers,
    *normalization.providers,
    # Default to fast but potentially inaccurate beam center finder
    beam_center_finder.beam_center_from_center_of_mass,
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
    'common',
    'conversions',
    'direct_beam',
    'i_of_q',
    'io',
    'loki',
    'normalization',
    'providers',
    'sans2d',
    'transmission_from_sample_run',
    'transmission_from_background_run',
]
