# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

"""
Components for diffraction experiments (powder and single crystal).
"""

import importlib.metadata

from . import uncertainty
from .correction import (
    normalize_by_monitor,
    normalize_by_proton_charge,
    normalize_by_vanadium,
)
from .correction import providers as correction_providers
from .filtering import crop_tof, filter_events
from .filtering import providers as filtering_providers
from .filtering import remove_bad_pulses
from .grouping import finalize_histogram, group_by_two_theta, merge_all_pixels
from .grouping import providers as grouping_providers
from .smoothing import lowpass

try:
    __version__ = importlib.metadata.version(__package__ or __name__)
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"

del importlib

providers = (
    *correction_providers,
    *filtering_providers,
    *grouping_providers,
    *uncertainty.providers,
)
"""Sciline providers for setting up a diffraction pipeline.

These implement basic diffraction data-reduction functionality and need to be
extended with instrument-specific and sub-technique-specific providers.
"""
del correction_providers, filtering_providers, grouping_providers

__all__ = [
    'crop_tof',
    'filter_events',
    'finalize_histogram',
    'group_by_two_theta',
    'lowpass',
    'merge_all_pixels',
    'normalize_by_monitor',
    'normalize_by_proton_charge',
    'normalize_by_vanadium',
    'remove_bad_pulses',
    'uncertainty',
]
