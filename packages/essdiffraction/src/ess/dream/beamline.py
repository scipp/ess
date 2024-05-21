# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

"""Beamline tools for DREAM."""

from ess.powder.types import NeXusDetectorDimensions, NeXusDetectorName

DETECTOR_BANK_SHAPES_DAY1 = {
    "endcap_backward": {
        "strip": 16,
        "wire": 16,
        "module": 11,
        "segment": 28,
        "counter": 2,
    },
    "endcap_forward": {
        "strip": 16,
        "wire": 16,
        "module": 5,
        "segment": 28,
        "counter": 2,
    },
    "mantle": {"wire": 32, "module": 5, "segment": 6, "strip": 256, "counter": 2},
    # TODO: missing "high_resolution" and "sans" detectors
}


def dream_detector_dimensions_day1(
    detector_name: NeXusDetectorName,
) -> NeXusDetectorDimensions[NeXusDetectorName]:
    """Logical dimensions of a NeXus DREAM detector for the day 1 configuration."""
    return NeXusDetectorDimensions(DETECTOR_BANK_SHAPES_DAY1[detector_name])


providers = (dream_detector_dimensions_day1,)
"""Sciline providers for DREAM detector handling."""
