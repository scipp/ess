# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

"""Beamline tools for DREAM."""

from ess.powder.types import DetectorDimensions, DetectorName


def dream_detector_dimensions(detector_name: DetectorName) -> DetectorDimensions:
    """Logical dimensions used by a DREAM detector.

    Parameters
    ----------
    detector_name:
        Name of a detector of DREAM.

    Returns
    -------
    :
        Logical dimensions used by the given DREAM detector.
    """
    base = ('module', 'segment', 'counter', 'wire', 'strip')
    if detector_name == DetectorName.high_resolution:
        return DetectorDimensions(base + ('sector',))
    return DetectorDimensions(base)


providers = (dream_detector_dimensions,)
"""Sciline providers for DREAM detector handling."""
