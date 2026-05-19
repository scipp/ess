# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)

"""Bragg peak detector handling for BIFROST."""

import scippnexus as snx
from ess.spectroscopy.types import (
    Analyzer,
    DetectorPositionOffset,
    EmptyDetector,
    NeXusComponent,
    NeXusTransformation,
    RunType,
)

from ..detector import get_base_calibrated_detector_bifrost


def get_calibrated_bragg_peak_detector(
    detector: NeXusComponent[snx.NXdetector, RunType],
    analyzer: Analyzer[RunType],
    *,
    transform: NeXusTransformation[snx.NXdetector, RunType],
    offset: DetectorPositionOffset[RunType],
) -> EmptyDetector[RunType]:
    """Extract the data array corresponding to the Bragg peak detector's signal field.

    Parameters
    ----------
    detector:
        Loaded NeXus detector.
    analyzer:
        Loaded analyzer parameters.
    transform:
        Transformation that determines the detector position.
    offset:
        Offset to add to the detector position.

    Returns
    -------
    :
        Detector with geometry coordinates.
    """
    return get_base_calibrated_detector_bifrost(
        detector, analyzer, transform=transform, offset=offset
    )


providers = (get_calibrated_bragg_peak_detector,)
