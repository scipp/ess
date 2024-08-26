# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
"""
Beamline parameters and utilities for POWGEN.
"""

import scipp as sc

from ...powder.types import (
    DetectorBankSizes,
    NeXusDetectorName,
)

DETECTOR_BANK_SIZES = {"powgen_detector": {"bank": 23, "column": 154, "row": 7}}


def map_detector_to_spectrum(
    data: sc.Dataset, *, detector_info: sc.Dataset
) -> sc.Dataset:
    """
    Transform 'detector' coords to 'spectrum'.

    Parameters
    ----------
    data:
        Input data whose 'detector' coord is transformed.
    detector_info:
        Defines mapping from detector numbers to spectra.

    Returns
    -------
    :
        `data` with 'detector' coord and dim replaced by 'spectrum'.
    """
    if not sc.identical(
        data.coords["detector"].to(
            dtype=detector_info.coords["detector"].dtype, copy=False
        ),
        detector_info.coords["detector"],
    ):
        raise sc.CoordError(
            "The 'detector' coords of `data` and `detector_info` do not match."
        )

    out = data.copy(deep=False)
    del out.coords["detector"]
    # Add 1 because spectrum numbers in the data start at 1 but
    # detector_info contains spectrum indices which start at 0.
    out.coords["spectrum"] = detector_info.coords["spectrum"] + sc.index(
        1, dtype=detector_info.coords["spectrum"].dtype
    )

    return out.rename_dims({"detector": "spectrum"})


def powgen_detector_dimensions(
    detector_name: NeXusDetectorName,
) -> DetectorBankSizes:
    """Dimensions used by POWGEN detectors."""
    return DetectorBankSizes(DETECTOR_BANK_SIZES[detector_name])


providers = (powgen_detector_dimensions,)
"""Sciline providers for POWGEN beamline processing."""
