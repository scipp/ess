# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)
"""
Beamline parameters and utilities for POWGEN.
"""

import scipp as sc


def map_detector_to_spectrum(data: sc.DataArray, *,
                             detector_info: sc.DataArray) -> sc.DataArray:
    if not sc.identical(
            data.coords['detector'].to(dtype=detector_info.coords['detector'].dtype,
                                       copy=False), detector_info.coords['detector']):
        raise sc.CoordError(
            "The 'detector' coords of `data` and `detector_info` do not match.")

    out = data.copy(deep=False)
    del out.coords['detector']
    # Add 1 because spectrum numbers in the data start at 1 but
    # detector_info contains spectrum indices which start at 0.
    out.coords['spectrum'] = (
        detector_info.coords['spectrum'] +
        sc.scalar(1, dtype=detector_info.coords['spectrum'].dtype))

    return out.rename_dims({'detector': 'spectrum'})
