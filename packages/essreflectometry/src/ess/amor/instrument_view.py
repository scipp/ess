# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
import scipp as sc
import scippneutron as scn


def instrument_view(da: sc.DataArray, pixel_size: float = 0.0035, **kwargs):
    """
    Instrument view for the Amor instrument.

    Parameters
    ----------
    da:
        Data to display.
    pixel_size:
        Pixel size.
    """
    return scn.instrument_view(da, pixel_size=pixel_size, **kwargs)
