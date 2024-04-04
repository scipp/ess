# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
import scipp as sc
import scippneutron as scn

from .beamline import instrument_view_components


def instrument_view(
    da: sc.DataArray, components: dict = None, pixel_size: float = 0.0035, **kwargs
):
    """
    Instrument view for the Amor instrument, which automatically populates a list of
    additional beamline components, and sets the pixel size.

    :param da: The input data for which to display the instrument view.
    :param components: A dict of additional components to display. By default, a
        set of components defined in `beamline.instrument_view_components()` are added.
    :param pixel_size: The detector pixel size. Default is 0.0035.
    """
    default_components = instrument_view_components(da)
    if components is not None:
        default_components = {**default_components, **components}

    return scn.instrument_view(
        da, components=default_components, pixel_size=pixel_size, **kwargs
    )
