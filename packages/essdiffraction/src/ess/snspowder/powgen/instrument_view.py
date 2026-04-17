# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

import scipp as sc
import scippneutron as scn


def instrument_view(
    da: sc.DataArray,
    positions: str = "position",
    pixel_size: float | None = None,
    components: dict | None = None,
    **kwargs,
):
    """
    Instrument view for the POWGEN instrument, with adjusted default arguments.

    Parameters
    ----------
    positions:
        Key for coord holding positions to use for pixels
    pixel_size:
        Custom pixel size to use for detector pixels
    components:
        Dictionary containing display names and corresponding
        settings (also a Dictionary) for additional components to display
        items with known positions to be shown
    kwargs:
        See :func:`scippneutron.instrument_view`
    """
    # TODO: the camera argument does not work with the Plopp instrument view
    # if 'camera' not in kwargs:
    #     kwargs = {
    #         **kwargs, 'camera': {
    #             'position': sc.vector(value=[-3, 3, 3],
    #                                   unit=da.coords[positions].unit)
    #         }
    #     }
    kwargs.setdefault('cbar', True)
    return scn.instrument_view(
        da, positions=positions, components=components, pixel_size=pixel_size, **kwargs
    )
