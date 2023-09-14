# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

from typing import List, Optional, Union

import plopp as pp
import scipp as sc

DREAM_DETECTOR_DIMENSIONS = ('module', 'segment', 'counter', 'wire', 'strip')
DREAM_PIXEL_SIZE = sc.scalar(1.0, unit='cm')


def _preprocess_data(
    data: sc.DataArray, to_be_flattened: List[str], dim: str, to: str
) -> sc.DataArray:
    """
    The 3D scatter visualization requires a flattened one-dimensional data array.
    This function flattens the data array along the dimensions that are known to be
    detector dimensions.
    Because flattening can only happen for contiguous dimensions, the data array is
    transposed to the correct dimension order before flattening.

    Parameters
    ----------
    data:
        Data to be flattened.
    to_be_flattened:
        List of dimensions to be flattened.
    dim:
        Dimension to be used for the slider (this will not be flattened)
    to:
        Name of the new dimension to which the data will be flattened.
    """
    if not to_be_flattened:
        # Need to return a copy here because `flatten` makes a copy below.
        return data.copy(deep=False)
    transpose = list(data.dims)
    if dim is not None:
        # Move slider dim to the end of the list to allow flattening of the other dims
        transpose.remove(dim)
        transpose.append(dim)
    return data.transpose(dims=transpose).flatten(dims=to_be_flattened, to=to)


def instrument_view(
    data: Union[sc.DataArray, sc.DataGroup],
    x: Optional[str] = None,
    y: Optional[str] = None,
    z: Optional[str] = None,
    pos: Optional[str] = None,
    dim: Optional[str] = None,
    pixel_size: Union[sc.Variable, float] = DREAM_PIXEL_SIZE,
    **kwargs,
):
    """
    Three-dimensional visualization of the DREAM instrument pixels.
    By default, the data counts will be integrated for all tofs/wavelengths.
    It is possible to add a tof/wavelength slider by specifying the ``dim`` and ``bins``
    arguments (see parameters below).

    Parameters
    ----------
    data:
        Data to visualize.
    x:
        The name of the coordinate that is to be used for the X positions.
        Default is 'x'.
    y:
        The name of the coordinate that is to be used for the Y positions.
        Default is 'y'.
    z:
        The name of the coordinate that is to be used for the Z positions.
        Default is 'z'.
    pos:
        The name of the vector coordinate that is to be used for the positions.
    dim:
        Dimension to use for the slider.
    pixel_size:
        Size of the pixels. If a float is provided, it will assume the same unit as the
        pixel coordinates.
    **kwargs:
        Additional arguments to pass to the plopp figure
        (see https://scipp.github.io/plopp/about/generated/plopp.scatter3d.html).
    """
    import plopp.widgets as pw
    from plopp.graphics import figure3d

    dims = [d for d in data.dims if (d in DREAM_DETECTOR_DIMENSIONS) and (d != dim)]
    to = 'pixel'
    if not isinstance(data, sc.DataArray):
        data = sc.concat(
            [
                _preprocess_data(data=da, to_be_flattened=dims, dim=dim, to=to)
                for da in data.values()
            ],
            dim=to,
        )
    else:
        data = _preprocess_data(data=data, to_be_flattened=dims, dim=dim, to=to)

    if pos is not None:
        if any((x, y, z)):
            raise ValueError(
                f'If pos ({pos}) is defined, all of '
                f'x ({x}), y ({y}), and z ({z}) must be None.'
            )
        coords = {
            (x := f'{pos}.x'): data.coords[pos].fields.x,
            (y := f'{pos}.y'): data.coords[pos].fields.y,
            (z := f'{pos}.z'): data.coords[pos].fields.z,
        }
    else:
        x = x if x is not None else 'x'
        y = y if y is not None else 'y'
        z = z if z is not None else 'z'
        coords = {k: data.coords[k] for k in (x, y, z)}

    # No need to make a copy here because one was made higher up with `preprocess_data`.
    data.coords.update(coords)

    if dim is not None:
        slider_widget = pw.SliceWidget(data, dims=[dim])
        slider_widget.controls[dim]['slider'].layout = {"width": "400px"}
        slider_node = pp.widget_node(slider_widget)
        nodes = [pw.slice_dims(data_array=data, slices=slider_node)]
    else:
        nodes = [pp.Node(data)]

    fig = figure3d(*nodes, x=x, y=y, z=z, pixel_size=pixel_size, **kwargs)
    tri_cutter = pw.TriCutTool(fig)
    fig.toolbar['cut3d'] = pw.ToggleTool(
        callback=tri_cutter.toggle_visibility,
        icon='cube',
        tooltip='Hide/show spatial cutting tool',
    )
    out = [fig, tri_cutter]
    if dim is not None:
        out.append(slider_widget)
    return pw.Box(out)
