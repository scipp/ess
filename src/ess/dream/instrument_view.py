# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

from typing import Optional, Union

import plopp as pp
import scipp as sc


def instrument_view(
    data: Union[sc.DataArray, sc.DataGroup],
    x: Optional[str] = None,
    y: Optional[str] = None,
    z: Optional[str] = None,
    pos: Optional[str] = None,
    dim: Optional[str] = None,
    pixel_size: Union[sc.Variable, float] = sc.scalar(1.0, unit='cm'),
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
    from plopp.graphics import figure3d
    import plopp.widgets as pw

    dims = list(data.dims)
    if dim is not None:
        dims.remove(dim)
    to = 'pixel'
    if not isinstance(data, sc.DataArray):
        data = sc.concat([da.flatten(dims=dims, to=to) for da in data.values()], dim=to)
    else:
        data = data.flatten(dims=dims, to=to)

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

    # No need to make a copy here because one was made higher up with `flatten`.
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
