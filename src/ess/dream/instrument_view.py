# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

import plopp as pp
import scipp as sc


def instrument_view(
    data: Union[sc.DataArray, sc.DataGroup],
    dim: Optional[str] = None,
    bins: Union[sc.Variable, int] = 50,
    pixel_size: Union[sc.Variable, float] = 10,
    **kwargs
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
    dim:
        Dimension to use for the slider.
    bins:
        Number of bins to use for the slider.
    pixel_size:
        Size of the pixels. If a float is provided, it will assume the same unit as the
        pixel coordinates.
    """
    from plopp.graphics import figure3d
    import plopp.widgets as pw

    if not isinstance(data, sc.DataArray):
        data = sc.concat([da.flatten(to='pixel') for da in data.values()], dim='pixel')
    else:
        data = data.flatten(to='pixel')

    if dim is not None:
        histogrammed = data.hist({dim: bins}) if data.bins is not None else data
        slider_widget = pw.SliceWidget(histogrammed, dims=[dim])
        slider_widget.controls[dim]['slider'].layout = {"width": "400px"}
        slider_node = pp.widget_node(slider_widget)
        nodes = [pw.slice_dims(data_array=histogrammed, slices=slider_node)]
    else:
        nodes = [pp.Node(data.hist() if data.bins is not None else data)]

    fig = figure3d(*nodes, x='x', y='y', z='z', pixel_size=pixel_size, **kwargs)
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
