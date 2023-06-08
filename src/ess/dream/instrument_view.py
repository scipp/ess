# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

import plopp as pp
import scipp as sc


def instrument_view(data, dim=None, bins=50, pixel_size=10, **kwargs):
    from plopp.graphics import figure3d
    import plopp.widgets as pw

    if not isinstance(data, sc.DataArray):
        data = sc.concat([da.flatten(to='pixel') for da in data.values()], dim='pixel')
    else:
        data = data.flatten(to='pixel')

    if dim is not None:
        histogrammed = data.hist({dim: bins})
        slider_widget = pw.SliceWidget(histogrammed, dims=[dim])
        slider_node = pp.widget_node(slider_widget)
        nodes = [pw.slice_dims(data_array=histogrammed, slices=slider_node)]
    else:
        nodes = [pp.Node(data.hist())]

    out = [figure3d(*nodes, x='x', y='y', z='z', pixel_size=pixel_size, **kwargs)]
    if dim is not None:
        out.append(slider_widget)
    return pw.Box(out)
