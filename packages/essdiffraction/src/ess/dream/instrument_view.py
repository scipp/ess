# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)

import plopp as pp
import scipp as sc
from plopp.core.typing import FigureLike


def _to_data_array(
    data: sc.DataArray | sc.DataGroup | dict, dim: str | None
) -> sc.DataArray:
    if isinstance(data, sc.DataArray):
        data = sc.DataGroup({"": data})
    pieces = []
    for da in data.values():
        da = da.drop_coords(set(da.coords) - {"position", dim})
        dims = list(da.dims)
        if (dim is not None) and (dim in dims):
            # Ensure that the dims to be flattened are contiguous
            da = da.transpose([d for d in dims if d != dim] + [dim])
            dims.remove(dim)
        flat = da.flatten(dims=dims, to="pixel")
        filtered = flat[sc.isfinite(flat.coords["position"])]
        pieces.append(
            filtered.assign_coords(
                {k: getattr(filtered.coords["position"].fields, k) for k in "xyz"}
            ).drop_coords("position")
        )
    return sc.concat(pieces, dim="pixel").squeeze()


def _slice_dim(
    da: sc.DataArray, slice_params: dict[str, tuple[int, int]]
) -> sc.DataArray:
    (params,) = slice_params.items()
    return da[params[0], params[1][0] : params[1][1]].sum(params[0])


def instrument_view(
    data: sc.DataArray | sc.DataGroup | dict,
    dim: str | None = None,
    pixel_size: float | sc.Variable | None = None,
    autoscale: bool = False,
    **kwargs,
) -> FigureLike:
    from plopp.widgets import ClippingPlanes, RangeSliceWidget, ToggleTool, VBar

    data = _to_data_array(data, dim)

    if dim is not None:
        range_slicer = RangeSliceWidget(data, dims=[dim])
        slider = range_slicer.controls[dim].slider
        slider.value = 0, data.sizes[dim]
        slider.layout = {"width": "600px"}
        slider_node = pp.widget_node(range_slicer)
        to_scatter = pp.Node(_slice_dim, da=data, slice_params=slider_node)

    else:
        to_scatter = pp.Node(data)

    kwargs.setdefault('cbar', True)
    fig = pp.scatter3dfigure(
        to_scatter,
        x="x",
        y="y",
        z="z",
        pixel_size=1.0 * sc.Unit("cm") if pixel_size is None else pixel_size,
        autoscale=autoscale,
        **kwargs,
    )

    clip_planes = ClippingPlanes(fig)
    fig.toolbar['cut3d'] = ToggleTool(
        callback=clip_planes.toggle_visibility,
        icon='layer-group',
        tooltip='Hide/show spatial cutting tool',
    )
    widgets = [clip_planes]
    if dim is not None:
        widgets.append(range_slicer)

        # def _maybe_update_value_cut(_):
        #     if any(cut._direction == "v" for cut in clip_planes.cuts):
        #         clip_planes.update_state()

        # range_slicer.observe(_maybe_update_value_cut, names='value')

    fig.bottom_bar.add(VBar(widgets))

    return fig
