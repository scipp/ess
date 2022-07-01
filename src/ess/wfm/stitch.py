# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)
import scipp as sc
from typing import Union


def _stitch_dense_data(item: sc.DataArray, frames: sc.Dataset, dim: str, new_dim: str,
                       bins: Union[int, sc.Variable]) -> Union[sc.DataArray, dict]:

    # Make empty data container
    if isinstance(bins, int):
        new_coord = sc.linspace(
            dim=new_dim,
            start=(frames["time_min"]["frame", 0] -
                   frames["time_correction"]["frame", 0]).value,
            stop=(frames["time_max"]["frame", -1] -
                  frames["time_correction"]["frame", -1]).value,
            num=bins + 1,
            unit=frames["time_min"].unit,
        )
    else:
        new_coord = bins

    dims = []
    shape = []
    for dim_ in item.dims:
        if dim_ != dim:
            dims.append(dim_)
            shape.append(item.sizes[dim_])
        else:
            dims.append(new_dim)
            shape.append(new_coord.sizes[new_dim] - 1)

    out = sc.DataArray(data=sc.zeros(dims=dims,
                                     shape=shape,
                                     with_variances=item.variances is not None,
                                     unit=item.unit),
                       coords={new_dim: new_coord})
    for group in ["coords", "attrs"]:
        for key in getattr(item, group):
            if key != dim:
                getattr(out, group)[key] = getattr(item, group)[key].copy()

    for i in range(frames.sizes["frame"]):
        section = item[dim, frames["time_min"].data["frame", i]:frames["time_max"].data[
            "frame", i]].rename_dims({dim: new_dim})
        section.coords[new_dim] = section.coords[dim] - frames["time_correction"].data[
            "frame", i]
        if new_dim != dim:
            del section.coords[dim]

        out += sc.rebin(section, new_dim, out.coords[new_dim])

    return out


def _stitch_event_data(item: sc.DataArray, frames: sc.Dataset, dim: str, new_dim: str,
                       bins: Union[int, sc.Variable]) -> Union[sc.DataArray, dict]:

    edges = sc.flatten(sc.transpose(sc.concat(
        [frames["time_min"].data, frames["time_max"].data], 'dummy'),
                                    dims=['frame', 'dummy']),
                       to=dim)

    binned = sc.bin(item, edges=[edges])

    for i in range(frames.sizes["frame"]):
        binned[dim, i * 2].bins.coords[dim] -= frames["time_correction"].data["frame",
                                                                              i]

    erase = None
    if new_dim != dim:
        binned.bins.coords[new_dim] = binned.bins.coords[dim]
        del binned.bins.coords[dim]
        erase = [dim]

    binned.masks['frame_gaps'] = (sc.arange(dim, 2 * frames.sizes["frame"] - 1) %
                                  2).astype(bool)
    binned.masks['frame_gaps'].unit = None

    new_edges = sc.concat([
        (frames["time_min"]["frame", 0] - frames["time_correction"]["frame", 0]).data,
        (frames["time_max"]["frame", -1] - frames["time_correction"]["frame", -1]).data
    ], new_dim)
    return sc.bin(binned, edges=[new_edges], erase=erase)


def _stitch_item(item: sc.DataArray, frames: sc.Dataset,
                 **kwargs) -> Union[sc.DataArray, dict]:

    if item.bins is not None:
        out = _stitch_event_data(item=item, frames=frames, **kwargs)
    else:
        out = _stitch_dense_data(item=item, frames=frames, **kwargs)

    # Update source position.
    # TODO: Note that this is a hack, to make sure the unit conversion to wavelength
    # computes the correct L1.
    # Because we cannot, in a nice way, tell the difference between cases where one
    # needs to set L1 (because `scatter=True` will be used in the conversion later),
    # and when one needs to set `Ltotal` instead (because `scatter=False` is required
    # for e.g. imaging).
    # Once we support general conversion graphs in the unit conversion of scippneutron,
    # we should stop modifying the coordinate here, and change to using a specialized
    # WFM conversion graph that looks for `wfm_chopper_mid_point` in the coords.
    if "source_position" in item.coords:
        del out.coords["source_position"]
    out.coords['source_position'] = frames["wfm_chopper_mid_point"].data
    return out


def stitch(
        data: Union[sc.DataArray, sc.Dataset],
        dim: str,
        frames: sc.Dataset,
        new_dim: str = 'tof',
        bins: Union[int, sc.Variable] = 256) -> Union[sc.DataArray, sc.Dataset, dict]:
    """
    Convert raw arrival time WFM data to time-of-flight by shifting each frame
    (described by the `frames` argument) by a time offset defined by the position
    of the WFM choppers.
    This process is also known as 'stitching' the frames.

    :param data: The DataArray or Dataset to be stitched.
    :param dim: The dimension along which the stitching will be performed.
    :param frames: The Dataset containing the information on the frame boundaries.
    :param new_dim: New dimension of the returned data, after stitching. Default: 'tof'.
    :param bins: Number or Variable describing the bins for the returned data. Default:
        256.
    """

    # TODO: for now, if frames depend on positions, we take the mean along the
    # position dimensions. We should implement the position-dependent stitching
    # in the future.
    frames = frames.copy()
    dims_to_reduce = list(set(frames.dims) - {'frame'})
    for _dim in dims_to_reduce:
        frames["time_min"] = sc.mean(frames["time_min"], _dim)
        frames["time_max"] = sc.mean(frames["time_max"], _dim)

    if isinstance(data, sc.Dataset):
        stitched = sc.Dataset()
        for i, (key, item) in enumerate(data.items()):
            stitched[key] = _stitch_item(item=item,
                                         dim=dim,
                                         frames=frames,
                                         new_dim=new_dim,
                                         bins=bins)
    else:
        stitched = _stitch_item(item=data,
                                dim=dim,
                                frames=frames,
                                new_dim=new_dim,
                                bins=bins)

    return stitched
