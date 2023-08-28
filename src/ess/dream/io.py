# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
import os
from typing import Any, Dict, Union

import scipp as sc
import scippnexus as snx


def load_nexus(path: Union[str, os.PathLike], load_pixel_shape: bool = False):
    """Load dream_mantle_test.nxs file."""
    definitions = snx.base_definitions()
    if not load_pixel_shape:
        definitions["NXdetector"] = FilteredDetector

    with snx.File(path, definitions=definitions) as f:
        dg = f['entry'][()]
    if 'mantel' in dg['instrument']:
        dg['instrument']['mantle'] = dg['instrument'].pop('mantel')
    # The file has (counter:2, hemisphere:2, module:5, segment:6, wire:32, strip:128)
    mantle = dg['instrument']['mantle']
    comps = mantle.data.bins.constituents
    for key in ['begin', 'end']:
        comps[key] = _reorder_voxels(comps[key])
    for key in [
        'detector_number',
        'x_pixel_offset',
        'y_pixel_offset',
        'z_pixel_offset',
    ]:
        mantle.coords[key] = _reorder_voxels(mantle.coords[key])

    mantle.data = sc.bins(**comps)
    # For convenience, combine (module,segment,counter) into "subsegment"
    dg['instrument']['mantle'] = mantle.fold(
        'detector_number', sizes={'subsegment': 5 * 6 * 2, 'wire': 32, 'strip': 256}
    )
    return dg


def _reorder_voxels(var: sc.Variable) -> sc.Variable:
    # Scipp only supports init from max 4-D NumPy arrays, workaround:
    flatsizes = {
        'counter': 2,
        'hemisphere': 2,
        'dummy': 5 * 6 * 32,
        'strip': 128,
    }
    var = var.fold('detector_number', sizes=flatsizes)
    lower = var['hemisphere', 0]
    upper = var['hemisphere', 1]
    # Reverse lower hemisphere strip
    lower = sc.array(dims=lower.dims, values=lower.values[..., ::-1], unit=lower.unit)
    # Concat hemispheres into common strip index
    # Move 'counter' inside 'segment' dim to match physical ordering
    dims = ('module', 'segment', 'counter', 'wire', 'strip')
    return (
        sc.concat([lower, upper], 'strip')
        .fold('dummy', sizes={'module': 5, 'segment': 6, 'wire': 32})
        .transpose(dims)
        .flatten(to='detector_number')
    )


def _skip(name: str, obj: Union[snx.Field, snx.Group]) -> bool:
    skip_classes = (snx.NXoff_geometry,)
    return isinstance(obj, snx.Group) and (obj.nx_class in skip_classes)


class FilteredDetector(snx.NXdetector):
    def __init__(
        self, attrs: Dict[str, Any], children: Dict[str, Union[snx.Field, snx.Group]]
    ):
        children = {
            name: child for name, child in children.items() if not _skip(name, child)
        }
        super().__init__(attrs=attrs, children=children)
