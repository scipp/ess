# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
import os
from typing import Union, Dict, Any
import scipp as sc
import scippnexus as snx


def load_nexus(path: Union[str, os.PathLike]):
    definitions = snx.base_definitions()
    definitions["NXdetector"] = FilteredDetector

    with snx.File(path, definitions=definitions) as f:
        dg = f['entry'][()]
    if 'mantel' in dg['instrument']:
        dg['instrument']['mantle'] = dg['instrument'].pop('mantel')
    # This file is (counter:2, hemisphere:2, module:5, segment:6, wire:32, strip:128)
    sizes = {
        'counter': 2,
        'hemisphere': 2,
        'module': 5,
        'segment': 6,
        'wire': 32,
        'strip': 128,
    }
    # Scipp only supports init from max 4-D NumPy arrays, workaround:
    flatsizes = {
        'counter': 2,
        'hemisphere': 2,
        'dummy': 5 * 6 * 32,
        'strip': 128,
    }
    mantle = dg['instrument']['mantle'].data.fold('detector_number', sizes=flatsizes)
    comps = mantle.bins.constituents
    for key in ['begin', 'end']:
        lower = comps[key]['hemisphere', 0]
        upper = comps[key]['hemisphere', 1]
        # Reverse lower hemisphere strip
        lower = sc.array(dims=lower.dims, values=lower.values[..., ::-1], unit=None)
        # Concat hemispheres into common strip index
        # Move 'counter' inside 'segment' dim to match physical ordering
        dims = ('module', 'segment', 'counter', 'wire', 'strip')
        comps[key] = (
            sc.concat([lower, upper], 'strip')
            .fold('dummy', sizes={'module': 5, 'segment': 6, 'wire': 32})
            .transpose(dims)
            .copy()
        )

    dg['instrument']['mantle'].data = sc.bins(**comps).flatten(to='detector_number')
    # For convenience, combine (module,segment,counter) into "subsegment"
    dg['instrument']['mantle'] = dg['instrument']['mantle'].fold(
        'detector_number', sizes={'subsegment': 5 * 6 * 2, 'wire': 32, 'strip': 256}
    )
    return dg


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
