# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

import os
from typing import Any, Dict, Union

import scipp as sc
import scippnexus as snx


def load_nexus(
    path: Union[str, os.PathLike],
    *,
    load_pixel_shape: bool = False,
    entry: str = 'entry',
    fold_detectors: bool = True,
) -> sc.DataGroup:
    """
    Load an unprocessed DREAM NeXus file.

    See https://confluence.esss.lu.se/pages/viewpage.action?pageId=462000005
    and the ICD DREAM interface specification for details.

    Notes (2023-12-06):

    - Mounting-unit, cassette, and counter roughly correspond to the azimuthal angle
      in the mantle detector. However, counter is reversed in the current files. This
      may also be the case in the other detectors.
    - The endcap detectors have a irregular structure that cannot be fully folded.
      This is not a problem but note again that the counter may be reversed. It is
      currently not clear if this is a bug in the file.
    - The high-resolution detector has a very odd numbering scheme. The SANS detector
      is using the same, but is not populated in the current files. The scheme
      attempts to follows some sort of physical ordering in space (x,y,z), but it
      looks partially messed up.

    Parameters
    ----------
    path:
        Path to the NeXus file.
    load_pixel_shape:
        If True, load the pixel shape from the file's NXoff_geometry group. This is
        often unused by would slow down file loading. Default is False.
    entry:
        Name of the entry to load. Default is "entry".
    fold_detectors:
        If True, fold the detector data to (partially) mimic the logical detector
        structure. Default is True.

    Returns
    -------
    :
        A data group with the loaded file contents.
    """
    definitions = snx.base_definitions()
    if not load_pixel_shape:
        definitions["NXdetector"] = FilteredDetector

    with snx.File(path, definitions=definitions) as f:
        dg = f[entry][()]
    dg = snx.compute_positions(dg)
    return fold_nexus_detectors(dg) if fold_detectors else dg


def fold_nexus_detectors(dg: sc.DataGroup) -> sc.DataGroup:
    """
    Fold the detector data in a DREAM NeXus file.

    The detector banks in the returned data group will have a multi-dimensional shape,
    following the logical structure as far as possible. Note that the full structure
    cannot be folded, as some dimensions are irregular.
    """
    dg = dg.copy()
    dg['instrument'] = dg['instrument'].copy()
    instrument = dg['instrument']
    mantle = instrument['mantle_detector']
    mantle['mantle_event_data'] = mantle['mantle_event_data'].fold(
        dim='detector_number',
        sizes={
            'wire': 32,
            'mounting_unit': 5,
            'cassette': 6,
            'counter': 2,
            'strip': 256,
        },
    )
    for direction in ('backward', 'forward'):
        endcap = instrument[f'endcap_{direction}_detector']
        endcap[f'endcap_{direction}_event_data'] = endcap[
            f'endcap_{direction}_event_data'
        ].fold(
            dim='detector_number',
            sizes={
                'strip': 16,
                'wire': 16,
                'sector': 5 if direction == 'forward' else 11,
                'sumo_cass_ctr': -1,  # sumo*cassette*counter, irregular, cannot fold
            },
        )
    high_resolution = instrument['high_resolution_detector']
    high_resolution['high_resolution_event_data'] = high_resolution[
        'high_resolution_event_data'
    ].fold(
        dim='detector_number',
        sizes={
            'strip': 32,
            'other': -1,  # some random order that is hard to follow
        },
    )
    sans = instrument['sans_detector']
    sans['sans_event_data'] = sans['sans_event_data'].fold(
        dim='detector_number',
        sizes={
            'strip': 32,
            'other': -1,  # some random order that is hard to follow
        },
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
