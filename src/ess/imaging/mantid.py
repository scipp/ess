# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

import operator
from functools import reduce

import scipp as sc
import scippneutron as scn


def load_component_info_to_2d(geometry_file, sizes, advanced_geometry=False):
    """Load geometry information from a mantid Instrument Definition File
    or a NeXuS file containing instrument geometry. and reshape into 2D
    physical dimensions.

    This function requires mantid-framework to be installed

    :param geometry_file: IDF or NeXus file
    :param sizes: Dict of dim to size for output
    :return Dataset containing items for positions,
        rotations and shapes
    :raises ImportError if mantid cannot be imported
    :raises ValueError if sizes argument invalid
    """
    from mantid.simpleapi import LoadEmptyInstrument

    ws = LoadEmptyInstrument(Filename=geometry_file, StoreInADS=False)
    source_pos, sample_pos = scn.mantid.make_component_info(ws)
    geometry = sc.Dataset()
    geometry["source_position"] = source_pos
    geometry["sample_position"] = sample_pos
    pos, rot, shp = scn.mantid.get_detector_properties(
        ws,
        source_pos,
        sample_pos,
        spectrum_dim='spectrum',
        advanced_geometry=advanced_geometry,
    )
    pos_shape = pos.shape[0]
    reshape_volume = reduce(operator.mul, sizes.values(), 1)
    if not pos_shape == reshape_volume:
        raise ValueError(
            f'file contains {pos_shape} spectra, but you are attempting\
            to reshape to an output with volume\
            {reshape_volume} via sizes argument'
        )
    fold_args = {'dim': 'spectrum', 'dims': sizes.keys(), 'shape': sizes.values()}
    pos2d = sc.fold(pos, **fold_args)
    geometry["position"] = pos2d
    if rot is not None:
        rot2d = sc.fold(rot, **fold_args)
        geometry["rotation"] = rot2d
    if shp is not None:
        shp2d = sc.fold(shp, **fold_args)
        geometry["shape"] = shp2d
    # Could be upgraded, but logic complex for mapping to fields of vector
    # We therefore limit the element coord generation to x, y, z only
    if set(sizes.keys()).issubset({'x', 'y', 'z'}):
        for u, v in zip(sizes.keys(), reversed(list(sizes.keys()))):
            geometry[u] = getattr(pos2d.fields, u)[v, 0]
    return geometry
