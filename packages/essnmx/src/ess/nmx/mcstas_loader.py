# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from typing import Iterable, NewType, Optional

import scipp as sc
import scippnexus as snx

PixelIDs = NewType("PixelIDs", sc.Variable)
InputFilepath = NewType("InputFilepath", str)
NMXData = NewType("NMXData", sc.DataGroup)

# McStas Configurations
MaximumProbability = NewType("MaximumProbability", int)
DefaultMaximumProbability = MaximumProbability(100_000)


def _retrieve_event_list_name(keys: Iterable[str]) -> str:
    prefix = "bank01_events_dat_list"

    # (weight, x, y, n, pixel id, time of arrival)
    mandatory_fields = 'p_x_y_n_id_t'

    for key in keys:
        if key.startswith(prefix) and mandatory_fields in key:
            return key

    raise ValueError("Can not find event list name.")


def _copy_partial_var(
    var: sc.Variable, idx: int, unit: Optional[str] = None, dtype: Optional[str] = None
) -> sc.Variable:
    """Retrieve a property from a variable."""
    var = var['dim_1', idx].astype(dtype or var.dtype, copy=True)
    if unit:
        var.unit = sc.Unit(unit)
    return var


def load_mcstas_nexus(
    file_path: InputFilepath,
    max_probability: Optional[MaximumProbability] = None,
) -> NMXData:
    """Load McStas simulation result from h5(nexus) file.

    Parameters
    ----------
    file_path:
        File name to load.

    max_probability:
        The maximum probability to scale the weights.

    """

    from .mcstas_xml import read_mcstas_geometry_xml

    geometry = read_mcstas_geometry_xml(file_path)
    probability = max_probability or DefaultMaximumProbability

    with snx.File(file_path) as file:
        bank_name = _retrieve_event_list_name(file["entry1/data"].keys())
        var: sc.Variable
        var = file["entry1/data/" + bank_name]["events"][()].rename_dims(
            {'dim_0': 'event'}
        )

        weights = _copy_partial_var(var, idx=0, unit='counts')  # p
        id_list = _copy_partial_var(var, idx=4, dtype='int64')  # id
        t_list = _copy_partial_var(var, idx=5, unit='s')  # t

        weights = (probability / weights.max()) * weights

        loaded = sc.DataArray(data=weights, coords={'t': t_list, 'id': id_list})
        coords = geometry.to_coords()
        grouped = loaded.group(coords.pop('pixel_id'))
        da = grouped.fold(dim='id', sizes={'panel': len(geometry.detectors), 'id': -1})
        da.coords.update(coords)

        return NMXData(da)
