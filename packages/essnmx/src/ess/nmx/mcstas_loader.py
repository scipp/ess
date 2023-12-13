# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from typing import Iterable, NewType, Optional

import scipp as sc
import scippnexus as snx

from .detector import NumberOfDetectors

PixelIDs = NewType("PixelIDs", sc.Variable)
InputFilename = NewType("InputFilename", str)
NMXData = NewType("NMXData", sc.DataArray)

# McStas Configurations
MaximumProbability = NewType("MaximumProbability", int)
DefaultMaximumProbability = MaximumProbability(100_000)


def _find_all(name: str, *properties: str) -> bool:
    if not properties:
        return True
    return name.find(properties[0]) != -1 and _find_all(name, *properties[1:])


def _retrieve_event_list_name(keys: Iterable[str]) -> str:
    prefix = "bank01_events_dat_list"
    mandatory_keys = ('_p', '_x', '_y', '_n', '_id', '_t')

    for key in keys:
        if key.startswith(prefix) and _find_all(
            key.removeprefix(prefix), *mandatory_keys
        ):
            return key

    raise ValueError("Can not find event list name.")


def _copy_partial_var(
    var: sc.Variable, idx: int, unit: Optional[str] = None, dtype: Optional[str] = None
) -> sc.Variable:
    """Retrieve property from variable."""
    original_var = var['dim_1', idx].copy()
    var = original_var.astype(dtype) if dtype else original_var
    if unit:
        var.unit = sc.Unit(unit)
    return var


def _get_mcstas_pixel_ids() -> PixelIDs:
    """pixel IDs for each detector"""
    intervals = [(1, 1638401), (2000001, 3638401), (4000001, 5638401)]
    ids = [sc.arange('id', start, stop, unit=None) for start, stop in intervals]
    return PixelIDs(sc.concat(ids, 'id'))


def load_mcstas_nmx_file(
    file_name: InputFilename,
    max_prop: Optional[MaximumProbability] = None,
    num_panels: Optional[NumberOfDetectors] = None,
) -> NMXData:
    """Load McStas NMX data from h5 file.

    Parameters
    ----------
    file:
        The file to load.

    max_prop:
        The maximum probability to scale the weights.

    num_panels:
        The number of detector panels.
        The data will be folded into number of panels.

    """

    prop = max_prop or DefaultMaximumProbability
    panels = num_panels or NumberOfDetectors(3)

    with snx.File(file_name) as file:
        bank_name = _retrieve_event_list_name(file["entry1/data"].keys())
        var: sc.Variable
        var = file["entry1/data/" + bank_name]["events"][()].rename_dims(
            {'dim_0': 'event'}
        )

        weights = _copy_partial_var(var, idx=0, unit='counts')
        id_list = _copy_partial_var(var, idx=4, dtype='int64')
        t_list = _copy_partial_var(var, idx=5, unit='s')

        weights = (prop / weights.max()) * weights

        loaded = sc.DataArray(data=weights, coords={'t': t_list, 'id': id_list})
        grouped = loaded.group(_get_mcstas_pixel_ids())

        return NMXData(grouped.fold(dim='id', sizes={'panel': panels, 'id': -1}))
