# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from dataclasses import dataclass
from typing import Iterable, NewType, Optional

import scipp as sc
import scippnexus as snx

PixelIDs = NewType("PixelIDs", sc.Variable)
InputFilename = NewType("InputFilename", str)
Events = NewType("Events", sc.DataArray)

# McStas Configurations
MaximumProbability = NewType("MaximumProbability", int)
DefaultMaximumProbability = MaximumProbability(100_000)


def _retrieve_event_list_name(keys: Iterable[str]) -> str:
    for key in keys:
        if key.startswith("bank01_events_dat_list"):
            return key
    raise ValueError("Can not find event list name.")


def _copy_partial_var(
    var: sc.Variable, dim: str, idx: int, unit: str, dtype: Optional[str] = None
) -> sc.Variable:
    """Retrieve property from variable."""
    original_var = var[dim, idx].copy()
    var = original_var.astype(dtype) if dtype else original_var
    var.unit = sc.Unit(unit)
    return var


def _get_mcstas_pixel_ids() -> PixelIDs:
    """pixel IDs for each detector"""
    intervals = [(1, 1638401), (2000001, 3638401), (4000001, 5638401)]
    ids = [sc.arange('id', start, stop) for start, stop in intervals]
    return PixelIDs(sc.concat(ids, 'id'))


@dataclass
class NMXData:
    """Data class for NMX data"""

    events: Events
    all_pixel_ids: PixelIDs


def _load_mcstas_nmx_file(file: snx.File, max_prop: MaximumProbability) -> NMXData:
    """Load McStas NMX data from h5 file.

    ``max_pro`` is used to scale ``weights`` to derive more realistic number of events.
    """
    from functools import partial

    bank_name = _retrieve_event_list_name(file["entry1/data"].keys())
    var: sc.Variable
    var = file["entry1/data/" + bank_name]["events"][()].rename_dims({'dim_0': 'event'})

    copier = partial(_copy_partial_var, var, dim='dim_1')
    weights = copier(idx=0, unit='counts')
    id_list = copier(idx=4, unit='dimensionless', dtype="int64")
    t_list = copier(idx=5, unit='s')

    weights = (max_prop / weights.max()) * weights

    loaded = sc.DataArray(data=weights, coords={'t': t_list, 'id': id_list})
    pixel_ids = _get_mcstas_pixel_ids()
    return NMXData(Events(loaded), pixel_ids)


def load_nmx_file(
    file_name: InputFileName,
    max_prop: Optional[MaximumProbability] = None,
) -> NMXData:
    """Load NMX data from a file and generate pixel ID coordinate.

    Check an entry path in the file and handle the data accordingly.
    Nexus file should have an entry path called ``entry`` and
    McStas simulation file should have an entry path called ``entry1``.

    Pixel id coordinate are wrapped together with the data
    since they are different for simulation and real data.
    """
    with snx.File(file_name) as file:
        if "entry1" in file:  # McStas file
            return _load_mcstas_nmx_file(file, max_prop or DefaultMaximumProbability)
        elif 'entry' in file:  # Nexus file
            raise NotImplementedError("Measurement data loader is not implemented yet.")
        else:
            raise ValueError(f"Can not load {file_name} with NMX file loader.")
