# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from typing import Iterable, NewType, Optional

import scipp as sc
import scippnexus as snx

from .reduction import NMXData

_PROTON_CHARGE_SCALE_FACTOR = 1 / 10_000  # Arbitrary number to scale the proton charge
PixelIDs = NewType("PixelIDs", sc.Variable)
InputFilepath = NewType("InputFilepath", str)

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
    if unit is not None:
        var.unit = sc.Unit(unit)
    return var


def _retrieve_crystal_rotation(file: snx.File, unit: str) -> sc.Variable:
    """Retrieve crystal rotation from the file."""

    return sc.vector(
        value=[file[f"entry1/simulation/Param/XtalPhi{key}"][...] for key in "XYZ"],
        unit=unit,
    )


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
    maximum_probability = sc.scalar(
        max_probability or DefaultMaximumProbability, unit='counts'
    )

    with snx.File(file_path) as file:
        bank_name = _retrieve_event_list_name(file["entry1/data"].keys())
        var: sc.Variable
        var = file["entry1/data/" + bank_name]["events"][()].rename_dims(
            {'dim_0': 'event'}
        )  # ``dim_0``: event index, ``dim_1``: property index.

        weights = _copy_partial_var(var, idx=0, unit='counts')  # p
        id_list = _copy_partial_var(var, idx=4, dtype='int64')  # id
        t_list = _copy_partial_var(var, idx=5, unit='s')  # t
        crystal_rotation = _retrieve_crystal_rotation(
            file, geometry.simulation_settings.angle_unit
        )

    coords = geometry.to_coords()
    loaded = sc.DataArray(
        # Scale the weights so that the weights are
        # within the range of [0,``max_probability``].
        data=(maximum_probability / weights.max()) * weights,
        coords={'t': t_list, 'id': id_list},
    )
    grouped: sc.DataArray = loaded.group(coords.pop('pixel_id'))
    da: sc.DataArray = grouped.fold(
        dim='id', sizes={'panel': len(geometry.detectors), 'id': -1}
    )
    # Proton charge is proportional to the number of neutrons,
    # which is proportional to the number of events.
    # The scale factor is chosen by previous results
    # to be convenient for data manipulation in the next steps.
    # It is derived this way since
    # the protons are not part of McStas simulation,
    # and the number of neutrons is not included in the result.
    proton_charge = _PROTON_CHARGE_SCALE_FACTOR * da.bins.size().sum().value
    return NMXData(
        sc.DataGroup(
            weights=da,
            proton_charge=proton_charge,
            crystal_rotation=crystal_rotation,
            **coords,
        )
    )
