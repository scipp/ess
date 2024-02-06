# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from typing import Iterable, NewType, Optional

import scipp as sc
import scippnexus as snx

from .reduction import NMXData

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


def _retrieve_raw_event_data(file: snx.File) -> sc.Variable:
    """Retrieve events from the nexus file."""
    bank_name = _retrieve_event_list_name(file["entry1/data"].keys())
    # ``dim_0``: event index, ``dim_1``: property index.
    return file["entry1/data/" + bank_name]["events"][()].rename_dims(
        {'dim_0': 'event'}
    )


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


def event_weights_from_probability(
    *,
    probabilities: sc.Variable,
    id_list: sc.Variable,
    t_list: sc.Variable,
    pixel_ids: sc.Variable,
    max_probability: sc.Variable,
    num_panels: int,
) -> sc.DataArray:
    """Create event weights by scaling probability data.

    event_weights = max_probability * (probabilities / max(probabilities))

    Parameters
    ----------
    probabilities:
        The probabilities of the events.

    id_list:
        The pixel IDs of the events.

    t_list:
        The time of arrival of the events.

    pixel_ids:
        All possible pixel IDs of the detector.

    max_probability:
        The maximum probability to scale the weights.

    num_panels:
        The number of (detector) panels used in the experiment.

    """
    if max_probability.unit != sc.units.counts:
        raise ValueError("max_probability must have unit counts")

    weights = sc.DataArray(
        # Scale the weights so that the weights are
        # within the range of [0,``max_probability``].
        data=max_probability * (probabilities / probabilities.max()),
        coords={'t': t_list, 'id': id_list},
    )
    grouped: sc.DataArray = weights.group(pixel_ids)
    return grouped.fold(dim='id', sizes={'panel': num_panels, 'id': -1})


def proton_charge_from_weights(weights: sc.DataArray) -> sc.Variable:
    """Make up the proton charge from the weights.

    Proton charge is proportional to the number of neutrons,
    which is proportional to the number of events.
    The scale factor is manually chosen based on previous results
    to be convenient for data manipulation in the next steps.
    It is derived this way since
    the protons are not part of McStas simulation,
    and the number of neutrons is not included in the result.

    Parameters
    ----------
    weights:
        The event weights binned in detector panel and pixel id dimensions.

    """
    # Arbitrary number to scale the proton charge
    _proton_charge_scale_factor = sc.scalar(1 / 10_000, unit=None)

    return _proton_charge_scale_factor * weights.bins.size().sum().data


def load_mcstas_nexus(
    file_path: InputFilepath,
    max_probability: Optional[MaximumProbability] = None,
) -> NMXData:
    """Load McStas simulation result from h5(nexus) file.

    See :func:`~event_weights_from_probability` and
    :func:`~proton_charge_from_weights` for details.

    Parameters
    ----------
    file_path:
        File name to load.

    max_probability:
        The maximum probability to scale the weights.
        If not provided, ``DefaultMaximumProbability`` is used.

    """

    from .mcstas_xml import read_mcstas_geometry_xml

    geometry = read_mcstas_geometry_xml(file_path)
    coords = geometry.to_coords()
    maximum_probability = sc.scalar(
        max_probability or DefaultMaximumProbability, unit='counts'
    )

    with snx.File(file_path) as file:
        raw_data = _retrieve_raw_event_data(file)
        weights = event_weights_from_probability(
            probabilities=_copy_partial_var(raw_data, idx=0, unit='counts'),  # p
            max_probability=maximum_probability,
            id_list=_copy_partial_var(raw_data, idx=4, dtype='int64'),  # id
            t_list=_copy_partial_var(raw_data, idx=5, unit='s'),  # t
            pixel_ids=coords.pop('pixel_id'),
            num_panels=len(geometry.detectors),
        )
        proton_charge = proton_charge_from_weights(weights)
        crystal_rotation = _retrieve_crystal_rotation(
            file, geometry.simulation_settings.angle_unit
        )

    return NMXData(
        weights=weights,
        proton_charge=proton_charge,
        crystal_rotation=crystal_rotation,
        **coords,
    )
