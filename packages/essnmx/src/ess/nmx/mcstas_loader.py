# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from typing import Callable, Iterable, NewType, Optional

import scipp as sc
import scippnexus as snx

from .reduction import NMXData

PixelIDs = NewType("PixelIDs", sc.Variable)
InputFilepath = NewType("InputFilepath", str)

# McStas Configurations
MaximumProbability = NewType("MaximumProbability", int)
DefaultMaximumProbability = MaximumProbability(100_000)

ConvertedEventWeights = NewType("ConvertedEventWeights", sc.Variable)
McStasEventWeightsConverter = Callable[..., ConvertedEventWeights]
# It should be ``Callable[[MaximumProbability, sc.Variable], ConvertedEventWeights]``
# but sciline pipeline breaks if there is a list in the type arguments.
McStasEventWeightsConverter.__name__ = "McStasEventWeightsConverter"

ProtonCharge = NewType("ProtonCharge", sc.Variable)
McStasProtonChargeConverter = Callable[..., ProtonCharge]
# Should be ``Callable[[sc.DataArray], ProtonCharge]`` for the same reason as above.
McStasProtonChargeConverter.__name__ = "McStasProtonChargeConverter"


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
    max_probability: MaximumProbability, probabilities: sc.Variable
) -> ConvertedEventWeights:
    """Create event weights by scaling probability data.

    event_weights = max_probability * (probabilities / max(probabilities))

    Parameters
    ----------
    probabilities:
        The probabilities of the events.

    max_probability:
        The maximum probability to scale the weights.

    """
    maximum_probability = sc.scalar(max_probability, unit='counts')

    return ConvertedEventWeights(
        maximum_probability * (probabilities / probabilities.max())
    )


def _compose_event_data_array(
    *,
    weights: sc.Variable,
    id_list: sc.Variable,
    t_list: sc.Variable,
    pixel_ids: sc.Variable,
    num_panels: int,
) -> sc.DataArray:
    """Combine data with coordinates loaded from the nexus file.

    Parameters
    ----------
    weights:
        The weights of the events.

    id_list:
        The pixel IDs of the events.

    t_list:
        The time of arrival of the events.

    pixel_ids:
        All possible pixel IDs of the detector.

    num_panels:
        The number of (detector) panels used in the experiment.

    """

    events = sc.DataArray(data=weights, coords={'t': t_list, 'id': id_list})
    grouped: sc.DataArray = events.group(pixel_ids)
    return grouped.fold(dim='id', sizes={'panel': num_panels, 'id': -1})


def proton_charge_from_event_data(event_da: sc.DataArray) -> ProtonCharge:
    """Make up the proton charge from the event data array.

    Proton charge is proportional to the number of neutrons,
    which is proportional to the number of events.
    The scale factor is manually chosen based on previous results
    to be convenient for data manipulation in the next steps.
    It is derived this way since
    the protons are not part of McStas simulation,
    and the number of neutrons is not included in the result.

    Parameters
    ----------
    event_da:
        The event data binned in detector panel and pixel id dimensions.

    """
    # Arbitrary number to scale the proton charge
    _proton_charge_scale_factor = sc.scalar(1 / 10_000, unit=None)

    return ProtonCharge(_proton_charge_scale_factor * event_da.bins.size().sum().data)


def load_mcstas_nexus(
    file_path: InputFilepath,
    event_weights_converter: McStasEventWeightsConverter,
    proton_charge_converter: McStasProtonChargeConverter,
    max_probability: Optional[MaximumProbability] = None,
) -> NMXData:
    """Load McStas simulation result from h5(nexus) file.

    See :func:`~event_weights_from_probability` and
    :func:`~proton_charge_from_weights` for details.

    Parameters
    ----------
    file_path:
        File name to load.

    event_weights_converter:
        A function to convert probabilities to event weights.
        The function should accept the probabilities as the first argument,
        and return the converted event weights.

    proton_charge_converter:
        A function to convert the event weights to proton charge.
        The function should accept the event weights as the first argument,
        and return the proton charge.

    max_probability:
        The maximum probability to scale the weights.
        If not provided, ``DefaultMaximumProbability`` is used.

    """

    from .mcstas_xml import read_mcstas_geometry_xml

    geometry = read_mcstas_geometry_xml(file_path)
    coords = geometry.to_coords()

    with snx.File(file_path) as file:
        raw_data = _retrieve_raw_event_data(file)
        weights = event_weights_converter(
            max_probability or DefaultMaximumProbability,
            _copy_partial_var(raw_data, idx=0, unit='counts'),  # p
        )
        event_da = _compose_event_data_array(
            weights=weights,
            id_list=_copy_partial_var(raw_data, idx=4, dtype='int64'),  # id
            t_list=_copy_partial_var(raw_data, idx=5, unit='s'),  # t
            pixel_ids=coords.pop('pixel_id'),
            num_panels=len(geometry.detectors),
        )
        proton_charge = proton_charge_converter(event_da)
        crystal_rotation = _retrieve_crystal_rotation(
            file, geometry.simulation_settings.angle_unit
        )

    return NMXData(
        weights=event_da,
        proton_charge=proton_charge,
        crystal_rotation=crystal_rotation,
        **coords,
    )
