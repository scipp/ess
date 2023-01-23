# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
"""
File loading for POWGEN.
"""
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import List, Union

import scipp as sc
import scippneutron as scn
from scippneutron.conversion.graph import beamline, tof

from ...diffraction.filtering import remove_bad_pulses
from ...logging import get_logger

# TODO Normalization by and filtering based on proton charge
#  with subtraction of the empty instrument
#
# The vanadium test file for POWGEN has an average proton charge for good pulses
# of ~1.4e7pC while the empty instrument file has ~6e3pC and much fewer events.
# This means that events from the empty instrument measurement have 10**4 times
# the weight as events for vanadium and are sparsely distributed.
# When combining event lists, the resulting distribution
#   vana / vana_charge - empty / empty_charge
# is mostly flat with huge peaks at the events from `empty`.
# This distorts the result.
#
# In Mantid, this is avoided by filtering out _all_ empty instrument events
# because of the low proton charge.
# This does not happen with RemoveBadPulses which was attempted to replicate here
# as that only removed pulses below the average charge.
# Also, by visual inspection of the charge, there is a clear separation between
# good and bad pulses. So discarding all events does not seem correct.
#
# For now, we ignore the empty instrument data.


def load(filename: Union[str, Path]) -> sc.DataArray:
    """
    Load a data file for POWGEN.

    Parameters
    ----------
    filename:
        Input file name.

    Returns
    -------
    da:
        Loaded data.
    """
    return scn.load(
        filename,
        advanced_geometry=True,
        load_pulse_times=False,
        mantid_args={"LoadMonitors": True},
    )


@dataclass
class _AuxData:
    da: sc.DataArray
    name: str


def _load_aux_file(filename: Union[str, Path], *, data_name: str) -> _AuxData:
    get_logger('powgen').info('Loading %s from file %s.', data_name, filename)
    da = scn.load(filename,
                  advanced_geometry=False,
                  load_pulse_times=True,
                  mantid_args={'LoadMonitors': False})
    da.attrs['proton_charge'] = sc.scalar(
        da.attrs['proton_charge'].value.rename(time='pulse_time'))
    return _AuxData(da=da, name=data_name)


def _normalize_by_proton_charge_in_place(data: _AuxData):
    total_charge = data.da.meta['proton_charge'].value.data.sum()
    get_logger('powgen').info('Normalizing %s by proton charge %e%s', data.name,
                              total_charge.value, total_charge.unit)
    data.da /= total_charge


def _common_edges(*edges, dim: str) -> sc.Variable:
    """
    The data has separate bin edges for each spectrum:

        ^      |---|
    spectrum  |--|
        v        |--|
              < dim >

    This function computes common edges for all spectra
    for the combination of all inputs:

        ^     | --- |
    spectrum  |--   |
        v     |   --|
              < dim >
    """
    lo = sc.reduce([e[dim, 0] for e in edges]).min().min()
    hi = sc.reduce([e[dim, -1] for e in edges]).max().max()
    return sc.concat([lo, hi], dim)


def _replace_by_common_edges(data: List[_AuxData], dim: str):
    for d in data:
        if d.da.coords[dim].sizes != {dim: 2}:
            raise RuntimeError(
                f"Cannot process vanadium data, coordinate '{dim}' of dataset "
                f"'{d.name}' must have sizes {{{dim}: 2}}, "
                f"got {d.da.coords[dim].sizes}")
    edges = _common_edges(*(d.da.coords[dim] for d in data), dim=dim)
    for d in data:
        d.da.coords[dim] = edges


def load_and_preprocess_vanadium(
        vanadium_file: Union[str, Path],
        empty_instrument_file: Union[str, Path],
        proton_charge_filter_threshold: Real = 0.9) -> sc.DataArray:
    """
    Load and return data from a vanadium measurement.

    Subtracts events recorded for the instrument without sample.

    Attention
    ---------
    Empty instrument handling is not currently implemented!
    See the corresponding comment in the source code of
    ess.external.powgen.load.py

    Parameters
    ----------
    vanadium_file:
        File that contains the vanadium data.
    empty_instrument_file:
        File that contains data for the empty instrument.
        Must correspond to the same setup as `vanadium_file`.
    proton_charge_filter_threshold:
        See diffraction.filtering.remove_bad_pulses

    Returns
    -------
    :
        (Vanadium - empty instrument) with a wavelength coordinate.
    """
    data = [
        _load_aux_file(vanadium_file, data_name='vanadium'),
        # TODO See comment at the top of the file.
        # _load_aux_file(empty_instrument_file, data_name='empty instrument')
    ]
    data = [
        _AuxData(da=remove_bad_pulses(d.da,
                                      proton_charge=d.da.meta['proton_charge'].value,
                                      threshold_factor=proton_charge_filter_threshold),
                 name=d.name) for d in data
    ]
    _replace_by_common_edges(data, dim='tof')
    tof_to_wavelength = {**beamline.beamline(scatter=True), **tof.elastic("tof")}
    for d in data:
        _normalize_by_proton_charge_in_place(d)
        d.da = d.da.transform_coords('wavelength', graph=tof_to_wavelength)
    # TODO
    # return data[0].da.bins.concat(-data[1].da)
    return data[0].da
