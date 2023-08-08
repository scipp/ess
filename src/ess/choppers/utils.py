# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
import scipp as sc
from scipp.constants import pi


def cutout_angles_begin(chopper: sc.Dataset, unit="rad") -> sc.Variable:
    """
    Get the starting/opening angles of the chopper cutouts.

    :param chopper: The Dataset containing the chopper parameters.
    :param unit: Convert to this unit before returning. Default is `'rad'`.
    """
    if "cutout_angles_begin" in chopper:
        out = chopper["cutout_angles_begin"].data
    elif all(x in chopper for x in ["cutout_angles_width", "cutout_angles_center"]):
        out = (
            chopper["cutout_angles_center"].data
            - 0.5 * chopper["cutout_angles_width"].data
        )
    else:
        raise KeyError(
            "Chopper does not contain the information required to compute "
            "the cutout_angles_begin."
        )
    return sc.to_unit(out, unit, copy=False)


def cutout_angles_end(chopper: sc.Dataset, unit="rad") -> sc.Variable:
    """
    Get the ending/closing angles of the chopper cutouts.

    :param chopper: The Dataset containing the chopper parameters.
    :param unit: Convert to this unit before returning. Default is `'rad'`.
    """
    if "cutout_angles_end" in chopper:
        out = chopper["cutout_angles_end"].data
    elif all(x in chopper for x in ["cutout_angles_width", "cutout_angles_center"]):
        out = (
            chopper["cutout_angles_center"].data
            + 0.5 * chopper["cutout_angles_width"].data
        )
    else:
        raise KeyError(
            "Chopper does not contain the information required to compute "
            "the cutout_angles_end."
        )
    return sc.to_unit(out, unit, copy=False)


def cutout_angles_width(chopper: sc.Dataset, unit="rad") -> sc.Variable:
    """
    Get the angular widths of the chopper cutouts.

    :param chopper: The Dataset containing the chopper parameters.
    :param unit: Convert to this unit before returning. Default is `'rad'`.
    """
    if "cutout_angles_width" in chopper:
        out = chopper["cutout_angles_width"].data
    elif all(x in chopper for x in ["cutout_angles_begin", "cutout_angles_end"]):
        out = chopper["cutout_angles_end"].data - chopper["cutout_angles_begin"].data
    else:
        raise KeyError(
            "Chopper does not contain the information required to compute "
            "the cutout_angles_width."
        )
    return sc.to_unit(out, unit, copy=False)


def cutout_angles_center(chopper: sc.Dataset, unit="rad") -> sc.Variable:
    """
    Get the angular centers of the chopper cutouts.

    :param chopper: The Dataset containing the chopper parameters.
    :param unit: Convert to this unit before returning. Default is `'rad'`.
    """
    if "cutout_angles_center" in chopper:
        out = chopper["cutout_angles_center"].data
    elif all(x in chopper for x in ["cutout_angles_begin", "cutout_angles_end"]):
        out = 0.5 * (
            chopper["cutout_angles_begin"].data + chopper["cutout_angles_end"].data
        )
    else:
        raise KeyError(
            "Chopper does not contain the information required to compute "
            "the cutout_angles_center."
        )
    return sc.to_unit(out, unit, copy=False)


def angular_frequency(chopper: sc.Dataset) -> sc.Variable:
    """
    Get the angular frequency of the chopper.

    :param chopper: The Dataset containing the chopper parameters.
    """
    return (2.0 * sc.units.rad) * pi * chopper["frequency"].data


def time_open(chopper: sc.Dataset, unit: str = "us") -> sc.Variable:
    """
    Get the times when a chopper window is open.

    :param chopper: The Dataset containing the chopper parameters.
    :param unit: Convert to this unit before returning. Default is `'rad'`.
    """
    return sc.to_unit(
        (cutout_angles_begin(chopper) + sc.to_unit(chopper["phase"].data, "rad"))
        / angular_frequency(chopper),
        unit,
        copy=False,
    )


def time_closed(chopper: sc.Dataset, unit: str = "us") -> sc.Variable:
    """
    Get the times when a chopper window is closed.

    :param chopper: The Dataset containing the chopper parameters.
    :param unit: Convert to this unit before returning. Default is `'rad'`.
    """
    return sc.to_unit(
        (cutout_angles_end(chopper) + sc.to_unit(chopper["phase"].data, "rad"))
        / angular_frequency(chopper),
        unit,
        copy=False,
    )


def find_chopper_keys(da: sc.DataArray) -> list:
    """
    Scan the coords of the data container and return a list of all entries starting with
    `"chopper"`, ignoring case.
    TODO: This is a very brittle mechanism. In Nexus, choppers are identified with
    `NXdisk_chopper`. We could make use of these identifiers instead, but we currently
    do not store the NX attributes when loading files in `scn.load_nexus()`.

    :param da: The DataArray containing the coordinates to scan.
    """
    return [key for key in da.coords if key.lower().startswith("chopper")]
