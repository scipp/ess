# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
import scipp as sc

from . import utils


def make_chopper(
    frequency: sc.Variable,
    position: sc.Variable,
    phase: sc.Variable = None,
    cutout_angles_center: sc.Variable = None,
    cutout_angles_width: sc.Variable = None,
    cutout_angles_begin: sc.Variable = None,
    cutout_angles_end: sc.Variable = None,
    kind: str = None,
) -> sc.Dataset:
    """
    Create a Dataset that holds chopper parameters.
    This ensures the Dataset is compatible with the other functions in the choppers
    module.
    Defining a chopper's cutout angles can either constructed from an array of cutout
    centers and an array of angular widths, or two arrays containing the begin (leading)
    and end (closing) angles of the cutout windows.

    :param frequency: The rotational frequency of the chopper.
    :param position: The position vector of the chopper.
    :param phase: The chopper phase.
    :param cutout_angles_center: The centers of the chopper cutout angles.
    :param cutout_angles_width: The angular widths of the chopper cutouts.
    :param cutout_angles_begin: The starting/opening angles of the chopper cutouts.
    :param cutout_angles_end: The ending/closing angles of the chopper cutouts.
    :param kind: The chopper kind. Any string can be supplied, but WFM choppers should
        be given 'wfm' as their kind.
    """
    data = {"frequency": frequency, "position": position}
    if phase is not None:
        data["phase"] = phase
    if cutout_angles_center is not None:
        data["cutout_angles_center"] = cutout_angles_center
    if cutout_angles_width is not None:
        data["cutout_angles_width"] = cutout_angles_width
    if cutout_angles_begin is not None:
        data["cutout_angles_begin"] = cutout_angles_begin
    if cutout_angles_end is not None:
        data["cutout_angles_end"] = cutout_angles_end
    if kind is not None:
        data["kind"] = kind
    chopper = sc.Dataset(data=data)

    # Sanitize input parameters
    if (None not in [cutout_angles_begin, cutout_angles_end]) or (
        None not in [cutout_angles_center, cutout_angles_width]
    ):
        widths = utils.cutout_angles_width(chopper)
        if (sc.min(widths) < sc.scalar(0.0, unit=widths.unit)).value:
            raise ValueError("Negative window width found in chopper cutout angles.")
        if not sc.allsorted(utils.cutout_angles_begin(chopper), dim=widths.dim):
            raise ValueError("Chopper begin cutout angles are not monotonic.")
        if not sc.allsorted(utils.cutout_angles_end(chopper), dim=widths.dim):
            raise ValueError("Chopper end cutout angles are not monotonic.")

    return chopper
