# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)
import scipp as sc
from typing import Union
from .. import choppers as ch
import scipp.constants as constants


def get_frames(data: Union[sc.DataArray, sc.Dataset]) -> sc.Dataset:
    """
    Compute analytical frame boundaries and shifts based on chopper
    parameters and detector pixel positions.
    A set of frame boundaries is returned for each pixel.
    The frame shifts are the same for all pixels.
    See Schmakat et al. (2020);
    https://www.sciencedirect.com/science/article/pii/S0168900220308640
    for a description of the procedure.

    TODO: This currently ignores scattering paths, only the distance from
    source to pixel.
    For imaging, this is what we want, but for scattering techniques, we should
    use l1 + l2 in the future.
    """

    # Identify the WFM choppers based on their `kind` property
    wfm_choppers = {}
    for name in ch.find_chopper_keys(data):
        chopper = data.meta[name].value
        if chopper["kind"].value == "wfm":
            wfm_choppers[name] = chopper
    if len(wfm_choppers) != 2:
        raise RuntimeError("The number of WFM choppers is expected to be 2, "
                           "found {}".format(len(wfm_choppers)))
    # Find the near and far WFM choppers based on their positions relative to the source
    wfm_chopper_names = list(wfm_choppers.keys())
    if (sc.norm(wfm_choppers[wfm_chopper_names[0]]["position"].data -
                data.meta["source_position"]) <
            sc.norm(wfm_choppers[wfm_chopper_names[1]]["position"].data -
                    data.meta["source_position"])).value:
        near_index = 0
        far_index = 1
    else:
        near_index = 1
        far_index = 0
    near_wfm_chopper = wfm_choppers[wfm_chopper_names[near_index]]
    far_wfm_chopper = wfm_choppers[wfm_chopper_names[far_index]]

    # Compute distances for each detector pixel
    detector_positions = data.meta["position"] - data.meta["source_position"]

    # Container for frames information
    frames = sc.Dataset()

    # Distance between WFM choppers
    dz_wfm = sc.norm(far_wfm_chopper["position"].data -
                     near_wfm_chopper["position"].data)
    # Mid-point between WFM choppers
    z_wfm = 0.5 * (near_wfm_chopper["position"].data +
                   far_wfm_chopper["position"].data) - data.meta["source_position"]
    # Ratio of WFM chopper distances
    z_ratio_wfm = (
        sc.norm(far_wfm_chopper["position"].data - data.meta["source_position"]) /
        sc.norm(near_wfm_chopper["position"].data - data.meta["source_position"]))
    # Distance between detector positions and wfm chopper mid-point
    zdet_minus_zwfm = sc.norm(detector_positions - z_wfm)

    # Neutron mass to Planck constant ratio
    alpha = sc.to_unit(constants.m_n / constants.h, 'us/m/angstrom')

    # Frame time corrections: these are the mid-time point between the WFM choppers,
    # which is the same as the opening edge of the second WFM chopper in the case
    # of optically blind choppers.
    frames["time_correction"] = ch.time_open(far_wfm_chopper)

    # Find delta_t for the min and max wavelengths:
    # dt_lambda_max is equal to the time width of the WFM choppers windows
    dt_lambda_max = ch.time_closed(near_wfm_chopper) - ch.time_open(near_wfm_chopper)

    # t_lambda_max is found from the relation between t and delta_t: equation (2) in
    # Schmakat et al. (2020).
    t_lambda_max = (dt_lambda_max / dz_wfm) * zdet_minus_zwfm

    # t_lambda_min is found from the relation between lambda_N and lambda_N+1,
    # equation (3) in Schmakat et al. (2020).
    t_lambda_min = t_lambda_max * z_ratio_wfm - data.meta["source_pulse_length"] * (
        zdet_minus_zwfm /
        sc.norm(near_wfm_chopper["position"].data - data.meta["source_position"]))

    # dt_lambda_min is found from the relation between t and delta_t: equation (2)
    # in Schmakat et al. (2020), and using the expression for t_lambda_max.
    dt_lambda_min = dt_lambda_max * z_ratio_wfm - data.meta[
        "source_pulse_length"] * dz_wfm / sc.norm(near_wfm_chopper["position"].data -
                                                  data.meta["source_position"])

    # Compute wavelength information
    lambda_min = t_lambda_min / (alpha * zdet_minus_zwfm)
    lambda_max = t_lambda_max / (alpha * zdet_minus_zwfm)
    dlambda_min = dz_wfm * lambda_min / zdet_minus_zwfm
    dlambda_max = dz_wfm * lambda_max / zdet_minus_zwfm

    # Frame edges and resolutions for each pixel.
    # The frames do not stop at t_lambda_min and t_lambda_max, they also include the
    # fuzzy areas (delta_t) at the edges.
    frames["time_min"] = t_lambda_min - (0.5 *
                                         dt_lambda_min) + frames["time_correction"]
    frames["delta_time_min"] = dt_lambda_min

    frames["time_max"] = t_lambda_max + (0.5 *
                                         dt_lambda_max) + frames["time_correction"]
    frames["delta_time_max"] = dt_lambda_max
    frames["wavelength_min"] = lambda_min
    frames["wavelength_max"] = lambda_max
    frames["delta_wavelength_min"] = dlambda_min
    frames["delta_wavelength_max"] = dlambda_max

    frames["wfm_chopper_mid_point"] = 0.5 * (near_wfm_chopper["position"].data +
                                             far_wfm_chopper["position"].data)

    return frames
