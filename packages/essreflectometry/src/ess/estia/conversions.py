# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
import scipp as sc

from ..reflectometry.conversions import reflectometry_q
from ..reflectometry.types import (
    BeamDivergenceLimits,
    BeamSize,
    RawDetectorData,
    ReducibleData,
    RunType,
    WavelengthBins,
    YIndexLimits,
    ZIndexLimits,
)


def theta(divergence_angle, sample_rotation, detector_rotation):
    '''
    Angle of reflection.

    Computes the angle between the scattering direction of
    the neutron and the sample surface.
    '''
    return divergence_angle + detector_rotation - sample_rotation


def angle_of_divergence(
    theta, sample_rotation, angle_to_center_of_beam, natural_incidence_angle
):
    """
    Difference between the incident angle and the center of the incident beam.
    Useful for filtering parts of the beam that have too high divergence.

    This is always in the interval [-0.75 deg, 0.75 deg],
    but the divergence of the incident beam can also be reduced.
    """
    return (
        theta
        - sample_rotation
        - angle_to_center_of_beam
        - natural_incidence_angle.to(unit='rad')
    )


def wavelength(
    event_time_offset,
    # Other inputs
):
    "Converts event_time_offset to wavelength"
    # Use frame unwrapping from scippneutron
    pass


def _not_between(v, a, b):
    return (v < a) | (v > b)


def add_common_coords_and_masks(
    da: RawDetectorData[RunType],
    ylim: YIndexLimits,
    zlims: ZIndexLimits,
    bdlim: BeamDivergenceLimits,
    wbins: WavelengthBins,
    beam_size: BeamSize[RunType],
) -> ReducibleData[RunType]:
    "Adds coords and masks that are useful for both reference and sample measurements."
    da = da.transform_coords(
        ("wavelength", "theta", "angle_of_divergence", "Q"),
        {
            "divergence_angle": "pixel_divergence_angle",
            "wavelength": wavelength,
            "theta": theta,
            "angle_of_divergence": angle_of_divergence,
            "Q": reflectometry_q,
        },
        rename_dims=False,
        keep_intermediate=False,
    )
    da.masks["stripe_range"] = _not_between(da.coords["stripe"], *ylim)
    da.masks['z_range'] = _not_between(da.coords["z_index"], *zlims)
    da.bins.masks["divergence_too_large"] = _not_between(
        da.bins.coords["angle_of_divergence"],
        bdlim[0].to(unit=da.bins.coords["angle_of_divergence"].bins.unit),
        bdlim[1].to(unit=da.bins.coords["angle_of_divergence"].bins.unit),
    )
    da.bins.masks['wavelength'] = _not_between(
        da.bins.coords['wavelength'],
        wbins[0],
        wbins[-1],
    )
    # Correct for illumination of virtual source
    da /= sc.sin(da.bins.coords['theta'])
    return da


providers = (add_common_coords_and_masks,)
