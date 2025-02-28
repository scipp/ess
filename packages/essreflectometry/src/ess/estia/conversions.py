# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
import scipp as sc

from ..reflectometry.conversions import reflectometry_q
from ..reflectometry.types import (
    BeamDivergenceLimits,
    CoordTransformationGraph,
    WavelengthBins,
    YIndexLimits,
    ZIndexLimits,
)


def theta(
    divergence_angle: sc.Variable,
    sample_rotation: sc.Variable,
):
    '''
    Angle of reflection.

    Computes the angle between the scattering direction of
    the neutron and the sample surface.

    Parameters
    ------------
        divergence_angle:
            Divergence angle of the scattered beam.
        sample_rotation:
            Rotation of the sample from to its zero position.

    Returns
    -----------
    The reflection angle of the neutron.
    '''
    return divergence_angle + sample_rotation.to(unit=divergence_angle.unit)


def divergence_angle(
    position: sc.Variable,
    sample_position: sc.Variable,
    detector_rotation: sc.Variable,
    incident_angle_of_center_of_beam: sc.Variable,
):
    """
    Angle between the scattering ray and
    the ray that travels parallel to the sample surface
    when the sample rotation is zero.

    Parameters
    ------------
        position:
            Detector position where the neutron was detected.
        sample_position:
            Position of the sample.
        detector_rotation:
            Rotation of the detector from its zero position.
        incident_angle_of_center_of_beam:
            Angle between the normal of the sample surface
            and the x-axis of the coordinate system
            when the sample rotation is zero.
    Returns
    ----------
    The divergence angle of the scattered beam.
    """
    p = position - sample_position.to(unit=position.unit)
    return (
        sc.atan2(y=p.fields.x, x=p.fields.z)
        - detector_rotation.to(unit='rad')
        - incident_angle_of_center_of_beam.to(unit='rad')
    )


def wavelength(
    event_time_offset,
    # Other inputs
):
    "Converts event_time_offset to wavelength"
    # Use frame unwrapping from scippneutron
    raise NotImplementedError()


def coordinate_transformation_graph() -> CoordTransformationGraph:
    return {
        "wavelength": wavelength,
        "theta": theta,
        "divergence_angle": divergence_angle,
        "Q": reflectometry_q,
        "L1": lambda source_position, sample_position: sc.norm(
            sample_position - source_position
        ),  # + extra correction for guides?
        "L2": lambda position, sample_position: sc.norm(position - sample_position),
        "incident_angle_of_center_of_beam": lambda: sc.scalar(1.7, unit='deg').to(
            unit='rad'
        ),
    }


def _not_between(v, a, b):
    return (v < a) | (v > b)


def add_masks(
    da: sc.DataArray,
    ylim: YIndexLimits,
    zlims: ZIndexLimits,
    bdlim: BeamDivergenceLimits,
    wbins: WavelengthBins,
) -> sc.DataArray:
    """
    Masks the data by ranges in the detector
    coordinates ``z`` and ``y``, and by the divergence of the beam,
    and by wavelength.
    """
    da = da.assign_masks(
        stripe_range=_not_between(da.coords["stripe"], *ylim),
        z_range=_not_between(da.coords["z_index"], *zlims),
        divergence_too_large=_not_between(
            da.coords["divergence_angle"],
            bdlim[0].to(unit=da.coords["divergence_angle"].unit),
            bdlim[1].to(unit=da.coords["divergence_angle"].unit),
        ),
    )
    da = da.bins.assign_masks(
        wavelength=_not_between(
            da.bins.coords['wavelength'],
            wbins[0],
            wbins[-1],
        ),
    )
    return da


providers = (coordinate_transformation_graph,)
