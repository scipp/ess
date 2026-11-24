# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import scipp as sc

from ..reflectometry.conversions import reflectometry_q
from ..reflectometry.types import (
    BeamDivergenceLimits,
    BeamSize,
    CoordTransformationGraph,
    DetectorRotation,
    ReferenceRun,
    RunType,
    SampleRotation,
    SampleSize,
    WavelengthBins,
    YIndexLimits,
    ZIndexLimits,
)
from .geometry import Detector
from .types import (
    ChopperDistance,
    ChopperFrequency,
    ChopperPhaseOffset,
    ChopperSeparation,
    GravityToggle,
)


def theta(
    wavelength, pixel_divergence_angle, L2, sample_rotation, detector_rotation
) -> sc.Variable:
    '''
    Angle of reflection.

    Computes the angle between the scattering direction of
    the neutron and the sample surface.

    :math:`\\gamma^*` denotes the angle between the scattering direction
    and the horizontal plane.
    :math:`\\gamma` denotes the angle between the ray from sample position
    to detection position
    and the horizontal plane.
    :math:`L_2` is the length of the ray from sample position to detector position.
    :math:`v` is the velocity of the neutron at the sample.
    :math:`t` is the travel time from sample to detector.

    The parabolic trajectory of the neutron satisfies

    .. math::

        \\sin(\\gamma) L_2 = \\sin(\\gamma^*) v t - \\frac{g}{2} t^2

    and

    .. math::

        \\cos(\\gamma) L_2 = \\cos(\\gamma^*) vt

    where :math:`g` is the gravitational acceleration.

    The second equation tells us that the approximation :math:`L_2=vt`
    will have a small error if :math:`\\gamma` is close to 0 and
    the difference between :math:`\\gamma` and :math:`\\gamma^*` is small.

    Using this approximation we can solve the first equation,
    and by expressing :math:`v` in terms of the wavelength we get

    .. math::

        \\sin(\\gamma^*) =
        \\sin(\\gamma) + \\frac{g}{2} \\frac{L_2 \\lambda^2 m_n^2}{h^2}.

    Finally, the scattering angle is obtained by subtracting the sample rotation
    relative to the horizontal plane.
    '''
    c = sc.constants.g * sc.constants.m_n**2 / sc.constants.h**2 / 2
    out = (c * L2 * wavelength**2).to(unit='dimensionless') + sc.sin(
        pixel_divergence_angle.to(unit='rad', copy=False)
        + detector_rotation.to(unit='rad')
    )
    out = sc.asin(out, out=out)
    out -= sample_rotation.to(unit='rad')
    return out


def theta_no_gravity(
    wavelength, pixel_divergence_angle, sample_rotation, detector_rotation
) -> sc.Variable:
    '''
    Angle of reflection.

    Computes the angle between the scattering direction of
    the neutron and the sample surface while disregarding the
    effect of gravity.
    '''
    theta = (
        pixel_divergence_angle.to(unit='rad', copy=False)
        + detector_rotation.to(unit='rad')
        - sample_rotation.to(unit='rad')
    )
    if wavelength.bins:
        return sc.bins_like(wavelength, theta)
    return theta


def divergence_angle(theta, sample_rotation, detector_rotation) -> sc.Variable:
    """
    Difference between the incident angle and the center of the incident beam.
    Useful for filtering parts of the beam that have too high divergence.

    On the Amor instrument this is always in the interval [-0.75 deg, 0.75 deg],
    but the divergence of the incident beam can be made lower.
    """
    return (
        theta.to(unit='rad', copy=False)
        - detector_rotation.to(unit='rad')
        + sample_rotation.to(unit='rad')
    )


def direct_beam_theta(
    wavelength, pixel_divergence_angle, L2, sample_rotation, detector_rotation
) -> sc.Variable:
    """Incidence angle at the footprint aperture for a transmitted ray.

    If the ray leaves the sample position at angle gamma to the horizon,
    its incidence angle is mu - gamma. For a reflected ray it is gamma - mu.
    The gravity correction to gamma is the same in both cases.
    """
    out = theta(
        wavelength, pixel_divergence_angle, L2, sample_rotation, detector_rotation
    )
    out *= -1
    return out


def direct_beam_theta_no_gravity(
    wavelength, pixel_divergence_angle, sample_rotation, detector_rotation
) -> sc.Variable:
    """Incidence angle of a transmitted ray without gravity."""
    out = theta_no_gravity(
        wavelength, pixel_divergence_angle, sample_rotation, detector_rotation
    )
    out *= -1
    return out


def direct_beam_divergence_angle(
    theta, sample_rotation, detector_rotation
) -> sc.Variable:
    """Divergence of a transmitted ray in the reflected beam's convention.

    Reflection reverses the angular profile. With theta = mu - gamma for
    the direct beam, theta + nu - mu = -(gamma - nu) selects the corresponding
    incident rays before the detector rows are mirrored.
    """
    out = theta.to(unit='rad', copy=True)
    out += detector_rotation.to(unit='rad')
    out -= sample_rotation.to(unit='rad')
    return out


def wavelength(
    event_time_offset,
    pixel_divergence_angle,
    L1,
    L2,
    chopper_frequency,
    chopper_phase_offset,
) -> sc.Variable:
    '''
    Wavelength of the neutrons from their arrival time at the detector.

    The chopper lets a pulse of neutrons pass every half revolution,
    :math:`\\tau = 1 / (2 f)`, but the arrival times are recorded
    relative to a trigger that occurs once per revolution.
    The time of flight :math:`t` is therefore only known up to a multiple
    of :math:`\\tau`,

    .. math::

        t \\equiv t_\\text{event} + \\frac{2 \\phi - \\delta}{180^\\circ} \\tau
        \\pmod \\tau,

    where :math:`\\phi` is the phase between the trigger and the opening of
    the chopper and :math:`\\delta` is the divergence angle of the pixel.
    The term with :math:`\\delta` accounts for the chopper opening earlier or
    later for neutrons that travel at an angle to the center of the beam.

    The multiple of :math:`\\tau` is selected such that the wavelength

    .. math::

        \\lambda = \\frac{h}{m_n} \\frac{t}{L_1 + L_2}

    lies in the first frame, with a time of flight in [0, tau).
    This assumes the instrument suppresses neutrons from earlier pulses.
    The wavelength bins only select and histogram events; they do not change
    the frame assignment.
    '''
    if not sc.isfinite(chopper_frequency).value or chopper_frequency.value <= 0:
        raise ValueError('ChopperFrequency must be finite and positive.')
    half_period = (1 / (2 * chopper_frequency)).to(unit='ns')
    time_per_wavelength = ((L1 + L2) * sc.constants.m_n / sc.constants.h).to(
        unit='ns/angstrom'
    )
    half_turns = (
        2 * chopper_phase_offset.to(unit='deg') - pixel_divergence_angle.to(unit='deg')
    ) / sc.scalar(180.0, unit='deg')
    time_of_flight = event_time_offset.to(unit='ns', dtype='float64', copy=True)
    time_of_flight += half_turns * half_period
    time_of_flight %= half_period
    time_of_flight /= time_per_wavelength
    return time_of_flight.to(unit='angstrom', copy=False)


def coordinate_transformation_graph(
    gravity: GravityToggle,
    detector_rotation: DetectorRotation[RunType],
    sample_rotation: SampleRotation[RunType],
    chopper_phase_offset: ChopperPhaseOffset[RunType],
    chopper_frequency: ChopperFrequency[RunType],
    chopper_distance: ChopperDistance[RunType],
    chopper_separation: ChopperSeparation[RunType],
    sample_size: SampleSize[RunType],
    beam_size: BeamSize[RunType],
) -> CoordTransformationGraph[RunType]:
    """Build the coordinate transformation graph for reflected Amor events."""
    return {
        "wavelength": wavelength,
        "theta": theta if gravity else theta_no_gravity,
        "divergence_angle": divergence_angle,
        "Q": reflectometry_q,
        "L1": lambda chopper_distance: sc.abs(chopper_distance),
        "L2": lambda distance_in_detector: distance_in_detector + Detector.distance,
        "sample_rotation": lambda: sample_rotation.to(unit='rad'),
        "detector_rotation": lambda: detector_rotation.to(unit='rad'),
        "chopper_phase_offset": lambda: chopper_phase_offset,
        "chopper_frequency": lambda: chopper_frequency,
        "chopper_separation": lambda: chopper_separation,
        "chopper_distance": lambda: chopper_distance,
        "sample_size": lambda: sample_size,
        "beam_size": lambda: beam_size,
    }


def direct_beam_coordinate_transformation_graph(
    gravity: GravityToggle,
    detector_rotation: DetectorRotation[ReferenceRun],
    sample_rotation: SampleRotation[ReferenceRun],
    chopper_phase_offset: ChopperPhaseOffset[ReferenceRun],
    chopper_frequency: ChopperFrequency[ReferenceRun],
    chopper_distance: ChopperDistance[ReferenceRun],
    chopper_separation: ChopperSeparation[ReferenceRun],
    sample_size: SampleSize[ReferenceRun],
    beam_size: BeamSize[ReferenceRun],
) -> CoordTransformationGraph[ReferenceRun]:
    """Use transmitted-ray angles when preparing a direct-beam reference."""
    graph = coordinate_transformation_graph(
        gravity=gravity,
        detector_rotation=detector_rotation,
        sample_rotation=sample_rotation,
        chopper_phase_offset=chopper_phase_offset,
        chopper_frequency=chopper_frequency,
        chopper_distance=chopper_distance,
        chopper_separation=chopper_separation,
        sample_size=sample_size,
        beam_size=beam_size,
    )
    graph['theta'] = direct_beam_theta if gravity else direct_beam_theta_no_gravity
    graph['divergence_angle'] = direct_beam_divergence_angle
    return graph


def validate_wavelength_range(wavelength_bins, L1, L2, chopper_frequency) -> None:
    """Require a wavelength range contained in the first frame at every pixel."""
    frame_end = (
        (sc.constants.h / sc.constants.m_n / (2 * chopper_frequency * (L1 + L2)))
        .to(unit='angstrom')
        .min()
    )
    if (
        not sc.all(sc.isfinite(wavelength_bins)).value
        or wavelength_bins[0] < sc.scalar(0.0, unit='angstrom')
        or wavelength_bins[-1] > frame_end
    ):
        raise ValueError(
            'WavelengthBins must lie within the first chopper frame: '
            f'0 to {frame_end.value:g} angstrom for this frequency and geometry.'
        )


def add_coords(
    da: sc.DataArray,
    graph: dict,
) -> sc.DataArray:
    "Adds scattering coordinates to the raw detector data."
    return da.transform_coords(
        (
            "wavelength",
            "theta",
            "divergence_angle",
            "Q",
            "L1",
            "L2",
            "blade",
            "wire",
            "strip",
            "z_index",
            "sample_rotation",
            "detector_rotation",
            "sample_size",
            "beam_size",
            "chopper_separation",
            "chopper_distance",
        ),
        graph,
        rename_dims=False,
        keep_intermediate=False,
        keep_aliases=False,
    )


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
        strip_range=_not_between(da.coords["strip"], *ylim),
        z_range=_not_between(da.coords["z_index"], *zlims),
    )
    da = da.bins.assign_masks(
        divergence_too_large=_not_between(
            da.bins.coords["divergence_angle"],
            bdlim[0].to(unit=da.bins.coords["divergence_angle"].bins.unit),
            bdlim[1].to(unit=da.bins.coords["divergence_angle"].bins.unit),
        ),
        wavelength=_not_between(
            da.bins.coords['wavelength'],
            wbins[0],
            wbins[-1],
        ),
    )
    return da


providers = (coordinate_transformation_graph,)
