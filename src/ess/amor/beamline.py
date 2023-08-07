# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
import scipp as sc
from scipp.constants import g

from ..choppers import make_chopper
from ..logging import log_call


@log_call(
    instrument='amor', message='Constructing AMOR beamline from default parameters'
)
def make_beamline(
    sample_rotation: sc.Variable,
    beam_size: sc.Variable = None,
    sample_size: sc.Variable = None,
    detector_spatial_resolution: sc.Variable = None,
    gravity: sc.Variable = None,
    chopper_frequency: sc.Variable = None,
    chopper_phase: sc.Variable = None,
    chopper_1_position: sc.Variable = None,
    chopper_2_position: sc.Variable = None,
) -> dict:
    """
    Amor beamline components.

    Parameters
    ----------
    sample_rotation:
        Sample rotation (omega) angle.
    beam_size:
        Size of the beam perpendicular to the scattering surface. Default is `0.001 m`.
    sample_size:
        Size of the sample in direction of the beam. Default :code:`0.01 m`.
    detector_spatial_resolution:
        Spatial resolution of the detector. Default is `2.5 mm`.
    gravity:
        Vector representing the direction and magnitude of the Earth's gravitational
        field. Default is `[0, -g, 0]`.
    chopper_frequency:
        Rotational frequency of the chopper. Default is `6.6666... Hz`.
    chopper_phase:
        Phase offset between chopper pulse and ToF zero. Default is `-8. degrees of
        arc`.
    chopper_position:
        Position of the chopper. Default is `-15 m`.

    Returns
    -------
    :
        A dict.
    """
    if beam_size is None:
        beam_size = 2.0 * sc.units.mm
    if sample_size is None:
        sample_size = 10.0 * sc.units.mm
    if detector_spatial_resolution is None:
        detector_spatial_resolution = 0.0025 * sc.units.m
    if gravity is None:
        gravity = sc.vector(value=[0, -1, 0]) * g
    if chopper_frequency is None:
        chopper_frequency = sc.scalar(20 / 3, unit='Hz')
    if chopper_phase is None:
        chopper_phase = sc.scalar(-8.0, unit='deg')
    if chopper_1_position is None:
        chopper_1_position = sc.vector(value=[0, 0, -15.5], unit='m')
    if chopper_2_position is None:
        chopper_2_position = sc.vector(value=[0, 0, -14.5], unit='m')
    beamline = {
        'sample_rotation': sample_rotation,
        'beam_size': beam_size,
        'sample_size': sample_size,
        'detector_spatial_resolution': detector_spatial_resolution,
        'gravity': gravity,
    }
    # TODO: in scn.load_nexus, the chopper parameters are stored as coordinates
    # of a DataArray, and the data value is a string containing the name of the
    # chopper. This does not allow storing e.g. chopper cutout angles.
    # We should change this to be a Dataset, which is what we do here.
    beamline["source_chopper_2"] = sc.scalar(
        make_chopper(
            frequency=chopper_frequency,
            phase=chopper_phase,
            position=chopper_2_position,
        )
    )
    beamline["source_chopper_1"] = sc.scalar(
        make_chopper(
            frequency=chopper_frequency,
            phase=chopper_phase,
            position=chopper_1_position,
        )
    )
    return beamline


@log_call(instrument='amor', level='DEBUG')
def instrument_view_components(da: sc.DataArray) -> dict:
    """
    Create a dict of instrument view components, containing:
      - the sample
      - the source chopper

    Parameters
    ----------
    da:
        The DataArray containing the sample and source chopper coordinates.

    Returns
    -------
    :
        Dict of instrument view definitions.
    """
    return {
        "sample": {
            'center': da.meta['sample_position'],
            'color': 'red',
            'size': sc.vector(value=[0.2, 0.01, 0.2], unit=sc.units.m),
            'type': 'box',
        },
        "source_chopper_2": {
            'center': da.meta['source_chopper_2'].value["position"].data,
            'color': 'blue',
            'size': sc.vector(value=[0.5, 0, 0], unit=sc.units.m),
            'type': 'disk',
        },
        "source_chopper_1": {
            'center': da.meta['source_chopper_1'].value["position"].data,
            'color': 'blue',
            'size': sc.vector(value=[0.5, 0, 0], unit=sc.units.m),
            'type': 'disk',
        },
    }
