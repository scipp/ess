# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
import scipp as sc

from ..choppers import make_chopper
from ..logging import log_call
from ..types import (
    BeamSize,
    DetectorSpatialResolution,
    Gravity,
    Run,
    SampleRotation,
    SampleSize,
)
from .types import (
    BeamlineParams,
    Chopper1Position,
    Chopper2Position,
    ChopperFrequency,
    ChopperPhase,
)


@log_call(
    instrument='amor', message='Constructing AMOR beamline from default parameters'
)
def make_beamline(
    sample_rotation: SampleRotation[Run],
    beam_size: BeamSize[Run],
    sample_size: SampleSize[Run],
    detector_spatial_resolution: DetectorSpatialResolution[Run],
    gravity: Gravity,
    chopper_frequency: ChopperFrequency[Run],
    chopper_phase: ChopperPhase[Run],
    chopper_1_position: Chopper1Position[Run],
    chopper_2_position: Chopper2Position[Run],
) -> BeamlineParams[Run]:
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
    return BeamlineParams(beamline)


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


providers = [make_beamline]
