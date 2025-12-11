# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

"""Coordinate conversions for indirect spectroscopy."""

import numpy as np
import scipp as sc
import scipp.constants
import scippnexus as snx
from scippneutron.conversion.tof import (
    energy_from_wavelength,
    wavelength_from_tof,
    wavevector_from_wavelength,
)

from ..types import (
    EnergyQDetector,
    GravityVector,
    IncidentEnergyDetector,
    InelasticCoordTransformGraph,
    MonitorCoordTransformGraph,
    MonitorType,
    NormalizedIncidentEnergyDetector,
    Position,
    PrimarySpecCoordTransformGraph,
    RunType,
    SecondarySpecCoordTransformGraph,
    TofDetector,
    TofMonitor,
    WavelengthMonitor,
)


def incident_energy_from_wavelength(*, incident_wavelength: sc.Variable) -> sc.Variable:
    return energy_from_wavelength(wavelength=incident_wavelength)


def incident_wavelength_from_tof(
    *, sample_tof: sc.Variable, L1: sc.Variable
) -> sc.Variable:
    return wavelength_from_tof(tof=sample_tof, Ltotal=L1)


def incident_wavevector_from_incident_wavelength(
    *, incident_wavelength: sc.Variable, incident_beam: sc.Variable
) -> sc.Variable:
    return wavevector_from_wavelength(
        wavelength=incident_wavelength, beam=incident_beam
    )


def energy_transfer(
    *, incident_energy: sc.Variable, final_energy: sc.Variable
) -> sc.Variable:
    r"""Compute the energy transfer.

    Here, the energy transfer is defined the same as in ScippNeutron
    (https://scipp.github.io/scippneutron/user-guide/coordinate-transformations.html)
    to be

    .. math::

        \Delta E = E_i - E_f

    Parameters
    ----------
    incident_energy:
        The neutron energy :math:`E_i` before scattering.
    final_energy:
        The neutron energy :math:`E_f` after scattering.

    Returns
    -------
    :
        The energy transfer :math:`\Delta E`.
    """
    return incident_energy - final_energy.to(unit=incident_energy.unit, copy=False)


def lab_momentum_transfer_from_wavevectors(
    incident_wavevector: sc.Variable, final_wavevector: sc.Variable
) -> sc.Variable:
    r"""Compute the momentum transfer in the lab frame.

    Here, the momentum transfer is defined as

    .. math::

        \vec{Q} = \vec{k_i} - \vec{k_f}

    Parameters
    ----------
    incident_wavevector:
        The neutron wavevector :math:`\vec{k_i}` before scattering.
    final_wavevector:
        The neutron wavevector :math:`\vec{k_f}` after scattering.

    Returns
    -------
    :
        The momentum transfer :math:`\vec{Q}` in the lab frame.
    """
    return (
        incident_wavevector.to(unit=final_wavevector.unit, copy=False)
        - final_wavevector
    )


def lab_momentum_transfer_from_incident_energy(
    incident_energy: sc.Variable,
    incident_beam: sc.Variable,
    final_wavevector: sc.Variable,
) -> sc.Variable:
    r"""Compute the momentum transfer in the lab frame.

    Here, the momentum transfer is defined as

    .. math::

        \vec{Q} = \vec{k_i} - \vec{k_f}

    with

    .. math::

        \vec{k_i} = \frac{2\pi}{h} \hat{b}_i \sqrt{2 M_n E_i}

    Parameters
    ----------
    incident_energy:
        The neutron energy :math:`E_i` before scattering.
    incident_beam:
        The vector of the incident beam in the lab frame :math:`\vec{b}_i`.
    final_wavevector:
        The neutron wavevector :math:`\vec{k_f}` after scattering.

    Returns
    -------
    :
        The momentum transfer :math:`\vec{Q}` in the lab frame.
    """
    # Typical shapes:
    #  - incident_beam: ()
    #  - incident_energy: (incident_energy,)  (small)
    #  - final_wavevector: (detector,)  (large-ish but 1D)
    #  - result: (detector, incident_energy)
    incident_wavevector = (
        (2 * np.pi / sc.constants.h)
        * (incident_beam / sc.norm(incident_beam))
        * sc.sqrt(2 * sc.constants.m_n * incident_energy)
    )
    # Order of operations such that the final result has dim order described above.
    return -final_wavevector + incident_wavevector.to(
        unit=final_wavevector.unit, copy=False
    )


def rotate_to_sample_table_momentum_transfer(
    *,
    a3: sc.Variable,
    lab_momentum_transfer: sc.Variable,
    gravity: sc.Variable,
) -> sc.Variable:
    r"""Rotate the momentum transfer vector into the sample-table coordinate system

    Here, the momentum transfer is defined as

    .. math::

        \vec{Q} = \vec{k_i} - \vec{k_f}

    Note
    ----
    When a3 is zero, the sample-table and lab coordinate systems are the same.
    That is, Z is along the incident beam, Y is opposite the gravitational force,
    and X completes the right-handed coordinate system. The sample-table angle, a3,
    has a rotation vector along Y, such that a positive 90-degree rotation places the
    sample-table Z along the lab X.

    Parameters
    ----------
    a3:
        The rotation angle of the sample table around the laboratory Y axis
    lab_momentum_transfer:
        The momentum transfer in the laboratory coordinate system
    gravity:
        The gravity vector which indicates the vertical axis.

    Returns
    -------
    :
        The momentum transfer in the sample-table coordinate system.
    """
    vertical = -gravity / sc.norm(gravity)
    # negative a3 since we rotate coordinates, not axes here
    return sc.spatial.rotations_from_rotvecs(-a3 * vertical) * lab_momentum_transfer


def inelastic_coordinate_transformation_graph_at_sample(
    gravity: GravityVector,
) -> InelasticCoordTransformGraph:
    return InelasticCoordTransformGraph(
        {
            'energy_transfer': energy_transfer,
            'incident_energy': incident_energy_from_wavelength,
            'incident_wavelength': incident_wavelength_from_tof,
            'incident_wavevector': incident_wavevector_from_incident_wavelength,
            'gravity': lambda: gravity,
            'lab_momentum_transfer': lab_momentum_transfer_from_wavevectors,
            'sample_table_momentum_transfer': rotate_to_sample_table_momentum_transfer,
        }
    )


def add_inelastic_coordinates(
    data: NormalizedIncidentEnergyDetector[RunType], graph: InelasticCoordTransformGraph
) -> EnergyQDetector[RunType]:
    transformed = data.transform_coords(
        [
            'energy_transfer',
            'sample_table_momentum_transfer',
            # These are inputs, but we want to preserve them
            'a3',
            'a4',
        ],
        graph=graph,
        keep_aliases=False,
        keep_inputs=False,
        keep_intermediate=False,
        rename_dims=False,
    )
    return EnergyQDetector[RunType](transformed)


def add_incident_energy(
    data: TofDetector[RunType], graph: InelasticCoordTransformGraph
) -> IncidentEnergyDetector[RunType]:
    transformed = data.transform_coords(
        [
            'incident_energy',
            'incident_wavelength',
            # These are inputs, but we need them for binning:
            'a3',
            'a4',
            'final_energy',
            'final_wavevector',
        ],
        graph=graph,
        keep_aliases=False,
        keep_inputs=False,
        keep_intermediate=False,
        rename_dims=False,
    )
    return IncidentEnergyDetector[RunType](transformed)


def add_spectrometer_coords(
    data: sc.DataArray,
    primary_graph: PrimarySpecCoordTransformGraph[RunType],
    secondary_graph: SecondarySpecCoordTransformGraph[RunType],
) -> sc.DataArray:
    """Compute and add coordinates for the spectrometer.

    Parameters
    ----------
    data:
        Data array with beamline coordinates "position", "source_position", and
        "sample_position".
        Does not need to contain events or flight times.
    primary_graph:
        Coordinate transformation graph for the primary spectrometer.
    secondary_graph:
        Coordinate transformation graph for the secondary spectrometer.
        Must be a closure over analyzer parameters.
        And those parameters must have a compatible shape with ``data``.

    Returns
    -------
    :
        Input data with added spectrometer coordinates.
        This includes "final_energy", "secondary_flight_time", and "L1".
    """
    return data.transform_coords(
        (
            'final_energy',
            'final_wavevector',
            'incident_beam',
            'L1',
            'secondary_flight_time',
        ),
        graph={**primary_graph, **secondary_graph},
        keep_intermediate=False,
        keep_aliases=False,
        rename_dims=False,
    )


def monitor_coordinate_transformation_graph(
    source_position: Position[snx.NXsource, RunType],
) -> MonitorCoordTransformGraph[RunType]:
    from scippneutron.conversion.graph import beamline, tof

    return MonitorCoordTransformGraph[RunType](
        {
            **beamline.beamline(scatter=False),
            **tof.elastic_wavelength(start='tof'),
            'incident_wavelength': 'wavelength',
            'source_position': lambda: source_position,
        }
    )


def add_monitor_wavelength_coord(
    monitor: TofMonitor[RunType, MonitorType], graph: MonitorCoordTransformGraph
) -> WavelengthMonitor[RunType, MonitorType]:
    return WavelengthMonitor[RunType, MonitorType](
        monitor.transform_coords(
            'incident_wavelength',
            graph=graph,
            keep_intermediate=False,
            keep_aliases=False,
        )
    )


providers = (
    add_inelastic_coordinates,
    add_incident_energy,
    add_monitor_wavelength_coord,
    inelastic_coordinate_transformation_graph_at_sample,
    monitor_coordinate_transformation_graph,
)
