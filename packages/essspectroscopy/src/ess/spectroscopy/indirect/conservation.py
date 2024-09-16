from scipp import vector

from ..types import (
    EnergyTransfer,
    FinalWavenumber,
    FinalWavevector,
    IncidentWavenumber,
    IncidentWavevector,
    LabMomentumTransfer,
    LabMomentumTransferX,
    LabMomentumTransferY,
    LabMomentumTransferZ,
    SampleTableAngle,
    TableMomentumTransfer,
    TableMomentumTransferX,
    TableMomentumTransferY,
    TableMomentumTransferZ,
)
from .kf import providers as kf_providers
from .ki import providers as ki_providers

# Directions relative to the incident beam coordinate system
PERP, VERT, PARALLEL = [vector(v) for v in ([1, 0, 0], [0, 1, 0], [0, 0, 1])]


def lab_momentum_vector(
    ki: IncidentWavevector, kf: FinalWavevector
) -> LabMomentumTransfer:
    """Return the momentum transferred to the sample in the laboratory coordinate system

    The laboratory coordinate system is independent of sample angle

    Parameters
    ----------
    ki:
        incident wavevector of the neutron
    kf:
        final wavevector of the neutron

    Returns
    -------
    :
        The difference kf - ki
    """
    return kf - ki


def lab_momentum_x(q: LabMomentumTransfer) -> LabMomentumTransferX:
    """Return the X coordinate of the momentum transfer in the lab coordinate system"""
    from scipp import dot

    return dot(PERP, q)


def lab_momentum_y(q: LabMomentumTransfer) -> LabMomentumTransferY:
    """Return the Y coordinate of the momentum transfer in the lab coordinate system"""
    from scipp import dot

    return dot(VERT, q)


def lab_momentum_z(q: LabMomentumTransfer) -> LabMomentumTransferZ:
    """Return the Z coordinate of the momentum transfer in the lab coordinate system"""
    from scipp import dot

    return dot(PARALLEL, q)


def sample_table_momentum_vector(
    a3: SampleTableAngle, q: LabMomentumTransfer
) -> TableMomentumTransfer:
    """Rotate the momentum transfer vector into the sample-table coordinate system

    Notes
    -----
    When a3 is zero, the sample-table and lab coordinate systems are the same.
    That is, Z is along the incident beam, Y is opposite the gravitational force,
    and X completes the right-handed coordinate system. The sample-table angle, a3,
    has a rotation vector along Y, such that a positive 90-degree rotation places the
    sample-table Z along the lab X.

    Parameters
    ----------
    a3:
        The rotation angle of the sample table around the laboratory Y axis
    q:
        The momentum transfer in the laboratory coordinate system
    """
    from scipp.spatial import rotations_from_rotvecs

    # negative a3 since we rotate coordinates not axes here
    return rotations_from_rotvecs(-a3 * VERT) * q


def sample_table_momentum_x(q: TableMomentumTransfer) -> TableMomentumTransferX:
    """Return the X coordinate of the momentum transfer in the sample-table system"""
    from scipp import dot

    return dot(PERP, q)


def sample_table_momentum_y(q: TableMomentumTransfer) -> TableMomentumTransferY:
    """Return the Y coordinate of the momentum transfer in the sample-table system"""
    from scipp import dot

    return dot(VERT, q)


def sample_table_momentum_z(q: TableMomentumTransfer) -> TableMomentumTransferZ:
    """Return the Z coordinate of the momentum transfer in the sample-table system"""
    from scipp import dot

    return dot(PARALLEL, q)


def energy(ki: IncidentWavenumber, kf: FinalWavenumber) -> EnergyTransfer:
    """Calculate the energy transferred to the sample by a neutron"""
    from scipp.constants import hbar, neutron_mass

    return hbar * hbar * (ki * ki - kf * kf) / 2 / neutron_mass


providers = (
    *ki_providers,
    *kf_providers,
    lab_momentum_vector,
    lab_momentum_x,
    lab_momentum_y,
    lab_momentum_z,
    sample_table_momentum_vector,
    sample_table_momentum_x,
    sample_table_momentum_y,
    sample_table_momentum_z,
    energy,
)
