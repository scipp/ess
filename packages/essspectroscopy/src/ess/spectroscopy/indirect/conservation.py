from ess.spectroscopy.types import *
from scipp import vector

from .kf import providers as kf_providers
from .ki import providers as ki_providers

# Directions relative to the incident beam coordinate system
PERP, VERT, PARALLEL = [vector(v) for v in ([1, 0, 0], [0, 1, 0], [0, 0, 1])]


def lab_momentum_vector(
    ki: IncidentWavevector, kf: FinalWavevector
) -> LabMomentumTransfer:
    return kf - ki


def lab_momentum_x(q: LabMomentumTransfer) -> LabMomentumTransferX:
    from scipp import dot

    return dot(PERP, q)


def lab_momentum_y(q: LabMomentumTransfer) -> LabMomentumTransferY:
    from scipp import dot

    return dot(VERT, q)


def lab_momentum_z(q: LabMomentumTransfer) -> LabMomentumTransferZ:
    from scipp import dot

    return dot(PARALLEL, q)


def sample_table_momentum_vector(
    a3: SampleTableAngle, q: LabMomentumTransfer
) -> TableMomentumTransfer:
    """Rotate the momentum transfer vector into the sample-table coordinate system

    When a3 is zero, the sample-table and lab coordinate systems are the same. That is, Z is along the incident
    beam, Y is opposite the gravitational force, and X completes the right-handed coordinate system.
    The sample-table angle, a3, has a rotation vector along Y, such that a positive 90-degree rotation places the
    sample-table Z along the lab X.
    """
    from scipp.spatial import rotations_from_rotvecs

    # negative a3 since we rotate coordinates not axes here
    return rotations_from_rotvecs(-a3 * VERT) * q


def sample_table_momentum_x(q: TableMomentumTransfer) -> TableMomentumTransferX:
    from scipp import dot

    return dot(PERP, q)


def sample_table_momentum_y(q: TableMomentumTransfer) -> TableMomentumTransferY:
    from scipp import dot

    return dot(VERT, q)


def sample_table_momentum_z(q: TableMomentumTransfer) -> TableMomentumTransferZ:
    from scipp import dot

    return dot(PARALLEL, q)


# def energy(ei: IncidentEnergy, ef: FinalEnergy) -> EnergyTransfer:
#     return ei - ef


def energy(ki: IncidentWavenumber, kf: FinalWavenumber) -> EnergyTransfer:
    from scipp.constants import hbar, neutron_mass

    return hbar * hbar * (ki * ki - kf * kf) / 2 / neutron_mass


providers = [
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
]
