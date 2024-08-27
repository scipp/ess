from scipp import vector
from ess.spectroscopy.types import *
from .ki import providers as ki_providers
from .kf import providers as kf_providers


# scipp vectors are always float64 element type?
X, Y, Z = [vector(v) for v in ([1, 0, 0], [0, 1, 0], [0, 0, 1])]


def lab_momentum_vector(ki: IncidentWavevector, kf: FinalWavevector) -> LabMomentumTransfer:
    return kf - ki


def lab_momentum_x(q: LabMomentumTransfer) -> LabMomentumTransferX:
    from scipp import dot
    return dot(X, q)


def lab_momentum_y(q: LabMomentumTransfer) -> LabMomentumTransferY:
    from scipp import dot
    return dot(Y, q)


def lab_momentum_z(q: LabMomentumTransfer) -> LabMomentumTransferZ:
    from scipp import dot
    return dot(Z, q)


def energy(ei: IncidentEnergy, ef: FinalEnergy) -> EnergyTransfer:
    return ei - ef



providers = [
    *ki_providers,
    *kf_providers,
    lab_momentum_vector,
    lab_momentum_x,
    lab_momentum_y,
    lab_momentum_z,
    energy,
]