# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

from scipp import vector

from ..types import (
    LabMomentumTransfer,
    SampleTableAngle,
    TableMomentumTransfer,
)
from .kf import providers as kf_providers
from .ki import providers as ki_providers

# Directions relative to the incident beam coordinate system
PERP, VERT, PARALLEL = (vector(v) for v in ([1, 0, 0], [0, 1, 0], [0, 0, 1]))


# TODO remove
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


providers = (
    *ki_providers,
    *kf_providers,
    sample_table_momentum_vector,
)
