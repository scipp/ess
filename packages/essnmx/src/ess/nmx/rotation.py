# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
# Rotation related functions for NMX
import numpy as np
import scipp as sc
from numpy.typing import NDArray


def axis_angle_to_quaternion(
    *, x: float, y: float, z: float, theta: sc.Variable
) -> NDArray:
    """Convert axis-angle to queternions, [x, y, z, w].

    Parameters
    ----------
    x:
        X component of axis of rotation.
    y:
        Y component of axis of rotation.
    z:
        Z component of axis of rotation.
    theta:
        Angle of rotation, with unit of ``rad`` or ``deg``.

    Returns
    -------
    :
        A list of (normalized) quaternions, [x, y, z, w].

    Notes
    -----
    Axis of rotation (x, y, z) does not need to be normalized,
    but it returns a unit quaternion (x, y, z, w).

    """

    w: sc.Variable = sc.cos(theta.to(unit='rad') / 2)
    xyz: sc.Variable = -sc.sin(theta.to(unit='rad') / 2) * sc.vector([x, y, z])
    q = np.array([*xyz.values, w.value])
    return q / np.linalg.norm(q)


def quaternion_to_matrix(*, x: float, y: float, z: float, w: float) -> sc.Variable:
    """Convert quaternion to rotation matrix.

    Parameters
    ----------
    x:
        x(a) component of quaternion.
    y:
        y(b) component of quaternion.
    z:
        z(c) component of quaternion.
    w:
        w component of quaternion.

    Returns
    -------
    :
        A 3x3 rotation matrix.

    """
    from scipy.spatial.transform import Rotation

    return sc.spatial.rotations_from_rotvecs(
        rotation_vectors=sc.vector(
            Rotation.from_quat([x, y, z, w]).as_rotvec(),
            unit='rad',
        )
    )
