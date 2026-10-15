# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)

"""Coordinate conversions for POWTEX texture reduction."""

import numpy as np
import scipp as sc
import scippneutron as scn
from scippneutron.conversion.tof import dspacing_from_tof

from ..powtex.types import CalibrationData, RunType, TofDetector
from .types import CylindricalSurfaceTransformGraph, DspacingCylSurfaceDetector

PROJECTION_RADIUS = sc.scalar(800.0, unit="mm")
"""Radius of the cylindrical detector surface used for projection."""


def project_onto_cylindrical_surface(
    data: TofDetector[RunType],
    calibration_data: CalibrationData,
) -> DspacingCylSurfaceDetector[RunType]:
    """Project detector data to a cylindrical surface."""
    _ = calibration_data

    data.coords["projection_radius"] = PROJECTION_RADIUS

    # ToDo: implement proper dspacing calculation using difc, difa, tof
    # and add it to the graph
    ltotal = scn.Ltotal(data, scatter=True)
    two_theta = scn.two_theta(data)
    dspacing = dspacing_from_tof(
        tof=data.coords["tof"], Ltotal=ltotal, two_theta=two_theta
    )

    data = data.transform_coords(
        ["gamma", "two_theta", "proj_position"],
        graph=cylindrical_surface_projection_graph(),
        keep_intermediate=False,
    )

    for key in ["position", "tof", "sample_position", "source_position"]:
        data.coords.pop(key, None)
    data.coords["dspacing"] = dspacing
    return DspacingCylSurfaceDetector[RunType](data)


def cylindrical_surface_projection_graph() -> CylindricalSurfaceTransformGraph:
    """Return the graph projecting Cartesian positions onto a cylinder."""
    return CylindricalSurfaceTransformGraph({**cyl_coords_from_cartesian()})


def cyl_coords_from_cartesian() -> dict:
    """Return graph nodes converting Cartesian detector positions."""
    return {("gamma", "two_theta", "proj_position"): get_cylindrical_from_cartesian}


def get_cylindrical_from_cartesian(position, projection_radius):
    """Convert Cartesian detector positions to POWTEX cylindrical coordinates."""
    x = position.fields.x
    y = position.fields.y
    z = position.fields.z

    xy_radius = sc.sqrt(x**2 + y**2)

    # The implementation of gamma below is performed to be consistent with user
    # expectations, i.e. gamma = 0 corresponds to y = 0 and negative x and then
    # increases with increasing y.
    angle = sc.atan2(y=y, x=-x)
    gamma = sc.mod(angle, sc.scalar(2.0 * np.pi, unit=angle.unit)).to(unit="deg")
    two_theta = sc.atan2(y=xy_radius, x=z).to(unit="deg")

    # Determine the projection of the position onto the cylindrical surface with
    # the given radius.
    proj_x = projection_radius / xy_radius * x
    proj_y = projection_radius / xy_radius * y
    proj_z = projection_radius / xy_radius * z
    proj_position = sc.spatial.as_vectors(proj_x, proj_y, proj_z)

    return {
        # "rho": rho,
        # "proj_x": proj_x,
        # "proj_y": proj_y,
        # "proj_z": proj_z,
        "gamma": gamma,
        "two_theta": two_theta,
        "proj_position": proj_position,
    }


def get_cartesian_from_cylindrical(cylinder_radius, gamma, two_theta):
    """Convert cylindrical coordinates positions to Cartesian."""
    x = -cylinder_radius * sc.cos(gamma)
    y = cylinder_radius * sc.sin(gamma)
    z = cylinder_radius / sc.tan(two_theta)

    return (x, y, z)


providers = (cylindrical_surface_projection_graph,)
