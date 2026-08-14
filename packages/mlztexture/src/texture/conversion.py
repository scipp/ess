import scipp as sc
import scippneutron as scn
import numpy as np

from ess.packages.mlztexture.src.texture.types import (
    DspacingCylSurfaceDetector,
    CylindricalSurfaceTransformGraph,
)

from ess.packages.essdiffraction.src.ess.powder.types import (
    TofDetector,
    RunType,
    CalibrationData
)

from scippneutron.conversion.tof import dspacing_from_tof

# Radius of the cylindrical surface onto which the detector data will be projected.
# This is a constant value used in the projection calculations.
PROJECTION_RADIUS = sc.scalar(800.0, unit="mm")


def project_onto_cylindrical_surface(
    data: TofDetector[RunType],
    calibration_data: CalibrationData
) -> DspacingCylSurfaceDetector[RunType]:
    """
    Project the detector data to a cylindrical surface.
    """

    data.coords["projection_radius"] = PROJECTION_RADIUS

    # ToDo: implement proper dspacing calculation using difc, difa, tof
    # and add it to the graph
    Ltotal = scn.Ltotal(data, scatter=True)
    two_theta = scn.two_theta(data)
    dspacing = dspacing_from_tof(tof=data.coords["tof"], Ltotal=Ltotal, two_theta=two_theta)

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
    """
    ToDo
    """
    return CylindricalSurfaceTransformGraph(
        {
           **cyl_coords_from_cartesian(),
        }
    )


def cyl_coords_from_cartesian() -> dict:
    """ToDo"""
    modified_graph = {}
    modified_graph[("gamma", "two_theta", "proj_position")] = get_cylindrical_from_cartesian
    return modified_graph


def get_cylindrical_from_cartesian(position, projection_radius):
    x = position.fields.x
    y = position.fields.y
    z = position.fields.z

    xy_radius = sc.sqrt(x**2 + y**2)
    # the implementation of gamma below is performed to be consistent with user expectations,
    # i.e. gamma = 0 corresponds to y = 0 and negative x and then increases with increasting y
    angle = sc.atan2(y=y, x=-x)
    gamma = sc.mod(angle, sc.scalar(2 * np.pi, unit=angle.unit)).to(unit="deg")
    two_theta = sc.atan2(y=xy_radius, x=z).to(unit="deg")

    # determine the projection of the position onto the cylindrical surface with the given radius
    proj_x = projection_radius / xy_radius  * x
    proj_y = projection_radius / xy_radius * y
    proj_z = projection_radius / xy_radius * z

    proj_position = sc.spatial.as_vectors(proj_x, proj_y, proj_z)


    return {
        #"rho": rho,
        #"proj_x": proj_x,
        #"proj_y": proj_y,
        #"proj_z": proj_z,
        "proj_position": proj_position,
        "gamma": gamma,
        "two_theta": two_theta,
    }
