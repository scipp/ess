import sciline as sl
import scipp as sc
from typing import TypeVar, NewType

from ess.packages.essdiffraction.src.ess.powder.types import (
    RunType,
)

class CylindricalSurfaceDetector(sl.Scope[RunType, sc.DataArray], sc.DataArray):
    """Detector data with time-of-flight coordinate."""


class CylindricalSurfaceTransformGraph(sl.Scope[dict], dict):
    """Graph for transforming projecting 3D POWTEX data onto a cylindrical surface."""

ProjectionRadius = NewType("ProjectionRadius", sc.Variable)
"""Radius of the cylindrical surface."""