from collections.abc import Callable
from typing import NewType

import sciline
import scipp as sc

from ess.reduce.nexus.types import DetectorData, Filename, RunType, SampleRun
from ess.reduce.time_of_flight.types import DetectorTofData


class StreakClusteredData(sciline.Scope[RunType, sc.DataArray], sc.DataArray):
    """Detector data binned by streak"""


DetectorData = DetectorData
Filename = Filename
SampleRun = SampleRun
DetectorTofData = DetectorTofData


TwoThetaMaskFunction = NewType(
    'TwoThetaMaskFunction', Callable[[sc.Variable], sc.Variable]
)

TofCoordTransformGraph = NewType("TofCoordTransformGraph", dict)

PulseLength = NewType('PulseLength', sc.Variable)
'''Length of the neutron source pulse in time.'''

ModulationPeriod = NewType('ModulationPeriod', sc.Variable)
'''The effective period of the modulating chopper:
``1 / (K * F)`` where ``K`` is the number of chopper openings and
``F`` is the chopper frequency.'''

WavelengthDefinitionChopperDelay = NewType(
    'WavelengthDefinitionChopperDelay', sc.Variable
)
'''Wavelength definition chopper time delay relative to source pulse.'''

DHKLList = NewType('DHKLList', sc.Variable)
'''List of peak position estimates.'''
