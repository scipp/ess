from typing import NewType

import scipp as sc

# TODO What do they mean?
# Supermirror parameters
MValue = NewType('MValue', sc.Variable)
CriticalEdge = NewType('CriticalEdge', sc.Variable)
Alpha = NewType('Alpha', sc.Variable)
SupermirrorCalibrationFactor = NewType('SupermirrorCalibrationFactor', sc.Variable)
