from typing import NewType

from scipp import Variable

from ess.reduce.nexus import types as reduce_t

ArcNumber = NewType('ArcNumber', Variable)

# See https://github.com/scipp/essreduce/issues/105 about monitor names
FrameMonitor0 = reduce_t.Monitor1
FrameMonitor1 = reduce_t.Monitor2
FrameMonitor2 = reduce_t.Monitor3
FrameMonitor3 = reduce_t.Monitor4
PsdMonitor0 = reduce_t.Monitor5
PsdMonitor1 = reduce_t.Monitor6
MonitorType = reduce_t.MonitorType
