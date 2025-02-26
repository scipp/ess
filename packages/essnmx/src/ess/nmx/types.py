from typing import Any, NewType

import scipp as sc

FilePath = NewType("FilePath", str)
"""File name of a file containing the results of a McStas run"""

DetectorIndex = NewType("DetectorIndex", int | sc.Variable | sc.DataArray)
"""Index of the detector to load. Index ordered by the id:s of the pixels"""

DetectorName = NewType("DetectorName", str)
"""Name of the detector to load"""

DetectorBankPrefix = NewType("DetectorBankPrefix", str)
"""Prefix identifying the event data array containing
the events from the selected detector"""

MaximumCounts = NewType("MaximumCounts", int)
"""Maximum number of counts after scaling the event counts"""

MaximumProbability = NewType("MaximumProbability", sc.Variable)
"""Maximum probability to scale the McStas event counts"""

McStasWeight2CountScaleFactor = NewType("McStasWeight2CountScaleFactor", sc.Variable)
"""Scale factor to convert McStas weights to counts"""


RawEventProbability = NewType("RawEventProbability", sc.DataArray)
"""DataArray containing the event probabilities read from the McStas file,
has coordinates 'id' and 't' """

NMXRawEventCountsDataGroup = NewType("NMXRawEventCountsDataGroup", sc.DataGroup)
"""DataGroup containing the RawEventData and other metadata"""

ProtonCharge = NewType("ProtonCharge", sc.Variable)
"""The proton charge signal"""

CrystalRotation = NewType("CrystalRotation", sc.Variable)
"""Rotation of the crystal"""

DetectorGeometry = NewType("DetectorGeometry", Any)
"""Description of the geometry of the detector banks"""

TimeBinSteps = NewType("TimeBinSteps", int)
"""Number of bins in the binning of the time coordinate"""

PixelIds = NewType("PixelIds", sc.Variable)
"""The pixel ids of the detector"""

NMXReducedProbability = NewType("NMXReducedProbability", sc.DataArray)
"""Histogram of time-of-arrival and pixel-id."""

NMXReducedCounts = NewType("NMXReducedCounts", sc.DataArray)
"""Histogram of time-of-arrival and pixel-id."""

NMXReducedDataGroup = NewType("NMXReducedDataGroup", sc.DataGroup)
"""Histogram of time-of-arrival and pixel-id, with additional metadata."""
