# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)

"""NeXus input/output for DREAM.

Notes on the detector dimensions (2024-05-22):

See https://confluence.esss.lu.se/pages/viewpage.action?pageId=462000005
and the ICD DREAM interface specification for details.

- The high-resolution and SANS detectors have a very odd numbering scheme.
  The scheme attempts to follows some sort of physical ordering in space (x,y,z),
  but it is not possible to reshape the data into all the logical dimensions.
"""

import scipp as sc
from ess.powder.types import (
    Filename,
    LoadedNeXusDetector,
    NeXusDetectorName,
    RawDetectorData,
    RawSample,
    RawSource,
    ReducibleDetectorData,
    RunType,
    SamplePosition,
    SourcePosition,
)
from ess.reduce import nexus

DETECTOR_BANK_SIZES = {
    "endcap_backward_detector": {
        "strip": 16,
        "wire": 16,
        "module": 11,
        "segment": 28,
        "counter": 2,
    },
    "endcap_forward_detector": {
        "strip": 16,
        "wire": 16,
        "module": 5,
        "segment": 28,
        "counter": 2,
    },
    "mantle_detector": {
        "wire": 32,
        "module": 5,
        "segment": 6,
        "strip": 256,
        "counter": 2,
    },
    "high_resolution_detector": {
        "strip": 32,
        "other": -1,
    },
    "sans_detector": lambda x: x.fold(
        dim="detector_number",
        sizes={
            "strip": 32,
            "other": -1,
        },
    ),
}


def load_nexus_sample(file_path: Filename[RunType]) -> RawSample[RunType]:
    return RawSample[RunType](nexus.load_sample(file_path))


def dummy_load_sample(file_path: Filename[RunType]) -> RawSample[RunType]:
    """
    In test files there is not always a sample, so we need a dummy.
    """
    return RawSample[RunType](
        sc.DataGroup({'position': sc.vector(value=[0, 0, 0], unit='m')})
    )


def load_nexus_source(file_path: Filename[RunType]) -> RawSource[RunType]:
    return RawSource[RunType](nexus.load_source(file_path))


def load_nexus_detector(
    file_path: Filename[RunType], detector_name: NeXusDetectorName
) -> LoadedNeXusDetector[RunType]:
    out = nexus.load_detector(file_path=file_path, detector_name=detector_name)
    out.pop("pixel_shape", None)
    return LoadedNeXusDetector[RunType](out)


def get_source_position(
    raw_source: RawSource[RunType],
) -> SourcePosition[RunType]:
    return SourcePosition[RunType](raw_source["position"])


def get_sample_position(
    raw_sample: RawSample[RunType],
) -> SamplePosition[RunType]:
    return SamplePosition[RunType](raw_sample["position"])


def get_detector_data(
    detector: LoadedNeXusDetector[RunType],
    detector_name: NeXusDetectorName,
) -> RawDetectorData[RunType]:
    da = nexus.extract_detector_data(detector)
    if detector_name in DETECTOR_BANK_SIZES:
        da = da.fold(dim="detector_number", sizes=DETECTOR_BANK_SIZES[detector_name])
    return RawDetectorData[RunType](da)


def patch_detector_data(
    detector_data: RawDetectorData[RunType],
    source_position: SourcePosition[RunType],
    sample_position: SamplePosition[RunType],
) -> ReducibleDetectorData[RunType]:
    """
    Patch a detector data object with source and sample positions.
    Also adds variances to the event data if they are missing.
    """
    out = detector_data.copy(deep=False)
    if out.bins is not None:
        content = out.bins.constituents["data"]
        if content.variances is None:
            content.variances = content.values
    out.coords["sample_position"] = sample_position
    out.coords["source_position"] = source_position
    return ReducibleDetectorData[RunType](out)


providers = (
    load_nexus_sample,
    load_nexus_source,
    load_nexus_detector,
    get_source_position,
    get_sample_position,
    get_detector_data,
    patch_detector_data,
)
"""
Providers for loading and processing DREAM NeXus data.
"""
