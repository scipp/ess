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

DETECTOR_BANK_RESHAPING = {
    "endcap_backward_detector": lambda x: x.fold(
        dim="detector_number",
        sizes={
            "strip": 16,
            "wire": 16,
            "module": 11,
            "segment": 28,
            "counter": 2,
        },
    ),
    "endcap_forward_detector": lambda x: x.fold(
        dim="detector_number",
        sizes={
            "strip": 16,
            "wire": 16,
            "module": 5,
            "segment": 28,
            "counter": 2,
        },
    ),
    "mantle_detector": lambda x: x.fold(
        dim="detector_number",
        sizes={
            "wire": 32,
            "module": 5,
            "segment": 6,
            "strip": 256,
            "counter": 2,
        },
    ),
    "high_resolution_detector": lambda x: x.fold(
        dim="detector_number",
        sizes={
            "strip": 32,
            "other": -1,
        },
    ),
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


def load_nexus_source(file_path: Filename[RunType]) -> RawSource[RunType]:
    return RawSource[RunType](nexus.load_source(file_path))


def load_nexus_detector(
    file_path: Filename[RunType], detector_name: NeXusDetectorName
) -> LoadedNeXusDetector[RunType]:
    return LoadedNeXusDetector[RunType](
        nexus.load_detector(file_path=file_path, detector_name=detector_name)
    )


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
    if detector_name in DETECTOR_BANK_RESHAPING:
        da = DETECTOR_BANK_RESHAPING[detector_name](da)
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


# def load_nexus(
#     path: Union[str, os.PathLike],
#     *,
#     load_pixel_shape: bool = False,
#     entry: str = 'entry',
#     fold_detectors: bool = True,
# ) -> sc.DataGroup:
#     """
#     Load an unprocessed DREAM NeXus file.

#     See https://confluence.esss.lu.se/pages/viewpage.action?pageId=462000005
#     and the ICD DREAM interface specification for details.

#     Notes (2023-12-06):

#     - Mounting-unit, cassette, and counter roughly correspond to the azimuthal angle
#       in the mantle detector. However, counter is reversed in the current files. This
#       may also be the case in the other detectors.
#     - The endcap detectors have a irregular structure that cannot be fully folded.
#       This is not a problem but note again that the counter may be reversed. It is
#       currently not clear if this is a bug in the file.
#     - The high-resolution detector has a very odd numbering scheme. The SANS detector
#       is using the same, but is not populated in the current files. The scheme
#       attempts to follows some sort of physical ordering in space (x,y,z), but it
#       looks partially messed up.

#     Parameters
#     ----------
#     path:
#         Path to the NeXus file.
#     load_pixel_shape:
#         If True, load the pixel shape from the file's NXoff_geometry group. This is
#         often unused by would slow down file loading. Default is False.
#     entry:
#         Name of the entry to load. Default is "entry".
#     fold_detectors:
#         If True, fold the detector data to (partially) mimic the logical detector
#         structure. Default is True.

#     Returns
#     -------
#     :
#         A data group with the loaded file contents.
#     """
#     definitions = snx.base_definitions()
#     if not load_pixel_shape:
#         definitions["NXdetector"] = FilteredDetector

#     with snx.File(path, definitions=definitions) as f:
#         dg = f[entry][()]
#     dg = snx.compute_positions(dg)
#     return fold_nexus_detectors(dg) if fold_detectors else dg


# def fold_nexus_detectors(dg: sc.DataGroup) -> sc.DataGroup:
#     """
#     Fold the detector data in a DREAM NeXus file.

#     The detector banks in the returned data group will have a multi-dimensional shape,
#     following the logical structure as far as possible. Note that the full structure
#     cannot be folded, as some dimensions are irregular.
#     """
#     dg = dg.copy()
#     dg['instrument'] = dg['instrument'].copy()
#     instrument = dg['instrument']
#     mantle = instrument['mantle_detector']
#     mantle['mantle_event_data'] = mantle['mantle_event_data'].fold(
#         dim='detector_number',
#         sizes={
#             'wire': 32,
#             'mounting_unit': 5,
#             'cassette': 6,
#             'counter': 2,
#             'strip': 256,
#         },
#     )
#     for direction in ('backward', 'forward'):
#         endcap = instrument[f'endcap_{direction}_detector']
#         endcap[f'endcap_{direction}_event_data'] = endcap[
#             f'endcap_{direction}_event_data'
#         ].fold(
#             dim='detector_number',
#             sizes={
#                 'strip': 16,
#                 'wire': 16,
#                 'sector': 5 if direction == 'forward' else 11,
#                 'sumo_cass_ctr': -1,  # sumo*cassette*counter, irregular, cannot fold
#             },
#         )
#     high_resolution = instrument['high_resolution_detector']
#     high_resolution['high_resolution_event_data'] = high_resolution[
#         'high_resolution_event_data'
#     ].fold(
#         dim='detector_number',
#         sizes={
#             'strip': 32,
#             'other': -1,  # some random order that is hard to follow
#         },
#     )
#     sans = instrument['sans_detector']
#     sans['sans_event_data'] = sans['sans_event_data'].fold(
#         dim='detector_number',
#         sizes={
#             'strip': 32,
#             'other': -1,  # some random order that is hard to follow
#         },
#     )
#     return dg


# def _skip(name: str, obj: Union[snx.Field, snx.Group]) -> bool:
#     skip_classes = (snx.NXoff_geometry,)
#     return isinstance(obj, snx.Group) and (obj.nx_class in skip_classes)


# class FilteredDetector(snx.NXdetector):
#     def __init__(
#         self, attrs: Dict[str, Any], children: Dict[str, Union[snx.Field, snx.Group]]
#     ):
#         children = {
#             name: child for name, child in children.items() if not _skip(name, child)
#         }
#         super().__init__(attrs=attrs, children=children)
