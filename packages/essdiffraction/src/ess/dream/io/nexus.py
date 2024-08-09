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

from ess import powder

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
    "high_resolution_detector": {"strip": 32, "other": -1},
    "sans_detector": lambda x: x.fold(
        dim="detector_number",
        sizes={"strip": 32, "other": -1},
    ),
}


def dream_detector_bank_sizes() -> powder.types.DetectorBankSizes | None:
    return powder.types.DetectorBankSizes(DETECTOR_BANK_SIZES)


providers = (*powder.nexus.providers, dream_detector_bank_sizes)
"""
Providers for loading and processing NeXus data.
"""
