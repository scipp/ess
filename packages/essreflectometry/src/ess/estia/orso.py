# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from ess.reflectometry.orso import OrsoCorrectionList


def orso_estia_corrections() -> OrsoCorrectionList:
    return OrsoCorrectionList(
        [
            "chopper ToF correction",
            "footprint correction",
            "supermirror calibration",
        ]
    )


providers = (orso_estia_corrections,)
