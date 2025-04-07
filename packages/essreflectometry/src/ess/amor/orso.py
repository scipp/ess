# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
from ess.reflectometry.orso import OrsoCorrectionList


def orso_amor_corrections() -> OrsoCorrectionList:
    return OrsoCorrectionList(
        [
            "chopper ToF correction",
            "footprint correction",
            "supermirror calibration",
        ]
    )


providers = (orso_amor_corrections,)
