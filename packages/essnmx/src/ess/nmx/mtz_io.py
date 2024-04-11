# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
from typing import NewType

import gemmi
import numpy as np
import pandas as pd

MTZFilepath = NewType("MTZFilepath", str)
"""Path to the mtz file"""

RawMtz = NewType("RawMtz", gemmi.Mtz)
"""The mtz file as a gemmi object"""

MtzDataFrame = NewType("MtzDataFrame", pd.DataFrame)
"""The mtz file as a pandas DataFrame"""


def read_mtz_file(file_path: MTZFilepath) -> RawMtz:
    '''read mtz file'''

    return RawMtz(gemmi.read_mtz_file(file_path))


def mtz_to_pandas(mtz: RawMtz) -> MtzDataFrame:
    return MtzDataFrame(
        pd.DataFrame(  # Recommended in the gemmi documentation.
            data=np.array(mtz, copy=False), columns=mtz.column_labels()
        )
    )
