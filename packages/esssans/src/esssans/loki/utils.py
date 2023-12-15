# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

from typing import Any, Dict, List
from sciline import ParamTable

from ..types import (
    Filename,
    RunID,
)


def make_parameter_tables(runs: Dict[Any, List[str]]) -> List[ParamTable]:
    tables = []

    counter = 0
    for key, fnames in runs.items():
        nruns = len(fnames)
        indices = list(range(counter, counter + nruns))
        table = ParamTable(RunID[key], {Filename[key]: fnames}, index=indices)
        tables.append(table)
        counter += nruns

    return tables
