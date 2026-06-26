# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)

import pytest
import scipp as sc
from pydantic import ValidationError

from ess.reduce.parameter_models import DspacingEdges, Scale


def test_dspacing_edges_create_scipp_bin_edges() -> None:
    model = DspacingEdges(start=0.1, stop=1.1, num_bins=2)

    assert sc.identical(
        model.get_edges(),
        sc.linspace(dim='dspacing', start=0.1, stop=1.1, num=3, unit='Å'),
    )


def test_log_edges_require_positive_start() -> None:
    with pytest.raises(ValidationError, match="start must be positive"):
        DspacingEdges(start=0.0, stop=1.0, scale=Scale.LOG)
