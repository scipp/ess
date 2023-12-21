# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
import pytest
import sciline as sl
import scipp as sc


@pytest.fixture
def mcstas_workflow() -> sl.Pipeline:
    from ess.nmx.data import small_mcstas_sample
    from ess.nmx.mcstas_loader import (
        DefaultMaximumProbability,
        InputFilepath,
        MaximumProbability,
        load_mcstas_nexus,
    )

    return sl.Pipeline(
        [load_mcstas_nexus],
        params={
            InputFilepath: small_mcstas_sample(),
            MaximumProbability: DefaultMaximumProbability,
        },
    )


def test_pipeline_builder(mcstas_workflow: sl.Pipeline) -> None:
    from ess.nmx.data import small_mcstas_sample
    from ess.nmx.mcstas_loader import InputFilepath

    assert mcstas_workflow.get(InputFilepath).compute() == small_mcstas_sample()


def test_pipeline_mcstas_loader(mcstas_workflow: sl.Pipeline) -> None:
    """Test if the loader graph is complete."""
    from ess.nmx.mcstas_loader import NMXData

    nmx_data = mcstas_workflow.compute(NMXData)
    assert isinstance(nmx_data, sc.DataArray)
