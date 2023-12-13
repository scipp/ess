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
        InputFilename,
        MaximumProbability,
        load_mcstas_nmx_file,
    )
    from ess.nmx.workflow import collect_default_parameters

    return sl.Pipeline(
        [load_mcstas_nmx_file],
        params={
            **collect_default_parameters(),
            InputFilename: small_mcstas_sample(),
            MaximumProbability: DefaultMaximumProbability,
        },
    )


def test_pipeline_builder(mcstas_workflow: sl.Pipeline) -> None:
    from ess.nmx.data import small_mcstas_sample
    from ess.nmx.mcstas_loader import InputFilename

    assert mcstas_workflow.get(InputFilename).compute() == small_mcstas_sample()


def test_pipeline_mcstas_loader(mcstas_workflow: sl.Pipeline) -> None:
    """Test if the loader graph is complete."""
    from ess.nmx.mcstas_loader import NMXData

    nmx_data = mcstas_workflow.compute(NMXData)
    assert isinstance(nmx_data, sc.DataArray)
