# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
import pytest
import sciline as sl
import scipp as sc


@pytest.fixture
def mcstas_workflow() -> sl.Pipeline:
    from ess.nmx.data import small_mcstas_sample
    from ess.nmx.loader import (
        DefaultMaximumProbability,
        InputFileName,
        MaximumProbability,
    )
    from ess.nmx.workflow import collect_default_parameters, providers

    return sl.Pipeline(
        list(providers),
        params={
            **collect_default_parameters(),
            InputFileName: small_mcstas_sample(),
            MaximumProbability: DefaultMaximumProbability,
        },
    )


def test_pipeline_builder(mcstas_workflow: sl.Pipeline) -> None:
    from ess.nmx.data import small_mcstas_sample
    from ess.nmx.loader import InputFileName

    assert mcstas_workflow.get(InputFileName).compute() == small_mcstas_sample()


def test_pipeline_mcstas_loader(mcstas_workflow: sl.Pipeline) -> None:
    """Test if the loader graph is complete."""
    from ess.nmx.loader import NMXData

    nmx_data = mcstas_workflow.compute(NMXData)
    assert isinstance(nmx_data.events, sc.DataArray)


def test_pipeline_mcstas_grouping(mcstas_workflow: sl.Pipeline) -> None:
    """Test if the data reduction graph is complete."""
    from ess.nmx.reduction import GroupedByPixelID

    assert isinstance(mcstas_workflow.compute(GroupedByPixelID), sc.DataArray)
