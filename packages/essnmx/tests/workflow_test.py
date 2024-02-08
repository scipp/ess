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
        McStasEventWeightsConverter,
        McStasProtonChargeConverter,
        event_weights_from_probability,
        load_mcstas_nexus,
        proton_charge_from_event_data,
    )
    from ess.nmx.reduction import TimeBinSteps, bin_time_of_arrival

    return sl.Pipeline(
        [load_mcstas_nexus, bin_time_of_arrival],
        params={
            InputFilepath: small_mcstas_sample(),
            MaximumProbability: DefaultMaximumProbability,
            TimeBinSteps: TimeBinSteps(50),
            McStasEventWeightsConverter: event_weights_from_probability,
            McStasProtonChargeConverter: proton_charge_from_event_data,
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
    assert isinstance(nmx_data, sc.DataGroup)
    assert nmx_data.sizes['panel'] == 3
    assert nmx_data.sizes['id'] == 1280 * 1280


def test_pipeline_mcstas_reduction(mcstas_workflow: sl.Pipeline) -> None:
    """Test if the loader graph is complete."""
    from ess.nmx.reduction import NMXReducedData

    nmx_reduced_data = mcstas_workflow.compute(NMXReducedData)
    assert isinstance(nmx_reduced_data, sc.DataGroup)
    assert nmx_reduced_data.sizes['panel'] == 3
    assert nmx_reduced_data.sizes['t'] == 50
