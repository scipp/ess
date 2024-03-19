# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
import pathlib
from typing import Generator

import pytest
import scipp as sc
import scippnexus as snx
from scipp.testing import assert_allclose, assert_identical

from ess.nmx.data import small_mcstas_2_sample, small_mcstas_3_sample
from ess.nmx.mcstas_loader import (
    DefaultMaximumProbability,
    InputFilepath,
    event_weights_from_probability,
    load_mcstas_nexus,
    proton_charge_from_event_data,
)
from ess.nmx.reduction import NMXData


def check_scalar_properties_mcstas_2(dg: NMXData):
    """Test helper for NMXData loaded from McStas 2.

    Expected numbers are hard-coded based on the sample file.
    """

    assert_identical(
        dg.proton_charge,
        sc.scalar(1e-4 * dg['weights'].bins.size().sum().data.values, unit=None),
    )
    assert_identical(dg.crystal_rotation, sc.vector([20, 0, 90], unit='deg'))
    assert_identical(dg.sample_position, sc.vector(value=[0, 0, 0], unit='m'))
    assert_identical(
        dg.source_position, sc.vector(value=[-0.53123, 0.0, -157.405], unit='m')
    )
    assert dg.sample_name == sc.scalar("sampleMantid")


def check_nmxdata_properties(dg: NMXData, fast_axis, slow_axis, npanels=None) -> None:
    assert isinstance(dg, sc.DataGroup)
    assert (
        dg.shape
        == (
            npanels,
            1280 * 1280,
        )
        if npanels
        else (1280 * 1280,)
    )
    # Check maximum value of weights.
    assert_identical(
        dg.weights.max().data,
        sc.scalar(DefaultMaximumProbability, unit='counts', dtype=float),
    )
    assert_allclose(dg.fast_axis, fast_axis, atol=sc.scalar(0.005))
    assert_identical(dg.slow_axis, slow_axis)


def test_file_reader_mcstas2(mcstas_2_deprecation_warning_context) -> None:
    with mcstas_2_deprecation_warning_context():
        file_path = InputFilepath(small_mcstas_2_sample())

    fast_axis = sc.vectors(
        dims=['panel'], values=((1.0, 0.0, -0.01), (-0.01, 0.0, -1.0), (0.01, 0.0, 1.0))
    )
    slow_axis = sc.vectors(
        dims=['panel'], values=((0.0, 1.0, 0.0), (0.0, 1.0, 0.0), (0.0, 1.0, 0.0))
    )

    entry_path = "entry1/data/bank01_events_dat_list_p_x_y_n_id_t"
    with snx.File(file_path) as file:
        raw_data = file[entry_path]["events"][()]
        data_length = raw_data.sizes['dim_0']

    dg = load_mcstas_nexus(
        file_path=file_path,
        event_weights_converter=event_weights_from_probability,
        proton_charge_converter=proton_charge_from_event_data,
        detector_bank_name='bank01',
    )
    check_scalar_properties_mcstas_2(dg)
    assert dg.weights.bins.size().sum().value == data_length
    check_nmxdata_properties(dg, fast_axis, slow_axis, npanels=3)


def check_scalar_properties_mcstas_3(dg: NMXData):
    """Test helper for NMXData loaded from McStas 3.

    Expected numbers are hard-coded based on the sample file.
    """

    assert_identical(
        dg.proton_charge,
        sc.scalar(1e-4 * dg['weights'].bins.size().sum().data.values, unit=None),
    )
    assert_identical(dg.crystal_rotation, sc.vector([0, 0, 0], unit='deg'))
    assert_identical(dg.sample_position, sc.vector(value=[0, 0, 0], unit='m'))
    assert_identical(
        dg.source_position, sc.vector(value=[-0.53123, 0.0, -157.405], unit='m')
    )
    assert dg.sample_name == sc.scalar("sampleMantid")


@pytest.mark.parametrize(
    'bank_id, fast_axis, slow_axis',
    (
        # Expected values are provided by the IDS
        # based on the simulation settings of the sample file.
        ('01', (1.0, 0.0, -0.01), (0.0, 1.0, 0.0)),
        ('02', (-0.01, 0.0, -1.0), (0.0, 1.0, 0.0)),
        ('03', (0.01, 0.0, 1.0), (0.0, 1.0, 0.0)),
    ),
)
def test_file_reader_mcstas3(bank_id, fast_axis, slow_axis) -> None:
    file_path = InputFilepath(small_mcstas_3_sample())
    entry_paths = [f"entry1/data/bank{bank_id}_events_dat_list_p_x_y_n_id_t"]
    with snx.File(file_path) as file:
        raw_datas = [file[entry_path]["events"][()] for entry_path in entry_paths]
        raw_data = sc.concat(raw_datas, dim='dim_0')
        data_length = raw_data.sizes['dim_0']

    dg = load_mcstas_nexus(
        file_path=file_path,
        event_weights_converter=event_weights_from_probability,
        proton_charge_converter=proton_charge_from_event_data,
        detector_bank_name=f'bank{bank_id}',
    )
    check_scalar_properties_mcstas_3(dg)
    assert dg.weights.bins.size().sum().value == data_length
    check_nmxdata_properties(dg, sc.vector(fast_axis), sc.vector(slow_axis))


@pytest.fixture(params=[small_mcstas_2_sample, small_mcstas_3_sample])
def tmp_mcstas_file(
    tmp_path: pathlib.Path,
    request: pytest.FixtureRequest,
    mcstas_2_deprecation_warning_context,
) -> Generator[pathlib.Path, None, None]:
    import os
    import shutil

    if request.param == small_mcstas_2_sample:
        with mcstas_2_deprecation_warning_context():
            original_file_path = request.param()
    else:
        original_file_path = request.param()

    tmp_file = tmp_path / pathlib.Path('file.h5')
    shutil.copy(original_file_path, tmp_file)
    yield tmp_file
    os.remove(tmp_file)


def test_file_reader_mcstas_additional_fields(tmp_mcstas_file: pathlib.Path) -> None:
    """Check if additional fields names do not break the loader."""
    import h5py

    entry_path = "entry1/data/bank01_events_dat_list_p_x_y_n_id_t"
    new_entry_path = entry_path + '_L'

    with h5py.File(tmp_mcstas_file, 'r+') as file:
        dataset = file[entry_path]
        del file[entry_path]
        file[new_entry_path] = dataset

    dg = load_mcstas_nexus(
        file_path=InputFilepath(str(tmp_mcstas_file)),
        event_weights_converter=event_weights_from_probability,
        proton_charge_converter=proton_charge_from_event_data,
        detector_bank_name='bank01',
    )

    assert isinstance(dg, sc.DataGroup)
