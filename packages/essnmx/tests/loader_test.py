# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
import pathlib
from typing import Generator

import pytest
import scipp as sc

from ess.nmx.data import small_mcstas_sample


def test_file_reader_mcstas() -> None:
    import scippnexus as snx

    from ess.nmx.mcstas_loader import (
        DefaultMaximumProbability,
        InputFilepath,
        load_mcstas_nexus,
    )

    file_path = InputFilepath(small_mcstas_sample())
    da = load_mcstas_nexus(file_path)

    entry_path = "entry1/data/bank01_events_dat_list_p_x_y_n_id_t"
    with snx.File(file_path) as file:
        raw_data = file[entry_path]["events"][()]
        data_length = raw_data.sizes['dim_0']

    expected_weight_max = sc.scalar(DefaultMaximumProbability, unit='1', dtype=float)

    assert isinstance(da, sc.DataArray)
    assert da.shape == (3, 1280 * 1280)
    assert da.bins.size().sum().value == data_length
    assert sc.identical(da.data.max(), expected_weight_max)


@pytest.fixture
def tmp_mcstas_file(tmp_path: pathlib.Path) -> Generator[pathlib.Path, None, None]:
    import os
    import shutil

    tmp_file = tmp_path / pathlib.Path('file.h5')
    shutil.copy(small_mcstas_sample(), tmp_file)
    yield tmp_file
    os.remove(tmp_file)


def test_file_reader_mcstas_additional_fields(tmp_mcstas_file: pathlib.Path) -> None:
    """Check if additional fields names do not break the loader."""
    import h5py

    from ess.nmx.mcstas_loader import InputFilepath, load_mcstas_nexus

    entry_path = "entry1/data/bank01_events_dat_list_p_x_y_n_id_t"
    new_entry_path = entry_path + '_L'

    with h5py.File(tmp_mcstas_file, 'r+') as file:
        dataset = file[entry_path]
        del file[entry_path]
        file[new_entry_path] = dataset

    da = load_mcstas_nexus(InputFilepath(str(tmp_mcstas_file)))

    assert isinstance(da, sc.DataArray)
    assert da.shape == (3, 1280 * 1280)
