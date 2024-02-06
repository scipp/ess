# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
import pathlib
from typing import Generator

import pytest
import scipp as sc

from ess.nmx.data import small_mcstas_sample


def test_file_reader_mcstas() -> None:
    import numpy as np
    import scippnexus as snx

    from ess.nmx.mcstas_loader import (
        DefaultMaximumProbability,
        InputFilepath,
        load_mcstas_nexus,
    )

    file_path = InputFilepath(small_mcstas_sample())
    dg = load_mcstas_nexus(file_path)

    entry_path = "entry1/data/bank01_events_dat_list_p_x_y_n_id_t"
    with snx.File(file_path) as file:
        raw_data = file[entry_path]["events"][()]
        data_length = raw_data.sizes['dim_0']

    expected_weight_max = sc.scalar(
        DefaultMaximumProbability, unit='counts', dtype=float
    )

    assert isinstance(dg, sc.DataGroup)
    assert dg.shape == (3, 1280 * 1280)
    assert sc.identical(dg.sample_position, sc.vector(value=[0, 0, 0], unit='m'))
    assert dg.weights.bins.size().sum().value == data_length
    assert sc.identical(dg.weights.max().data, expected_weight_max)
    # Expected coordinate values are provided by the IDS
    # based on the simulation settings of the sample file.
    # The expected values are rounded to 2 decimal places.
    assert np.all(
        np.round(dg.fast_axis.values, 2)
        == sc.vectors(
            dims=['panel'],
            values=[(1.0, 0.0, -0.01), (-0.01, 0.0, -1.0), (0.01, 0.0, 1.0)],
        ).values,
    )
    assert sc.identical(
        dg.slow_axis,
        sc.vectors(
            dims=['panel'], values=[[0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]]
        ),
    )


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

    dg = load_mcstas_nexus(InputFilepath(str(tmp_mcstas_file)))

    assert isinstance(dg, sc.DataGroup)
    assert dg.shape == (3, 1280 * 1280)
