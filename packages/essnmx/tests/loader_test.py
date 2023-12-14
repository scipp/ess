# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
import scipp as sc


def test_file_reader_mcstas() -> None:
    import scippnexus as snx

    from ess.nmx.data import small_mcstas_sample
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
