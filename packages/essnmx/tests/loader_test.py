# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
import scipp as sc


def test_file_reader_mcstas() -> None:
    import scippnexus as snx

    from ess.nmx.data import small_mcstas_sample
    from ess.nmx.loader import DefaultMaximumProbability, InputFileName, load_nmx_file

    file_path = InputFileName(small_mcstas_sample())
    entry_path = "entry1/data/bank01_events_dat_list_p_x_y_n_id_t"
    da = load_nmx_file(file_path).events

    with snx.File(file_path) as file:
        raw_data: sc.Variable = file[entry_path]["events"][()]

    weights = raw_data['dim_1', 0].copy()
    weights.unit = '1'
    expected_data = (DefaultMaximumProbability / weights.max()) * weights

    assert isinstance(da, sc.DataArray)
    assert list(da.values) == list(expected_data.values)
    assert list(da.coords['id'].values) == list(raw_data['dim_1', 4].values)
    assert list(da.coords['t'].values) == list(raw_data['dim_1', 5].values)
    assert da.unit == '1'
    assert da.coords['id'].unit == 'dimensionless'
    assert da.coords['t'].unit == 's'
