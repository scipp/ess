# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import pathlib

import pytest
import scipp as sc

from ess.nmx.data import get_small_mcstas
from ess.nmx.mcstas.load import load_raw_event_data, raw_event_data_chunk_generator


@pytest.fixture(params=[get_small_mcstas])
def mcstas_file_path(request: pytest.FixtureRequest) -> pathlib.Path:
    return request.param()


def test_generator_loading_at_once(mcstas_file_path) -> None:
    from ess.nmx.mcstas.load import detector_name_from_index

    detector_name = detector_name_from_index(0)
    whole_chunk = next(
        raw_event_data_chunk_generator(
            mcstas_file_path, detector_name=detector_name, chunk_size=-1
        )
    )
    loaded_data = load_raw_event_data(
        mcstas_file_path, detector_name=detector_name, bank_prefix=None
    )
    assert sc.identical(whole_chunk, loaded_data)


def test_generator_loading_warns_if_too_small(mcstas_file_path) -> None:
    from ess.nmx.mcstas.load import detector_name_from_index

    detector_name = detector_name_from_index(0)
    with pytest.warns(UserWarning, match="The chunk size may be too small"):
        next(
            raw_event_data_chunk_generator(
                mcstas_file_path, detector_name=detector_name, chunk_size=1
            )
        )
