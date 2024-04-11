# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
import gemmi
import pytest

from ess.nmx.data import get_small_mtz_samples
from ess.nmx.mtz_io import mtz_to_pandas, read_mtz_file


@pytest.fixture(params=get_small_mtz_samples())
def file_path(request) -> str:
    return request.param


def test_gemmi_mtz(file_path: str) -> None:
    mtz = read_mtz_file(file_path)
    assert mtz.spacegroup == gemmi.SpaceGroup('C 1 2 1')  # Hard-coded value
    assert len(mtz.columns[0]) == 100  # Number of samples, hard-coded value


@pytest.fixture
def gemmi_mtz_object(file_path: str) -> gemmi.Mtz:
    return read_mtz_file(file_path)


def test_mtz_to_pandas_dataframe(gemmi_mtz_object: gemmi.Mtz) -> None:
    df = mtz_to_pandas(gemmi_mtz_object)
    assert set(df.columns) == set(gemmi_mtz_object.column_labels())
    # Check if the test data are not all-same
    first_column_name, second_column_name = df.columns[0:2]
    assert not all(df[first_column_name] == df[second_column_name])

    # Check if the data are the same
    for column in gemmi_mtz_object.columns:
        assert column.label in df.columns
        assert all(df[column.label] == column.array)
