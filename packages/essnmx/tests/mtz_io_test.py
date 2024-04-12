# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
import gemmi
import pytest
import sciline as sl

from ess.nmx.data import get_small_mtz_samples
from ess.nmx.mtz_io import DEFAULT_SPACE_GROUP_DESC  # P 21 21 21
from ess.nmx.mtz_io import (
    MergedMtzDataFrame,
    MTZFileIndex,
    MTZFilepath,
    RawMtz,
    get_reciprocal_asu,
    get_space_group,
    merge_mtz_dataframes,
    mtz_to_pandas,
    read_mtz_file,
    reduce_merged_mtz_dataframe,
    reduce_single_mtz,
)


@pytest.fixture(params=get_small_mtz_samples())
def file_path(request) -> str:
    return request.param


def test_gemmi_mtz(file_path: str) -> None:
    mtz = read_mtz_file(MTZFilepath(file_path))
    assert mtz.spacegroup == gemmi.SpaceGroup("C 1 2 1")  # Hard-coded value
    assert len(mtz.columns[0]) == 100  # Number of samples, hard-coded value


@pytest.fixture
def gemmi_mtz_object(file_path: str) -> gemmi.Mtz:
    return read_mtz_file(MTZFilepath(file_path))


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


def test_mtz_to_reduced_pandas_dataframe(gemmi_mtz_object: gemmi.Mtz) -> None:
    df = reduce_single_mtz(RawMtz(gemmi_mtz_object))
    for expected_colum in ["hkl", "d", "resolution", "I_div_SIGI", *"HKL"]:
        assert expected_colum in df.columns

    for hkl_column in "HKL":
        assert hkl_column in df.columns
        assert df[hkl_column].dtype == int

    assert "hkl_eq" not in df.columns  # It should be done on merged dataframes


@pytest.fixture
def mtz_series() -> sl.Series[MTZFileIndex, RawMtz]:
    return sl.Series(
        row_dim=MTZFileIndex,
        items={
            MTZFileIndex(i_file): read_mtz_file(MTZFilepath(file_path))
            for i_file, file_path in enumerate(get_small_mtz_samples())
        },
    )


def test_get_space_group(mtz_series: sl.Series[MTZFileIndex, RawMtz]) -> None:
    assert (
        get_space_group(mtz_series).short_name() == "C2"
    )  # Expected value in test files


def test_get_space_group_with_spacegroup_desc(
    mtz_series: sl.Series[MTZFileIndex, RawMtz],
) -> None:
    assert (
        get_space_group(mtz_series, DEFAULT_SPACE_GROUP_DESC).short_name() == "P212121"
    )


@pytest.fixture
def conflicting_mtz_series(
    mtz_series: sl.Series[MTZFileIndex, RawMtz],
) -> sl.Series[MTZFileIndex, RawMtz]:
    mtz_series[MTZFileIndex(0)].spacegroup = gemmi.SpaceGroup(DEFAULT_SPACE_GROUP_DESC)
    # Make sure the space groups are different
    assert (
        mtz_series[MTZFileIndex(0)].spacegroup.short_name()
        != mtz_series[MTZFileIndex(1)].spacegroup.short_name()
    )

    return mtz_series


def test_get_space_group_conflict_raises(
    conflicting_mtz_series: sl.Series[MTZFileIndex, RawMtz],
) -> None:
    reg = r"Multiple space groups found:.+P 21 21 21.+C 1 2 1"
    with pytest.raises(ValueError, match=reg):
        get_space_group(conflicting_mtz_series)


def test_get_space_conflict_but_desc_provided(
    conflicting_mtz_series: sl.Series[MTZFileIndex, RawMtz],
) -> None:
    assert (
        get_space_group(conflicting_mtz_series, DEFAULT_SPACE_GROUP_DESC).short_name()
        == "P212121"
    )


@pytest.fixture
def merged_mtz_dataframe(
    mtz_series: sl.Series[MTZFileIndex, RawMtz],
) -> MergedMtzDataFrame:
    """Tests if the merged data frame has the expected columns."""
    reduced_mtz_series = sl.Series(
        row_dim=MTZFileIndex,
        items={i_file: reduce_single_mtz(mtz) for i_file, mtz in mtz_series.items()},
    )
    return merge_mtz_dataframes(reduced_mtz_series)


def test_reduce_merged_mtz_dataframe(
    mtz_series: sl.Series[MTZFileIndex, RawMtz],
    merged_mtz_dataframe: MergedMtzDataFrame,
) -> None:
    space_gr = get_space_group(mtz_series)
    rapio_asu = get_reciprocal_asu(space_gr)

    nmx_df = reduce_merged_mtz_dataframe(
        merged_mtz_df=merged_mtz_dataframe,
        rapio_asu=rapio_asu,
        sg=space_gr,
    )
    assert "hkl_eq" not in merged_mtz_dataframe.columns
    assert "hkl_eq" in nmx_df.columns
