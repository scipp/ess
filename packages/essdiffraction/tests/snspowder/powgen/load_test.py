# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
import pytest

from ess.snspowder.powgen import data, load_calibration


@pytest.mark.skip(
    reason='mantid.LoadDiffCal causes SEGFAULT on CI but seems to work fine elsewhere'
)
def test_load_calibration_loads_required_data():
    loaded = load_calibration(
        data.powgen_tutorial_mantid_calibration_file(),
        instrument_filename='POWGEN_Definition_2011-02-25.xml',
    )

    assert 'difa' in loaded
    assert 'difc' in loaded
    assert 'tzero' in loaded
    assert 'mask' in loaded
    assert 'detector' in loaded.coords
    assert loaded.dims == ['detector']


@pytest.mark.skip(
    reason='mantid.LoadDiffCal causes SEGFAULT on CI but seems to work fine elsewhere'
)
def test_load_calibration_requires_instrument_definition():
    with pytest.raises(ValueError, match='calibration'):
        load_calibration(data.powgen_tutorial_mantid_calibration_file())


@pytest.mark.skip(
    reason='mantid.LoadDiffCal causes SEGFAULT on CI but seems to work fine elsewhere'
)
def test_load_calibration_can_only_have_1_instrument_definition():
    with pytest.raises(ValueError, match='instrument_name'):
        load_calibration(
            data.powgen_tutorial_mantid_calibration_file(),
            instrument_name='POWGEN',
            instrument_filename='POWGEN_Definition_2011-02-25.xml',
        )
