import scipp as sc

from ess.diffraction.peaks import dspacing_peak_positions_from_cif


def test_retreived_peak_list_has_expected_units():
    d = dspacing_peak_positions_from_cif(
        'codid::9011998',
        sc.scalar(50, unit='barn'),
    )
    assert d.unit == 'angstrom'


def test_intensity_threshold_is_taken_into_account():
    d = dspacing_peak_positions_from_cif(
        'codid::9011998',
        sc.scalar(50, unit='barn'),
    )
    assert len(d) > 0

    d = dspacing_peak_positions_from_cif(
        'codid::9011998',
        sc.scalar(50, unit='Mbarn'),
    )
    assert len(d) == 0
