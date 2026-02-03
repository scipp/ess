import scipp as sc

from ess.diffraction.peaks import dspacing_peaks_from_cif


def test_retreived_peak_list_has_expected_units():
    d = dspacing_peaks_from_cif(
        'codid::9011998',
        sc.scalar(50, unit='barn'),
    )
    assert d.coords['dspacing'].unit == 'angstrom'
    assert d.unit == 'barn'


def test_intensity_threshold_is_taken_into_account():
    d = dspacing_peaks_from_cif(
        'codid::9011998',
        sc.scalar(50, unit='barn'),
    )
    assert len(d) > 0

    d = dspacing_peaks_from_cif(
        'codid::9011998',
        sc.scalar(50, unit='Mbarn'),
    )
    assert len(d) == 0


def test_retreived_peak_list_with_temperature_kwarg():
    d = dspacing_peaks_from_cif(
        'codid::9011998', sc.scalar(50, unit='barn'), uiso_temperature=300
    )
    assert d.coords['uiso_temperature'] == 300
