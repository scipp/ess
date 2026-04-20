import NCrystal as NC
import scipp as sc


def dspacing_peaks_from_cif(cif, intensity_threshold=None, **kwargs) -> sc.DataArray:
    """
    Retrieves a data array representing the bragg peaks of the given material.

    The material is represented by a cif file or a codid or similar.
    The number of peaks retrieved can be controlled by setting the intensity
    threshold argument to only retrieve prominent peaks.

    Parameters
    ------------
    cif:
        Generalised CIF source, either in the form of text data, a file path, a
        database ID (in the form of a string like "codid::9008460" or "mpid::87)",
        for either the Crystallography Open Database
        (https://www.crystallography.net/cod/) or the Materials Project
        (https://www.materialsproject.org/).

        For more details, see :py:`ncrystal.cifutils.CIFSource`.

    intensity_threshold:
        The minimum intensity of the peaks to return.
        Intensity is here defined as the squared structure factor
        times the multiplicity of the peak.
        The ``intensity_threshold`` must be convertible to unit ``barn``.

    kwargs:
        Can be anything that :py:`NCrystal.NCMATComposer.from_cif` supports.
        For example: ``uiso_temperature`` or ``override_spacegroup``.

    Returns
    ------------
        Data array representing peak amplitudes and peak positions in ``dspacing``.
        The full NCrystal information object is added as a coordinate with the name ``info``.
        The input arguments are added as scalar coordinates.

    Examples
    --------
    >>> from ess.diffraction.peaks import dspacing_peaks_from_cif
    >>> dspacing_peaks_from_cif(
    ...     'codid::9008460',
    ...     uiso_temperature=400,
    ...     override_spacegroup='F d -3 m:1',
    ... )
    <scipp.DataArray>
    Dimensions: Sizes[peaks:162, ]
    Coordinates:
    * cif                        string        <no unit>  ()  "codid::9008460"
    * dspacing                  float64             [Å]  (peaks)  [2.33803, 2.02479, ..., 0.243756, 0.243756]
    * info                     PyObject        <no unit>  ()  <NCrystal.core.Info object at ...>
    * override_spacegroup        string        <no unit>  ()  "F d -3 m:1"
    * uiso_temperature            int64  [dimensionless]  ()  400
    Data:
                                float64           [barn]  (peaks)  [13.3631, 9.59562, ..., 0.000556426, 0.000556426]

    """  # noqa: E501
    info = NC.NCMATComposer.from_cif(cif, **kwargs).load('comp=bragg').info
    min_intensity = (
        intensity_threshold.to(unit='barn').value
        if intensity_threshold is not None
        else 0
    )
    dims = ['peaks']
    peaks = [hkl for hkl in info.hklObjects() if (hkl.f2 * hkl.mult) >= min_intensity]
    dspacing = sc.array(
        dims=dims,
        values=[hkl.d for hkl in peaks],
        unit='angstrom',
    )
    out = sc.DataArray(
        sc.array(dims=dims, values=[hkl.f2 * hkl.mult for hkl in peaks], unit='barn'),
        coords={
            'dspacing': dspacing,
            'cif': sc.scalar(cif),
            'info': sc.scalar(info),
            **{name: sc.scalar(value) for name, value in kwargs.items()},
        },
    )
    if intensity_threshold is not None:
        out.coords['intensity_threshold'] = intensity_threshold
    return out
