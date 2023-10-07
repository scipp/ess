# standard library imports
import itertools
import os
from typing import Optional, Type, Union

# related third party imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.gridspec as gridspec
from IPython.display import display, HTML
from scipy.optimize import leastsq  # needed for fitting of turbidity

# local application imports
import scippneutron as scn
import scippnexus as snx
import scipp as sc
from ess.loki.nurf import utils
from scipp.ndimage import median_filter


def normalize_uv(
    *, sample: sc.DataArray , reference: sc.DataArray , dark: sc.DataArray 
) -> sc.DataArray : 
    """Calculates the absorbance of the UV signal.

    Parameters
    ----------
    sample: sc.DataArray 
        DataArray containing sample UV signal, one spectrum or multiple spectra.
    reference: sc.DataArray 
        DataArray containing reference UV signal, one spectrum expected.
    dark: sc.DataArray 
        DataArray containing dark UV signal, one spectrum expected.

    Returns
    ----------
    normalized: sc.DataArray 
        DataArray that contains the normalized UV signal, one spectrum or mulitple spectra.

    """

    normalized = sc.log10(
        (reference - dark) / (sample - dark)
    )  # results in DataArrays with multiple spectra

    return normalized

def load_and_normalize_uv(name: str) -> sc.DataArray :
    """Loads the UV data from the corresponding entry in the LoKI.nxs filename and 
    calculates the absorbance of each UV spectrum.
    For an averaged spectrum based on all UV spectra in the file, use average_uv.

    Parameters
    ----------
    name: str
        Filename, e.g. 066017.nxs

    Returns
    ----------
    normalized: sc.DataArray 
        DataArray that contains the normalized UV signal, one spectrum or mulitple spectra.

    """
    uv_dict = utils.load_nurfloki_file(name, 'uv')
    normalized = normalize_uv(**uv_dict)  # results in DataArrays with multiple spectra
    # provide source of each spectrum in the file
    normalized.attrs['source'] = sc.scalar(name).broadcast(['spectrum'], [normalized.sizes['spectrum']])

    return normalized

def average_uv(name: str) -> sc.DataArray :
    """Processses all UV spectra in a single LoKI.nxs and averages them to one corrected
       UV spectrum.

    Parameters
    ----------
    name: str
        Filename for a LoKI.nxs file containting UV entry.

     Returns
    ----------
    normalized: 
        One averaged UV spectrum. Averaged over all UV spectra contained in the file
        under UV entry data. Preserves original source.

    """
    #uv_dict = utils.load_nurfloki_file(name, 'uv')
    #normalized = normalize_uv(**uv_dict) 
    normalized = load_and_normalize_uv(name)

    # returns averaged uv spectrum
    return normalized.groupby('source').mean('spectrum').squeeze()


def gather_uv_avg_set(filelist:list) -> sc.Dataset:
    """Creates a sc.DataSet for set of given filenames for an experiment composed of 
    multiple, separated UV measurements over time. Multiple UV spectra in each file will be averaged.
    Function will fail, if not all files contain the same number of UV spectra.

    Parameters
    ----------
    filelist: list of str
        List of filenames containing UV data

     Returns
    ----------
    uv_spectra_set: sc.Dataset
        DataSet of multiple UV DataArrays, where the UV signal for each experiment was
        averaged

    """

    uv_spectra_set = sc.Dataset({name: average_uv(name) for name in filelist})
    return uv_spectra_set



def gather_uv_set(filelist: list) -> Union[sc.Dataset, sc.DataArray]:
    """Gathers from multiple input LoKI.files the UV spectra. 
    If all LoKI.nxs files contain the same number of UV spectra, the function returns
     a sc.Dataset, if not a sc.DataArray is returned. The source attribute provides
     the reference of origin.
    """

    #check first if numbers of uv spectra in each file are the same
    num_uv_spectra=np.empty(len(filelist))
    for count, name in enumerate(filelist):
        uv_da=load_and_normalize_uv(name)
        num_uv_spectra[count]=uv_da.sizes['spectrum']

    # compare all entries with the first entry 
    if np.all(num_uv_spectra == num_uv_spectra[0]):
        # prepare as output an sc.Dataset
        res=sc.Dataset({name:load_and_normalize_uv(name) for name in filelist})

    else:
        data_arrays = []
        for name in filelist:
            da = load_and_normalize_uv(name)
            #da.attrs['source'] = sc.scalar(name).broadcast(['spectrum'], [da.sizes['spectrum']])
            data_arrays.append(da)
            res = sc.concat(data_arrays, dim='spectrum')

    return res


def uv_peak_int(uv_da: sc.DataArray , wavelength: Optional[sc.Variable] = None, tol=None) -> dict:
    """Extract uv peak intensity for a given wavelength and a given interval.
    If no wavelength is given, 280 nm is chosen as value.
    If no tolerance is given, a tolerance of 0.5 nm is chosen.

    UV peak intensity is calculated for a given wavelength in two different ways:
    1. Interpolation of intensity around given wavelength, selection of interpolated intensity value for requested wavelength ("one_wavelength")
    2. Selection of 2*tol wavelength interval around requested wavelength. Average over all intensity values in this interval. ("wl_interval")
    
    Parameters
    ----------
    uv_da: sc.DataArray 
        DataArray containing uv spectra
    wavelength: sc.Variable
        Wavelength with a unit
    tol: float
        Tolerance, 2*tol defines the interval around the given wavelength

    Returns
    ----------
    uv_peak_int: dict
        Dictionary that contains the peak intensity for the requested wavelength, the peak intensity averaged over the requested interval, the requested wavelength with its unit, and the tolerance

    """
    assert (
        "wavelength" in uv_da.dims
    ), "sc.DataArray  is missing the wavelength dimension"  # assert that 'wavelength' is a dimension in the uv_da sc.DataArray 

    # set default value for wavelength:
    if wavelength is None:
        wavelength = sc.scalar(280, unit='nm')
    else:
        if not isinstance(wavelength, sc.Variable):
            raise TypeError("Wavelength needs to be of type sc.Variable.")
        assert(wavelength.unit==uv_da.coords["wavelength"].unit)

    # set default value for tolerance:
    if tol is None:
        tol = sc.scalar(0.5, unit='nm')
    else:
        if not isinstance(tol,sc.Variable):
            raise TypeError("Tol needs to be of type sc.Variable.")

    # filter spectrum values for the specified interval, filtered along the wavelength 
    # dimension
    uv_da_filt = uv_da[
        "wavelength",
        (wavelength - tol) : (wavelength + tol) ,
    ]
    # average intensity value in interval
    uv_int_mean_interval = uv_da_filt.mean(dim="wavelength")

    # interpolation approach
    from scipp.interpolate import interp1d

    uv_interp = interp1d(uv_da, "wavelength")
    x = sc.linspace(
        dim="wavelength", start=wavelength, stop=wavelength, num=1, unit=wavelength.unit)
    uv_int_one_wl = uv_interp(x)

    # prepare a dict for output
    uv_peak_int = {
        "one_wavelength": uv_int_one_wl,   
        "wl_interval": uv_int_mean_interval,    
        "wavelength": wavelength,
        "tol": tol,
    }

    return uv_peak_int

def turbidity(wl: np.ndarray, b: np.ndarray, m: np.ndarray)-> np.ndarray:
    """Function describing turbidity tau. tau = b* lambda **(-m)
        Fitting parameters: b, m. b corresponds to the baseline found for higher wavelengths (flat line in UV spectrum), m corresponds to the slope.
        lambda: wavelength

    Parameters
    ----------
    b: np.ndarray
        Offset, baseline

    m: np.ndarray
        Slope

    wl: np.ndarray
        UV wavelengths

    Returns
    ----------
    y: np.ndarray
        Turbidity

    """

    y = b * wl ** (-m)
    return y


def residual(p: list, x:np.ndarray, y:np.ndarray)-> np.ndarray:
    """Calculates the residuals between fitted turbidity and measured UV data

    Parameters
    ----------
    p: list
        Fit parameters for turbidity

    x: np.ndarray
        x values, here: UV wavelength

    y: np.ndarray
        y values, here: UV intensity

    Returns
    ----------
    y - turbidity(x, *p): np.ndarray
        Difference between measured UV intensity values and fitted turbidity

    """

    return y - turbidity(x, *p)




def uv_turbidity_fit(
    uv_da: sc.DataArray ,
    fit_llim: Optional[sc.Variable] = None,
    fit_ulim: Optional[sc.Variable] = None, 
    b_llim: Optional[sc.Variable] = None,
    b_ulim: Optional[sc.Variable] = None,
    m=None
) -> sc.DataArray :
    """Fit turbidity to the experimental data. Turbidity: tau=b * wavelength^(-m) Parameters of interest: b, m.
        b is the baseline and m is the slope. b can be obtained by averaging over the flat range of the UV spectrum
        in the higher wavelength range.
        m: make an educated guess. Advice: Limit fitting range to wavelengths after spectroscopy peaks.

    Parameters
    ----------
    uv_da: sc.DataArray 
        UV sc.DataArray  containing one or more normalized UV spectra
    fit_llim: sc.Variable
        Lower wavelength limit of fit range for turbidity
    fit_ulim: sc.Variable
        Upper wavelength limit of fit range for turbidity
    b_llim: sc.Variable
        Lower wavelength limit of fit range for b
    b_ulim: sc.Variable
        Upper wavelength limit of fit range for b
    m: int
        Educated guess start value of slope parameter in turbidity, default: 0.01


    Returns:
    ----------
    uv_da_turbcorr: sc.DataArray 
        uv_da dataarray where each spectrum was corrected for a fitted turbidity, export for all wavelength values

    """
    # obtain unit of wavelength:
    #if wl_unit is None:
    #    wl_unit = uv_da.coords["wavelength"].unit
    #else:
    #    if not isinstance(wl_unit, sc.Unit):
    #        raise TypeError
    #    assert (
    #        wl_unit == uv_da.coords["wavelength"].unit
    #    )  # we check that the given unit corresponds to the unit for the wavelength
    if not isinstance(uv_da, sc.DataArray ):
        raise TypeError('uv_da must be of type sc.DataArray ')

    if fit_llim is None:
        fit_llim=sc.scalar(350, unit='nm')
    if fit_ulim is None:
        fit_ulim=sc.scalar(600, unit='nm')

    if b_llim is None:
        b_llim=sc.scalar(500, unit='nm')
    if b_ulim is None:
        b_ulim=sc.scalar(800, unit='nm')

    if fit_llim is not None and fit_ulim is not None:
        assert (fit_llim < fit_ulim).value, "fit_llim < fit_ulim"
    if b_llim is not None and b_ulim is not None:
        assert (b_llim < b_ulim).value, "b_llim<b_ulim"

    if not b_llim.unit==b_ulim.unit:
        raise ValueError("Use same unit for fit range.")
    if not fit_llim.unit==fit_ulim.unit:
        raise ValueError("Use same unit for fit range.")
    
    if m is None:
        m = 0.01

    # select the UV wavelength range for fitting the turbidity
    uv_da_filt = uv_da["wavelength", fit_llim  : fit_ulim ]

    # How many spectra are in the file?
    num_spec = uv_da_filt.sizes["spectrum"]

    # offset, choose wavelength range for b0 extraction and average over the selected wavelength range
    b0 = (
        uv_da["wavelength", b_llim : b_ulim ]
        .mean(dim="wavelength")
        .values
    )
    # create np.ndarray of same shape as b, but with values of m
    m0 = np.full(b0.shape, m)

    # create dummy array
    uv_da_turb_corr_dat = np.zeros(uv_da.data.shape)

    res_popt = np.empty([num_spec, 2])
    res_pcov = np.empty([num_spec, 1])

    # Perform the fitting for each spectrum:
    for i in range(num_spec):
        # start parameters
        p0 = [b0[i], m0[i]]
        popt, pcov = leastsq(
            residual,
            p0,
            args=(
                uv_da_filt.coords["wavelength"].values,
                uv_da_filt["spectrum", i].values,
            ),
        )

        # calculation of each spectrum corrected for fitted turbidity
        uv_da_turb_corr_dat[i, :] = uv_da["spectrum", i].values - turbidity(
            uv_da["spectrum", i].coords["wavelength"].values, popt[0], popt[1]
        )
        # don't lose fit parameters
        res_popt[i, :] = popt
        res_pcov[i, 0] = pcov

    # Prepare for the new data uv_da corrected for a fitted turbidity
    uv_da_turbcorr = uv_da.copy()
    # Replace data in the dataarray
    uv_da_turbcorr.data.values = uv_da_turb_corr_dat

    # Collect the results of the fitting and store them with the dataarray
    uv_da_turbcorr.attrs["fit-slope_m"] = sc.array(
        dims=["spectrum"], values=res_popt[:, 1]
    )
    uv_da_turbcorr.attrs["fit-offset_b"] = sc.array(
        dims=["spectrum"], values=res_popt[:, 0]
    )
    uv_da_turbcorr.attrs["fit-pcov"] = sc.array(
        dims=["spectrum"], values=res_pcov[:, 0]
    )

    # track range parameters
    uv_da_turbcorr.attrs["turbidity_fit_range"] = sc.array(dims=["spectrum"], values=[fit_llim.value, fit_ulim.value], unit=fit_llim.unit)
    uv_da_turbcorr.attrs["b_fit_range"] = sc.array(dims=["spectrum"], values=[b_llim.value, b_ulim.value], unit=b_llim.unit)

    return uv_da_turbcorr
 

def uv_multi_turbidity_fit(
    filelist: dict,
    fit_llim: Optional[sc.Variable] = None,
    fit_ulim: Optional[sc.Variable] = None,
    b_llim: Optional[sc.Variable] = None,
    b_ulim: Optional[sc.Variable] = None,
    m=None 
)-> sc.DataArray:
    """Applies turbidity correction to UV spectra for a set of  LoKI.nxs files.
    b: offset, m: slope. Same values are applied to all given spectra.
    filelist: dict 
        Dict of sc.DataArrays, key: filename, value: sc.DataArray

    """
   
    uv_collection = {}
    for name, uv_da in filelist.items():
     
        uv_da_turbcorr = uv_turbidity_fit(
            uv_da,
            fit_llim=fit_llim,
            fit_ulim=fit_ulim,
            b_llim=b_llim,
            b_ulim=b_ulim,
            m=m
        )

        # if utils.load_nurfloki_file and normalize_uv are used, source attribute does not exist
        # same holds if mean is applied to a sc.DataArray
        # this is in preparation to rewrite uv_multi_turbidity to accept dataarrays or dict of dataarrays, not filelist
        if not "source" in uv_da_turbcorr.attrs.keys():
            # append names as attributes
            uv_da_turbcorr.attrs["source"] = sc.array(
                dims=["spectrum"], values=[name] * uv_da_turbcorr.sizes["spectrum"]
            )
       
        uv_collection[f"{name}"] = uv_da_turbcorr

        # print(name,uv_da_turbcorr.data.shape,uv_da_turbcorr.data.values.ndim  )

    multi_uv_turb_corr_da = sc.concat(
        [uv_collection[f"{name}"] for name in filelist], dim="spectrum"
    )
    #display(multi_uv_turb_corr_da)

    
    return multi_uv_turb_corr_da






