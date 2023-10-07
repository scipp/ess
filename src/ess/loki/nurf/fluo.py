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
import scippnexus as snx
import scipp as sc
from ess.loki.nurf import utils
from scipp.ndimage import median_filter
from utils import nurf_median_filter



def normalize_fluo(
    *, sample: sc.DataArray, reference: sc.DataArray, dark: sc.DataArray
) -> sc.DataArray:
    """Calculates the corrected fluo signal for each given fluo spectrum in a given sc.DataArray

    Parameters
    ----------
    sample: sc.DataArray
        DataArray containing sample fluo signal, one spectrum or multiple spectra.
    reference: sc.DataArray
        DataArray containing reference fluo signal, one spectrum expected.
    dark: sc.DataArray
        DataArray containing dark fluo signal, one spectrum expected.

    Returns
    ----------
    final_fluo: sc.DataArray
        DataArray that contains the calculated fluo signal, one spectrum or mulitple spectra.

    """

    # all spectra in this file are converted to final_fluo
    # More explanation on the dark here.
    # We keep dark here for consistency reasons. 
    # The dark measurement is necessary for this type of detector. Sometimes, 
    # one can use the fluorescence emission without the reference. In that case 
    # having the dark is important.
    # In most cases the reference should have no fluorescence emission, 
    # basically flat. In more complex solvent the reference may have some 
    # intrinsic fluorescence that would need to be substracted.


    final_fluo = (sample - dark) - (reference - dark)

    return final_fluo


def load_and_normalize_fluo(name) -> sc.DataArray :
    """Loads the fluo data from the corresponding entry in the LoKI.nxs filename and 
    calculates the final fluo spectrum of each spectrum.
    
    Parameters
    ----------
    name: str
        Filename, e.g. 066017.nxs

    Returns
    ----------
    normalized: sc.DataArray 
        DataArray that contains the normalized fluo signal, one spectrum or mulitple spectra.

    """
    fluo_dict = utils.load_nurfloki_file(name, 'fluorescence')
    normalized = normalize_fluo(**fluo_dict)
    # provide source of each spectrum in the file
    #normalized.attrs['source'] = sc.scalar(name).broadcast(['spectrum'], [normalized.sizes['spectrum']])

    return normalized 



def fluo_maxint_max_wavelen(
    flist_num: list,
    wllim: Optional[sc.Variable] = None,
    wulim: Optional[sc.Variable] = None,
    medfilter=True,
    kernel_size=15,
):
    """For a given list of files this function extracts for each file the maximum intensity and the corresponding
    wavelength of each fluo spectrum.

    Parameters
    ----------
    flist_num: list
        List of filename for a LoKI.nxs file containting Fluo entry.
    wllim: sc.Variable
        Lower wavelength limit where the search for the maximum should begin
    wulim: sc.Variable
        Upper wavelength limit where the search for the maximum should end
    medfilter: bool
        If medfilter=False, not medfilter is applied. Default: True
        A medfilter is applied to the fluo spectra as fluo is often more noisy
    kernel_size: int or sc.Variable
        kernel for median_filter along the wavelength dimension. Default: 15

    Returns
    ----------
    fluo_int_dict: dict
        A dictionary of nested dictionaries. For each found monowavelength, there are nested dictionaries
        for each file containing the maximum intensity "intensity_max" and the corresponding wavelength "wavelength_max"

    """

    from collections import defaultdict

    fluo_int_dict = defaultdict(dict)

    # set default parameter
    if wllim is None:
        wllim=sc.scalar(300, unit='nm')
    if wulim is None:
        wulim=sc.scalar(400, unit='nm')
    if (wllim is not None and wulim is not None):
        assert wllim.value < wulim.value, "wllim < wulim"
    if not wllim.unit==wulim.unit:
        raise ValueError("Use same unit for wavelength range.")
       

    for name in flist_num:
        fluo_dict = utils.load_nurfloki_file(name, 'fluorescence')
        fluo_da = normalize_fluo(**fluo_dict)
        # check for the unit
        #if wl_unit is None:
        #    wl_unit = fluo_da.coords["wavelength"].unit

        print(f'Number of fluo spectra in {name}: {fluo_da.sizes["spectrum"]}')
        print(f"This is the fluo dataarray for {name}.")
        display(fluo_da)

        fluo_filt_max = fluo_peak_int(
            fluo_da,
            wllim=wllim,
            wulim=wulim,
            medfilter=medfilter,
            kernel_size=kernel_size,
        )
        print(f"This is the fluo filt max intensity and wavelength dataset for {name}.")
        display(fluo_filt_max)

        # print(f'{name} Unique mwl:', np.unique(fluo_filt_max.coords['monowavelengths'].values))
        unique_monowavelen = np.unique(fluo_filt_max.coords["monowavelengths"].values)
        unique_monowavelen_unit = np.unique(fluo_filt_max.coords["monowavelengths"].unit)

        # create entries into dict
        for mwl in unique_monowavelen:
            fluo_int_max = fluo_filt_max["intensity_max"][
                fluo_filt_max.coords["monowavelengths"] == mwl * unique_monowavelen_unit
            ].values 
            fluo_wavelen_max = fluo_filt_max["wavelength_max"][
                fluo_filt_max.coords["monowavelengths"] == mwl * unique_monowavelen_unit
            ].values  

            # I collect the values in a dict with nested dicts, separated by wavelength
            fluo_int_dict[f"{mwl}{unique_monowavelen_unit}"][f"{name}"] = {
                "intensity_max": fluo_int_max,
                "wavelength_max": fluo_wavelen_max,
            }

    return fluo_int_dict


def fluo_peak_int(
    fluo_da: sc.DataArray,
    wllim: Optional[sc.Variable] = None,
    wulim: Optional[sc.Variable] = None,
    medfilter=True,
    kernel_size=15,
) -> sc.Dataset:
    """Main task: Extract for a given wavelength range [wllim, wulim] the maximum fluo intensity and its corresponding wavelength position.
    A median filter is automatically applied to the fluo data and data is extracted after its application.
    TODO: Check with Cedric if it is ok to use max intensity values after filtering
   
    Parameters
    ----------
    fluo_da: sc.DataArray
        DataArray containing uv spectra
    wllim: sc.Variable
        Wavelength range lower limit
    wulim: sc.Variable
        Wavelength range upper limit
    medfilter: bool
        If medfilter=False, not medfilter is applied. Default: True
        A medfilter is applied to the fluo spectra as fluo is often more noisy
    kernel_size: int or sc.Variable
        kernel for median_filter along the wavelength dimension. Expected dims
        'spectrum' and 'wavelength' in sc.DataArray
        Default kernel_size in wavelength direction: 15

    Returns
    ----------
    fluo_filt_max: sc.Dataset
        A new dataset for each spectrum max. intensity value and corresponding wavelength position


    """
    if not isinstance(fluo_da, sc.DataArray):
        raise TypeError("fluo_da has to be an sc.DataArray!")

    # set default parameter
    if wllim is None:
        wllim=sc.scalar(300, unit='nm')
    if wulim is None:
        wulim=sc.scalar(400, unit='nm')
    if (wllim is not None and wulim is not None):
        assert wllim.value < wulim.value, "wllim < wulim, lower wavelength limit needs to be smaller than upper wavelength limit"
    if not wllim.unit==wulim.unit:
        raise ValueError("Use same unit for wavelength range.")


    # apply nurf_median_filter with kernel_size along the wavelength dimension
    if (medfilter is True and kernel_size is not None):
        fluo_da=utils.nurf_median_filter(fluo_da, kernel_size=kernel_size)
    elif (medfilter is True and kernel_size is None):
        kernel_size=15
        fluo_da=utils.nurf_median_filter(fluo_da, kernel_size=kernel_size)


    # let's go and filter
    # filter spectrum values for the specified interval, filtered along the wavelength dimension
    fluo_filt = fluo_da["wavelength", wllim  : wulim ]

    # there is no function in scipp similar to xr.argmax
    # access to data np.ndarray in fluo_filt
    fluo_filt_data = fluo_filt.values
    # max intensity values along the wavelength axis, here axis=1, but they have seen a median filter. TODO: check with Cedric, is it okay to continue with this intensity value?
    fluo_filt_max_int = fluo_filt.data.max("wavelength").values
    # corresponding indices
    fluo_filt_max_idx = fluo_filt_data.argmax(axis=1)
    # corresponding wavelength values for max intensity values in each spectrum
    fluo_filt_max_wl = fluo_filt.coords["wavelength"].values[fluo_filt_max_idx]

    # new sc.Dataset
    fluo_filt_max = sc.Dataset()
    fluo_filt_max["intensity_max"] = sc.Variable(
        dims=["spectrum"], values=fluo_filt_max_int
    )
    fluo_filt_max["wavelength_max"] = sc.Variable(
        dims=["spectrum"], values=fluo_filt_max_wl, unit=wllim.unit
    )
    # adding source information to each data entry in the dataset
    fluo_filt_max["intensity_max"].attrs['source']=fluo_da.attrs['source'].broadcast(['spectrum'], [fluo_da.sizes['spectrum']])
    fluo_filt_max["wavelength_max"].attrs['source']=fluo_da.attrs['source'].broadcast(['spectrum'], [fluo_da.sizes['spectrum']])


    # add previous information to this dataarray
    # TODO: Can we copy multiple coordinates from one array to another?
    # add information from previous fluo_da to the new dataarray
    fluo_filt_max.coords["integration_time"] = fluo_da.coords["integration_time"]
    fluo_filt_max.coords["is_dark"] = fluo_da.coords["is_dark"]
    fluo_filt_max.coords["is_reference"] = fluo_da.coords["is_reference"]
    fluo_filt_max.coords["is_data"] = fluo_da.coords["is_data"]
    fluo_filt_max.coords["monowavelengths"] = fluo_da.coords["monowavelengths"]
    fluo_filt_max.coords["time"] = fluo_da.coords["time"]

    return fluo_filt_max



