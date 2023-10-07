# This file contains functions that will be deprecated. 
# NUrf scipp graveyard

def load_uv(name):
    """Loads the UV data from the corresponding entry in the LoKI.nxs filename.

    Parameters
    ----------
    name: str
        Filename, e.g. 066017.nxs

    Returns:
    ----------
    uv_dict: dict
        Dictionary that contains UV data signal (data) from the sample, the reference,
        and the dark measurement.
        Keys: sample, reference, dark

    """

    # load the nexus and extract the uv entry
    with snx.File(name) as f:
        uv = f["entry/instrument/uv"][()]

    # separation
    uv_dict = split_sample_dark_reference(uv)

    return uv_dict


def load_fluo(name):
    """Loads the data contained in the fluo entry of a LoKI.nxs file

    Parameters
    ----------
    name: str
        Filename, e.g. 066017.nxs

    Returns
    ----------
    fluo_dict: dict
        Dictionary of sc.DataArrays. Keys: data, reference, dark. Data contains the fluo signals of the sample.

    """

    with snx.File(name) as f:
        fluo = f["entry/instrument/fluorescence"][()]

    # separation
    fluo_dict = split_sample_dark_reference(fluo)

    return fluo_dict


# possibilites for median filters, I did not benchmark, apparently median_filter
# could be the faster and medfilt2d is faster than medfilt
#from scipy.ndimage import median_filter
#from scipy.signal import medfilt
# This function will be replaced by scipp owns medilter.
def apply_medfilter(
    da: sc.DataArray, kernel_size: Optional[int] = None
) -> sc.DataArray:
    #TODO: Rewrite this function with median filters offered by scipp.
    """Applies a median filter to a sc.DataArray that contains fluo or uv spectra to
    remove spikes
    Filter used: from scipy.ndimage import median_filter. This filter function is maybe
    subject to change for another scipy function.
    Caution:  The scipy mean (median?) filter will just ignore errorbars on counts
    according to Neil Vaytet.

    Parameters
    ----------
    da: sc.DataArray
        DataArray that contains already fluo or uv spectra in data.

    kernel_size: int
        Scalar giving the size of the median filter window. Elements of kernel_size should be odd. Default size is 3

    Returns
    ----------
    da_mfilt: sc.DataArray
        New sc.DataArray with median filtered data

    """
    if kernel_size is None:
        kernel_size = 3
    else:
        assert type(kernel_size) is int, "kernel_size must be an integer"

    # check if kernel_size is odd
    if (kernel_size % 2) != 1:
        raise ValueError("kernel_size should be odd.")

    # extract all spectreal values
    data_spectrum = da.values

    # apply a median filter
    # yes, we can apply a median filter to an array (vectorize it), but the kernel needs to be adapted to the  ndim off the array
    # no filterin in the first dimension (spectrum), but in the second (wavelength)
    # magic suggested by Simon
    # this will make [kernel_size] for 1d data, and [1, kernel_size] for 2d data
    ksize = np.ones_like(da.shape)
    ksize[-1] = kernel_size
    data_spectrum_filt = median_filter(data_spectrum, size=ksize)

    # create a new sc.DataArray where the data is replaced by the median filtered data
    # make a deep copy
    da_mfilt = da.copy()

    # replace original data with filtered data
    da_mfilt.values = data_spectrum_filt

    # graphical comparison
    legend_props = {"show": True, "loc": (1.04, 0)}
    figs, axs = plt.subplots(1, 2, figsize=(16, 5))
    # out1=sc.plot(da['spectrum',7], ax=axs[0])
    # out2=sc.plot(da_mfilt['spectrum',7], ax=axs[1])
    out1 = sc.plot(
        sc.collapse(da, keep="wavelength"),
        linestyle="dashed",
        marker=".",
        grid=True,
        legend=legend_props,
        title=f"Before median filter",
        ax=axs[0],
    )
    out2 = sc.plot(
        sc.collapse(da_mfilt, keep="wavelength"),
        linestyle="dashed",
        marker=".",
        grid=True,
        legend=legend_props,
        title=f"After median filter - kernel_size: {kernel_size}",
        ax=axs[1],
    )
    # display(figs)

    return da_mfilt
