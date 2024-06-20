# standard library imports
import itertools
import os
from typing import Optional

# related third party imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.gridspec as gridspec
from IPython.display import display, HTML
# possibilites for median filters, I did not benchmark, apparently median_filter
# could be the faster and medfilt2d is faster than medfilt
from scipy.ndimage import median_filter
from scipy.signal import medfilt
from scipy.optimize import leastsq  # needed for fitting of turbidity

# local application imports
import scippneutron as scn
import scippnexus as snx
import scipp as sc


def split_sample_dark_reference(da):
    """Separate incoming dataarray into the three contributions: sample, dark, reference.

    Parameters
    ----------
    da: scipp.DataArray
            sc.DataArray that contains spectroscopy contributions sample, dark, reference

    Returns:
    ----------
    da_dict: dict
            Dictionary that contains spectroscopy data signal (data) from the sample,
            the reference, and the dark measurement.
            Keys: sample, reference, dark

    """
    assert isinstance(da, sc.DataArray)

    dark = da[da.coords["is_dark"]].squeeze()
    ref = da[da.coords["is_reference"]].squeeze()
    sample = da[da.coords["is_data"]]    
       
    #TODO Instead of a dict a sc.Dataset? 
    return {"sample": sample, "reference": ref, "dark": dark}


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


def normalize_uv(
    *, sample: sc.DataArray, reference: sc.DataArray, dark: sc.DataArray
) -> sc.DataArray: 
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


def load_and_normalize_uv(name):
    """Loads the UV data from the corresponding entry in the LoKI.nxs filename and calculates the absorbance of each UV spectrum.
        For an averaged spectrum based on all UV spectra in the file, use process_uv.

    Parameters
    ----------
    name: str
        Filename, e.g. 066017.nxs

    Returns:
    ----------
    normalized: sc.DataArray
        DataArray that contains the normalized UV signal, one spectrum or mulitple spectra.

    """
    uv_dict = load_uv(name)
    normalized = normalize_uv(**uv_dict)  # results in DataArrays with multiple spectra

    return normalized


def process_uv(name):
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
        under UV entry data.

    """

    uv_dict = load_uv(name)
    normalized = normalize_uv(**uv_dict) 

    # returns averaged uv spectrum
    return normalized.mean("spectrum")


def plot_uv(name):
    """Plots multiple graphs of UV data contained in a LoKI.nxs file.
        First graph: Raw UV spectra
        Second graph:  Plot of dark and reference
        Third graph: Individual normalized UV spectra
        Fourth graph: Averaged UV spectrum over all found data in LoKI.nxs file.

    Parameters
    ----------
    name: str
        Filename for a LoKI.nxs file containting UV entry.

    """
    # extracting and preparing uv data
    uv_dict = load_uv(name)
    normalized = normalize_uv(**uv_dict)  # results in DataArrays with multiple spectra

    # legend property
    legend_props = {"show": True, "loc": 1}
    figs, axs = plt.subplots(1, 4, figsize=(16, 3))

    # How to plot raw data spectra in each file?
    out1 = sc.plot(
        sc.collapse(uv_dict["sample"], keep="wavelength"),
        linestyle="dashed",
        marker=".",
        grid=True,
        legend=legend_props,
        title=f"{name} - raw data",
        ax=axs[0],
    )

    # How to plot dark and reference?
    out4 = sc.plot(
        {"dark": uv_dict["dark"], "reference": uv_dict["reference"]},
        linestyle="dashed",
        marker=".",
        grid=True,
        legend=legend_props,
        title=f"{name} - dark and reference",
        ax=axs[1],
    )

    # How to plot individual calculated spectra from one .nxs file
    out2 = sc.plot(
        sc.collapse(normalized, keep="wavelength"),
        linestyle="dashed",
        marker=".",
        grid=True,
        legend=legend_props,
        title=f"{name} - calculated",
        ax=axs[2],
    )

    # How to plot averaged spectra (.mean and NOT .sum)
    out3 = sc.plot(
        sc.collapse(normalized.mean("spectrum"), keep="wavelength"),
        linestyle="dashed",
        marker=".",
        grid=True,
        legend=legend_props,
        title=f"{name} - averaged",
        ax=axs[3],
    )
    # display(figs)


def gather_uv_set(flist_num):
    """Creates a sc.DataSet for set of given filenames for an experiment composed of 
    multiple, separated UV measurements over time.
    Parameters
    ----------
    flist_num: list of int
        List of filenames as numbers (ILL style) containing UV data

     Returns
    ----------
    uv_spectra_set: sc.DataSet
        DataSet of multiple UV DataArrays, where the UV signal for each experiment was averaged

    """

    uv_spectra_set = sc.Dataset({name: process_uv(name) for name in flist_num})
    return uv_spectra_set


def plot_uv_set(flist_num, lambda_min=None, lambda_max=None, vmin=None, vmax=None):
    """Plots a set of averaged UV spectra

    Parameters
    ----------
    flist_num: list of int
        List of filenames as numbers (ILL style) containing UV data
    lambda_min: float
        Minimum wavelength
    lambda_max: float
        Maximum wavelength
    vmin: float
        Minimum y-value
    vmax: float
        Maximum y-value

    Returns
    ----------
    fig_all_spectra: scipp.plotting.objects.Plot

    """

    # creates a dataset based on flist_num, plots all of them
    uv_spectra_set = gather_uv_set(flist_num)
    # How to plot multiple UV spectra and zoom in.
    # set figure size and legend props
    figure_size = (7, 4)
    legend_props = {"show": True, "loc": (1.04, 0)}
    fig_all_spectra = uv_spectra_set.plot(
        linestyle="dashed",
        marker=".",
        grid=True,
        legend=legend_props,
        figsize=figure_size,
    )
    if vmin is not None and vmax is not None:
        fig_all_spectra.ax.set_ylim(vmin, vmax)
    if lambda_min is not None and lambda_max is not None:
        fig_all_spectra.ax.set_xlim(lambda_min, lambda_max)
    display(fig_all_spectra)
    return fig_all_spectra


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


def plot_fluo(name):
    """Plots all fluo spectra contained in a LoKI.nxs
        Currently, we separate between good and bad fluo spectra collected during a ILL experiment. This differentiation can later go for LoKI.
        First graph: Plot of all raw fluo spectra
        Second graph: Plot dark and reference for the fluo path
        Third graph: All bad fluo spectra recorded during the ILL experiment
        Fourth graph: All good fluo spectra recorded during the ILL experiment
        Fifth graph: All good fluo spectra for wavelength range  250nm - 600nm


    Parameters
    ----------
    name: str
        Filename of a LoKI.nxs file

    """
    # Plots fluo spectra in LOKI.nxs of file name
    # Input: name, str

    fluo_dict = load_fluo(name)  # this is dictionary with scipp.DataArrays in it
    final_fluo = normalize_fluo(**fluo_dict)  # this is one scipp.DataArray

    # set figure size and legend props
    figure_size = (8, 4)
    legend_props = {"show": True, "loc": (1.04, 0)}

    # plot all fluo raw spectra
    out1 = sc.plot(
        sc.collapse(fluo_dict["sample"]["spectrum", :], keep="wavelength"),
        linestyle="dashed",
        grid=True,
        legend=legend_props,
        figsize=figure_size,
        title=f"{name}, raw fluo spectrum - all",
    )  # legend={"show": True, "loc": (1.0, 1.0)} #figsize=(width, height)
    display(out1)

    # plot raw and dark spectra for fluo part
    out2 = sc.plot(
        {"dark": fluo_dict["dark"], "reference": fluo_dict["reference"]},
        linestyle="dashed",
        grid=True,
        legend=legend_props,
        figsize=figure_size,
        title=f"{name}, dark and reference",
    )  # legend={"show": True, "loc": (1.0, 1.0)} #figsize=(width, height))
    display(out2)

    # specific for ILL data, every second sppectrum good, pay attention to range() where
    #  the selection takes place
    only_bad_spectra = {}  # make empty dict
    for i in range(0, fluo_dict["sample"].sizes["spectrum"], 2):
        only_bad_spectra[f"spectrum-{i}"] = fluo_dict["sample"]["spectrum", i]
    out3 = sc.plot(
        only_bad_spectra,
        linestyle="dashed",
        grid=True,
        legend=legend_props,
        figsize=figure_size,
        title=f"{name} - all bad raw spectra",
    )
    display(out3)

    only_good_spectra = {}  # make empty dict
    for i in range(1, fluo_dict["sample"].sizes["spectrum"], 2):
        only_good_spectra[f"spectrum-{i}"] = fluo_dict["sample"]["spectrum", i]
    out4 = sc.plot(
        only_good_spectra,
        linestyle="dashed",
        grid=True,
        legend=legend_props,
        figsize=figure_size,
        title=f"{name} - all good raw spectra",
    )
    display(out4)

    # plot only good spectra with wavelength in legend name
    only_good_fspectra = {}  # make empty dict
    for i in range(1, fluo_dict["sample"].sizes["spectrum"], 2):
        mwl = str(final_fluo.coords["monowavelengths"][i].value) + "nm"
        only_good_fspectra[f"spectrum-{i}, {mwl}"] = final_fluo["spectrum", i]
    out5 = sc.plot(
        only_good_fspectra,
        linestyle="dashed",
        grid=True,
        legend=legend_props,
        figsize=figure_size,
        title=f"{name} - all good final spectra",
    )
    display(out5)

    # zoom in final spectra selection
    out6 = sc.plot(
        only_good_fspectra,
        linestyle="dashed",
        grid=True,
        legend=legend_props,
        figsize=figure_size,
        title=f"{name} - all good final spectra",
    )
    lambda_min = 250
    lambda_max = 600
    out6.ax.set_xlim(lambda_min, lambda_max)
    display(out6)

    fig_handles = [out1, out2, out3, out4, out5, out5, out6]
    modify_plt_app(fig_handles)


def markers():
    """Creates a list of markers for plots"""
    return ["+", ".", "o", "*", "1", "2", "x", "P", "v", "X", "^", "d"]


def line_colors(number_of_lines):
    """
    Creates a number of colors based on the number of spectra/lines  to plot.

    Parameters
    ----------
    number_of_lines: int
        number of available spectra in a plot


    Returns
    ----------
    colors: list of tuples
    """

    # create better line colors
    start = 0.0
    stop = 1.0
    number_of_lines = number_of_lines
    cm_subsection = np.linspace(start, stop, number_of_lines)
    colors = [cm.jet(x) for x in cm_subsection]
    return colors


def modify_plt_app(fig_handles):
    """Modifies scipp plots. Nicer markers and rainbow colors for multiple curves in a scipp plot object.

    Parameters
    ----------
    fig_handles: list of scipp.plotting.objects.Plot

    """
    # Modify scipp plots afterwards
    for pl1 in fig_handles:
        pl1.fig.tight_layout()
        number_lines = len(pl1.ax.get_lines())
        colors = line_colors(number_lines)
        for i in range(0, number_lines):
            pl1.ax.get_lines()[i].set_color(colors[i])
            pl1.ax.get_legend().legendHandles[i].set_color(colors[i])
            pl1.ax.get_lines()[i].set_marker(markers()[i])
            pl1.ax.get_lines()[i].set_markersize(5)
            pl1.ax.get_lines()[i].set_markevery(5)
            pl1.ax.get_legend().legendHandles[i].set_marker(markers()[i])
            pl1.ax.get_lines()[i].set_linewidth(1)


def export_uv(name, path_output):
    """Export normalized all uv data and an averaged uv spectrum in a LoKI.nxs file to .dat file
    
    Attention: Current output format follows custom format for an individual user, not
    any other software.

    Parameters
    ----------
    name: str
        Filename of LoKI.nxs file that contains the UV data for export

    path_output: str
        Absolute path to output folder

    Returns
    ----------
    Tab-limited .dat file with columns wavelength, dark, reference, (multiple) raw uv spectra, (multiple) corrected uv spectra, one averaged uv spectrum

    """

    uv_dict = load_uv(name)  
    normalized = normalize_uv(
        **uv_dict
    ) 
   
    normalized_avg = normalized.mean("spectrum")

    # prepare for export as .dat files
    output_filename = f"{name}_uv.dat"

    # puzzle the header together
    l = "".join(
        ["dark_{0}\t".format(i) for i in range(uv_dict['dark'].ndim)
        ]
        )
    m="".join(
        ["reference_{0}\t".format(i) for i in range(uv_dict['reference'].ndim)
        ]
        )

    n="".join(
        [   
            "uv_raw_spectra_{0}\t".format(i)
            for i, x in enumerate(range(uv_dict['sample'].sizes["spectrum"]))
        ]
        )
    o="".join(    
        [   
            "uv_norm_spectra_{0}\t".format(i)
            for i, x in enumerate(range(normalized.sizes["spectrum"]))
        ]
        )
    
    p = "uv_spectra_avg\t" 
  

    hdrtxt = "wavelength [nm]\t"
    final_header = hdrtxt + l + m + n + o + p


    data_to_save = np.column_stack(
        (
            normalized.coords["wavelength"].values.transpose(),

            # raw data
            uv_dict['dark'].values.transpose(),
            uv_dict['reference'].values.transpose(),
            uv_dict['sample'].values.transpose(),

            # reduced data
            normalized.values.transpose(),
            normalized_avg.values.transpose(),
            
        )
    )
    path_to_save = os.path.join(path_output, output_filename)

    # dump the content
    with open(path_to_save, "w") as f:
        np.savetxt(f, data_to_save, fmt="%.10f", delimiter="\t", header=final_header)


def export_fluo(name, path_output):
    """Export corrected fluo data contained in a LoKI.nxs file to .dat file

    Attention: Current output format follows custom format for an individual user, not
    any other software.

    Parameters
    ----------
    name: str
        Filename of LoKI.nxs file that contains the fluo data for export

    path_output: str
        Absolute path to output folder

    Returns
    ----------
    Tab-limited .dat file with columns wavelength, dark, reference, multiple raw fluo 
    spectra, and multiple normalized fluo spectra.
    Header of each fluo spectrum column contains the incident excitation energy.

    """
    # export of all calculated fluo data in a LOKI.nxs name to .dat
    # input: filename of LOKI.nxs: name, str, path_output: absolut path to output folder, str

    fluo_dict = load_fluo(name)
    final_fluo = normalize_fluo(**fluo_dict)

    # prepare for export as .dat files
    output_filename = f"{name}_fluo.dat"
    path_to_save = os.path.join(path_output, output_filename)

    
    
    l = "".join(
        ["dark_{0}\t".format(i) for i in range(fluo_dict['dark'].ndim)
        ]
        )
    m = "".join(
        ["reference_{0}\t".format(i) for i in range(fluo_dict['reference'].ndim)
        ]
        )
    
    n= "".join([f"raw_{i}nm\t" for i in final_fluo.coords["monowavelengths"].values])

    o= "".join([f"norm_{i}nm\t" for i in final_fluo.coords["monowavelengths"].values])



    hdrtxt = "wavelength [nm]\t"
    final_header = hdrtxt + l + m + n + o
 
    data_to_save = np.column_stack(
        (
            final_fluo.coords["wavelength"].values.transpose(),
            # dark
            fluo_dict['dark'].values.transpose(),
            # reference
            fluo_dict['reference'].values.transpose(),
            # sample
            fluo_dict['sample'].values.transpose(),
            # final fluo spectra
            final_fluo.data["spectrum", :].values.transpose(),
        )
    )

    # dump the content
    with open(path_to_save, "w") as f:
        np.savetxt(
            f,
            data_to_save,
            fmt="".join(["%.5f\t"] + ["%.5e\t"] * (fluo_dict['dark'].ndim + 
                        fluo_dict['reference'].ndim + 2*final_fluo.sizes["spectrum"])),
            delimiter="\t",
            header=final_header,
        )


def fluo_maxint_max_wavelen(
    flist_num,
    wllim=300,
    wulim=400,
    wl_unit=None,
    medfilter=True,
    kernel_size=15,
):
    """For a given list of files this function extracts for each file the maximum intensity and the corresponding
    wavelength of each fluo spectrum.

    Parameters
    ----------
    flist_num: list
        List of filename for a LoKI.nxs file containting Fluo entry.
    wllim: float
        Lower wavelength limit where the search for the maximum should begin
    wulim: float
        Upper wavelength limit where the search for the maximum should end
    wl_unit: sc.Unit
        Unit of the wavelength
    medfilter: bool
        A medfilter is applied to the fluo spectra as fluo is often more noisy
    kernel_size: int, uneven, min>=3
        The width of the median filter, uneven, and at least a value of 3.

    Returns
    ----------
    fluo_int_dict: dict
        A dictionary of nested dictionaries. For each found monowavelength, there are nested dictionaries
        for each file containing the maximum intensity "intensity_max" and the corresponding wavelength "wavelength_max"

    """

    from collections import defaultdict

    fluo_int_dict = defaultdict(dict)

    for name in flist_num:
        fluo_dict = load_fluo(name)
        fluo_da = normalize_fluo(**fluo_dict)
        # check for the unit
        if wl_unit is None:
            wl_unit = fluo_da.coords["wavelength"].unit


        print(f'Number of fluo spectra in {name}: {fluo_da.sizes["spectrum"]}')
        print(f"This is the fluo dataarray for {name}.")
        display(fluo_da)

        fluo_filt_max = fluo_peak_int(
            fluo_da,
            wllim=wllim,
            wulim=wulim,
            wl_unit=wl_unit,
            medfilter=medfilter,
            kernel_size=kernel_size,
        )
        print(f"This is the fluo filt max intensity and wavelength dataset for {name}.")
        display(fluo_filt_max)

        # print(f'{name} Unique mwl:', np.unique(fluo_filt_max.coords['monowavelengths'].values))
        unique_monowavelen = np.unique(fluo_filt_max.coords["monowavelengths"].values)

        # create entries into dict
        for mwl in unique_monowavelen:
            fluo_int_max = fluo_filt_max["intensity_max"][
                fluo_filt_max.coords["monowavelengths"] == mwl * wl_unit
            ].values 
            fluo_wavelen_max = fluo_filt_max["wavelength_max"][
                fluo_filt_max.coords["monowavelengths"] == mwl * wl_unit
            ].values  

            # I collect the values in a dict with nested dicts, separated by wavelength
            fluo_int_dict[f"{mwl}{wl_unit}"][f"{name}"] = {
                "intensity_max": fluo_int_max,
                "wavelength_max": fluo_wavelen_max,
            }

    return fluo_int_dict


def fluo_plot_maxint_max_wavelen(fluo_int_dict: dict):
    """Plots for fluorescence maximum intensity and corressponding wavelength as function of pertubation.
        Pertubation parameter: currently filename as unique identifier

    Parameters
    ----------
    fluo_int_dict: dict
        Contains per found monowavelength and per measurement the value of the max fluo intensity for a selected wavelength range and the corresponding maximum wavelength.

    Returns
    ----------

    """

    # separate information
    max_int_dict = {}
    max_int_wavelen_dict = {}

    monowavelen = fluo_int_dict.keys()

    # prepare the plots
    monowavelen = len(fluo_int_dict.keys())
    print("Number of keys", monowavelen)

    fig, ax = plt.subplots(nrows=monowavelen, ncols=2, figsize=(10, 7))
    plt.subplots_adjust(
        left=None, bottom=0.05, right=None, top=0.95, wspace=0.2, hspace=0.5
    )

    for j, mwl in enumerate(fluo_int_dict):

        for name in fluo_int_dict[mwl]:
            max_int_dict[name] = fluo_int_dict[mwl][name]["intensity_max"]
            max_int_wavelen_dict[name] = fluo_int_dict[mwl][name]["wavelength_max"]

        for i, fname in enumerate(max_int_dict):

            x_val = [i] * len(max_int_dict[fname])
            ax[j, 0].set_title(f"monowavelength: {mwl}: max. intensity")
            ax[j, 0].plot(x_val, max_int_dict[fname], marker="o", linestyle="None")
            ax[j, 0].grid(True)
            ax[j, 0].set_ylabel("Max. peak intensity [counts]")
            print(mwl, fname, "max int", max_int_dict[fname])

            ax[j, 1].set_title(f"monowavelength: {mwl}: wavelength")
            ax[j, 1].plot(
                x_val, max_int_wavelen_dict[fname], marker="o", linestyle="None"
            )
            ax[j, 1].set_ylabel("Peak wavelength [nm]")
            ax[j, 1].grid(True)
            # ax[1].set_ylabel(f'Wavelength [{unit_str}]')
            # ax[j,1].set_ylim(bottom=0.95*int(mwl[:-4]))

        ax[j, 1].plot(
            np.linspace(
                0,
                len(max_int_wavelen_dict.keys()) - 1,
                num=len(max_int_wavelen_dict.keys()),
            ),
            [int(mwl[:-4])] * len(max_int_wavelen_dict.keys()),
            "--",
        )
        # overwrite xticks in ax0
        ax[j, 0].set_xticks(
            np.arange(0, len(max_int_dict.keys()), 1),
            labels=[f"{name}" for name in max_int_dict],
            rotation=90,
        )
        # overwrite xticks in ax1
        ax[j, 1].set_xticks(
            np.arange(0, len(max_int_wavelen_dict.keys()), 1),
            labels=[f"{name}" for name in max_int_wavelen_dict],
            rotation=90,
        )


def apply_medfilter(
    da: sc.DataArray, kernel_size: Optional[int] = None
) -> sc.DataArray:
    #TODO: Rewrite this function with median filters offered by scipp.
    """Applies a median filter to a sc.DataArray that contains fluo or uv spectra to remove spikes
    Filter used: from scipy.ndimage import median_filter. This filter function is maybe subject to change for another scipy function.
    Caution:  The scipy mean (median?) filter will just ignore errorbars on counts according to Neil Vaytet.

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


def uv_peak_int(uv_da: sc.DataArray, wavelength=None, wl_unit=None, tol=None):
    """Extract uv peak intensity for a given wavelength [wl_unit] and a given interval [wl_unit].
    First version: interval around wavelength of width 2*tol, values are averaged and then we need interpolation to get value at requested wavelength??? not sure yet how to realise this
    If no wavelength is given, 280 is chosen as value
    If no wl_unit is given, the unit of the wavelength coordinate of the sc.DataArray is chosen, e.g. [nm]. Other option to generate a unit: wl_unit=sc.Unit('nm')

    Parameters
    ----------
    uv_da: sc.DataArray
        DataArray containing uv spectra
    wavelength: float
        Wavelength
    wl_unit: sc.Unit
        Unit of the wavelength
    tol: float
        Tolerance, 2*tol defines the interval around the given wavelength

    Returns
    ----------
    uv_peak_int: dict
        Dictionary that contains the peak intensity for the requested wavelength, the peak intensity averaged over the requested interval, the requested wavelength with its unit, and the tolerance

    """
    assert (
        "wavelength" in uv_da.dims
    ), "sc.DataArray is missing the wavelength dimension"  # assert that 'wavelength' is a dimension in the uv_da sc.DataArray

    # obtain unit of wavelength:
    if wl_unit is None:
        wl_unit = uv_da.coords["wavelength"].unit
    else:
        if not isinstance(wl_unit, sc.Unit):
            raise TypeError
        assert (
            wl_unit == uv_da.coords["wavelength"].unit
        )  # we check that the given unit corresponds to the unit for the wavelength

    # set default value for wavelength:
    if wavelength is None:
        wavelength = 280

    # set default value for tolerance:
    if tol is None:
        tol = 0.5

    # filter spectrum values for the specified interval, filtered along the wavelength dimension
    uv_da_filt = uv_da[
        "wavelength",
        (wavelength - tol) * wl_unit : (wavelength + tol) * wl_unit,
    ]

    # try out interpolation
    from scipp.interpolate import interp1d

    uv_interp = interp1d(uv_da, "wavelength")
    # get new x values, in particular I want the value at one wavelength
    x2 = sc.linspace(
        dim="wavelength", start=wavelength, stop=wavelength, num=1, unit=wl_unit
    )
    # alternative
    # x2 = sc.array(dims=['wavelength'], values=[wavelength], unit=wl_unit)
    # this gives us the intensity value for one specific wavelength
    uv_int_one_wl = uv_interp(x2)

    # now we want to have as well the wavelength for the interval
    uv_int_mean_interval = uv_da_filt.mean(dim="wavelength")

    # prepare a dict for output
    uv_peak_int = {
        "one_wavelength": uv_int_one_wl,
        "wl_interval": uv_int_mean_interval,
        "wavelength": wavelength,
        "unit": wl_unit,
        "tol": tol,
    }

    return uv_peak_int


def plot_uv_peak_int(
    uv_da: sc.DataArray,
    name,
    wavelength=None,
    wl_unit=None,
    tol=None,
    medfilter=False,
    kernel_size=None,
):
    """Plotting of extracted uv peak intensity for a given wavelength [wl_unit] and a given interval [wl_unit] in file name
    First version: interval around wavelength of width 2*tol, values are averaged and then we need interpolation to get value at requested wavelength??? not sure yet how to realise this
    If no wavelength is given, 280 is chosen as value
    If no wl_unit is given, the unit of the wavelength coordinate of the sc.DataArray is chosen, e.g. [nm]. Other option to generate a unit: wl_unit=sc.Unit('nm')
    Kernel size determined the window for the medfilter

    Parameters
    ----------
    uv_da: sc.DataArray
        DataArray containing uv spectra
    name: str
        Filename of a file containting uv data
    wavelength: float
        Wavelength
    wl_unit: sc.Unit
        Unit of the wavelength
    tol: float
        Tolerance, 2*tol defines the interval around the given wavelength
    medfilter: boolean
        If medfilter==False, not medfilter is applied
    kernel_size: int
        kernel for medianfilter

    Returns
    ----------
    uv_peak_int: dict
        Dictionary that contains the peak intensity for the requested wavelength, the peak intensity averaged over the requested interval, the requested wavelength with its unit, and the tolerance

    """
    if not isinstance(uv_da, sc.DataArray):
        raise TypeError

    if medfilter is not False:
        # apply medianfilter
        uv_da = apply_medfilter(uv_da, kernel_size=kernel_size)

    # process
    uv_peak_int_dict = uv_peak_int(
        uv_da, wavelength=wavelength, wl_unit=wl_unit, tol=tol
    )
    # append name to dict
    uv_peak_int_dict["filename"] = name

    print(uv_peak_int_dict["one_wavelength"])
    print(uv_peak_int_dict["wl_interval"]["spectrum", :])

    # set figure size and legend props
    # figs, axs = plt.subplots(1, 2, figsize=(10 ,3))
    fig = plt.figure(figsize=(12, 4))

    gs = gridspec.GridSpec(nrows=1, ncols=2, width_ratios=[1, 1], wspace=0.4)
    ax0 = fig.add_subplot(gs[0, 0])
    out1 = sc.plot(
        uv_peak_int_dict["one_wavelength"]["spectrum", :].squeeze(),
        linestyle="none",
        grid=True,
        title=f"UV peak intensity for {uv_peak_int_dict['wavelength']} {uv_peak_int_dict['unit']} in {name} ",
        ax=ax0,
    )
    ax0.set_ylabel("UV int")
    ax1 = fig.add_subplot(gs[0, 1])
    out2 = sc.plot(
        uv_peak_int_dict["wl_interval"]["spectrum", :],
        linestyle="none",
        grid=True,
        title=f"UV peak intensity for intervall [{uv_peak_int_dict['wavelength']-uv_peak_int_dict['tol']} , {uv_peak_int_dict['wavelength']+uv_peak_int_dict['tol']}]{uv_peak_int_dict['unit']} in {name} ",
        ax=ax1
    )
    ax1.set_ylabel("UV int")


def uv_quick_data_check(
    filelist,
    wavelength=None,
    wl_unit=None,
    tol=None,
    medfilter=False,
    kernel_size=None,
):
    """Plots uv peak intensity for a given wavelength with unit wl_unit for a given 
    filelist in order of items in filelist. Per recorded spectrum in the file a data 
    point is shown.

    """
    # Currently I cannot rely on having the same number of spectra in each file currently, sc.Dataset will not work
    # The following does not work if the number of spectra in each Loki file is not the same :()
    # out= sc.Dataset({name:load_and_normalize_uv(name) for name in filelist})

    # Let's try dictionary comprehension ... yeap that works
    uv_dict = {name: load_and_normalize_uv(name) for name in filelist}

    # apply medianfilter
    if medfilter is not False:
        # apply_medfilter expects a sc.DataArray
        uv_dict = {
            name: apply_medfilter(uv_dict[name], kernel_size=kernel_size)
            for name in filelist
        }

    # extract peak intensities
    # function: uv_peak_int(uv_da: sc.DataArray, wavelength=wavelength, wl_unit=wl_unit, tol=tol)
    # this process will return a dictionary of dictionaries
    uv_peak_int_dict = {
        name: uv_peak_int(
            uv_dict[name], wavelength=wavelength, wl_unit=wl_unit, tol=tol
        )
        for name in filelist
    }

    # prepare plots
    figure_size = (20, 10)
    fig = plt.figure(figsize=figure_size)
    gs = gridspec.GridSpec(
        nrows=2,
        ncols=3,
        width_ratios=[1, 1, 1],
        wspace=0.3,
        height_ratios=[1, 1],
    )
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])

    # Testing but so far unsuccessful
    # print(uv_peak_int_dict['066017.nxs']['one_wavelength'].data)
    # display(sc.plot(sc.collapse(uv_peak_int_dict['066017.nxs']['one_wavelength'].data,keep='spectrum')))

    uv_peak_one_wl = {}  # make empty dict
    for name in filelist:
        # uv_peak_one_wl[f'{name}']=uv_peak_int_dict[f'{name}']['one_wavelength']['spectrum',:]
        # print(uv_peak_one_wl[f'{name}'])

        ax0.plot(
            uv_peak_int_dict[f"{name}"]["one_wavelength"]["spectrum", :].values,
            "o",
            label=f"{name}",
        )
        ax0.legend()

        ax1.plot(
            uv_peak_int_dict[f"{name}"]["wl_interval"]["spectrum", :].values,
            "o",
            label=f"{name}",
        )
        ax1.legend()

    ax0.grid(visible=True)
    ax0.set_title(
        f"UV peak intensity for {uv_peak_int_dict[filelist[0]]['wavelength']} {uv_peak_int_dict[filelist[0]]['unit']} "
    )
    ax0.set_xlabel("spectrum")

    ax1.plot(
        uv_peak_int_dict[filelist[0]]["wl_interval"]["spectrum", :].values,
        "o",
        label=f"{name}",
    )
    ax1.legend()
    ax1.grid(visible=True)
    ax1.set_title(
        f"UV peak intensity for intervall [{uv_peak_int_dict[filelist[0]]['wavelength']-uv_peak_int_dict[filelist[0]]['tol']} , {uv_peak_int_dict[filelist[0]]['wavelength']+uv_peak_int_dict[filelist[0]]['tol']}]{uv_peak_int_dict[filelist[0]]['unit']}"
    )
    print(
        uv_peak_int_dict[filelist[0]]["wl_interval"]["spectrum", :].values,
        uv_peak_int_dict[filelist[0]]["wl_interval"]["spectrum", :].values,
    )
    ax1.set_xlabel("spectrum")

    display(fig)


def plot_multiple_uv_peak_int(
    filelist,
    wavelength=None,
    wl_unit=None,
    tol=None,
    medfilter=False,
    kernel_size=None,
):
    """Plots uv peak intensity for a given wavelength with unit wl_unit for a given filelist in order of items in filelist
    Explorative state. I fall back to matplotlib because it is complicated. Try later with sc.plot.
    Currently, I cannot rely on all files having the same number of spectra. I cannot create a sc.Dataset. I am unsure how to catch this with Python.
    No averaging in uv spectra when calculating the uv peak intensity within one file.
    Medfilter can be activated

    #TODO: Get rid of the for loop and and make filename an attribute when loading nexus files, filename as new dimension?
    """

    # currently I cannot rely on having the same number of spectra in each file currently, sc.Dataset will not work
    fig = plt.figure(figsize=(15, 5))
    gs = gridspec.GridSpec(
        nrows=1, ncols=2, width_ratios=[1, 1], wspace=0.3, height_ratios=[1]
    )
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])

    for i, name in enumerate(filelist):
        uv_dict = load_uv(name)
        uv_da = normalize_uv(**uv_dict)

        # check for medfilter
        if medfilter is not False:
            # apply medianfilter
            uv_da = apply_medfilter(uv_da, kernel_size=kernel_size)

        # process
        uv_peak_int_dict = uv_peak_int(
            uv_da, wavelength=wavelength, wl_unit=wl_unit, tol=tol
        )
        # append name to dict
        uv_peak_int_dict["filename"] = name

        num_spec = uv_peak_int_dict["wl_interval"]["spectrum", :].values.size

        # prints number of spectra in each file and artifical x-value
        # print(num_spec,i*np.ones(num_spec))

        # to offset each spectrum on the x-axis, I create an artifical x-axis at the moment, TODO: could be later replaced by the pertubation parmeter (time, concentration)
        ax0.plot(
            i * np.ones(num_spec),
            uv_peak_int_dict["one_wavelength"]["spectrum", :].values,
            "o",
            label=f"{name}",
        )
        ax0.legend()
        ax0.grid(visible=True)
        ax0.set_title(
            f"UV peak intensity for {uv_peak_int_dict['wavelength']} {uv_peak_int_dict['unit']} "
        )

        ax1.plot(
            i * np.ones(num_spec),
            uv_peak_int_dict["wl_interval"]["spectrum", :].values,
            "o",
            label=f"{name}",
        )
        ax1.legend()
        ax1.grid(visible=True)
        ax1.set_title(
            f"UV peak intensity for intervall [{uv_peak_int_dict['wavelength']-uv_peak_int_dict['tol']} , {uv_peak_int_dict['wavelength']+uv_peak_int_dict['tol']}]{uv_peak_int_dict['unit']}"
        )

    # overwrite xticks in ax0
    ax0.set_xticks(
        np.arange(0, len(filelist), 1),
        labels=[f"{name}" for name in filelist],
        rotation=90,
    )
    # overwrite xticks in ax1
    ax1.set_xticks(
        np.arange(0, len(filelist), 1),
        labels=[f"{name}" for name in filelist],
        rotation=90,
    )

    display(fig)


def turbidity(wl, b, m):
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


def residual(p, x, y):
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
    uv_da: sc.DataArray,
    wl_unit=None,
    fit_llim=None,
    fit_ulim=None,
    b_llim=None,
    b_ulim=None,
    m=None,
    plot_corrections=True,
):
    """Fit turbidity to the experimental data. Turbidity: tau=b * wavelength^(-m) Parameters of interest: b, m.
        b is the baseline and m is the slope. b can be obtained by averaging over the flat range of the UV spectrum
        in the higher wavelength range.
        m: make an educated guess. Advice: Limit fitting range to wavelengths after spectroscopy peaks.

    Parameters
    ----------
    uv_da: sc.DataArray
        UV sc.DataArray containing one or more normalized UV spectra
    wl_unit: sc.Unit
        Wavelength unit
    fit_llim: int
        Lower wavelength limit of fit range for turbidity
    fit_ulim: int
        Upper wavelength limit of fit range for turbidity
    b_llim: int
        Lower wavelength limit of fit range for b
    b_ulim: int
        Upper wavelength limit of fit range for b
    m: int
        Educated guess start value of slope parameter in turbidity
    plot_corrections: bool
        If true, plot single contribitions for turbidity corrections

    Returns:
    ----------
    uv_da_turbcorr: sc.DataArray
        uv_da dataarray where each spectrum was corrected for a fitted turbidity, export for all wavelength values

    """
    # obtain unit of wavelength:
    if wl_unit is None:
        wl_unit = uv_da.coords["wavelength"].unit
    else:
        if not isinstance(wl_unit, sc.Unit):
            raise TypeError
        assert (
            wl_unit == uv_da.coords["wavelength"].unit
        )  # we check that the given unit corresponds to the unit for the wavelength

    if fit_llim is None:
        fit_llim = 400
    if fit_ulim is None:
        fit_ulim = 800

    if fit_llim is not None and fit_ulim is not None:
        assert fit_llim < fit_ulim, "fit_llim < fit_ulim"
    if b_llim is not None and b_ulim is not None:
        assert b_llim < b_ulim, "b_llim<b_ulim"

    if m is None:
        m = 0.01

    # select the UV wavelength range for fitting the turbidity
    uv_da_filt = uv_da["wavelength", fit_llim * wl_unit : fit_ulim * wl_unit]

    # How many spectra are in the file?
    num_spec = uv_da_filt.sizes["spectrum"]

    # offset, choose wavelength range for b0 extraction and average over the slected wavelength range
    b0 = (
        uv_da["wavelength", b_llim * wl_unit : b_ulim * wl_unit]
        .mean(dim="wavelength")
        .values
    )
    # create np.ndarray of same shape as b, but with values of m
    m0 = np.full(b0.shape, m)

    # create dummy array
    uv_da_turb_corr_dat = np.zeros(uv_da.data.shape)

    res_popt = np.empty([num_spec, 2])
    res_pcov = np.empty([num_spec, 1])
    # Perform the fitting
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
    # Is this a good way to replace the data in the sc.DataArray? It works and I don't have all methods available like in xarray, but is it slow?
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

    # display(uv_da_turbcorr)

    # Switch on plots for verification
    if plot_corrections:

        # Plotting results as sc.plot
        fig2, ax2 = plt.subplots(ncols=2, figsize=(12, 5))
        out0 = sc.plot(
            sc.collapse(uv_da, keep="wavelength"),
            grid=True,
            title="before correction",
            ax=ax2[0],
        )
        out1 = sc.plot(
            sc.collapse(uv_da_turbcorr, keep="wavelength"),
            grid=True,
            title="after correction",
            ax=ax2[1],
        )

        fig, ax = plt.subplots(ncols=num_spec + 1, figsize=(18, 5))
        out3 = sc.plot(
            sc.collapse(uv_da_filt, keep="wavelength"),
            grid=True,
            title="Selection for turbidity",
            ax=ax[-1],
        )

        for i in range(num_spec):
            # collect the fitting parameters for each spectrum to avoid new fitting
            popt = [
                uv_da_turbcorr.attrs["fit-offset_b"]["spectrum", i].values,
                uv_da_turbcorr.attrs["fit-slope_m"]["spectrum", i].values,
            ]

            ax[i].plot(
                uv_da.coords["wavelength"].values,
                uv_da["spectrum", i].values,
                "s",
                label=f"Full UV raw data {i}",
            )
            ax[i].plot(
                uv_da.coords["wavelength"].values,
                uv_da["spectrum", i].values
                - turbidity(uv_da.coords["wavelength"].values, popt[0], popt[1]),
                "x",
                label=f"Whole UV spectrum, fitted turbidity subtracted {i}",
            )
            ax[i].plot(
                uv_da.coords["wavelength"].values,
                turbidity(uv_da.coords["wavelength"].values, popt[0], popt[1]),
                "^",
                label=f"Full turbidity {i}",
            )

            ax[i].plot(
                uv_da_filt.coords["wavelength"].values,
                turbidity(uv_da_filt.coords["wavelength"].values, popt[0], popt[1]),
                ".",
                label=f"Fitted turbidity {i}, b={popt[0]:.3f}, m={popt[1]:.3f}",
            )

            # ax[i].plot(uv_da['wavelength', b_llim*wl_unit: b_ulim*wl_unit]['spectrum',i].coords['wavelength'].values, uv_da['wavelength',
            #   b_llim*wl_unit: b_ulim*wl_unit]['spectrum',i].values,'v', label=f'Selection for b0 {i}')
            # No need to slice in the spectrum dimension for the x values
            ax[i].plot(
                uv_da["wavelength", b_llim * wl_unit : b_ulim * wl_unit]
                .coords["wavelength"]
                .values,
                uv_da["wavelength", b_llim * wl_unit : b_ulim * wl_unit][
                    "spectrum", i
                ].values,
                "v",
                label=f"Selection for b0 {i}",
            )

            ax[i].grid(True)
            ax[i].set_xlabel("Wavelength [nm]")
            ax[i].set_ylabel("Absorbance")
            ax[i].legend()
            ax[i].set_title(f"Spectrum {str(i)}")
            # set limits, np.isfinite filters out inf (and nan) values
            ax[i].set_ylim(
                [
                    -0.5,
                    1.1
                    * uv_da["spectrum", i]
                    .values[np.isfinite(uv_da["spectrum", i].values)]
                    .max(),
                ]
            )

        # display(fig2)

    return uv_da_turbcorr


def uv_multi_turbidity_fit(
    filelist,
    wl_unit=sc.Unit("nm"),
    fit_llim=300,
    fit_ulim=850,
    b_llim=450,
    b_ulim=700,
    m=0.1,
    plot_corrections=False,
):
    """Applies turbidity correction to uv spectra for a set of  LoKI.nxs files."""

    uv_collection = {}
    for name in filelist:
        uv_dict = load_uv(name)
        uv_da = normalize_uv(**uv_dict)
        uv_da_turbcorr = uv_turbidity_fit(
            uv_da,
            wl_unit=wl_unit,
            fit_llim=fit_llim,
            fit_ulim=fit_ulim,
            b_llim=b_llim,
            b_ulim=b_ulim,
            m=m,
            plot_corrections=plot_corrections,
        )
        # append names as attributes
        uv_da_turbcorr.attrs["name"] = sc.array(
            dims=["spectrum"], values=[name] * uv_da_turbcorr.sizes["spectrum"]
        )
        uv_collection[f"{name}"] = uv_da_turbcorr

        # print(name,uv_da_turbcorr.data.shape,uv_da_turbcorr.data.values.ndim  )

    multi_uv_turb_corr_da = sc.concat(
        [uv_collection[f"{name}"] for name in filelist], dim="spectrum"
    )
    display(multi_uv_turb_corr_da)

    fig, ax = plt.subplots(ncols=3, figsize=(21, 7))
    legend_props = {"show": True, "loc": 1}
    num_spectra = multi_uv_turb_corr_da.sizes["spectrum"]

    out = sc.plot(
        sc.collapse(multi_uv_turb_corr_da, keep="wavelength"),
        grid=True,
        ax=ax[0],
        legend=legend_props,
        title=f"All turbidity corrected UV spectra for {num_spectra} spectra",
    )
    ax[0].set_ylim(
        [
            -1,
            1.2
            * multi_uv_turb_corr_da.data.values[
                np.isfinite(multi_uv_turb_corr_da.data.values)
            ].max(),
        ]
    )

    out2 = sc.plot(
        sc.collapse(multi_uv_turb_corr_da.attrs["fit-offset_b"], keep="spectrum"),
        title=f"All fit-offset b for {num_spectra} spectra",
        ax=ax[1],
        grid=True,
    )

    # ax0.set_xticks(np.arange(0,len(filelist),1), labels=[f'{name}' for name in filelist], rotation=90)
    secx = ax[1].secondary_xaxis(-0.2)
    secx.set_xticks(
        np.arange(0, num_spectra, 1),
        labels=[f"{name}" for name in multi_uv_turb_corr_da.attrs["name"].values],
        rotation=90,
    )
    out3 = sc.plot(
        sc.collapse(multi_uv_turb_corr_da.attrs["fit-slope_m"], keep="spectrum"),
        title=f"All fit-slope m for {num_spectra} spectra",
        ax=ax[2],
        grid=True,
    )
    secx2 = ax[2].secondary_xaxis(-0.2)
    secx2.set_xticks(
        np.arange(0, num_spectra, 1),
        labels=[f"{name}" for name in multi_uv_turb_corr_da.attrs["name"].values],
        rotation=90,
    )

    display(fig)

    return multi_uv_turb_corr_da


def fluo_peak_int(
    fluo_da: sc.DataArray,
    wllim=None,
    wulim=None,
    wl_unit=None,
    medfilter=True,
    kernel_size=None,
):
    """Main task: Extract for a given wavelength range [wllim, wulim]*wl_unit the maximum fluo intensity and its corresponding wavelength position.
    A median filter is automatically applied to the fluo data and data is extracted after its application.
    TODO: Check with Cedric if it is ok to use max intensity values after filtering

    Parameters
    ----------
    fluo_da: sc.DataArray
        DataArray containing uv spectra
    wllim: float
        Wavelength range lower limit
    wulim: float
        Wavelength range upper limit
    wl_unit: sc.Unit
        Unit of the wavelength
    medfilter: boolean
        If medfilter=False, not medfilter is applied. Default: True
    kernel_size: int
        kernel for medianfilter. Default value is applied if no value is given.

    Returns
    ----------
    fluo_filt_max: sc.Dataset
        A new dataset for each spectrum max. intensity value and corresponding wavelength position


    """
    if not isinstance(fluo_da, sc.DataArray):
        raise TypeError("fluo_da has to be an sc.DataArray!")

    # apply medfiler with kernel_size
    if medfilter is True:
        fluo_da = apply_medfilter(fluo_da, kernel_size=kernel_size)

    # obtain unit of wavelength:
    if wl_unit is None:
        wl_unit = fluo_da.coords["wavelength"].unit
    else:
        if not isinstance(wl_unit, sc.Unit):
            raise TypeError
        assert (
            wl_unit == fluo_da.coords["wavelength"].unit
        )  # we check that the given unit corresponds to the unit for the wavelength

    # set defaul values for wllim and wulim for wavelength interval where to search for the maximum
    if wllim is None:
        wllim = 300
    if wulim is None:
        wulim = 400

    assert (
        wllim < wulim
    ), "Lower wavelength limit needs to be smaller than upper wavelength limit"

    # let's go and filter
    # filter spectrum values for the specified interval, filtered along the wavelength dimension
    fluo_filt = fluo_da["wavelength", wllim * wl_unit : wulim * wl_unit]

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
        dims=["spectrum"], values=fluo_filt_max_wl, unit=wl_unit
    )

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


def plot_fluo_peak_int(
    fluo_da: sc.DataArray,
    name,
    wllim=None,
    wulim=None,
    wl_unit=None,
    medfilter=True,
    kernel_size=None,
):
    """Plot max intensity value found in a given wavelength interval and corresponding wavelength, both as function monowavelengths in one file "name" """

    # extract max int value and corresponding wavelength position
    fluo_filt_max = fluo_peak_int(
        fluo_da,
        wllim=wllim,
        wulim=wulim,
        wl_unit=wl_unit,
        medfilter=medfilter,
        kernel_size=kernel_size,
    )
    # attach filename as attribute to dataarray
    fluo_da.attrs["name"] = sc.scalar(name)
    display(fluo_da)

    fig = plt.figure(figsize=(20, 10))
    gs = gridspec.GridSpec(
        nrows=2,
        ncols=3,
        width_ratios=[1, 1, 1],
        wspace=0.3,
        height_ratios=[1, 1],
    )
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])

    out0 = sc.plot(
        fluo_filt_max["intensity_max"]["spectrum", :],
        grid=True,
        labels={"spectrum": "monowavelengths"},
        title=f"Sample: {name}",
        ax=ax0,
    )
    out1 = sc.plot(
        fluo_filt_max["wavelength_max"]["spectrum", :],
        grid=True,
        labels={"spectrum": "monowavelengths"},
        title=f"Sample: {name}",
        ax=ax1,
    )
    display(fig)


def plot_fluo_multiple_peak_int(
    filelist,
    wllim=None,
    wulim=None,
    wl_unit=None,
    medfilter=True,
    kernel_size=None,
):
    """Plot multiple max peak intensities for given wavelength range and corresponding
    position of found maximum for a series of fluo measurements.

    Parameters
    ----------
    filelist: list
        List of complete filenames for LoKI.nxs containing fluo spectra
    wllim: float
        Wavelength range lower limit
    wulim: float
        Wavelength range upper limit
    wl_unit: sc.Unit
        Unit of the wavelength
    medfilter: boolean
        If medfilter=False, no medfilter is applied
    kernel_size: int
        kernel for medianfilter

    Returns
    ----------


    """

    # setting the scene for the markers
    marker = itertools.cycle(markers())

    print(filelist)

    figure_size = (15, 5)
    fig, ax = plt.subplots(
        nrows=1, ncols=2, figsize=figure_size, constrained_layout=True
    )

    unique_mwl = []
    ds_list = []
    for name in filelist:
        fluo_dict = load_fluo(name)
        fluo_da = normalize_fluo(**fluo_dict)
        # extract max int value and corresponding wavelength position, median filter is applied
        fluo_filt_max = fluo_peak_int(
            fluo_da,
            wllim=wllim,
            wulim=wulim,
            wl_unit=wl_unit,
            medfilter=medfilter,
            kernel_size=kernel_size,
        )
        # attach filename as attribute to dataset, TODO: should this happen in fluo_peak_int ?
        fluo_filt_max.attrs["name"] = sc.scalar(name)
        # display(fluo_filt_max)
        ds_list.append(fluo_filt_max)
        unique_mwl.append(np.unique(fluo_filt_max.coords["monowavelengths"].values))
        # print(fluo_filt_max)

        # same marker for both plots for the same file
        markerchoice = next(marker)

        ax[0].plot(
            fluo_filt_max.coords["monowavelengths"].values,
            fluo_filt_max["intensity_max"].values,
            label=f"{name}",
            linestyle="None",
            marker=markerchoice,
            markersize=10,
        )
        ax[0].set_ylabel("Max. Intensity")
        ax[0].set_title("Fluo - max. intensity")

        ax[1].plot(
            fluo_filt_max.coords["monowavelengths"].values,
            fluo_filt_max["wavelength_max"].values,
            label=f"{name}",
            linestyle="None",
            marker=markerchoice,
            markersize=10,
        )
        unit_str = str(fluo_filt_max["wavelength_max"].unit)

        ax[1].set_ylabel(f"Wavelength [{unit_str}]")
        ax[1].set_title("Fluo - corresponding wavelength")

    # show the lowest monowavelength as lower boundary on the y-axis
    ax[1].set_ylim(bottom=0.9 * np.min(fluo_filt_max.coords["monowavelengths"].values))

    # plot the found monowavelengths as additional visual information on the y-axis
    for mwl in np.unique(unique_mwl):
        ax[1].plot(
            np.unique(unique_mwl),
            np.full(np.shape(np.unique(unique_mwl)), mwl),
            "--",
            label=f"{mwl}{sc.Unit('nm')}",
        )
    # ax[1].legend(loc='upper right', bbox_to_anchor=(1.05, 1.05))

    for axes in ax:
        # axes.legend(loc='upper right', bbox_to_anchor=(1.1, 1.00))
        axes.legend(bbox_to_anchor=(1.04, 1))
        axes.grid(True)
        axes.set_xlabel("Monowavelengths")

    display(fig)


def plot_fluo_spectrum_selection(
    flist_num: list,
    spectral_idx: int,
    kernel_size: Optional[int] = None,
    wllim: Optional[float] = None,
    wulim: Optional[float] = None,
    wl_unit: Optional[sc.Unit] = None,
) -> dict:
    """This function extracts a specific fluo spectrum from all given files. Ideally, the user provides the number index.
        A median filter can be applied to the input data. A lower and upper wavelength range will be used for a zoomed-in image.
        Selected spectra are all plotted in one graph.

    Parameters
    ----------
    flist_num: list
        List of LoKI.nxs file containting fluo entry.
    spectral_idx: int
        Index of spectrum for selection
    kernel_size: int
        Scalar giving the size of the median filter window. Elements of kernel_size should be odd. Default size is 3
    wllim: float
        Lower wavelength limit
    wulim: float
        Upper wavelength limit
    wl_unit: sc.Unit
        Wavelength unit of a fluo spectum

    Returns
    ----------
    fluo_spec_idx_ds: sc.Dataset
        sc.Dataset containing selected spectra for each input LoKI.nxs

    """
    # This is maybe not necessary, because scipp catches it later, but I can catch it earlier.
    if not isinstance(spectral_idx, int):
        raise TypeError("Spectral index should be of type int.")

    if wllim is None:
        wllim = 300
    if wulim is None:
        wulim = 400

    # obtain unit of wavelength, I extract it from the first element in the flist_num,
    # but I have to load it
    fluo_dict = load_fluo(flist_num[0])
    final_fluo = normalize_fluo(**fluo_dict)
    if wl_unit is None:
        wl_unit = final_fluo.coords["wavelength"].unit
    else:
        if not isinstance(wl_unit, sc.Unit):
            raise TypeError("wl_unit should be of type sc.Unit.")
        assert (
            wl_unit == final_fluo.coords["wavelength"].unit
        )  # we check that the given unit corresponds to the unit for the wavelength

    # prepare data
    fluo_spec_idx_ds = sc.Dataset()
    for name in flist_num:
        fluo_dict = load_fluo(name)  # this is dictionary with scipp.DataArrays in it
        fluo_normalized = normalize_fluo(**fluo_dict)  # this is one scipp.DataArray
        final_fluo = apply_medfilter(
            fluo_normalized, kernel_size=kernel_size
        )  # apply medfilter
        fluo_spec_idx_ds[name] = final_fluo[
            "spectrum", spectral_idx
        ]  # append data array to dataset

    # Plotting, TODO: should it be decoupled form the loading?
    legend_props = {"show": True, "loc": (1.04, 0)}
    fig = plt.figure(figsize=(17, 5))
    gs = gridspec.GridSpec(nrows=1, ncols=2, width_ratios=[1, 1], wspace=0.5)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    out0 = sc.plot(
        fluo_spec_idx_ds,
        grid=True,
        linestyle="dashed",
        title=f"Normalized fluo spectra, selected spectrum #{spectral_idx}",
        legend=legend_props,
        ax=ax0,
    )  # would also work with fluo_spec_idx_dict
    out1 = sc.plot(
        fluo_spec_idx_ds["wavelength", wllim * wl_unit : wulim * wl_unit],
        grid=True,
        linestyle="dashed",
        title=f"Normalized fluo spectra, selected spectrum #{spectral_idx}",
        legend=legend_props,
        ax=ax1,
    )
    display(fig)

    return fluo_spec_idx_ds
