# standard library imports
import itertools
import os
from typing import Optional, Type

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
from scipp.ndimage import median_filter

from ess.loki.nurf import uv, fluo, utils


def markers(number_lines):
    """Creates a list of markers for plots"""
    marker_list=["+", ".", "o", "*", "1", "2", "x", "P", "v", "X", "^", "d", "3", "4"]
    repetition= int(np.ceil(number_lines/marker_list.__len__()))
    return marker_list*repetition

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
            pl1.ax.get_lines()[i].set_marker(markers(number_lines)[i])
            pl1.ax.get_lines()[i].set_markersize(5)
            pl1.ax.get_lines()[i].set_markevery(5)
            pl1.ax.get_legend().legendHandles[i].set_marker(markers(number_lines)[i])
            pl1.ax.get_lines()[i].set_linewidth(1)


################################################
#
#  UV plot functions
#
################################################

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
    uv_dict = utils.load_nurfloki_file(name, 'uv')
    normalized = uv.normalize_uv(**uv_dict)  # results in DataArrays with multiple spectra
    #normalized =  uv.load_and_normalize_uv(name)

    # legend property
    legend_props = {"show": True, "loc": 1}
    figs, axs = plt.subplots(1, 4, figsize=(16, 3))

    # How to plot raw data spectra in each file?
    #TODO: You could put all the common options (of the 4 plots) into a dict, 
    # and then use **options here, do avoid the duplication and make changes easier.
    # --> I am not sure currently, I wait for Simon's advice

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
    #out4 = sc.plot(
    #    {"dark": uv_dict["dark"], "reference": uv_dict["reference"]},
    #    linestyle="dashed",
    #    marker=".",
    #    grid=True,
    #    legend=legend_props,
    #    title=f"{name} - dark and reference",
    #    ax=axs[1],
    #)
    # How to plot dark and reference?
    to_plot = {}
    for group in ('dark', 'reference'):
        for key, da in sc.collapse(uv_dict[group]["spectrum", :], keep="wavelength").items():
            to_plot[f'{group}-{key}'] = da

    out4= sc.plot(to_plot, 
        linestyle="dashed",
        grid=True,
        marker='.',
        legend=legend_props,
        #figsize=figure_size2,
        title=f"{name}, dark and reference spectra - all",
        ax=axs[1],
        )
    #display(out4)


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
    #display(out2)

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
    #display(out3)
    #display(figs)

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
    medfilter: bool
        If medfilter=False, not medfilter is applied. Default: True
        A medfilter is applied to the fluo spectra as fluo is often more noisy
    kernel_size: int or sc.Variable
        kernel for median_filter along the wavelength dimension. Expected dims
        'spectrum' and 'wavelength' in sc.DataArray

    Returns
    ----------
    uv_peak_int: dict
        Dictionary that contains the peak intensity for the requested wavelength, the peak intensity averaged over the requested interval, the requested wavelength with its unit, and the tolerance

    """
    if not isinstance(uv_da, sc.DataArray):
        raise TypeError

    if medfilter is not False:
        # apply medfiler with kernel_size along the wavelength dimension
        if ('spectrum' and 'wavelength') in fluo_da.dims:
            kernel_size_sc={'spectrum':1, 'wavelength':kernel_size}
        else:
            raise ValueError('Dimensions spectrum and wavelength expected.')
        uv_da=median_filter(uv_da, size=kernel_size_sc)


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
    uv_dict = {name: uv.load_and_normalize_uv(name) for name in filelist}

    # apply medianfilter
    if medfilter is not False:
        # apply medfilter with kernel_size along the wavelength dimension
        if ('spectrum' and 'wavelength') in fluo_da.dims:
            kernel_size_sc={'spectrum':1, 'wavelength':kernel_size}
        else:
            raise ValueError('Dimensions spectrum and wavelength expected.')
        uv_dict = {
            name: median_filter(uv_dict[name], size=kernel_size_sc)
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
    fig = plt.figure(figsize=(24, 7))
    gs = gridspec.GridSpec(
        nrows=1, ncols=2, width_ratios=[1, 1], wspace=0.2, height_ratios=[1], top=0.9, bottom=0.15, left=0.1, right=0.9
    )
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])



    for i, name in enumerate(filelist):
        uv_dict= utils.load_nurfloki_file(name, 'uv') 
        uv_da = uv.normalize_uv(**uv_dict)

        # check for medfilter
        if medfilter is not False:
            # apply medianfilter
            uv_da = utils.nurf_median_filter( uv_da,kernel_size=kernel_size )

        # process
        uv_peak_int_dict=uv.uv_peak_int(uv_da , wavelength=wavelength, tol=tol)
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
        ax0.legend(  loc="best")
        ax0.set_ylabel("Abs")
        ax0.grid(visible=True)
        ax0.set_title(
            f"UV peak intensity for {uv_peak_int_dict['wavelength'].value} {uv_peak_int_dict['wavelength'].unit} "
        )

        ax1.plot(
            i * np.ones(num_spec),
            uv_peak_int_dict["wl_interval"]["spectrum", :].values,
            "o",
            label=f"{name}",
        )
        ax1.legend(loc="best")
        ax1.set_ylabel("Abs")
        ax1.grid(visible=True)
        ax1.set_title(
            f"UV peak intensity for intervall [{uv_peak_int_dict['wavelength'].value-uv_peak_int_dict['tol'].value} , {uv_peak_int_dict['wavelength'].value+uv_peak_int_dict['tol'].value}]{uv_peak_int_dict['wavelength'].unit}"
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



    # a quick hack for the poster
    cm = 1/2.54  # centimeters in inches
    #figsize_b=8
    #figsize_a=1.333*figsize_b
    figsize_a=15
    figsize_b= figsize_a*0.75
    
    fig3, ax0 = plt.subplots(1, 1, constrained_layout=True, figsize=(figsize_a*cm, figsize_b*cm) )

    
    for i, name in enumerate(filelist):
        uv_dict= utils.load_nurfloki_file(name, 'uv') 
        uv_da = uv.normalize_uv(**uv_dict)

        # check for medfilter
        if medfilter is not False:
            # apply medianfilter
            uv_da = utils.nurf_median_filter( uv_da,kernel_size=kernel_size )

        # process
        uv_peak_int_dict=uv.uv_peak_int(uv_da , wavelength=wavelength, tol=tol)
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
        ax0.legend( bbox_to_anchor=(1.05, 1.03), loc="upper left" )
        ax0.set_ylabel("Abs")
        ax0.grid(visible=True)
        ax0.set_title(
            f"UV peak intensity for {uv_peak_int_dict['wavelength'].value} {uv_peak_int_dict['wavelength'].unit} "
        )


    # overwrite xticks in ax0
    ax0.set_xticks(
        np.arange(0, len(filelist), 1),
        labels=[f"{name}" for name in filelist],
        rotation=90,
    )
  

    return fig3
    #display(fig)


def plot_fluo_peak_int(
    fluo_da: sc.DataArray,
    wllim: Optional[sc.Variable] = None,
    wulim: Optional[sc.Variable] = None,
    medfilter=True,
    kernel_size=15,
):
    """Plot max intensity value found in a given wavelength interval and 
    corresponding wavelength, both as function monowavelengths in one 
    file "name"
    
    """

    # extract max int value and corresponding wavelength position
    fluo_filt_max = fluo.fluo_peak_int(
        fluo_da,
        wllim=wllim,
        wulim=wulim,
        medfilter=medfilter,
        kernel_size=kernel_size,
    )
    # attach filename as attribute to dataarray
    #fluo_da.attrs["name"] = sc.scalar(name) #not necessary anymore
    #print('This is the fluo_da scipp array received as input.')
    #display(fluo_da)

    source = np.unique(fluo_da.attrs['source'].values)
    name = (lambda x: x)(*source)   #https://stackoverflow.com/questions/33161448/getting-only-element-from-a-single-element-list-in-python


    fig = plt.figure(figsize=(22, 7))
    gs = gridspec.GridSpec(
        nrows=1,
        ncols=2,
        width_ratios=[1, 1],
        wspace=0.3,
        height_ratios=[1],
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

    x = np.arange(len(fluo_filt_max.coords['monowavelengths'].values))  # the label locations
    out0.ax.set_xticks(x)
    # Set ticks labels for x-axis
    out0.ax.set_xticklabels(fluo_filt_max.coords['monowavelengths'].values)

    out1 = sc.plot(
        fluo_filt_max["wavelength_max"]["spectrum", :],
        grid=True,
        labels={"spectrum": "monowavelengths"},
        title=f"Sample: {name}",
        ax=ax1,
    )
    out1.ax.set_xticks(x)
    out1.ax.set_xticklabels(fluo_filt_max.coords['monowavelengths'].values)
    #display(fig)


def plot_fluo_multiple_peak_int(
    filelist: list,
    wllim: Optional[sc.Variable] = None,
    wulim: Optional[sc.Variable] = None,
    medfilter=True,
    kernel_size: Optional[int] = None,
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
    medfilter: boolean
        If medfilter=False, no medfilter is applied
    kernel_size: int
        kernel for medianfilter

    Returns
    ----------

    """

    # setting the scene for the markers
    marker = itertools.cycle(markers(15))

    print(filelist)

    #mpr,a;
    figure_size = (17, 5)
    fig, ax = plt.subplots(
        nrows=1, ncols=2, figsize=figure_size, constrained_layout=True
    )

    #poster hack
    #cm = 1/2.54  # centimeters in inches
    #figsize_b=8
    #figsize_a=1.333*figsize_b
    #figsize_a=13
    #figsize_b= figsize_a*1.6
    
    #fig, ax = plt.subplots(2, 1, constrained_layout=True, figsize=(figsize_a*cm, figsize_b*cm) )


    unique_mwl = []
    ds_list = []
    for name in filelist:
        fluo_dict=utils.load_nurfloki_file(name,'fluorescence')
        fluo_da = fluo.normalize_fluo(**fluo_dict)
        # extract max int value and corresponding wavelength position, median filter is applied
        fluo_filt_max = fluo.fluo_peak_int(
            fluo_da,
            wllim=wllim,
            wulim=wulim,
            medfilter=medfilter,
            kernel_size=kernel_size,
        )
        # attach filename as attribute to dataset, TODO: should this happen in fluo_peak_int ?
        #fluo_filt_max.attrs["name"] = sc.scalar(name)
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
        #ax[0].set_title("Fluo - max. intensity")
        ax[0].set_title(r'Fluo - $\mathrm{Int}_{\mathrm{max}}$')

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
        #ax[1].set_title("Fluo - corresponding wavelength")
        ax[1].set_title(r'Fluo - $\lambda_{\mathrm{max}}$')

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
        #axes.legend(loc='upper right', bbox_to_anchor=(1.1, 1.00))
        axes.legend(loc='best')#, bbox_to_anchor=(1.06, 1.03))
        axes.grid(True)
        axes.set_xlabel("Monowavelengths [nm]")


    #return fig
    #display(fig)


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
    fluo_dict=utils.load_nurfloki_file(flist_num[0],'fluorescence')
    final_fluo = fluo.normalize_fluo(**fluo_dict)
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
        fluo_dict=utils.load_nurfloki_file(name,'fluorescence') 
        fluo_normalized = fluo.normalize_fluo(**fluo_dict)
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



def plot_uv_turbidity_fit(uv_da: sc.DataArray, 
    fit_llim: Optional[sc.Variable] = None,
    fit_ulim: Optional[sc.Variable] = None,
    b_llim: Optional[sc.Variable] = None,
    b_ulim: Optional[sc.Variable] = None,
    m=None):

    # carry out the fit
    uv_da_turbcorr= uv.uv_turbidity_fit(
    uv_da, 
    fit_llim=fit_llim, 
    fit_ulim=fit_ulim,
    b_llim=b_llim,
    b_ulim=b_ulim,
    m=m
    )
    # get the values for the fit ranges
    fit_llim, fit_ulim=uv_da_turbcorr.attrs["turbidity_fit_range"]
    b_llim, b_ulim= uv_da_turbcorr.attrs["b_fit_range"]
  
    # select the UV wavelength range for fitting the turbidity
    uv_da_filt = uv_da["wavelength", fit_llim  : fit_ulim ]

    # How many spectra are in the file?
    num_spec = uv_da_filt.sizes["spectrum"]

 
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

    # Plotting intermediate steps
    fig, ax = plt.subplots(ncols=num_spec + 1, figsize=(18, 5))
    out3 = sc.plot(
        sc.collapse(uv_da_filt, keep="wavelength"),
        grid=True,
        title="Selection for turbidity fit",
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
            - uv.turbidity(uv_da.coords["wavelength"].values, popt[0], popt[1]),
            "x",
            label=f"Whole UV spectrum, fitted turbidity subtracted {i}",
        )
        ax[i].plot(
            uv_da.coords["wavelength"].values,
            uv.turbidity(uv_da.coords["wavelength"].values, popt[0], popt[1]),
            "^",
            label=f"Full turbidity {i}",
        )

        ax[i].plot(
            uv_da_filt.coords["wavelength"].values,
            uv.turbidity(uv_da_filt.coords["wavelength"].values, popt[0], popt[1]),
            ".",
            label=f"Fitted turbidity {i}, b={popt[0]:.3f}, m={popt[1]:.3f}, [{fit_llim.value}:{fit_ulim.value}]{fit_llim.unit}",
        )

        # ax[i].plot(uv_da['wavelength', b_llim*wl_unit: b_ulim*wl_unit]['spectrum',i].coords['wavelength'].values, uv_da['wavelength',
        #   b_llim*wl_unit: b_ulim*wl_unit]['spectrum',i].values,'v', label=f'Selection for b0 {i}')
        # No need to slice in the spectrum dimension for the x values
        ax[i].plot(
            uv_da["wavelength", b_llim : b_ulim ]
            .coords["wavelength"]
            .values,
            uv_da["wavelength", b_llim:b_ulim]["spectrum", i].values,
            "v",
            label=f"Selection for b0 {i}: [{b_llim.value}:{b_ulim.value}]{b_llim.unit}",
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

    display(fig2)

def plot_uv_multi_turbidity_fit(
    filelist,
    fit_llim: Optional[sc.Variable] = None,
    fit_ulim: Optional[sc.Variable] = None,
    b_llim: Optional[sc.Variable] = None,
    b_ulim: Optional[sc.Variable] = None,
    m=None 
):

    multi_uv_turb_corr_da=uv.uv_multi_turbidity_fit(filelist,
    fit_llim=fit_llim,
    fit_ulim=fit_ulim,
    b_llim=b_llim,
    b_ulim=b_ulim,
    m=m )

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
        labels=[f"{name}" for name in multi_uv_turb_corr_da.attrs["source"].values],
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
        labels=[f"{name}" for name in multi_uv_turb_corr_da.attrs["source"].values],
        rotation=90,
    )





################################################
#
#  Fluo plot functions
#
################################################


def plot_fluo(name: str):
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

    fluo_dict=utils.load_nurfloki_file(name,'fluorescence')  
    final_fluo = fluo.normalize_fluo(**fluo_dict)  

    # set figure size and legend props
    figure_size = (8, 4)
    legend_props = {"show": True, "loc": (1.04, 0)}

    all_fluo_raw_spectra={}
    #should this be more consistent? 
    #possible via fluo_dict["sample"] or final_fluo
    for i in range(1, fluo_dict["sample"].sizes["spectrum"]):
        mwl = str(final_fluo.coords["monowavelengths"][i].value) + "nm"
        all_fluo_raw_spectra[f"spectrum:{i}, {mwl}"] = final_fluo["spectrum", i]

    # plot all fluo raw spectra
    out1 = sc.plot(
        #sc.collapse(fluo_dict["sample"]["spectrum", :], keep="wavelength"),
        all_fluo_raw_spectra,
        linestyle="dashed",
        grid=True,
        marker='.',
        legend=legend_props,
        figsize=figure_size,
        title=f"{name}, raw fluo spectrum - all",
    )  
    display(out1)

    # plot raw and dark spectra for fluo part
    figure_size2 = (10, 6)
    legend_props = {"show": True, "loc": (1.04, 0)}

    to_plot = {}
    for group in ('dark', 'reference'):
        for key, da in sc.collapse(fluo_dict[group]["spectrum", :], keep="wavelength").items():
            to_plot[f'{group}-{key}'] = da
    out2= sc.plot(to_plot, 
        linestyle="dashed",
        grid=True,
        marker='.',
        legend=legend_props,
        figsize=figure_size2,
        title=f"{name}, dark and reference spectra - all"
        )
    display(out2)

    # specific for ILL data, every second sppectrum good, pay attention to range() where
    #  the selection takes place
    only_bad_spectra = {}  # make empty dict
    for i in range(0, fluo_dict["sample"].sizes["spectrum"], 2):
        only_bad_spectra[f"spectrum-{i}, {mwl}"] = fluo_dict["sample"]["spectrum", i]
    out3 = sc.plot(
        only_bad_spectra,
        linestyle="dashed",
        grid=True,
        marker='.',
        legend=legend_props,
        figsize=figure_size,
        title=f"{name} - all bad raw spectra",
    )
    display(out3)

    only_good_spectra = {}  # make empty dict
    for i in range(1, fluo_dict["sample"].sizes["spectrum"], 2):
        only_good_spectra[f"spectrum-{i}, {mwl}"] = fluo_dict["sample"]["spectrum", i]
    out4 = sc.plot(
        only_good_spectra,
        linestyle="dashed",
        grid=True,
        marker='.',
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
        marker='.',
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
        marker='.',
        legend=legend_props,
        figsize=figure_size,
        title=f"{name} - all good final spectra",
    )
    lambda_min = 250
    lambda_max = 600
    out6.ax.set_xlim(lambda_min, lambda_max)
    display(out6)

    fig_handles = [out1, out2, out3, out4, out5, out5, out6]
    #fig_handles=[out3]
    modify_plt_app(fig_handles)


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


