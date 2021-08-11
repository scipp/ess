# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import scipp as sc
from .wfm import get_frames


def time_distance_diagram(data: sc.DataArray, **kwargs) -> plt.Figure:
    """
    Plot the time-distance diagram for a WFM beamline.
    The expected input is a Dataset or DataArray containing the chopper cascade
    information as well as the description of the source pulse.
    This internally calls the `get_frames` method which is used to compute the
    frame properties for stitching.
    """

    # Get the frame properties
    frames = get_frames(data, **kwargs)

    # Find detector pixel furthest away from source
    detector_pos_norm = sc.norm(data.meta["position"])
    source_pos = sc.norm(data.meta["source_position"]).value
    furthest_detector_pos = sc.max(detector_pos_norm).value
    pulse_rectangle_height = furthest_detector_pos / 50.0

    # Create figure and axes
    fig, ax = plt.subplots(1, 1, figsize=(9, 7))
    ax.grid(True, color='lightgray', linestyle="dotted")
    ax.set_axisbelow(True)

    # Draw a light grey rectangle from the origin to t_0 + pulse_length + t_0
    # The second t_0 should in fact be the end of the pulse tail, but since this
    # information is not needed for computing the frame properties, it may
    # not be present in the description of the beamline.
    # So we fake this by simply using t_0 again at the end of the pulse.
    ax.add_patch(
        Rectangle((0, source_pos),
                  (2.0 * data.meta["source_pulse_t_0"] +
                   data.meta["source_pulse_length"]).value,
                  -pulse_rectangle_height,
                  lw=1,
                  fc='lightgrey',
                  ec='k',
                  zorder=10))
    # Draw a dark grey rectangle from t_0 to t_0 + pulse_length to represent the usable
    # pulse.
    ax.add_patch(
        Rectangle((data.meta["source_pulse_t_0"].value, source_pos),
                  data.meta["source_pulse_length"].value,
                  -pulse_rectangle_height,
                  lw=1,
                  fc='grey',
                  ec='k',
                  zorder=11))
    # Indicate source pulse and add the duration.
    ax.text(data.meta["source_pulse_t_0"].value,
            -pulse_rectangle_height,
            "Source pulse ({} {})".format(data.meta["source_pulse_length"].value,
                                          data.meta["source_pulse_length"].unit),
            ha="left",
            va="top",
            fontsize=6)

    # Plot the chopper openings as segments
    for name, chopper in data.meta["choppers"].value.items():
        yframe = sc.norm(chopper.position).value
        time_open = chopper.time_open.values
        time_close = chopper.time_close.values
        for fnum in range(len(time_open)):
            ax.plot([time_open[fnum], time_close[fnum]], [yframe] * 2,
                    color="C{}".format(fnum))
        ax.text(2.0 * time_close[-1] - time_open[-1],
                yframe,
                name,
                ha="left",
                va="center")

    # Plot the shades of possible neutron paths
    for i in range(frames.sizes["frame"]):

        col = "C{}".format(i)
        frame = frames["frame", i]
        for dim in data.meta["position"].dims:
            frame = frame[dim, 0]

        # Minimum wavelength
        lambda_min = np.array(
            [[(data.meta["source_pulse_t_0"] + data.meta["source_pulse_length"] -
               frame['delta_time_min']).value, source_pos],
             [(data.meta["source_pulse_t_0"] + data.meta["source_pulse_length"]).value,
              source_pos],
             [(frame["time_min"] + frame["delta_time_min"]).value,
              furthest_detector_pos], [frame["time_min"].value, furthest_detector_pos]])

        # Maximum wavelength
        lambda_max = np.array([[data.meta["source_pulse_t_0"].value, source_pos],
                               [(data.meta["source_pulse_t_0"] +
                                 frame['delta_time_max']).value, source_pos],
                               [frame["time_max"].value, furthest_detector_pos],
                               [(frame["time_max"] - frame["delta_time_max"]).value,
                                furthest_detector_pos]])

        ax.plot(np.concatenate((lambda_min[:, 0], lambda_min[0:1, 0])),
                np.concatenate((lambda_min[:, 1], lambda_min[0:1, 1])),
                color=col,
                lw=1)

        ax.plot(np.concatenate((lambda_max[:, 0], lambda_max[0:1, 0])),
                np.concatenate((lambda_max[:, 1], lambda_max[0:1, 1])),
                color=col,
                lw=1)

        ax.fill(
            [lambda_max[0, 0], lambda_max[-1, 0], lambda_min[2, 0], lambda_min[1, 0]],
            [lambda_max[0, 1], lambda_max[-1, 1], lambda_min[2, 1], lambda_min[1, 1]],
            alpha=0.3,
            color=col,
            zorder=-5)

        ax.fill(lambda_min[:, 0], lambda_min[:, 1], color='w', zorder=-4)
        ax.fill(lambda_max[:, 0], lambda_max[:, 1], color='w', zorder=-4)

        ax.text(sc.mean(sc.concatenate(frame["time_min"], frame["time_max"],
                                       'none')).value,
                furthest_detector_pos,
                "Frame {}".format(i + 1),
                ha="center",
                va="top")

    # Add thick solid line for the detector position, spanning the entire width
    ax.plot([0, sc.max(frames["time_max"].data).value], [furthest_detector_pos] * 2,
            lw=3,
            color='grey')
    ax.text(0.0, furthest_detector_pos, "Detector", va="bottom", ha="left")

    # Set axis labels
    ax.set_xlabel("Time [microseconds]")
    ax.set_ylabel("Distance [m]")

    return fig
