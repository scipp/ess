# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

"""Grouping helpers for POWTEX texture reduction."""

from .types import (
    DetectorView,
    DetectorViewData
)

import plotly.graph_objects as go
from plotly.subplots import make_subplots

import plotly.express as px
import numpy as np


def detector_plot(
    detector_view_data: DetectorViewData,
) -> DetectorView:
    """Plot the detector view data."""

    gamma = detector_view_data.coords["gamma"].values
    two_theta = detector_view_data.coords["two_theta"].values

    counts = detector_view_data.data.values
    x = detector_view_data.coords["x"].values
    y = detector_view_data.coords["y"].values
    z = detector_view_data.coords["z"].values

    hover_3D = '<br>'.join(['2\u03b8: %{customdata[0]}\u00B0',
                        '\u03b3: %{customdata[1]}\u00B0',
                        'z: %{customdata[2]:.3f} mm',
                        'Counts: %{customdata[3]}',])

    fig = go.Figure(go.Surface(
        x=np.broadcast_to(x[:, None], counts.shape),
        y=np.broadcast_to(y[:, None], counts.shape),
        z=np.broadcast_to(z[None, :], counts.shape),
        customdata=np.dstack((np.broadcast_to(two_theta[None, :], counts.shape).T,
                                np.broadcast_to(gamma[:, None], counts.shape).T,
                                np.broadcast_to(z[None, :], counts.shape).T,
                                counts.T)),
        hovertemplate=hover_3D + '<extra></extra>', #extra removes trace
        colorscale='Portland',
        surfacecolor=counts
    ))

    # Set axis labels
    # the default view is set here
    my_camera = dict(
        up=dict(x=0, y=1, z=0),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=0.25, y=0.75, z=-1.75)
    )

    scene_dict = dict(
        xaxis_title='x, mm',
        yaxis_title='y, mm',
        zaxis_title='z, mm',
        #xaxis=dict(range=[-800, 800]),
        #yaxis=dict(range=[-800, 800]),
        #zaxis=dict(range=[-800, 800])
    )

    fig.update_layout(
        scene=scene_dict,
        title='3D Surface Plot',
        scene_camera=my_camera
    )

    return DetectorView(fig.show())


providers = (
    detector_plot
)
