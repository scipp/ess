import scipp as sc


def maximum_resolution_achievable(
    events: sc.DataArray,
    coarse_x_bin_edges: sc.Variable,
    coarse_y_bin_edges: sc.Variable,
    time_bin_edges: sc.Variable,
    max_tries: int = 10,
    max_pixels_x: int = 2048,
    max_pixels_y: int = 2048,
    raise_if_not_maximum: bool = False,
):
    """
    Estimates the maximum resolution achievable
    given a desired binning in time.
    The maximum achievable resolution is defined
    as the resolution in ``xy`` such that
    there is at least one event in every ``xyt`` pixel.

    Parameters
    -------------
    events:
        1D DataArray containing events with associated x, y, and t coordinates.
        The names of the coordinates must not be `x`, `y` and `t`,
        the names of the coordinates are taken from the provided ``bin_edges``
        for each respective dimension.
    coarse_x_bin_edges:
        Minimum acceptable resolution in ``x``.
    coarse_y_bin_edges:
        Minimum acceptable resolution in ``y``.
    time_bin_edges:
        Desired resolution in ``t``.
    max_tries:
        The maximum number of iterations before giving up.
    max_pixels_x:
        The maximum number of pixels in ``x``.
    max_pixels_y:
        The maximum number of pixels in ``y``.
    raise_if_not_maximum:
        Often it is not important to find the exact maximum resolution.
        Therefore this parameter is ``False`` by default, and the function
        returns an estimate of the maximum resolution.
        If you want the returned resolution to be exactly the maximum resolution,
        set the value of this parameter to ``True``.

    Returns
    -------------
        The bin edges in x respectively y that define the
        maximum achievable resolution.
    """

    lower_nx = coarse_x_bin_edges.size
    lower_ny = coarse_y_bin_edges.size
    upper_nx = max_pixels_x
    upper_ny = max_pixels_y

    nx = int(2**0.5 * lower_nx) + 1
    ny = int(2**0.5 * lower_ny) + 1
    events = events.bin({time_bin_edges.dim: time_bin_edges})

    for _ in range(max_tries):
        xbins = sc.linspace(
            coarse_x_bin_edges.dim, coarse_x_bin_edges[0], coarse_x_bin_edges[-1], nx
        )
        ybins = sc.linspace(
            coarse_y_bin_edges.dim, coarse_y_bin_edges[0], coarse_y_bin_edges[-1], ny
        )
        min_counts_per_pixel = (
            events.bin(
                {
                    coarse_x_bin_edges.dim: xbins,
                    coarse_y_bin_edges.dim: ybins,
                }
            )
            .bins.size()
            .min()
        )

        if min_counts_per_pixel.value > 0:
            lower_nx = nx
            lower_ny = ny
            nx = max(min(round((upper_nx * nx) ** 0.5), nx * 2), lower_nx + 1)
            ny = max(min(round((upper_ny * ny) ** 0.5), ny * 2), lower_ny + 1)
        else:
            upper_nx = nx
            upper_ny = ny
            nx = min(round((lower_nx * nx) ** 0.5), upper_nx - 1)
            ny = min(round((lower_ny * ny) ** 0.5), upper_nx - 1)

        if upper_nx - lower_nx < 2 and upper_ny - lower_ny < 2:
            break

    if raise_if_not_maximum and upper_nx - lower_nx >= 2 and upper_ny - lower_ny >= 2:
        raise RuntimeError(
            'Maximal resolution was not found. Increase `max_tries` to search longer. '
            'Or set `raise_if_not_maximum=False` if it is not necessary to locate the '
            'maximum exactly.'
        )

    return (
        sc.linspace(
            coarse_x_bin_edges.dim,
            coarse_x_bin_edges[0],
            coarse_x_bin_edges[-1],
            lower_nx,
        ),
        sc.linspace(
            coarse_y_bin_edges.dim,
            coarse_y_bin_edges[0],
            coarse_y_bin_edges[-1],
            lower_ny,
        ),
    )
