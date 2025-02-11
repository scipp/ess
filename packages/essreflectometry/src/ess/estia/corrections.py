import scipp as sc


def correct_by_footprint(da: sc.DataArray) -> sc.DataArray:
    "Corrects the data by the size of the footprint on the sample."
    return da / sc.sin(da.coords['theta'])
