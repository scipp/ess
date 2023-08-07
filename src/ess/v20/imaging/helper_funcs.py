# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
import csv
import glob
import os
import re

import numpy as np
import scipp as sc


def read_x_values(tof_file, **kwargs):
    """
    Reads the TOF values from the CSV into a list.
    If usecols is defined, we use the column requested by the user.
    If not, as there may be more than one column in the file (typically the
    counts are also stored in the file alongside the TOF bins), we search for
    the first column with monotonically increasing values.
    The first argument is the name of the file to be loaded.
    All subsequent arguments are forwarded to numpy's loadtxt.
    (see https://numpy.org/doc/stable/reference/generated/numpy.loadtxt.html).

    :param tof_file: Name of the file to be read.
    :type tof_file: str
    """
    data = np.loadtxt(tof_file, **kwargs)
    if data.ndim == 1:
        return data

    # Search for the first column with monotonically increasing values
    for i in range(data.shape[1]):
        if np.all(data[1:, i] > data[:-1, i], axis=0):
            return data[:, i]

    raise RuntimeError(
        "Cannot automatically determine time-of-flight column. "
        "No column with monotonically increasing values was "
        "found in file " + tof_file
    )


def _load_images(image_dir, extension, loader):
    image_dir = os.path.abspath(image_dir)
    if not os.path.isdir(image_dir):
        raise ValueError(image_dir + " is not directory")
    filenames = glob.glob(image_dir + f"/*.{extension}")
    # Sort the filenames by converting the digits in the strings to integers
    filenames.sort(key=lambda f: int(re.sub(r'\D', '', f)))
    return loader(filenames)


def _load_fits(image_dir):
    def loader(filenames):
        from astropy.io import fits

        stack = []
        path_length = len(filenames) + 1
        nfiles = len(filenames)
        count = 0
        print(f"Loading {nfiles}'")
        for filename in filenames:
            count += 1
            print(
                '\r{0}: Image {1}, of {2}'.format(
                    filename[path_length:], count, nfiles
                ),
                end="",
            )
            img = None
            handle = fits.open(filename, mode='readonly')
            img = handle[0].data.copy()
            handle.close()
            stack.append(np.flipud(img.data))
        return np.array(stack)

    return _load_images(image_dir, 'fits', loader)


def _load_tiffs(tiff_dir):
    import tifffile

    return _load_images(tiff_dir, 'tiff', lambda f: tifffile.imread(f))


def export_tiff_stack(dataset, key, base_name, output_dir, x_len, y_len, tof_values):
    import tifffile

    to_save = dataset[key]

    num_bins = 1 if len(to_save.shape) == 1 else to_save.shape[0]
    stack_data = np.reshape(to_save.values, (x_len, y_len, num_bins))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Writing tiffs
    for i in range(stack_data.shape[2]):
        tifffile.imsave(
            os.path.join(output_dir, '{:s}_{:04d}.tiff'.format(base_name, i)),
            stack_data[:, :, i].astype(np.float32),
        )
    print('Saved {:s}_{:04d}.tiff stack.'.format(base_name, 0))

    # Write out tofs as CSV
    tof_vals = [tof_values[0], tof_values[-1]] if num_bins == 1 else tof_values

    with open(
        os.path.join(output_dir, 'tof_of_tiff_{}.txt'.format(base_name)), 'w'
    ) as tofs:
        writer = csv.writer(tofs, delimiter='\t')
        writer.writerow(['tiff_bin_nr', 'tof'])
        tofs = tof_vals
        tof_data = list(zip(list(range(len(tofs))), tofs))
        writer.writerows(tof_data)
    print('Saved tof_of_tiff_{}.txt.'.format(base_name))


def _image_to_variable(
    image_dir, loader, dtype=np.float64, with_variances=True, reshape=False
):
    """
    Loads all images from the directory into a scipp Variable.
    """
    stack = loader(image_dir)

    if stack.size == 0:
        raise RuntimeError(f'No image files found in {image_dir}')

    data = stack.astype(dtype)
    if reshape:
        data = data.reshape(stack.shape[0], stack.shape[1] * stack.shape[2])
        dims = ["t", "spectrum"]
    else:
        dims = ["t", "y", "x"]
    var = sc.Variable(dims=dims, values=data, unit=sc.units.counts)
    if with_variances:
        var.variances = data
    return var


def tiffs_to_variable(image_dir, dtype=np.float64, with_variances=True, reshape=False):
    """
    Loads all tiff images from the directory into a scipp Variable.
    """
    return _image_to_variable(
        image_dir, _load_tiffs, dtype, with_variances, reshape=reshape
    )


def fits_to_variable(fits_dir, dtype=np.float64, with_variances=True, reshape=False):
    """
    Loads all fits images from the directory into a scipp Variable.
    """
    return _image_to_variable(
        fits_dir, _load_fits, dtype, with_variances, reshape=reshape
    )


def make_detector_groups(nx_original, ny_original, nx_target, ny_target):
    element_width_x = nx_original // nx_target
    element_width_y = ny_original // ny_target

    x = sc.Variable(dims=['x'], values=np.arange(nx_original) // element_width_x)
    y = sc.Variable(dims=['y'], values=np.arange(ny_original) // element_width_y)
    grid = x + nx_target * y
    return sc.Variable(["spectrum"], values=np.ravel(grid.values))
