# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

"""Utilities for loading example data for POWGEN."""

from pathlib import Path

import sciline as sl
import scipp as sc
import scippneutron as scn
import scippnexus as snx
import os
from ess.packages.essdiffraction.src.ess.powder.types import (
    AccumulatedProtonCharge,
    CalibrationData,
    CalibrationFilename,
    DetectorBankSizes,
    ElasticCoordTransformGraph,
    Filename,
    GravityVector,
    MonitorCoordTransformGraph,
    MonitorType,
    Position,
    ProtonCharge,
    RawDataAndMetadata,
    RunType,
    GeneralRunType,
    TofDetector,
    WavelengthDetector,
    WavelengthMonitor,
    FilenameMLZ,
    FilepathMLZ,
    CalibrationRun,
    MaskingData,
    TofMask,
    TwoThetaMask,
    WavelengthMask,
    MaskedDetectorIDs,
    CorrectedDspacing
    #RawTofDataMLZ
)

from ess.packages.essdiffraction.src.ess.powder.masking import with_pixel_mask_filenames
from ess.packages.essreduce.src.ess.reduce.data import Entry, make_registry


# class TofDetector(sl.Scope[RunType, sc.DataArray], sc.DataArray):
#     """
#     Detector with a time-of-flight coordinate
#     """


# class TofMonitor(sl.Scope[RunType, MonitorType, sc.DataArray], sc.DataArray):
#     """Monitor data with time-of-flight coordinate."""


_registry = make_registry(
    "ess/powgen",
    version="1",
    files={
        # Files loadable with Mantid
        "PG3_4844_event.nxs": "md5:d5ae38871d0a09a28ae01f85d969de1e",
        "PG3_4866_event.nxs": "md5:3d543bc6a646e622b3f4542bc3435e7e",
        "PG3_5226_event.nxs": "md5:58b386ebdfeb728d34fd3ba00a2d4f1e",
        "PG3_FERNS_d4832_2011_08_24.cal": "md5:c181221ebef9fcf30114954268c7a6b6",
        # Zipped Scipp HDF5 files
        "PG3_4844_event.zip": Entry(
            alg="md5", chk="a644c74f5e740385469b67431b690a3e", unzip=True
        ),
        "PG3_4866_event.zip": Entry(
            alg="md5", chk="5bc49def987f0faeb212a406b92b548e", unzip=True
        ),
        "PG3_FERNS_d4832_2011_08_24.zip": Entry(
            alg="md5", chk="0fef4ed5f61465eaaa3f87a18f5bb80d", unzip=True
        ),
        "PG3_FERNS_d4832_2011_08_24_spectrum.h5": "md5:7aee0b40deee22d57e21558baa7a6a1a",  # noqa: E501
        # Smaller files for unit tests
        "TEST_PG3_4844_event.h5": "md5:03ea1018c825b0d90a67b4fc7932ea3d",
        "TEST_PG3_4866_event.h5": "md5:4a8df454871cec7de9ac624ebdc97095",
        "TEST_PG3_FERNS_d4832_2011_08_24_spectrum.h5": "md5:b577a054e9aa7b372df79bf4489947d0",  # noqa: E501
    },
)


def powgen_tutorial_mantid_sample_file() -> Path:
    return _registry.get_path("PG3_4844_event.nxs")


def powgen_tutorial_mantid_vanadium_file() -> Path:
    return _registry.get_path("PG3_4866_event.nxs")


def powgen_tutorial_mantid_empty_instrument_file() -> Path:
    return _registry.get_path("PG3_5226_event.nxs")


def powgen_tutorial_mantid_calibration_file() -> Path:
    return _registry.get_path("PG3_FERNS_d4832_2011_08_24.cal")


def powgen_tutorial_sample_file(*, small: bool = False) -> Path:
    """
    Return the path to the POWGEN sample file.

    Parameters
    ----------
    small:
        If True, return a smaller file for unit tests.
        The small version of the file was created using the following code, which keeps
        only 7 columns out of 154 (154 / 7 = 22):

        .. code-block:: python

            import scipp as sc

            fname = 'PG3_4844_event.h5'
            dg = sc.io.load_hdf5(fname)

            sizes = {"bank": 23, "column": 154, "row": 7}

            def foldme(x, dim):
                return x.fold(dim=dim, sizes=sizes)['column', ::22].flatten(
                    dims=list(sizes.keys()), to=dim)

            small = sc.DataGroup({
                'data': foldme(dg['data'], 'spectrum'),
                'detector_info': sc.Dataset(
                    coords={key: foldme(c, 'detector')
                    for key, c in dg['detector_info'].coords.items()})
            })
            sc.io.save_hdf5(small, 'TEST_PG3_4844_event.h5')
    """
    prefix = "TEST_" if small else ""
    ext = ".h5" if small else ".zip"
    return _registry.get_path(f"{prefix}PG3_4844_event{ext}")


def powgen_tutorial_vanadium_file(*, small: bool = False) -> Path:
    """
    Return the path to the POWGEN vanadium file.

    Parameters
    ----------
    small:
        If True, return a smaller file for unit tests.
        The small version of the file was created using the following code, which keeps
        only 7 columns out of 154 (154 / 7 = 22):

        .. code-block:: python

            import scipp as sc

            fname = 'PG3_4866_event.h5'
            dg = sc.io.load_hdf5(fname)

            sizes = {"bank": 23, "column": 154, "row": 7}

            def foldme(x, dim):
                return x.fold(dim=dim, sizes=sizes)['column', ::22].flatten(
                    dims=list(sizes.keys()), to=dim)

            small = sc.DataGroup({
                'data': foldme(dg['data'], 'spectrum'),
                'proton_charge': dg['proton_charge']['pulse_time', ::10]
            })
            sc.io.save_hdf5(small, 'TEST_PG3_4866_event.h5')
    """
    prefix = "TEST_" if small else ""
    ext = ".h5" if small else ".zip"
    return _registry.get_path(f"{prefix}PG3_4866_event{ext}")


def powgen_tutorial_calibration_file(*, small: bool = False) -> Path:
    """
    Return the path to the POWGEN calibration file.

    Parameters
    ----------
    small:
        If True, return a smaller file for unit tests.
        The small version of the file was created using the following code, which keeps
        only 7 columns out of 154 (154 / 7 = 22):

        .. code-block:: python

            import scipp as sc

            fname = 'PG3_FERNS_d4832_2011_08_24_spectrum.h5'
            dg = sc.io.load_hdf5(fname)

            sizes = {"bank": 23, "column": 154, "row": 7}

            def foldme(x, dim):
                return x.fold(dim=dim, sizes=sizes)['column', ::22].flatten(
                    dims=list(sizes.keys()), to=dim)

            small = sc.Dataset(
                data={k: foldme(a, 'spectrum') for k, a in ds.items()},
                coords={k: foldme(c, 'spectrum') for k, c in ds.coords.items()}
            )
            sc.io.save_hdf5(small, 'TEST_PG3_FERNS_d4832_2011_08_24_spectrum.h5')
    """
    prefix = "TEST_" if small else ""
    return _registry.get_path(f"{prefix}PG3_FERNS_d4832_2011_08_24_spectrum.h5")


def pooch_load(filename: Filename[RunType]) -> RawDataAndMetadata[RunType]:
    """Load a file with pooch.

    If the file is a zip archive, it is extracted and a path to the contained
    file is returned.

    The loaded data holds both the events and any metadata from the file.
    """
    return RawDataAndMetadata[RunType](sc.io.load_hdf5(filename))


def load_powgen_mlz(
    filename: FilenameMLZ[RunType],
    datapath: FilepathMLZ[RunType]
) -> TofDetector[RunType]: #RawDataAndMetadata[RunType]:
    """Load a file.

    The loaded data holds both the events and useful metadata from the file.
    This loader can be applied to already preprocessed POWGEN event files,
    not the raw files from the instrument. To generate an appropriate file,
    the following steps should be taken:
    - use convert_to_scipp_nxs() defined in utils.py to extract appropriate
    fields from the original NeXus files.
    - the appropriate fields (including 'positions') are oextracted from
    the outcome of scn.load_with_mantid().
    """
    filepath = Path(os.path.join(datapath, filename))
    dg = sc.io.load_hdf5(filepath)
    out = dg["data"].squeeze()
    del dg
    out.coords.pop("tof", None)
    det_size = {"bank": 40, "column": 154, "row": 7}
    out = out.fold(dim="spectrum", sizes=det_size)
    if "pulse_time" in out.data.bins.coords:
        out.data = out.data.bins.drop_coords("pulse_time")
    return TofDetector[RunType](out) #RawDataAndMetadata[RunType](sc.io.load_hdf5(filepath))


def pooch_load_calibration(
    filename: CalibrationFilename,
    detector_dimensions: DetectorBankSizes,
) -> CalibrationData:
    """Load the calibration data for the POWGEN test data."""
    if filename is None:
        return CalibrationFilename(None)
    ds = sc.io.load_hdf5(filename)
    ds = sc.Dataset(
        {
            key: da.fold(dim='spectrum', sizes=detector_dimensions)
            for key, da in ds.items()
        }
    )
    return CalibrationData(ds)


def load_powgen_calibration_mlz(
    filename: FilenameMLZ[CalibrationRun],
    datapath: FilepathMLZ[CalibrationRun]
) -> CalibrationData:
    """Load a diamond calibration file.

    The loaded data holds ...
    """
    filepath = Path(os.path.join(datapath, filename))
    ds = sc.io.load_hdf5(filepath)
    # print(f"Loading calibration data from {filepath}")
    det_size = {"bank": 40, "column": 154, "row": 7}
    out = sc.Dataset(
        {
            key: da.fold(dim='spectrum', sizes=det_size)
            for key, da in ds.items()
        }
    )
    del ds
    return CalibrationData(out)


def load_powgen_tof_mask_mlz() -> MaskingData[TofMask]:
    """Load the TOF mask for POWGEN."""
    mask_function = None
    #mask_function = lambda x: (x < sc.scalar(0.0, unit="us")) | (x > sc.scalar(16666.67, unit="us"))
    return MaskingData[TofMask](mask_function)


def load_powgen_wavelength_mask_mlz() -> MaskingData[WavelengthMask]:
    """Load the wavelength mask for POWGEN."""
    mask_function = lambda x: (
        x < sc.scalar(0.3, unit="angstrom") # extra cut as in powtex
    ) | (
        x > sc.scalar(1.25, unit="angstrom") # extra cut as in powtex
    ) | (
        (x > sc.scalar(1.03, unit="angstrom")) & (x < sc.scalar(1.07, unit="angstrom"))
    )
    return MaskingData[WavelengthMask](mask_function)


def load_powgen_two_theta_mask_mlz() -> MaskingData[TwoThetaMask]:
    """Load the two-theta mask for POWGEN."""

    deg = lambda value: sc.scalar(value, unit="degree").to(unit="rad")

    mask_function = lambda x: (
        x < deg(10.0) # extra cut as in powtex
    ) | (
        x > deg(150.0) ) | (
        (x > deg(13.6)) & (x < deg(16.6))
    ) | (
        (x > deg(23.6)) & (x < deg(26.6))
    ) | (
        (x > deg(34.3)) & (x < deg(37.3))
    ) | (
        (x > deg(45.2)) & (x < deg(48.2))
    ) | (
        (x > deg(56.0)) & (x < deg(61.0))
    ) | (
        (x > deg(69.8)) & (x < deg(72.8))
    ) | (
        (x > deg(83.5)) & (x < deg(86.5))
    ) | (
        (x > deg(97.1)) & (x < deg(101.1))
    ) | (
        (x > deg(113.0)) & (x < deg(116.0))
    ) | (
        (x > deg(131.0)) & (x < deg(134.0))
    ) | (
        (x > deg(146.0)) & (x < deg(150.0))
    )
    return MaskingData[TwoThetaMask](mask_function)


def extract_raw_data(
    dg: RawDataAndMetadata[RunType], sizes: DetectorBankSizes
) -> TofDetector[RunType]:
    """Return the events from a loaded data group."""
    # Remove the tof binning and dimension, as it is not needed and it gets in the way
    # of masking.
    out = dg["data"].squeeze()
    out.coords.pop("tof", None)
    out = out.fold(dim="spectrum", sizes=sizes)
    return TofDetector[RunType](out)


def extract_raw_data_mlz(
    dg: RawDataAndMetadata[RunType] #, sizes: DetectorBankSizes
) -> TofDetector[RunType]:
    """Return the events from a loaded data group."""
    # Remove the tof binning and dimension, as it is not needed and it gets in the way
    # of masking.
    out = dg["data"].squeeze()
    out.coords.pop("tof", None)
    det_size = {"bank": 40, "column": 154, "row": 7}
    out = out.fold(dim="spectrum", sizes=det_size)
    return TofDetector[RunType](out)


def extract_proton_charge(dg: RawDataAndMetadata[RunType]) -> ProtonCharge[RunType]:
    """Return the proton charge from a loaded data group."""
    return ProtonCharge[RunType](dg["proton_charge"])


def extract_accumulated_proton_charge(
    data: CorrectedDspacing[RunType],
) -> AccumulatedProtonCharge[RunType]:
    """Return the stored accumulated proton charge from a loaded data group."""
    return AccumulatedProtonCharge[RunType](data.coords["gd_prtn_chrg"])


def source_position(dg: RawDataAndMetadata[RunType]) -> Position[snx.NXsource, RunType]:
    """Return the source position from a loaded data group."""
    return Position[snx.NXsource, RunType](dg["data"].coords["source_position"])


def sample_position(dg: RawDataAndMetadata[RunType]) -> Position[snx.NXsample, RunType]:
    """Return the sample position from a loaded data group."""
    return Position[snx.NXsample, RunType](dg["data"].coords["sample_position"])


def source_position_mlz(dg: TofDetector[RunType]) -> Position[snx.NXsource, RunType]:
    """Return the source position from a loaded data group."""
    return Position[snx.NXsource, RunType](dg["data"].coords["source_position"])


def sample_position_mlz(dg: TofDetector[RunType]) -> Position[snx.NXsample, RunType]:
    """Return the sample position from a loaded data group."""
    return Position[snx.NXsample, RunType](dg["data"].coords["sample_position"])


def _coordinate_transformation_graph(
    source_position: sc.Variable,
    sample_position: sc.Variable,
    gravity: sc.Variable,
    scatter: bool,
) -> dict:
    return {
        **scn.conversion.graph.beamline.beamline(scatter=scatter),
        **scn.conversion.graph.tof.elastic("tof"),
        'source_position': lambda: source_position,
        'sample_position': lambda: sample_position,
        'gravity': lambda: gravity,
    }


def detector_coordinate_transformation_graph(
    source_position: Position[snx.NXsource, RunType],
    sample_position: Position[snx.NXsample, RunType],
    gravity: GravityVector,
) -> ElasticCoordTransformGraph[RunType]:
    """Generate a coordinate transformation graph for detectors.

    Parameters
    ----------
    source_position:
        Position of the neutron source.
    sample_position:
        Position of the sample.
    gravity:
        Gravity vector.

    Returns
    -------
    :
        A dictionary graph for the transformation.
    """
    return ElasticCoordTransformGraph[RunType](
        _coordinate_transformation_graph(
            source_position, sample_position, gravity, scatter=True
        )
    )


def monitor_coordinate_transformation_graph(
    source_position: Position[snx.NXsource, RunType],
    sample_position: Position[snx.NXsample, RunType],
    gravity: GravityVector,
) -> MonitorCoordTransformGraph[RunType]:
    """Generate a coordinate transformation graph for monitors.

    Parameters
    ----------
    source_position:
        Position of the neutron source.
    sample_position:
        Position of the sample.
    gravity:
        Gravity vector.

    Returns
    -------
    :
        A dictionary graph for the transformation.
    """
    return MonitorCoordTransformGraph[RunType](
        _coordinate_transformation_graph(
            source_position, sample_position, gravity, scatter=False
        )
    )


def convert_detector_to_wavelength(
    da: TofDetector[RunType],
    graph: ElasticCoordTransformGraph[RunType],
) -> WavelengthDetector[RunType]:
    return WavelengthDetector[RunType](
        da.transform_coords("wavelength", graph=graph, keep_intermediate=False)
    )


def convert_monitor_to_wavelength(
    monitor: TofMonitor[RunType, MonitorType],
    graph: MonitorCoordTransformGraph[RunType],
) -> WavelengthMonitor[RunType, MonitorType]:
    return WavelengthMonitor[RunType, MonitorType](
        monitor.transform_coords("wavelength", graph=graph, keep_intermediate=False)
    )


providers = (
    pooch_load,
    pooch_load_calibration,
    extract_accumulated_proton_charge,
    extract_proton_charge,
    extract_raw_data,
    sample_position,
    source_position,
    detector_coordinate_transformation_graph,
    monitor_coordinate_transformation_graph,
    convert_detector_to_wavelength,
    convert_monitor_to_wavelength,
)
"""Sciline Providers for loading POWGEN data."""
