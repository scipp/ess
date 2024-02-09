# Design document data workflow for the NMX instrument at ESS

This is a description of the data workflow for the NMX instrument at ESS.

The [NMX](https://europeanspallationsource.se/instruments/nmx) Macromolecular Diffractometer is a time-of-flight (TOF) quasi-Laue diffractometer optimised for small samples and large unit cells dedicated to the structure determination of biological macromolecules by crystallography. 
The main scientific driver is to locate the hydrogen atoms relevant to the function of the macromolecule.

## Data reduction

![work_flow](https://github.com/scipp/essnmx/new/main/docs/about/NMX_work_flow.png)


### From single event data to binned image-like data
From single event data to binned image-like data
The first step in the data reduction is to reduce the data from single event data to image-like data.
Therefore the [essNMX](https://github.com/scipp/essnmx) package is used. 
First, the time of arrival (TOA) is converted into time of flight (TOF).

Then the single events get binned into pixels and then histogramed in the TOF dimension. 
These data will be written be added with some meta and instrument data to an HDF5 file.

### Spot finding and integration
For the next five steps of the data reduction from spot finding to spot integration, we use the [programme](https://dials.github.io/index.html) [DIALS](https://onlinelibrary.wiley.com/doi/10.1002/pro.4224) 

First, we use [dials.import](https://dials.github.io/documentation/programs/dials_import.html) to convert image data files into a format compatible with dials. It the metadata and filenames of each image to establish relationships between different sets of images. Once all images are processed, the program generates an experiment object file, outlining the connections between the files. The images to be processed are designated as command-line arguments. Occasionally, there may be a restriction on the maximum number of arguments allowed on the command line, and the number of files could surpass this limit. In such cases, image filenames can be entered through stdin, as demonstrated in the examples below. 
The Format class for NMX is at modules/dxtbx/src/dxtbx/format/FormatNMX.py where beam-line-specific parameters and file format information are stored.

```console
dials.import *.nxs
```

In the next step, a [search for strong pixel](https://dials.github.io/documentation/programs/dials_find_spots.html) is performed. Therefore the intensity of a pixel or pixel group is compared with its local surroundings. With the information of strong pixels, strong spots are defined. for these spots, the centroids and intensities will be calculated. the results can be visualised in the image viewer or the [browser](https://toastisme.github.io/dials_browser_experiment_viewer/)

```console
dials.find_spots imported.expt find_spots.phil
```

In the [indexing](https://dials.github.io/documentation/programs/dials_index.html) step the unit cell is determined. a list of indexed reflexes and an instrument model including a crystal model is returned. One-dimensional and three-dimensional fast Fourier transform-based methods are available.

As input parameters the imported.exp and strong.refl files are used. more parameters such as unit cell and spacegroup can be given.

```console
dials.index imported.expt strong.refl space_group=P1 unit_cell=a,b,c,alpha,beta,gamma
```



After indexing the instrument geometry is getting [refined](https://dials.github.io/documentation/programs/dials_refine.html).
```console
dials.refine indexed.refl indexed.expt detector.panels=hierarchical
```

The last step in DIALS is to integrate(https://dials.github.io/documentation/programs/dials_integrate.html) each reflex. Currently, in the dimension of the image, a simple summation is used and in the TOF dimension, a profile-fitting approach is used.

```console
dev.dials.simple_tof_integrate refined.expt refined.refl
```




### Scaling
Currently [LSCALE](https://scripts.iucr.org/cgi-bin/paper?S0021889898015350) can be used in a docker container which makes it indented from the OS. The source code is available on [Zenodo](https://zenodo.org/records/4381992). LSCALE is a program for scaling and normalisation of Laue intensity data. 
Since LSCALE is not maintained anymore we currently develop a Python-based [alternative](https://github.com/mlund/pyscale) to LSCALE.

start docker desktop
```console
docker run -it -v $HOME:/mnt/host -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=host.docker.internal:0 lscale
```
```console
lscale < lscale.com > lscale.out
```

### AIMLESS and CTRUNCATE

[AIMLESS](https://www.ccp4.ac.uk/html/aimless.html) is used to scale multiple observations of reflections together, and merges multiple observations into an average intensity.


[CTRUNCATE](https://www.ccp4.ac.uk/html/ctruncate.html) converts measured intensities into structure factors. CTRUNCATE includes corrections for weak reflections to avoid negative intensities due to background corrections. 

```console
Start CCP4 GUI
go to all programs
select Aimless
select scaled *.mtz file
```
usually, standard parameters are fine but parameters can be modified.

This results in a final *mtz file which can be used in a standard protein crystallographic program to solve and refine the structure.
