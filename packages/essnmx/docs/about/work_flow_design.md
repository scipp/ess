# Design document data workflow for the NMX instrument at ESS

This is a description of the data workflow for the NMX instrument at ESS.

The [NMX](https://europeanspallationsource.se/instruments/nmx) Macromolecular Diffractometer is a time-of-flight (TOF) quasi-Laue diffractometer optimised for small samples and large unit cells dedicated to the structure determination of biological macromolecules by crystallography. 
The main scientific driver is to locate the hydrogen atoms relevant for the function of the macromolecule.

## Data reduction

![work_flow](https://github.com/scipp/essnmx/new/main/docs/about/NMX_work_flow.png)


### From single event data to binned image like data
From single event data to binned image like data
The first step in the data reduction is to reduce the data from single event data to image like data.
Therefore the [essNMX](https://github.com/scipp/essnmx) package is used. 
Tirst the time of arrival (TOA) is converted into time of flight (TOF).

Then the single events get binned into pixels and then histogramed in the TOF dimension. 
These  data will be written be added with some meta and instrument data to an HDF5 file.
