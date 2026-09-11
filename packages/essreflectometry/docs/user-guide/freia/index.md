# FREIA

The initial FREIA workflow loads final-detector McStas events and uses the generic
ESSreduce analytical frame-unwrapping workflow to compute wavelengths. The notebook
guides below use the example download helpers in `ess.freia.data` and visualize the
detector with Scipp and Plopp. Choppers use the fixed **WFM** simulation configuration.
The wavelength lookup-table guide reads only detector geometry from the file, so it
can also be run on simulations without detector events.

```{toctree}
:maxdepth: 1

freia-mcstas-visualization
freia-wavelength-lookup-table
```
