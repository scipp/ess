# User Guide

## Overview

```{mermaid}
graph TD
    A[Sample Run] --> B([SANS Workflow])
    B --> C["I(Qx, Qy) in event mode"]
    D[Runs with He3 cell at a few time points] --> E([SANS Workflow])
    E --> F[wavelength-dependent He3 cell transmission fraction at a few time points]
    F --> G([<font color=black>He3 Cell Workflow])
    G --> H[<font color=black>time- and wavelength-dependent transmission function]
    C --> I([Polarization Correction])
    H --> I
    I --> J["Corrected I(Qx, Qy) in 4 spin channels"]

    style B fill:green
    style D fill:green
    style E fill:green
    style F fill:green
    style G fill:yellowgreen
    style H fill:yellowgreen
```

## Content

```{toctree}
---
maxdepth: 1
---

workflow
sans-polarization-analysis-methodology
inverse_of_polarization_matrices
zoom
installation
```
