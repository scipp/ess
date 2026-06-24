# API Reference

## ESSdiffraction

### Submodules

```{eval-rst}
.. currentmodule:: ess.diffraction

.. autosummary::
   :toctree: ../generated/modules
   :template: module-template.rst
   :recursive:

   pdf
   peaks
```

## ESSpowder

### Module Attributes

```{eval-rst}
.. currentmodule:: ess.powder

.. autosummary::
   :toctree: ../generated/attributes

   providers

.. autosummary::
   :toctree: ../generated/classes

   RunNormalization

.. autosummary::
   :toctree: ../generated/functions

   with_pixel_mask_filenames
```

### Submodules

```{eval-rst}
.. autosummary::
   :toctree: ../generated/modules
   :template: module-template.rst
   :recursive:

   calibration
   conversion
   correction
   filtering
   grouping
   logging
   masking
   smoothing
   types
```

## DREAM

### Workflows

```{eval-rst}
.. currentmodule:: ess.dream

.. autosummary::
   :toctree: ../generated/functions

   DreamGeant4MonitorHistogramWorkflow
   DreamGeant4MonitorIntegratedWorkflow
   DreamGeant4ProtonChargeWorkflow
   DreamGeant4Workflow
   DreamPowderWorkflow
   DreamWorkflow
```

### Top-level functions

```{eval-rst}
.. currentmodule:: ess.dream

.. autosummary::
   :toctree: ../generated/functions

   instrument_view
   load_geant4_csv
```

### Top-level classes

```{eval-rst}
.. currentmodule:: ess.dream

.. autosummary::
   :toctree: ../generated/classes

   InstrumentConfiguration
```

### Submodules

```{eval-rst}
.. autosummary::
   :toctree: ../generated/modules
   :template: module-template.rst
   :recursive:

   beamline
   data
   diagnostics
   io
```

## BEER

### Workflows

```{eval-rst}
.. currentmodule:: ess.beer

.. autosummary::
   :toctree: ../generated/functions

   BeerMcStasWorkflowPulseShaping
   BeerMcStasWorkflowPulseShapingAnalytical
   BeerModMcStasWorkflowKnownPeaks
   BeerModMcStasWorkflow
   BeerPowderWorkflow
   BeerPowderWorkflowAnalytical
   BeerPowderMcStasWorkflow
   BeerPowderMcStasWorkflowAnalytical
```

### Top-level functions

```{eval-rst}
.. currentmodule:: ess.beer

.. autosummary::
   :toctree: ../generated/functions

   dhkl_peaks_from_cif
   load_beer_mcstas
```

### Submodules

```{eval-rst}
.. autosummary::
   :toctree: ../generated/modules
   :template: module-template.rst
   :recursive:

   clustering
   conversions
   data
   io
   mcstas
   workflow
```

## MAGIC

### Workflows

```{eval-rst}
.. currentmodule:: ess.magic

.. autosummary::
   :toctree: ../generated/functions

   MagicWorkflow
```

### Submodules

```{eval-rst}
.. autosummary::
   :toctree: ../generated/modules
   :template: module-template.rst
   :recursive:

   beamline
   data
   types
   workflow
```

## SNS powder

```{eval-rst}
.. currentmodule:: ess.snspowder

.. autosummary::
   :toctree: ../generated/modules
   :template: module-template.rst
   :recursive:

   powgen
```
