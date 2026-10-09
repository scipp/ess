# API Reference

The prototype is not yet packaged as `mlzdiffraction`. Until a normal
`pyproject.toml` and package namespace are added, the importable API reference
uses the current namespace-package path under `src`.

## MLZdiffraction

### Shared POWTEX submodules

```{eval-rst}
.. currentmodule:: src.powtex

.. autosummary::
   :toctree: ../generated/modules
   :recursive:

   data
   masking
   types
```

### POWTEX workflows

```{eval-rst}
.. currentmodule:: src.powder

.. autosummary::
   :toctree: ../generated/functions
   :recursive:

   PowderReduceP2D
```
```{eval-rst}
.. currentmodule:: src.texture

.. autosummary::
   :toctree: ../generated/functions
   :recursive:

   TextureReduce
```

### Powder submodules

```{eval-rst}
.. currentmodule:: src.powder

.. autosummary::
   :toctree: ../generated/modules
   :recursive:

   conversion
   correction
   data
   grouping
   masking
   types
```

### Texture submodules

```{eval-rst}
.. currentmodule:: src.texture

.. autosummary::
   :toctree: ../generated/modules
   :recursive:

   conversion
   correction
   grouping
   histogramming
   types
```
