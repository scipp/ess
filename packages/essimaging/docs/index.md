:::{image} _static/logo.svg
:class: only-light
:alt: ESSimaging
:width: 60%
:align: center
:::
:::{image} _static/logo-dark.svg
:class: only-dark
:alt: ESSimaging
:width: 60%
:align: center
:::

```{raw} html
   <style>
    .transparent {display: none; visibility: hidden;}
    .transparent + a.headerlink {display: none; visibility: hidden;}
   </style>
```

```{role} transparent
```

# {transparent}`ESSimaging`

<div style="font-size:1.2em;font-style:italic;color:var(--pst-color-text-muted);text-align:center;">
  Imaging data reduction for the European Spallation Source
  </br></br>
</div>

## Quick links

::::{grid} 3

:::{grid-item-card} Odin
:link: odin/index.md
:img-top: _static/odin.svg

:::

:::{grid-item-card} Test beamline
:link: tbl/index.md

:::

:::{grid-item-card} Ymir
:link: ymir/index.md
:img-top: _static/ymir.svg

:::

::::

## Installation

To install ESSimaging and all of its dependencies, use

`````{tab-set}
````{tab-item} pip
```sh
pip install essimaging
```
````
````{tab-item} conda
```sh
conda install -c conda-forge essimaging
```
````
`````

## Get in touch

- If you have questions that are not answered by these documentation pages, ask on [discussions](https://github.com/scipp/essimaging/discussions). Please include a self-contained reproducible example if possible.
- Report bugs (including unclear, missing, or wrong documentation!), suggest features or view the source code [on GitHub](https://github.com/scipp/essimaging).

```{toctree}
---
hidden:
---

odin/index
tbl/index
ymir/index
tools/index
api-reference/index
developer/index
about/index
```
