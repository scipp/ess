:::{image} _static/logo.svg
:class: only-light
:alt: ESSreflectometry
:width: 60%
:align: center
:::
:::{image} _static/logo-dark.svg
:class: only-dark
:alt: ESSreflectometry
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

# {transparent}`ESSreflectometry`

<div style="font-size:1.2em;font-style:italic;color:var(--pst-color-text-muted);text-align:center;">
  Reflectometry data reduction for the European Spallation Source
  </br></br>
</div>

## Quick links

::::{grid} 3

:::{grid-item-card} ESTIA
:link: user-guide/estia/index.md

:::

:::{grid-item-card} Amor
:link: user-guide/amor/index.md

:::

::::{grid-item-card} Offspec
:link: user-guide/offspec/index.md

:::

::::

:::{include} user-guide/installation.md
:heading-offset: 1
:::

## Get in touch

- If you have questions that are not answered by these documentation pages, ask on [discussions](https://github.com/scipp/essreflectometry/discussions). Please include a self-contained reproducible example if possible.
- Report bugs (including unclear, missing, or wrong documentation!), suggest features or view the source code [on GitHub](https://github.com/scipp/essreflectometry).

```{toctree}
---
hidden:
---

user-guide/index
api-reference/index
developer/index
about/index
```
