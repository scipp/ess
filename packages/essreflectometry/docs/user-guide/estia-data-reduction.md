# Polarization data reduction for ESTIA

Based on https://confluence.ess.eu/display/ESTIA/Polarised+Neutron+Reflectometry+%28PNR%29+-+Reduction+Notes

## Model

Intensity in the detector is related to the reflectivity of the sample by the model
```math
\begin{bmatrix}
I^{+} \\
I^{-}
\end{bmatrix}
\big(\lambda, j\big)
=
D(\lambda, j)
\begin{bmatrix}
1 - a^{\uparrow} & 1 - a^{\downarrow} \\
a^{\uparrow} & a^{\downarrow}
\end{bmatrix}
\begin{bmatrix}
(1 - f_2) & f_2 \\ f_2 & (1 - f_2)
\end{bmatrix}
\begin{bmatrix}
R^{\uparrow\uparrow} & R^{\downarrow\uparrow} \\
R^{\uparrow\downarrow} & R^{\downarrow\downarrow}
\end{bmatrix}
\begin{bmatrix}
(1 - f_1) & f_1 \\ f_1 & (1 - f_1)
\end{bmatrix}
\begin{bmatrix}
1 - p^{\uparrow} \\
1 - p^{\downarrow}
\end{bmatrix}
```
where

* $I^+$ is the intensity of the neutron beam transmitted by the analyzer
and $I^-$ is the intensity of the neutron beam reflected by the analyzer,
* $R^\cdot$ are the reflectivities of the sample,
  - $R^{\uparrow\uparrow}$ is the fraction of incoming neutrons with spin up that are reflected with spin up,
  - $R^{\uparrow\downarrow}$ is the fraction of incoming neutrons with spin up that are reflected with spin down,
  - etc..
* $a^\uparrow$ is the analyzer reflectivity for spin up neutrons and $a^\downarrow$ is the analyzer reflectivity for spin down neutrons,
* $p^\uparrow$ is the polarizer reflectivity for spin up neutrons and $p^\downarrow$ is the polarizer reflectivity for spin down neutrons,
* $f_1$ is the probability of spin flip by the polarizer spin flipper, $f_2$ is the probability of spin flip by the analyzer spin flipper
* $D$ represents the inhomogeneity from the beam- and detector efficiency (and all other polarization unrelated terms).

## Reducing a measurement

If the sample is measured at two different flipper settings $f_1=0, f_2=0$ and $f_1=1, f_2=0$, then we have four measurement in total:
```math
\begin{bmatrix}
I^{0+} \\
I^{0-}
\end{bmatrix}
\big(\lambda, j\big)
=
D(\lambda, j)
\begin{bmatrix}
1 - a^{\uparrow} & 1 - a^{\downarrow} \\
a^{\uparrow} & a^{\downarrow}
\end{bmatrix}
\begin{bmatrix}
R^{\uparrow\uparrow} & R^{\downarrow\uparrow} \\
R^{\uparrow\downarrow} & R^{\downarrow\downarrow}
\end{bmatrix}
\begin{bmatrix}
1 - p^{\uparrow} \\
1 - p^{\downarrow}
\end{bmatrix}
```
```math
\begin{bmatrix}
I^{1+} \\
I^{1-}
\end{bmatrix}
\big(\lambda, j\big)
=
D(\lambda, j)
\begin{bmatrix}
1 - a^{\uparrow} & 1 - a^{\downarrow} \\
a^{\uparrow} & a^{\downarrow}
\end{bmatrix}
\begin{bmatrix}
R^{\uparrow\uparrow} & R^{\downarrow\uparrow} \\
R^{\uparrow\downarrow} & R^{\downarrow\downarrow}
\end{bmatrix}
\begin{bmatrix}
1 - p^{\downarrow} \\
1 - p^{\uparrow}
\end{bmatrix}.
```

To simplify the above, collect the terms in the matrix $\mathbf{a}$
```math
\begin{bmatrix}
I^{0+} \\
I^{0-} \\
I^{1+} \\
I^{1-}
\end{bmatrix}
\big(\lambda, j\big)
=
D(\lambda, j)
\mathbf{a}(\lambda)
\begin{bmatrix}
R^{\uparrow\uparrow} \\
R^{\uparrow\downarrow} \\
R^{\downarrow\uparrow} \\
R^{\downarrow\downarrow}
\end{bmatrix}
(Q(\lambda, j)).
```

To compute the reflectivities, integrate over a region of (almost) constant $Q$
```math
\int_{Q\in[q_{n}, q_{n+1}]}
\mathbf{a}^{-1}(\lambda)
\begin{bmatrix}
I^{0+} \\
I^{0-} \\
I^{1+} \\
I^{1-}
\end{bmatrix}
\big(\lambda, j\big)
d\lambda dj
\approx
\int_{Q\in[q_{n}, q_{n+1}]}
D(\lambda, j)
d\lambda dj
\begin{bmatrix}
R^{\uparrow\uparrow} \\
R^{\uparrow\downarrow} \\
R^{\downarrow\uparrow} \\
R^{\downarrow\downarrow}
\end{bmatrix}
(q_{n+\frac{1}{2}}).
```
The integral on the righ-hand-side can be evaluated using the reference measurement, call evaluated integral $\bar{D}(q_{n+{\frac{1}{2}}})$.
$R$ was moved outside of the integral because if $Q$ is almost constant so is $R(Q)$.

Finally we have
```math
\int_{Q\in[q_{n}, q_{n+1}]}
\mathbf{a}^{-1}(\lambda)
\bar{D}^{-1}(q_{n+{\frac{1}{2}}})
\begin{bmatrix}
I^{0+} \\
I^{0-} \\
I^{1+} \\
I^{1-}
\end{bmatrix}
\big(\lambda, j\big)
d\lambda dj
\approx
\begin{bmatrix}
R^{\uparrow\uparrow} \\
R^{\uparrow\downarrow} \\
R^{\downarrow\uparrow} \\
R^{\downarrow\downarrow}
\end{bmatrix}
(q_{n+\frac{1}{2}}).
```

### How to use the reference measurement to compute the integral over $D(\lambda, j)$?

For a reference measurement using flipper setting $f_1=0, f_2=0$ we have
```math
\begin{bmatrix}
I_{ref}^{+} \\
I_{ref}^{-}
\end{bmatrix}
\big(\lambda, j\big)
=
D(\lambda, j)
\begin{bmatrix}
1 - a^{\uparrow} & 1 - a^{\downarrow} \\
a^{\uparrow} & a^{\downarrow}
\end{bmatrix}
\begin{bmatrix}
R_{ref}^{\uparrow\uparrow} & R_{ref}^{\downarrow\uparrow} \\
R_{ref}^{\uparrow\downarrow} & R_{ref}^{\downarrow\downarrow}
\end{bmatrix}
\begin{bmatrix}
1 - p^{\uparrow} \\
1 - p^{\downarrow}
\end{bmatrix}.
```
But in practice, the analyzer/polarizer will be efficient enough to make only one of $I_{ref}^\pm$ have enough intensity to be useful. For example:
```math
\frac{I_{ref}^{+}(\lambda, j)}{r^+(\lambda, j)}
=
D(\lambda, j)
```
where $r^+$ is a known term involving the reflectivity of the supermirror and the pol-/analyzer efficiencies.
The expression for $D$ above can be used to evaluate integrals of $D$,
but in this case only in the region of the detector where the transmitted beam hits, because we only got data in that region from our reference measurement.

To measure $D$ for the entire detector we need to make several reference measurements with different flipper settings so that every part of the detector is illuminated in at least one measurement.
It might be unecessary to use all 4 flipper settings, but to illustrate the idea imagine we make reference measurements using all 4 flipper settings:
```math
\begin{bmatrix}
I_{ref}^{00+} \\
I_{ref}^{00-}
\end{bmatrix}
\big(\lambda, j\big)
=
D(\lambda, j)
\begin{bmatrix}
1 - a^{\uparrow} & 1 - a^{\downarrow} \\
a^{\uparrow} & a^{\downarrow}
\end{bmatrix}
\begin{bmatrix}
R_{ref}^{\uparrow\uparrow} & R_{ref}^{\downarrow\uparrow} \\
R_{ref}^{\uparrow\downarrow} & R_{ref}^{\downarrow\downarrow}
\end{bmatrix}
\begin{bmatrix}
1 - p^{\uparrow} \\
1 - p^{\downarrow}
\end{bmatrix}
```
```math
\begin{bmatrix}
I_{ref}^{01+} \\
I_{ref}^{01-}
\end{bmatrix}
\big(\lambda, j\big)
=
D(\lambda, j)
\begin{bmatrix}
1 - a^{\uparrow} & 1 - a^{\downarrow} \\
a^{\uparrow} & a^{\downarrow}
\end{bmatrix}
\begin{bmatrix}
0 & 1 \\ 1 & 0
\end{bmatrix}
\begin{bmatrix}
R_{ref}^{\uparrow\uparrow} & R_{ref}^{\downarrow\uparrow} \\
R_{ref}^{\uparrow\downarrow} & R_{ref}^{\downarrow\downarrow}
\end{bmatrix}
\begin{bmatrix}
1 - p^{\uparrow} \\
1 - p^{\downarrow}
\end{bmatrix}
```
```math
\begin{bmatrix}
I_{ref}^{10+} \\
I_{ref}^{10-}
\end{bmatrix}
\big(\lambda, j\big)
=
D(\lambda, j)
\begin{bmatrix}
1 - a^{\uparrow} & 1 - a^{\downarrow} \\
a^{\uparrow} & a^{\downarrow}
\end{bmatrix}
\begin{bmatrix}
R_{ref}^{\uparrow\uparrow} & R_{ref}^{\downarrow\uparrow} \\
R_{ref}^{\uparrow\downarrow} & R_{ref}^{\downarrow\downarrow}
\end{bmatrix}
\begin{bmatrix}
0 & 1 \\ 1 & 0
\end{bmatrix}
\begin{bmatrix}
1 - p^{\uparrow} \\
1 - p^{\downarrow}
\end{bmatrix}
```
```math
\begin{bmatrix}
I_{ref}^{11+} \\
I_{ref}^{11-}
\end{bmatrix}
\big(\lambda, j\big)
=
D(\lambda, j)
\begin{bmatrix}
1 - a^{\uparrow} & 1 - a^{\downarrow} \\
a^{\uparrow} & a^{\downarrow}
\end{bmatrix}
\begin{bmatrix}
0 & 1 \\ 1 & 0
\end{bmatrix}
\begin{bmatrix}
R_{ref}^{\uparrow\uparrow} & R_{ref}^{\downarrow\uparrow} \\
R_{ref}^{\uparrow\downarrow} & R_{ref}^{\downarrow\downarrow}
\end{bmatrix}
\begin{bmatrix}
0 & 1 \\ 1 & 0
\end{bmatrix}
\begin{bmatrix}
1 - p^{\uparrow} \\
1 - p^{\downarrow}
\end{bmatrix}.
```

Summing all 8 measurements gives us an expression for $D$ that ought to be valid for the entire detector:
```math
\frac{
I_{ref}^{00+}(\lambda, j) +
I_{ref}^{00-}(\lambda, j) +
I_{ref}^{01+}(\lambda, j) +
I_{ref}^{01-}(\lambda, j) +
I_{ref}^{10+}(\lambda, j) +
I_{ref}^{10-}(\lambda, j) +
I_{ref}^{11+}(\lambda, j) +
I_{ref}^{11-}(\lambda, j)
}{
r^{00+}(\lambda, j) +
r^{00-}(\lambda, j) +
r^{01+}(\lambda, j) +
r^{01-}(\lambda, j) +
r^{10+}(\lambda, j) +
r^{10-}(\lambda, j) +
r^{11+}(\lambda, j) +
r^{11-}(\lambda, j)
}
=
D(\lambda, j).
```
