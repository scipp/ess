# Reflectometry data reduction procedure

The goal of the reflectometry data reduction is to compute the sample reflectivity $R(Q)$ as a function of the momentum transfer $Q$.

## Preliminaries

The detector data consists of a list $EV$ of neutron detector events.
For each neutron event in the event list we know what wavelength $\lambda$ it had and we know the pixel number $j$ of the detector pixel that it hit.
The detector pixel positions are known and so is the position of the sample and the orientation of the sample.
From that information we can compute the reflection angle (assumed equal to the incidence angle) $\theta$, and the momentum transfer $Q$ caused by the interaction with the sample.

The purpose of this text is not to describe how the event coordinates $Q$ and $\theta$ are derived from the raw event data and the geometry information, so for now just take those relations for given.

To avoid overcomplicating the description it is assumed that the sample- and reference measurements were made over the same length of time, and it is assumed the neutron intensity from the source did not vary between the two measurements.

## Model

The sample reflectivity is related to the intensity of neutron counts in the detector by the model

$$
I_{sam}(\lambda, j) = F(\theta(\lambda, j, \mu_{sam}), w_{sam}) \cdot R(Q(\lambda, \theta(\lambda, j, \mu_{sam}))) \cdot I_{ideal}(\lambda, j)
$$ (model)
where $I_{sam}(\lambda, j)$ represents the number of neutrons detected in the $j$ pixel of the detector having a wavelength in the interval $[\lambda, \lambda + d\lambda]$. $I_{ideal}$ represents the number of neutrons that would have been detected if the sample was a perfect reflector and large enough so that the footprint of the focused beam on the sample was small compared to the sample. $F(\theta, w)$ is the fraction of the focused beam that hits the sample. It depends on the incidence angle $\theta$ and on the size of the sample represented by $w$. $\mu_{sam}$ is the sample rotation.

The ideal intensity is estimated from a reference measurement on a neutron supermirror.
How it is computed will be described later, for now assume it exists.

## Estimating $R(Q)$
Move $F$ to the left-hand-side of equation {eq}`model` and integrate over all $\lambda$ and $j$ contributing to one particular $Q$-bin $[q_{i}, q_{i+1}]$

$$
\int_{Q(\lambda, \theta(\lambda, j, \mu_{sam})) \in [q_{i}, q_{i+1}]} \frac{I_{sam}(\lambda, j)}{F(\theta(\lambda, j, \mu_{sam}), w_{sam})} d\lambda \ dj = \\
\int_{Q(\lambda, \theta(\lambda, j, \mu_{sam})) \in [q_{i}, q_{i+1}]}  I_{ideal}(\lambda, j) R(Q(\lambda, \theta(\lambda, j, \mu_{sam}))) d\lambda  \ dj.
$$
Notice that if the $Q$ binning is sufficiently fine then $R(Q)$ is approximately constant in the integration region.
Assuming the binning is fine enough $R(Q)$ can be moved outside the integral and isolated so that

$$
 R(Q_{i+\frac{1}{2}}) \approx \frac{\int_{Q(\lambda, j, \mu_{sam}) \in [q_{i}, q_{i+1}]} \frac{I_{sam}(\lambda, j)}{F(\theta(\lambda, j, \mu_{sam}), w_{sam})} d\lambda \ dj }{\int_{Q(\lambda, \theta(\lambda, j, \mu_{sam})) \in [q_{i}, q_{i+1}]}  I_{ideal}(\lambda, j) d\lambda  \ dj} := \frac{I_{measured}(Q_{i+\frac{1}{2}})}{I_{ideal}(Q_{i+\frac{1}{2}})}
$$
for $Q_{i+\frac{1}{2}} \in [q_{i}, q_{i+1}]$.


## The reference intensity $I_{ideal}$
$I_{ideal}$ is estimated from a reference measurement on a neutron supermirror with known reflectivity curve.
The reference measurement intensity is modeled the same way the sample measurement was

$$
I_{ref}(\lambda, j) = F(\theta(\lambda, j, \mu_{ref}), w_{ref}) \cdot R_{supermirror}(Q(\lambda, \theta(\lambda, j, \mu_{ref}))) \cdot I_{ideal}(\lambda, j)
$$
but in this case $R_{supermirror}(Q)$ is known.

This leads to

$$
I_{ideal}(Q_{i+\frac{1}{2}}) = \int_{Q(\lambda, \theta(\lambda, j, \mu_{sam})) \in [q_{i}, q_{i+1}]} \frac{I_{ref}(\lambda, j)}{F(\theta(\lambda, j, \mu_{ref}), w_{ref}) R_{supermirror}(Q(\lambda, \theta(\lambda, j, \mu_{ref})))}
 d\lambda  \ dj.
$$

## Estimating intensities from detector counts
The neutron counts are Poisson distributed.
This implies that the intensity integrals are equal to the expected number of neutron detector counts in the integration region.
The expected number of counts can be estimated by the empirically observed count:

$$
I_{measured}(Q_{i+\frac{1}{2}}) = \int_{Q(\lambda, \theta(\lambda, j, \mu_{sam})) \in [q_{i}, q_{i+1}]} \frac{I_{sam}(\lambda, j)}{F(\theta(\lambda, j, \mu_{sam}), w_{sam})} d\lambda \ dj = \\
E\bigg[ \sum_{\substack{k \in EV_{sam} \\ Q(\lambda_{k}, \theta(\lambda_{k}, j_{k}, \mu_{sam})) \in [q_{i}, q_{i+1}]}} \frac{1}{F(\theta(\lambda_{k}, j_{k}, \mu_{sam}), w_{sam})} \bigg] \approx
\sum_{\substack{k \in EV_{sam} \\ Q(\lambda_{k}, \theta(\lambda_{k}, j_{k}, \mu_{sam})) \in [q_{i}, q_{i+1}]}} \frac{1}{F(\theta(\lambda_{k}, j_{k}, \mu_{sam}), w_{sam})}
$$
where $EV_{sam}$ refers to the event list from the sample experiment.

We also know that the variance of the counts is the same as the expected count, so it can also be estimated as the empirically observed count:

$$
V\bigg[ \sum_{\substack{k \in EV_{sam} \\ Q(\lambda_{k}, \theta(\lambda_{k}, j_{k}, \mu_{sam})) \in [q_{i}, q_{i+1}]}} \frac{1}{F(\theta(\lambda_{k}, j_{k}, \mu_{sam}), w_{sam})} \bigg] \approx
\sum_{\substack{k \in EV_{sam} \\ Q(\lambda_{k}, \theta(\lambda_{k}, j_{k}, \mu_{sam})) \in [q_{i}, q_{i+1}]}} \frac{1}{F(\theta(\lambda_{k}, j_{k}, \mu_{sam}), w_{sam})}.
$$

The same estimates are used to approximate the ideal intensity:

$$
I_{ideal}(Q_{i+\frac{1}{2}}) \approx \sum_{\substack{k \in EV_{ref} \\ Q(\lambda_{k}, \theta(\lambda_{k}, j_{k}, \mu_{sam})) \in [q_{i}, q_{i+1}]}} \frac{1}{F(\theta(\lambda_{k}, j_{k}, \mu_{ref}), w_{ref}) R_{supermirror}(Q(\lambda_{k}, \theta(\lambda_{k}, j_{k}, \mu_{ref})))}
$$

### More efficient evaluation of the reference intensity
The above expression for the reference intensity is cumbersome to compute because it is a sum over the reference measurement event list, and the reference measurement is large compared to the sample measurement.

Therefore we back up a bit. Consider the expression for the reference intensity, replacing the integrand with a generic $I(\lambda, j)$ it looks something like:

$$
I_{ideal}(Q_{i+\frac{1}{2}}) = \int_{Q(\lambda, \theta(\lambda, j, \mu_{sam}))  \in [q_{i}, q_{i+1}]} I(\lambda, j) \ d\lambda  \ dj.
$$

In the previous section we approximated the integral by summing over all events in the reference measurement.

Alternatively, we could define a $\lambda$ grid with edges $\lambda_{k}$ for $k=1\ldots N$ and approximate the integration region as the union of a subset of the grid cells:

$$
\int_{Q(\lambda, \theta(\lambda, j, \mu_{sam}))  \in [q_{i}, q_{i+1}]} I(\lambda, j) \ d\lambda  \ dj
\approx \sum_{Q(\bar{\lambda}_{k+\frac{1}{2}},\ \theta(\bar{\lambda}_{k+\frac{1}{2}}, j, \mu_{sam}))  \in [q_{i}, q_{i+1}]} I_{k+\frac{1}{2},j}
$$
where

$$
I_{k+\frac{1}{2},j} = \int_{\lambda \in [\lambda_{k}, \lambda_{k+1}]} I(\lambda, j) \ d\lambda
$$
and $\bar{\lambda}_{k+\frac{1}{2}} = (\lambda_{k} + \lambda_{k+1}) / 2$.

Why would this be more efficient than the original approach? Note that $I_{k+\frac{1}{2}, j}$ does not depend on $\mu_{sam}$, and that it can be computed once and reused for all sample measurements.
This allows us to save computing time for each new sample measurement, as long as $|EV_{ref}| >> NM$ where $M$ is the number of detector pixels and $N$ is the size of the $\lambda$ grid.

However, ideally computing the reference intensity should be quick compared to reducing the sample measurement. And since a reasonable value for $N$ is approximately $500$, and $M\approx 30000$, and a sample measurement is likely less than $10$ million events, the cost of computing the reference measurement is still considerable compared to reducing the sample measurement.

Therefore there's one more approximation that is used to further reduce the cost of computing the reference intensity.

The Amor detector has three logical dimensions, `blade`, `wire` and `stripe`. It happens to be the case that $\theta(\lambda, j)$ is almost the same for all $j$ belonging to the same `stripe` of the detector.
We can express this as

$$
\theta(\lambda, j, \mu_{sam}) \approx \bar{\theta}(\lambda, \mathrm{bladewire}(j), \mu_{sam})
$$
where $\bar{\theta}$ is an approximation for $\theta$ that only depends on the blade and the wire of the pixel where the neutron was detected.
Then the above expression for the reference intensity can be rewritten as

$$
\int_{Q(\lambda, \bar{\theta}(\lambda, z, \mu_{sam}))  \in [q_{i}, q_{i+1}]} \int_{\mathrm{bladewire}(j) = z} I(\lambda, j) \ dj \ d\lambda  \ dz
\approx \sum_{Q(\bar{\lambda}_{k+\frac{1}{2}}, \bar{\theta}(\bar{\lambda}_{k+\frac{1}{2}}, z, \mu_{sam}))  \in [q_{i}, q_{i+1}]} I_{k+\frac{1}{2},z}
$$
where

$$
I_{k+\frac{1}{2},z} = \int_{\lambda \in [\lambda_{k}, \lambda_{k+1}]} \int_{\mathrm{bladewire}(j) = z} I(\lambda, j) \ dj \ d\lambda .
$$
Like before, the benefit of doing this is that

$$
 \int_{\lambda \in [\lambda_{k}, \lambda_{k+1}]} \int_{\mathrm{bladewire}(j) = z} I(\lambda, j) \ dj \ d\lambda
$$
can be pre-computed because it doesn't depend on $\mu_{sam}$.
But unlike before $I_{k+\frac{1}{2},z}$ now has a much more manageable size, about 64x smaller than the first attempt.
This makes it comfortably smaller than the sample measurement.
