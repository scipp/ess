# Focused reflectometry data reduction

The goal of the reflectometry data reduction is to compute the reflectivity $R(Q)$ of the sample as a function of the momentum transfer $Q$.

Based on {cite}`STAHN201644`.


## Target audience

The target audience of this text is anyone who wants to know the reasoning behind the data reduction workflow implemented for the focusing reflectometry instruments ESTIA at ESS and AMOR at PSI.
It is not necessary to read this to be able to use the workflow, for that we recommend the tutorial notebooks as they will give a much more practically useful introduction, but if you are going to develop new features it is probably good to at least skim it.


## Preliminaries

The detector data is a list $EV$ of detected neutron events.
For each event in the list we know its wavelength $\lambda$ and the pixel number $j$ of the detector pixel that it hit.
The detector pixel positions are known and so is the position and the orientation of the sample.
From this information we can compute the reflection angle $\theta$, and the momentum transfer $Q$ caused by the interaction with the sample.

The purpose of this text is not to describe how the event coordinates wavelength, $Q$ and $\theta$ are derived from the raw detector data and the instrument geometry, so for now just take those for given.
For more details see the implementations for the respective instruments [Amor] and [Estia].

To simplify the description it is assumed that the sample- and reference measurements were made over the same length of time, and it is assumed the brightness of the source did not change between the measurements.


## Model of event intensity in the detector

The reflectivity of the sample is related to the intensity of neutron counts in the detector by the model

$$
I_{\text{sam}}(\lambda, j) = F(\theta(\lambda, j, \mu_{\text{sam}}), w_{\text{sam}}) \cdot R(Q(\lambda, \theta(\lambda, j, \mu_{\text{sam}}))) \cdot I_{\text{ideal}}(\lambda, j)
$$ (model)
where $I_{\text{sam}}(\lambda, j)$ represents the expected number of neutrons detected in the $j$ pixel of the detector per unit of wavelength at the wavelength value $\lambda$. $I_{\text{ideal}}$ represents the expected number of neutrons detected if the sample was a perfect reflector and large enough so that the footprint of the focused beam on the sample was small compared to the sample. $F(\theta, w)$ is the fraction of the beam that hits the sample. It depends on the incidence angle $\theta$ and on the size of the sample represented by $w,$ and $\mu_{\text{sam}}$ is the sample rotation.

The model does not hold for any $\lambda,j$. For example, there might be a region in the detector where we can see part of the direct beam. To make this explicit, let $M_{sam}$ represent the region of $\lambda,j$  where the model is expected to hold (the "region of interest").

The ideal intensity $I_{ideal}$ will be estimated from a reference measurement on a neutron supermirror.
How that is done will be described in more detail later, for now assume it is a known quantity.


## Estimating $R(Q)$
Move $F$ to the left-hand-side of equation {eq}`model` and integrate over all $\lambda, j\in M$ contributing to the $Q$-bin $[q_{i}, q_{i+1}]$

$$
\int_{M \cap Q(\lambda, \theta(\lambda, j, \mu_{\text{sam}})) \in [q_{i}, q_{i+1}]} \frac{I_{\text{sam}}(\lambda, j)}{F(\theta(\lambda, j, \mu_{\text{sam}}), w_{\text{sam}})} d\lambda \ dj = \\
\int_{M \cap Q(\lambda, \theta(\lambda, j, \mu_{\text{sam}})) \in [q_{i}, q_{i+1}]}  I_{\text{ideal}}(\lambda, j) R(Q(\lambda, \theta(\lambda, j, \mu_{\text{sam}}))) d\lambda  \ dj.
$$
Notice that if the $Q$ binning is sufficiently fine then $R(Q)$ is approximately constant in the integration region.
Assuming the binning is fine enough $R(Q)$ can be moved outside the integral and isolated so that

$$
 R(Q_{i+\frac{1}{2}}) \approx \frac{\int_{M \cap Q(\lambda, j, \mu_{\text{sam}}) \in [q_{i}, q_{i+1}]} \frac{I_{\text{sam}}(\lambda, j)}{F(\theta(\lambda, j, \mu_{\text{sam}}), w_{\text{sam}})} d\lambda \ dj }{\int_{M \cap Q(\lambda, \theta(\lambda, j, \mu_{\text{sam}})) \in [q_{i}, q_{i+1}]}  I_{\text{ideal}}(\lambda, j) d\lambda  \ dj} =: \frac{I_{measured}(Q_{i+\frac{1}{2}})}{I_{\text{ideal}}(Q_{i+\frac{1}{2}})}
$$ (reflectivity)
for $Q_{i+\frac{1}{2}} \in [q_{i}, q_{i+1}]$.

For the integral to make sense the region of interest $M$ has to be contained in the region where {eq}`model` holds, $M\subset M_{sam}$, but there might be other constraints limiting the region of interest $M$ even more, so it is left undefined for now.


## The reference intensity $I_{\text{ideal}}$
$I_{\text{ideal}}$ is estimated from a reference measurement on a neutron supermirror with known reflectivity curve.
The reference measurement intensity $I_{\text{ref}}$ is modeled the same way the sample measurement was

$$
I_{\text{ref}}(\lambda, j) = F(\theta(\lambda, j, \mu_{\text{ref}}), w_{\text{ref}}) \cdot R_{\text{supermirror}}(Q(\lambda, \theta(\lambda, j, \mu_{\text{ref}}))) \cdot I_{\text{ideal}}(\lambda, j)
$$
but in this case $R_{\text{supermirror}}(Q)$ is known.

As before, the model does not hold for any $\lambda,j$. Let $M_{ref}$ represent the region of $\lambda,j$  where the model is expected to hold. $M_{ref}$ is typically not the same as $M_{sam}$. For example, if the supermirror reflectivity is not known for all $Q$ we are not going to have a model for the reference intensity at the $\lambda,j$ corresponding to those $Q$ and that is reflected in $M_{ref}$ but not in $M_{sam}$.

Using the definition in {eq}`reflectivity`

$$
I_{\text{ideal}}(Q_{i+\frac{1}{2}}) = \int_{M \cap Q(\lambda, \theta(\lambda, j, \mu_{\text{sam}})) \in [q_{i}, q_{i+1}]} \frac{I_{\text{ref}}(\lambda, j)}{F(\theta(\lambda, j, \mu_{\text{ref}}), w_{\text{ref}}) R_{\text{supermirror}}(Q(\lambda, \theta(\lambda, j, \mu_{\text{ref}})))}
 d\lambda  \ dj.
$$
For this integral to make sense $M\subset M_{ref}$, so now we have an additional constraint on $M$ to keep in mind.


## Estimating intensities from detector counts

The number of neutron counts in the detector is a Poisson process where the expected number of neutrons per pixel and unit of wavelength are the measurement intensities $I_{sam}$ and $I_{ref}$ defined above.
The expected intensity can be estimated by the measured intensity:

$$
I_{measured}(Q_{i+\frac{1}{2}}) = \int_{M\cap Q(\lambda, \theta(\lambda, j, \mu_{\text{sam}})) \in [q_{i}, q_{i+1}]} \frac{I_{\text{sam}}(\lambda, j)}{F(\theta(\lambda, j, \mu_{\text{sam}}), w_{\text{sam}})} d\lambda \ dj \\
 \approx \sum_{\substack{k \in EV_{\text{sam}} \\ Q(\lambda_{k}, \theta(\lambda_{k}, j_{k}, \mu_{\text{sam}})) \in [q_{i}, q_{i+1}]\\ (\lambda_k, j_k)\in M}} \frac{1}{F(\theta(\lambda_{k}, j_{k}, \mu_{\text{sam}}), w_{\text{sam}})}
$$
where $EV_{\text{sam}}$ refers to the event list from the sample experiment.

The sum is compound Poisson distributed and for such random variables the variance can be estimated by the sum of squared summands

$$
V\bigg[ \sum_{\substack{k \in EV_{\text{sam}} \\ Q(\lambda_{k}, \theta(\lambda_{k}, j_{k}, \mu_{\text{sam}})) \in [q_{i}, q_{i+1}]\\ (\lambda_k, j_k)\in M}} \frac{1}{F(\theta(\lambda_{k}, j_{k}, \mu_{\text{sam}}), w_{\text{sam}})} \bigg] \approx
\sum_{\substack{k \in EV_{\text{sam}} \\ Q(\lambda_{k}, \theta(\lambda_{k}, j_{k}, \mu_{\text{sam}})) \in [q_{i}, q_{i+1}]\\ (\lambda_k, j_k)\in M}} \frac{1}{F(\theta(\lambda_{k}, j_{k}, \mu_{\text{sam}}), w_{\text{sam}})^2}.
$$

The same estimates are used to approximate the ideal intensity:

$$
I_{\text{ideal}}(Q_{i+\frac{1}{2}}) \approx \sum_{\substack{k \in EV_{\text{ref}} \\ Q(\lambda_{k}, \theta(\lambda_{k}, j_{k}, \mu_{\text{sam}})) \in [q_{i}, q_{i+1}]\\ (\lambda_k, j_k)\in M}} \frac{1}{F(\theta(\lambda_{k}, j_{k}, \mu_{\text{ref}}), w_{\text{ref}}) R_{\text{supermirror}}(Q(\lambda_{k}, \theta(\lambda_{k}, j_{k}, \mu_{\text{ref}})))}
$$
The above expressions and {eq}`reflectivity` lets us express $R(Q_{i+{\frac{1}{2}}})$ and its uncertainty in terms of the neutron counts in the detector.


## More efficient evaluation of the reference intensity
The above expression for the reference intensity is cumbersome to compute because it is a sum over the reference measurement event list, and the reference measurement is large compared to the sample measurement.

Therefore we back up a bit. Consider the expression for the reference intensity, replacing the integrand with a generic $I(\lambda, j)$ it looks something like:

$$
I_{\text{ideal}}(Q_{i+\frac{1}{2}}) = \int_{Q(\lambda, \theta(\lambda, j, \mu_{\text{sam}}))  \in [q_{i}, q_{i+1}]} I(\lambda, j) \ d\lambda  \ dj.
$$

In the previous section we approximated the integral by summing over all events in the reference measurement.

Alternatively, we could define a $\lambda$ grid with edges $\lambda_{k}$ for $k=1\ldots N$ and approximate the integration region as the union of a subset of the grid cells:

$$
\int_{Q(\lambda, \theta(\lambda, j, \mu_{\text{sam}}))  \in [q_{i}, q_{i+1}]} I(\lambda, j) \ d\lambda  \ dj
\approx \sum_{Q(\bar{\lambda}_{k+\frac{1}{2}},\ \theta(\bar{\lambda}_{k+\frac{1}{2}}, j, \mu_{\text{sam}}))  \in [q_{i}, q_{i+1}]} I_{k+\frac{1}{2},j}
$$
where

$$
I_{k+\frac{1}{2},j} = \int_{\lambda \in [\lambda_{k}, \lambda_{k+1}]} I(\lambda, j) \ d\lambda
$$
and $\bar{\lambda}_{k+\frac{1}{2}} = (\lambda_{k} + \lambda_{k+1}) / 2$.

Why would this be more efficient than the original approach? Note that $I_{k+\frac{1}{2}, j}$ does not depend on $\mu_{\text{sam}}$, and that it can be computed once and reused for all sample measurements.
This allows us to save computing time for each new sample measurement, as long as $|EV_{\text{ref}}| >> NM$ where $M$ is the number of detector pixels and $N$ is the size of the $\lambda$ grid.

However, ideally computing the reference intensity should be quick compared to reducing the sample measurement. And since a reasonable value for $N$ is approximately $500$, and $M\approx 30000$, and a sample measurement is likely less than $10$ million events, the cost of computing the reference measurement is still considerable compared to reducing the sample measurement.

Therefore there's one more approximation that is used to further reduce the cost of computing the reference intensity.
The description of the final approximation is instrument specific.
In the next section it is described specifically for the Amor instrument.


### Evaluating the reference intensity for the Amor instrument

The Amor detector has three logical dimensions, `blade`, `wire` and `strip`. It happens to be the case that $\theta(\lambda, j)$ is almost the same for all $j$ belonging to the same `strip` of the detector.
We can express this as

$$
\theta(\lambda, j, \mu_{\text{sam}}) \approx \bar{\theta}(\lambda, \mathrm{bladewire}(j), \mu_{\text{sam}})
$$
where $\bar{\theta}$ is an approximation for $\theta$ that only depends on the blade and the wire of the pixel where the neutron was detected.
Then the above expression for the reference intensity can be rewritten as

$$
\int_{Q(\lambda, \bar{\theta}(\lambda, z, \mu_{\text{sam}}))  \in [q_{i}, q_{i+1}]} \int_{\mathrm{bladewire}(j) = z} I(\lambda, j) \ dj \ d\lambda  \ dz
\approx \sum_{Q(\bar{\lambda}_{k+\frac{1}{2}}, \bar{\theta}(\bar{\lambda}_{k+\frac{1}{2}}, z, \mu_{\text{sam}}))  \in [q_{i}, q_{i+1}]} I_{k+\frac{1}{2},z}
$$
where

$$
I_{k+\frac{1}{2},z} = \int_{\lambda \in [\lambda_{k}, \lambda_{k+1}]} \int_{\mathrm{bladewire}(j) = z} I(\lambda, j) \ dj \ d\lambda .
$$
Like before, the benefit of doing this is that

$$
 \int_{\lambda \in [\lambda_{k}, \lambda_{k+1}]} \int_{\mathrm{bladewire}(j) = z} I(\lambda, j) \ dj \ d\lambda
$$
can be pre-computed because it doesn't depend on $\mu_{\text{sam}}$.
But unlike before $I_{k+\frac{1}{2},z}$ now has a much more manageable size, about 64x smaller than the first attempt.
This makes it comfortably smaller than the sample measurement.

