# Workflow Design

## Introduction

### Traditional data-reduction workflows

Traditionally, users are supplied with a toolbox of algorithms and optionally a reduction script or a notebook that uses those algorithms.
Conceptually this looks similar to the following:

```python
sample = load_sample(run=12345)
background = load_background(run=12300)
direct_beam = load_direct_beam(run=10000)
mask_detectors(sample)
mask_detectors(background)
sample_monitors = preprocess_monitors(sample)
background_monitors = preprocess_monitors(background)
transmission_fraction = transmission_fraction(**sample_monitors)
sample_iofq = compute_i_of_q(sample, direct_beam, transmission_fraction)
transmission_fraction = transmission_fraction(**background_monitors)
background_iofq = compute_i_of_q(background, direct_beam, transmission_fraction)
iofq = subtract_background(sample_iofq, background_iofq)
```

This is an *imperative workflow*, where the user specifies the order of operations and the dependencies between them.
This is not ideal for a number of reasons:

- The user has to know the order of operations and the dependencies between them.
- The user has to know which algorithms to use.
- The user has to know which parameters to use for each algorithm.
- The user has to know which data to use for each algorithm.
- The user can easily introduce mistakes into a workflow, e.g., by using the wrong order of operations, or by overwriting data.
  This is especially problematic in Jupyter notebooks, where the user can easily run cells out of order.

Our most basic programming models provide little help to the user.
For example we typically write components of reduction workflows as functions of `scipp.Variable` or `scipp.DataArray` objects:

```python
def transmission_fraction(
    incident_monitor: sc.DataArray,
    transmission_monitor: sc.DataArray,
"""
Compute transmission fraction from incident and transmission monitors.

Parameters
----------
incident_monitor:
    Incident monitor.
transmission_monitor:
    Transmission monitor.
"""
) -> sc.DataArray:
    return transmission_monitor / incident_monitor
```

Here, we rely on naming of function parameters as well as docstrings to convey the meaning of the parameters, and it is up to the user to pass the correct inputs.
While good practices such as keyword-only arguments can help, this is still far from a scalable and maintainable solution.

As an improvement, we could adopt an approach with more specific domain types, e.g.,

```python
def transmission_fraction(
    incident_monitor: IncidentMonitor,
    transmission_monitor: TransmissionMonitor,
) -> TransmissionFraction:
    return transmission_monitor / incident_monitor
```

We could now run [mypy](https://mypy-lang.org/) on reduction scipts to ensure that the correct types are passed to each function.
However, this is not practical with dynamic workflows, i.e., when users modifying workflows in a Jupyter notebooks on the fly.
Aside from this, such an approach would still not help with several of the other issues listed above.


### High-level summary of proposed approach

We propose an architecture combining *domain-driven design* with *dependency injection*.
Dependency injection aids in building a declarative workflow.
We define domain-specific concepts that are meaningful to the (instrument) scientist.
Simple functions provide workflow components that define relations between these domain concepts.

Concretely, we propose to define specific domain types, such as `IncidentMonitor`, `TransmissionMonitor`, and `TransmissionFraction` in the example above.
However, instead of the user having to pass these to functions, we use dependency injection to provide them to the functions.
In essence this will build a workflow's task graph.

From the [Guice documentation](https://github.com/google/guice/wiki/MentalModel#injection) (Guice is a dependency injection framework for Java):

> "This is the essence of dependency injection. If you need something, you don't go out and get it from somewhere, or even ask a class to return you something. Instead, you simply declare that you can't do your work without it, and rely on Guice to give you what you need.
>
> This model is backwards from how most people think about code: it's a more *declarative model* rather than an *imperative one*. This is why dependency injection is often described as a kind of inversion of control (IoC)."
> (emphasis added)


### Domain-driven design

Domain-Driven Design (DDD) is an approach to software development that aims to make software more closely match the domain it is used in.
The obvious benefit of this is that it makes it easier for domain experts to understand and modify the software.

How should we define the domain for the purpose of data reduction?
Looking at, e.g., Mantid, we see that the domain is defined as data reduction for any type of neutron scattering experiment.
This has led to more than 1000 algorithms, making it hard for users to know how to use them.
Furthermore, while algorithms provide some sort of domain-specific language, the data types are generic.

What we propose here is to define the domain more narrowly, highly specific to a technique or even specific to an instrument.
This will reduce the scope to cover in the domain-specific language.
By making data types specific to the domain, we provide nouns for the domain-specific language.


### Dependency injection

Dependency injection is a common technique for implementing the [inversion of control](https://en.wikipedia.org/wiki/Inversion_of_control) principle.
It makes components of a system more loosely coupled, and makes it easier to replace components, including for testing purposes.
Dependency injection can be performed manually, but there are also frameworks that can help with this.

## Architecture

### In a nutshell

1. The user will define building blocks of a workflow using highly specific domain types for the type-hints, such as `IncidentMonitor`, `TransmissionMonitor`, and `TransmissionFraction`, e.g.,

   ```python
   def transmission_fraction(
       incident_monitor: IncidentMonitor,
       transmission_monitor: TransmissionMonitor,
   ) -> TransmissionFraction:
       return transmission_monitor / incident_monitor
   ```

2. The user passes a set of building blocks to the system, which assembles a dependency graph based on the type-hints.
3. The user requests a specific output from the system using one of the domain types.
   This may be computed directly, or the system may construct a `dask` graph to compute the output.

Depending on the level of expertise of the user and the level of control they need, step 1.) or step 1.) and 2.) will be omitted, as we will provide pre-defined building blocks and sets of building blocks for common use cases.

### Parameter handling

Generally, the user must provide configuration parameters to a workflow.
In many cases there are defaults that can be used.
In either case, these parameters must be associated with the correct step in the workflow.
This is complicated by the non-linear nature of the workflow.
A flat list of parameters has been used traditionally, relying entirely on parameter naming.
This is problematic for two reasons:
First, certain basic workflow steps may be used in multiple places.
Second, workflows frequently contain nested steps, which may have the same parameters (or not).
This makes the process of setting parameters somewhat opaque and error-prone.
Furthermore, it relies on a hand-written higher-level workflow to set parameters for nested steps, mapping between globally-uniquely-named parameters and the parameters of the nested steps.
These, in turn, require complicated testing.

A hierarchical parameter system could provide an alternative, but makes it harder to set "global" parameters.
For example, we may want to use the same wavelength-binning for all steps in the workflow.

We propose to handle parameters as dependencies of workflow steps.
That is, the dependency-injection system is used to provide parameters to workflow steps.
Parameters are identified via their type, i.e., we will require defining a domain-specific type for each parameter, such as `WavelengthBinning`.
For nested workflows, we can use a child injector, which provides a scope for parameters.
Parent-scopes can be searched for parameters that are not found in the child-scope, providing a mechanism for "global" parameters.

### Metadata handling

There have been a number of dicussions around metadata handling.
For example, the support (or non-support) of an arbitrary `attrs` dict as part of `scipp.Variable` and `scipp.DataArray`.
Furthermore, we may have metadata that is part of the data-catalog, which may partially overlap with the metadata that is part of the data itself.
The current conclusion is that any attempt to handle metadata in a generic and automatic way will not work.
Therefore, if a user wants to provide metadata for a workflow result, they must do so *explicitly* by specifying functions that can assemble that metadata.
As with regular results, this can be done by injecting the input metadata into the function that computes the result's metadata.

### Domain-specific types

`typing.NewType` can be used as a simple way of creating a domain-specific type for type-checking.
For example, we can use it to create a type for a `scipp.DataArray` that represents a transmission monitor.
This avoids a more complex solution such as creating a wrapper class or a subclass:

```python
import typing

TransmissionMonitor = typing.NewType('TransmissionMonitor', scipp.DataArray)
```

Note that this does not create a new type, but rather a new name for an existing type.
That is, `isinstance(monitor, TransmissionMonitor)` does not work, since `TransmissionMonitor` is not a type.
Furthermore, operations will revert to the underlying type, e.g., `monitor * 2` will return a `scipp.DataArray`.
For this application this would actually be desired behavior:
Applying an operation to a domain type will generally result in a different type, so falling back to the underlying type is the correct behavior and forces the user to be explicit about the type of the result.

### Model workflow

We define a model workflow, which we will use to illustrate the architecture.

```xmermaid
%%{init: {"flowchart": {"defaultRenderer": "elk"}} }%%
graph TD
    ExperimentId-->DirectBeam
    ExperimentId-->PixelMask
    subgraph "sample reduction"
    RawSample([RawSample])
    MaskedSample([MaskedSample])
    Sample([Sample])
    SampleIofQ([SampleIofQ])
    SampleIncidentMonitor
    SampleTransmissionMonitor
    SampleTransmissionFraction
    SampleSolidAngle
    end
    subgraph "background reduction"
    RawBackground([RawBackground])
    MaskedBackground([MaskedBackground])
    Background([Background])
    BackgroundIofQ([BackgroundIofQ])
    BackgroundIncidentMonitor
    BackgroundTransmissionMonitor
    BackgroundTransmissionFraction
    BackgroundSolidAngle
    end
    SampleRunId-->RawSample
    DirectRunId-->RawDirect
    BackgroundRunId-->RawBackground
    RawSample-->SampleIncidentMonitor
    RawSample-->SampleTransmissionMonitor
    RawDirect-->DirectIncidentMonitor
    RawDirect-->DirectTransmissionMonitor
    RawBackground-->BackgroundIncidentMonitor
    RawBackground-->BackgroundTransmissionMonitor
    SampleIncidentMonitor-->SampleTransmissionFraction
    SampleTransmissionMonitor-->SampleTransmissionFraction
    BackgroundIncidentMonitor-->BackgroundTransmissionFraction
    BackgroundTransmissionMonitor-->BackgroundTransmissionFraction
    DirectIncidentMonitor-->SampleTransmissionFraction
    DirectTransmissionMonitor-->SampleTransmissionFraction
    DirectIncidentMonitor-->BackgroundTransmissionFraction
    DirectTransmissionMonitor-->BackgroundTransmissionFraction
    RawSample==>MaskedSample==>Sample==>SampleIofQ
    RawBackground==>MaskedBackground==>Background==>BackgroundIofQ
    PixelMask-->MaskedSample
    PixelMask-->MaskedBackground
    BeamCenter-->Sample
    BeamCenter-->Background
    Sample-->SampleSolidAngle
    Background-->BackgroundSolidAngle
    QBinning-->SampleIofQ
    QBinning-->BackgroundIofQ
    SampleSolidAngle-->SampleIofQ
    BackgroundSolidAngle-->BackgroundIofQ
    SampleTransmissionFraction-->SampleIofQ
    BackgroundTransmissionFraction-->BackgroundIofQ
    DirectBeam-->SampleIofQ
    DirectBeam-->BackgroundIofQ
    SampleIofQ==>IofQ([IofQ])
    BackgroundIofQ==>IofQ
```

Note that the subgraphs for sample and background reduction are identical.
A number of details have been omitted for clarity.
For example, there are typically more parameters provided by the user.

![Model workflow](workflow-design-containers-and-modules.svg)


```xmermaid
graph TD
    subgraph params['Reduction Parameters']
    QBinning
    SampleRunID
    BackgroundRunID
    end

    subgraph exp["Experiment Config"]
    ExperimentId
    end

    subgraph scicat
    DirectBeam
    DetectorCalibration
    PixelMask
    end

    ExperimentId-->DirectBeam
    ExperimentId-->DetectorCalibration
    ExperimentId-->PixelMask

    subgraph data extraction
    RawData
    Detector
    IncidentMonitor
    TransmissionMonitor
    MetaData
    end

    subgraph reduction
    SolidAngle
    BeamCenter
    TransmissionFraction
    NormalizedIofQ
    end

    Detector-->SolidAngle
    Detector-->BeamCenter

    IncidentMonitor-->TransmissionFraction
    TransmissionMonitor-->TransmissionFraction

    Detector-->NormalizedIofQ
    DirectBeam-->NormalizedIofQ
    TransmissionFraction-->NormalizedIofQ
    QBinning-->NormalizedIofQ
    SolidAngle-->NormalizedIofQ
```

### Examples

1. Swap provider of `TransmissionFraction` for provider that handles wide-angle-correction
2. Swap provider of `BeamCenter` for provider that returns a constant value.
   This could also be by setting the value in the reduction config.
   Note that the injection sytem must make sure to only allow for unique sources of thruth.

## Dask

We can use this to build a dask graph.
This will allow for computing intermediate results without recomputing everything for every subresult.

## Multiple injectors

- Top level "experiment injector".
  Provides everything that is experiment-specific.
- Sub-injectors (child injectors) for sample and background reduction.
  They may pull things from the experiment injector, but otherwise provide an independent scope.
- More advanced example/problem: We may want to compute an experiment-level `BeamCenter`, based on a sample run, but it has to be made available to other sub-injectors.

```python
iofq = reduction.get(IfofQ)
iofq.compute()  # maybe

# Better
def save(iofq: IofQ, meta: ReductionMetadata):
    pass

# `call` would call dask.compute on the injected parameters
# or the task made from the function.
reduction.call(save)
```

We can also use injection for dataclasses to request multiple outputs.
It is not clear how to handle sub-injectors in the syntax.

Top-level container:

- scicat
- exp config
- reduction module
- global params

Two child containers, sample and background, each adding run config as well as run-specific user params, inheriting others from containing scope.
How would a user refer to those on the top level?
Is it allowed to access nested containers?

## Reducing multiple runs

## Notes

Instead of user calling `injector.get(IofQ)`, let them declare a function that needs inputs (could be multiple), the injection system can compute and inject all of them.
Instead of:

```python
sample_iofq = injector.get(IofQ)
iofq = injector.get(BackgroundSubtractedIofQ)
sample_iofq, iofq = dask.compute([sample_iofq, iofq])
```

Use:

```python
def process_results(sample_iofq: IofQ, iofq: BackgroundSubtractedIofQ):
    pass

injector.call_with_injection(process_results)
```


## TODO

- Validators, and validation ahead of computation?
- Can we inject the "runner" into the workflow?
  This could be dask (to build a task graph), or a tracer object, to build a tree for display in documentation.
  It is also very simple to hard-code building an injection-level task graph for visualization purposes.


Our top-level container is the composition root.
It is the only place where we should use `injector.get` directly, to get the top-level workflow.
Example:

```python
container = Injector()
workflow = container.get(BackgroundSubtractedIofQ)
workflow.run()
```

The container must contain bindings for all dependencies of the workflow.
In particular, it needs `SampleIofQ` and `BackgroundIofQ`.
These are provided by child containers, which are basically identical, aside from configuration.
The child containers could be created by the top-level container, which provides the configuration.
We need to translate from an `IofQ` which can be provided by a child container, to a `BackgroundIofQ` which is required by the top-level container.
This can be done by a binding in the top-level container, which uses the child container to get the `IofQ` and then converts it to a `BackgroundIofQ`.
This is a bit of a hack, but it is the only way to get the child container to provide the `IofQ` to the top-level container.
The top-level container can then use the `BackgroundIofQ` to create the `BackgroundSubtractedIofQ` workflow.

We can also directly create such a binding:

```python
def get_background_iofq(background: Injector):
    return BackgroundIofQ(background.get(IofQ))

container.bind(BackgroundIofQ, get_background_iofq)
```

As we will likely use this repeatedly, we can create a helper function:

```python
def bind_child(container, child, parent_type, child_type):
    """
    Bind a child injector to a parent injector.

    Example
    -------
        bind_child(container, child, IofQ, BackgroundIofQ)
    """
    def get_child(parent: parent_type) -> child_type:
        return child_type(child.get(child_type))
    container.bind(child_type, get_child)
```

```python
container.bind_from_child(BackgroundIofQ, IofQ, modules)
```

Should the child add the binding in the parent?
But then the child needs to know "what" it is.

```python
container = injector.Injector()
background = container.create_child_injector(modules)
```

Dependency injection is all about separation of concerns.
To make `BackgroundIofQ` we need a background run (given by `BackgroundRunID`) and a container that can provide `IofQ`.

```python
import injector
from typing import NewType

BackgroundContainer = NewType('BackgroundContainer', injector.Injector)

class BackgroundModule(injector.Module):
    def configure(self, binder: injector.Binder):
        # Binder does not have this method, but knows its parent, so we can solve this.
        background = binder.create_child_injector(background_config)
        binder.bind(BackgroundContainer, background, scope=injector.SingletonScope)

    @injector.provide
    def get_background_iofq(self, background: BackgroundContainer) -> BackgroundIofQ:
        return BackgroundIofQ(background.get(IofQ))
```

The same would be done for `SampleIofQ`.
But how can we handle multiple samples?
One solution is to make a new container for every sample, creating one workflow per sample, but then the background cannot be shared.
What we really want is to configure the container with a list of sample run IDs.
Then the container should provide us with a workflow that processes all of them, and returns a list of results.

```python
sample_modules = [SampleModule(run_id) for run_id in sample_run_ids]
container = injector.Injector([BackgroundModule, sample_modules])
container.get(BackgroundSubtractedIofQ)  # Returns a list of results.
```

How can we make this work?
We do not want to change the code that does the background subtraction.
That is, we have a list of `SampleIofQ`, but need to iterate over the list in all subsequent code.

We could use `injector.Binder.multibind` to create a list of `SampleIofQ`.
This may work, but we would need to change the code that does the background subtraction, but maybe that is the best solution?
And how can be bind the result, as list/dict seems to be reserved for multibind?

```python
import injector
from typing import NewType


def SampleModule(run_id):
    SampleContainer = NewType('SampleContainer', injector.Injector)
    class SampleModule(injector.Module):
        def configure(self, binder: injector.Binder):
            sample = binder.create_child_injector(sample_config(run_id))
            binder.bind(SampleContainer, sample, scope=injector.SingletonScope)

        @injector.provide
        def provide_sample_iofq(self, sample: SampleContainer) -> SampleIofQ:
            return SampleIofQ(sample.get(IofQ))

        # If we have multiple samples, we need to multibind them. Not like this.
        @injector.provide
        def provide_background_subtracted(self,
                                          sample: SampleIofQ,
                                          background: BackgroundIofQ
        ) -> BackgroundSubtractedIofQ:
            return BackgroundSubtractedIofQ(sample - background)

    return SampleModule
```

Can we use a special container type for automatic handling of workflow branch replicas?

```python
import injector
from typing import NewType

# Use a dedicated container type, specific to the meaning!
Runs = NewType('Runs', list)
# Question to ask: Why do we have different samples?
# This should be used as a name for the container type.
# Example 1: We grew different crystals, and want to compare them.
Crystals = NewType('Crystals', list)

def get_all_iofq(samples: Runs[IofQ]):
    # samples is injected with an iterable of IofQ, each from
    # a different child container.
```

We would then have a provider of a container of, e.g., `RunID`, and a template container that takes a single `RunID` and returns an `IofQ`.
The template container would be used to create a child container for each `RunID` in the list, with an automatic system of bindings for containers of results as well as index-based access to the results.

```python
import injector
from typing import NewType

Runs = NewType('Runs', list)
RunID = NewType('RunID', int)
IofQ = NewType('IofQ', float)

class Config(injector.Module):
    def configure(self, binder: injector.Binder):
        # Can we use bind or multibind, or do we need something custom?
        binder.bind(Runs[RunID], [RunID(1), RunID(2), RunID(3)])

# Is this a module, or do we need a new type, "ModuleTemplate",
# which is used internally to instantiate a module for each item in the list?
class Reduction(injector.Module):
    @injector.provide
    def get_iofq(self, run_id: RunID) -> IofQ:
        return IofQ(0.5* run_id)

reduction_module = from_template(Runs, template=Reduction)
container = injector.Injector([Config, reduction_module])
container.get(Runs[IofQ])  # Returns [IofQ(0.5), IofQ(1.0), IofQ(1.5)]
```

```python
import injector
from typing import NewType, TypeVar, Generic

# Config providers Runs[RunID], we need to create a container for each RunID.

T = TypeVar('T')

class Runs(Generic[T], list):
    pass

class Container:
    def get(self, key):
        # Check if key is Runs[T]
        # If so, get T from all child containers, and return a list.
        return Runs([c.get(T) for c in self.children])

container.get(Runs[IofQ])
```


```python
class MappingModule(injector.Module):
    @injector.provide
    def provide_iofq(self, Runs[Container]) -> Runs[IofQ]:
        return Runs([c.get(IofQ) for c in self.children])

get(Runs[IofQ])
```
