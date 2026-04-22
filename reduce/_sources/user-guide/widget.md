# Workflow Widget in Jupyter Notebooks

You can run a workflow with the ESSreduce widget inside a Jupyter notebook.

## Widget with All Workflows

`workflow_widget` builds a widget that lets you choose from the workflows registered in `ess.reduce.workflow.workflow_registry`.

To make workflows from other `ess` packages available, import the module that registers them before creating the widget, as shown in the [Voila example](#deploying-voila-application).

## Deploying Voila Application

The notebook needs to import `workflow_widget` and at least one module that registers workflows:


```python
import ess.dream  # Example: importing a package registers its workflows.
from ess.reduce.ui import workflow_widget

widget = workflow_widget()
```

You can then deploy the notebook as an application using `voila`:

```bash
voila {PATH_TO_THE_NOTEBOOK}
```

## Widget from a Workflow in a Notebook

```python
%matplotlib widget
from ess.reduce.ui import workflow_widget

workflow_widget()
```
