# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import glob
import os
import re
import uuid
from collections.abc import Callable

import h5py
import ipywidgets as widgets
import pandas as pd
import plopp as pp
import scipp as sc
from ipydatagrid import DataGrid, TextRenderer, VegaExpr
from IPython.display import display
from ipytree import Node, Tree
from traitlets import Bool

from ess import amor
from ess.amor.types import ChopperPhase
from ess.reflectometry.figures import wavelength_z_figure
from ess.reflectometry.types import (
    Filename,
    QBins,
    ReducedReference,
    ReducibleData,
    ReferenceRun,
    ReflectivityOverQ,
    SampleRun,
    SampleSize,
    WavelengthBins,
    YIndexLimits,
    ZIndexLimits,
)
from ess.reflectometry.workflow import with_filenames


def _get_selected_rows(grid):
    return (
        pd.concat(
            [
                grid.get_visible_data().iloc[s['r1'] : s['r2'] + 1]
                for s in grid.selections
            ]
        )
        if grid.selections
        else None
    )


class DetectorView(widgets.HBox):
    is_active_tab = Bool(False).tag(sync=True)

    def __init__(
        self, runs_table: DataGrid, run_to_filepath: Callable[[str], str], **kwargs
    ):
        super().__init__([], **kwargs)
        self.runs_table = runs_table
        self.run_to_filepath = run_to_filepath
        self.plot_log = widgets.VBox([])
        self.working_label = widgets.Label(
            "...working", layout=widgets.Layout(display='none')
        )
        self.children = (
            widgets.VBox(
                [
                    widgets.Label("Runs Table"),
                    self.runs_table,
                ],
                layout={"width": "35%"},
            ),
            widgets.VBox(
                [
                    widgets.HBox(
                        [
                            widgets.Label("Wavelength z-index counts distribution"),
                            self.working_label,
                        ]
                    ),
                    self.plot_log,
                ],
                layout={"width": "60%"},
            ),
        )

        def run_when_selected_row_changes(change):
            # Runs when there are no previous selections,
            # or the new selection is different from the old.
            if not change['old'] or (
                change['new'] and change['new'][0]['r1'] != change['old'][0]['r1']
            ):
                self.run_workflow()

        self.runs_table.observe(run_when_selected_row_changes, names='selections')

        def run_when_active_tab(change):
            if change['new']:
                self.run_workflow()

        self.observe(run_when_active_tab, 'is_active_tab')

    def run_workflow(self):
        selected_rows = _get_selected_rows(self.runs_table)
        if not self.is_active_tab or selected_rows is None:
            return

        self.working_label.layout.display = ''
        run = selected_rows.iloc[0]['Run']

        workflow = amor.AmorWorkflow()
        workflow[SampleSize[SampleRun]] = sc.scalar(10, unit='mm')
        workflow[SampleSize[ReferenceRun]] = sc.scalar(10, unit='mm')

        workflow[ChopperPhase[ReferenceRun]] = sc.scalar(7.5, unit='deg')
        workflow[ChopperPhase[SampleRun]] = sc.scalar(7.5, unit='deg')

        workflow[YIndexLimits] = (0, 64)
        workflow[ZIndexLimits] = (0, 16 * 32)
        workflow[WavelengthBins] = sc.geomspace(
            'wavelength',
            2,
            13.5,
            2001,
            unit='angstrom',
        )
        workflow[Filename[SampleRun]] = self.run_to_filepath(run)
        da = workflow.compute(ReducibleData[SampleRun])
        da.bins.data[...] = sc.scalar(1.0, variance=1.0, unit=da.bins.unit)
        da.bins.unit = 'counts'
        da.masks.clear()
        da.bins.masks.clear()
        p = wavelength_z_figure(da, wavelength_bins=workflow.compute(WavelengthBins))
        self.plot_log.children = (p,)
        self.working_label.layout.display = 'none'


class NexusExplorer(widgets.VBox):
    def __init__(
        self, runs_table: DataGrid, run_to_filepath: Callable[[str], str], **kwargs
    ):
        kwargs.setdefault('layout', {"width": "100%"})
        super().__init__(**kwargs)
        self.runs_table = runs_table
        self.run_to_filepath = run_to_filepath

        # Create tree widget for Nexus structure
        self.nexus_tree = Tree(
            layout=widgets.Layout(
                width='100%',
                height='100%',  # Fill the container height
            )
        )
        self.nexus_tree.nodes = [Node("Select a run to view its structure")]

        # Add selection handler to runs table
        self.runs_table.observe(self.update_nexus_view, names='selections')

        # Create content viewer widget
        self.nexus_content = widgets.Textarea(
            value='Select a node to view its content',
            layout=widgets.Layout(width='100%', height='600px'),
            disabled=True,  # Make it read-only
        )

        # Add selection handler to tree
        self.nexus_tree.observe(self.on_tree_select, names='selected_nodes')

        # Create the Nexus Explorer tab content
        self.children = (
            widgets.Label("Nexus Explorer"),
            widgets.HBox(
                [
                    widgets.VBox(
                        [
                            widgets.Label("Runs Table"),
                            self.runs_table,
                        ],
                        layout={"width": "30%"},
                    ),
                    widgets.VBox(
                        [
                            widgets.Label("File Structure"),
                            widgets.VBox(
                                [self.nexus_tree],
                                layout=widgets.Layout(
                                    width='100%',
                                    height='600px',
                                    min_height='100px',  # Min resize height
                                    max_height='1000px',  # Max resize height
                                    overflow_y='scroll',
                                    border='1px solid lightgray',
                                    resize='vertical',  # Add resize handle
                                ),
                            ),
                        ],
                        layout={"width": "35%"},
                    ),
                    widgets.VBox(
                        [
                            widgets.Label("Content"),
                            self.nexus_content,
                        ],
                        layout={"width": "35%"},
                    ),
                ]
            ),
        )

    def create_hdf5_tree(self, filepath):
        """Create a tree representation of an HDF5 file structure."""

        def create_node(name, obj, path=''):
            full_path = f"{path}/{name}" if path else name
            if isinstance(obj, h5py.Dataset):
                # For datasets, show shape and dtype
                display_name = f"{name} ({obj.shape}, {obj.dtype})"
                node = Node(display_name, opened=False, icon='file')
                node.nexus_path = full_path  # Store path as custom attribute
                return node
            else:
                # For groups, create parent node and add children
                parent = Node(name, opened=False, icon='folder')
                parent.nexus_path = full_path  # Store path as custom attribute
                # Just iterate over the keys directly
                for child_name in obj.keys():
                    parent.add_node(create_node(child_name, obj[child_name], full_path))
                return parent

        try:
            with h5py.File(filepath, 'r') as f:
                root_node = create_node('', f)
                return Tree(nodes=[root_node])
        except Exception as e:
            # Use explicit conversion flag
            return Tree(nodes=[Node(f"Error loading file: {e!s}")])

    def update_nexus_view(self, *_):
        """Update the Nexus file viewer based on selected run."""
        selected_rows = _get_selected_rows(self.runs_table)
        if selected_rows is None:
            self.nexus_tree.nodes = [Node("Select a run to view its structure")]
            return

        run = selected_rows.iloc[0]['Run']
        filepath = self.run_to_filepath(run)

        # Create and display the tree for this file
        new_tree = self.create_hdf5_tree(filepath)
        self.nexus_tree.nodes = new_tree.nodes

    def display_nexus_content(self, path, h5file):
        """Display the content of a Nexus entry."""
        try:
            item = h5file[path] if path else h5file
            content = []

            # Show attributes if any
            if len(item.attrs) > 0:
                content.append("Attributes:")
                for key, value in item.attrs.items():
                    content.append(f"  {key}: {value}")

            # Show dataset content if it's a dataset
            if isinstance(item, h5py.Dataset):
                content.append("\nDataset content:")
                data = item[()]
                if data.size > 100:  # Truncate large datasets
                    content.append(f"  Shape: {data.shape}")
                    content.append("  First few values:")
                    content.append(f"  {data.flatten()[:100]}")
                    content.append("  ...")
                else:
                    content.append(f"  {data}")

            self.nexus_content.value = '\n'.join(content)
        except Exception as e:
            # Use explicit conversion flag
            self.nexus_content.value = f"Error reading content: {e!s}"

    def on_tree_select(self, event):
        """Handle tree node selection."""
        if not event['new']:  # No selection
            self.nexus_content.value = "Select a node to view its content"
            return

        selected_node = event['new'][0]

        # Get the path from the custom attribute
        path = getattr(selected_node, 'nexus_path', selected_node.name)

        selected_rows = _get_selected_rows(self.runs_table)
        if selected_rows is None:
            return

        run = selected_rows.iloc[0]['Run']
        filepath = self.run_to_filepath(run)

        with h5py.File(filepath, 'r') as f:
            self.display_nexus_content(path, f)


class ReflectometryBatchReductionGUI:
    """GUI for batch reduction of reflectometry data."""

    def read_meta_data(self, path):
        'Reads metadata from the hdf5 file at ``path``'
        raise NotImplementedError()

    def sync_runs_table(self, db):
        'Returns the updated runs table based after metadata has been updated'
        raise NotImplementedError()

    def sync_reduction_table(self, db):
        'Returns the updated reduction table based after runs table has been updated'
        raise NotImplementedError()

    def sync_reference_table(self, db):
        'Returns the updated reference table based after runs table has been updated'
        raise NotImplementedError()

    def sync_custom_reduction_table(self):
        '''Returns the updated custom reduction table after
        the custom reduction table has been updated'''
        raise NotImplementedError()

    def display_results(self):
        '''Displays the results that are currently selected'''
        raise NotImplementedError()

    def run_workflow(self):
        raise NotImplementedError()

    def get_row_key(self, row):
        'Key determines if a result needs to be recomputed or not'
        raise NotImplementedError()

    def set_table_colors(self, table):
        template = 'row == {i} ? {reduced_color} : '
        expr = ''
        for i, (_, row) in enumerate(table.data.iterrows()):
            for row_key in self.results.keys():
                if self.get_row_key(row) == row_key:
                    expr += template.format(i=i, reduced_color="'lightgreen'")
        expr += "default_value"
        for renderer in table.renderers.values():
            renderer.background_color = VegaExpr(expr)
        table.default_renderer.background_color = VegaExpr(expr)

    @staticmethod
    def set_table_height(table, extra=0):
        height = (len(table.data) + 1) * (table.base_row_size + 1) + 5 + extra
        table.layout.height = f'{height}px'

    def set_result(self, metadata, result):
        self.results[self.get_row_key(metadata)] = result
        self.set_table_colors(self.reduction_table)
        self.set_table_colors(self.custom_reduction_table)
        self.set_table_colors(self.reference_table)

    def get_renderers_for_reduction_table(self):
        return {}

    def get_renderers_for_reference_table(self):
        return {}

    def get_renderers_for_custom_reduction_table(self):
        return {}

    def get_renderers_for_runs_table(self):
        return {}

    def log(self, message):
        out = widgets.Output()
        with out:
            display(message)
        self.text_log.children = (out, *self.text_log.children)

    def sync(self, *_):
        db = {}
        # db["settings"] = self.load_settings()
        db["run_number_min"] = int(self.run_number_min.value)
        db["run_number_max"] = int(self.run_number_max.value)
        db["meta"] = self.load_runs()
        db["user_runs"] = self.runs_table.data
        db["user_reduction"] = self.reduction_table.data
        db["user_reference"] = self.reference_table.data

        db["user_runs"] = self.sync_runs_table(db)
        db["user_reduction"] = self.sync_reduction_table(db)
        db["user_reference"] = self.sync_reference_table(db)

        self.runs_table.data = db["user_runs"]
        self.reduction_table.data = db["user_reduction"]
        self.reference_table.data = db["user_reference"]

        self.set_table_height(self.runs_table)
        self.set_table_height(self.reduction_table)
        self.set_table_height(self.custom_reduction_table)
        self.set_table_colors(self.reduction_table)
        self.set_table_colors(self.custom_reduction_table)
        self.set_table_colors(self.reference_table)

    @property
    def path(self):
        if self._path is None:
            raise ValueError("Path is not set")
        return self._path

    def __init__(self):
        self.text_log = widgets.VBox([])
        self.progress_log = widgets.VBox([])
        self.plot_log = widgets.VBox([])
        self._path = None
        self.log("init")

        self.results = {}

        self.runs_table = DataGrid(
            pd.DataFrame([]),
            editable=True,
            auto_fit_columns=True,
            column_visibility={"key": False},
            selection_mode="cell",
            renderers=self.get_renderers_for_runs_table(),
        )
        self.reduction_table = DataGrid(
            pd.DataFrame([]),
            editable=True,
            auto_fit_columns=True,
            column_visibility={"key": False},
            selection_mode="cell",
            renderers=self.get_renderers_for_reduction_table(),
        )
        self.reference_table = DataGrid(
            pd.DataFrame([]),
            editable=True,
            auto_fit_columns=True,
            column_visibility={"key": False},
            selection_mode="cell",
            renderers=self.get_renderers_for_reference_table(),
        )
        self.custom_reduction_table = DataGrid(
            pd.DataFrame([]),
            editable=True,
            auto_fit_columns=True,
            column_visibility={"key": False},
            selection_mode="cell",
            renderers=self.get_renderers_for_custom_reduction_table(),
        )

        self.runs_table.on_cell_change(self.sync)
        self.reduction_table.on_cell_change(self.sync)
        self.reference_table.on_cell_change(self.sync)

        self.custom_reduction_table.on_cell_change(
            lambda _: self.sync_custom_reduction_table()
        )

        self.proposal_number_box = widgets.Text(
            value="",
            placeholder="Proposal number or file path",
            description="Proposal no.:",
            layout=widgets.Layout(description_width="auto"),
            disabled=False,
        )

        def set_proposal_number_state(state):
            if state == "good":
                self.proposal_number_box.layout.border = '2px solid green'
            if state == "bad":
                self.proposal_number_box.layout.border = '2px solid red'

        def on_proposal_number_change(_):
            p = self.proposal_number_box.value
            if p.isdigit():
                # Handling proposal numbers is not yet implemented
                self._path = None
                set_proposal_number_state("bad")
            elif not os.path.isdir(p):
                self._path = None
                set_proposal_number_state("bad")
            else:
                self._path = p
                set_proposal_number_state("good")
                self.sync()

        set_proposal_number_state("bad")
        self.proposal_number_box.observe(on_proposal_number_change, names='value')

        reduce_button = widgets.Button(description="Reduce")
        plot_button = widgets.Button(description="Plot")

        def reduce_data(_):
            self.log("reduce data")
            self.run_workflow()

        def plot_results(_):
            self.log("plot results")
            self.display_results()

        reduce_button.on_click(reduce_data)
        plot_button.on_click(plot_results)

        add_row_button = widgets.Button(description="Add row")
        delete_row_button = widgets.Button(description="Remove row")

        def add_row(_):
            self.log("add row")
            row = _get_selected_rows(self.reduction_table)
            if row is None:
                row = pd.DataFrame(
                    [
                        {
                            'Sample': '',
                            'Angle': 0.0,
                            'Runs': (),
                            'QBins': 391,
                            'QStart': 0.01,
                            'QStop': 0.3,
                            'Scale': 1.0,
                        }
                    ]
                )
            # To avoid a flickering scrollbar
            # - increase table height with some margin before adding row
            # - adjust able height afterwards
            self.set_table_height(self.custom_reduction_table, extra=25)
            self.custom_reduction_table.data = pd.concat(
                [self.custom_reduction_table.data, row]
            )
            self.set_table_height(self.custom_reduction_table)
            self.set_table_colors(self.custom_reduction_table)

        def delete_row(_):
            self.log("delete row")
            self.custom_reduction_table.data = self.custom_reduction_table.data.iloc[
                :-1
            ]
            self.set_table_height(self.custom_reduction_table)

        add_row_button.on_click(add_row)
        delete_row_button.on_click(delete_row)
        data_buttons = widgets.HBox([reduce_button, plot_button])

        self.run_number_min = widgets.IntText(
            value=0, description='', layout=widgets.Layout(width='5em')
        )
        self.run_number_max = widgets.IntText(
            value=9999, description='', layout=widgets.Layout(width='5em')
        )
        self.run_number_min.observe(self.sync, names='value')
        self.run_number_max.observe(self.sync, names='value')
        run_number_filter = widgets.HBox(
            [self.run_number_min, widgets.Label("<=Run<="), self.run_number_max]
        )

        tab_data = widgets.VBox(
            [
                widgets.HBox(
                    [
                        widgets.VBox(
                            [
                                widgets.Label("Runs Table"),
                                run_number_filter,
                                self.runs_table,
                            ],
                            layout={"width": "35%"},
                        ),
                        widgets.VBox(
                            [
                                data_buttons,
                                widgets.VBox(
                                    [
                                        widgets.Label("Auto Reduction Table"),
                                        self.reduction_table,
                                    ],
                                    layout={'margin': '10px 0'},
                                ),
                                widgets.VBox(
                                    [
                                        widgets.Label("Manual Reduction Table"),
                                        widgets.HBox(
                                            [add_row_button, delete_row_button],
                                            layout={'margin': '5px 0'},
                                        ),
                                        self.custom_reduction_table,
                                    ],
                                    layout={'margin': '10px 0'},
                                ),
                            ],
                            layout={"width": "60%"},
                        ),
                    ]
                ),
                widgets.VBox(
                    [
                        widgets.VBox(
                            [widgets.Label("Progress"), self.progress_log],
                            layout={'width': '100%', 'margin': '10px 0'},
                        ),
                        widgets.VBox(
                            [widgets.Label("Plots"), self.plot_log],
                            layout={'width': '100%', 'margin': '10px 0'},
                        ),
                    ]
                ),
            ]
        )

        tab_settings = widgets.VBox(
            [
                widgets.Label("This is the settings tab"),
                widgets.Label("Reference runs"),
                self.reference_table,
            ],
            layout={"width": "100%"},
        )

        tab_log = widgets.VBox(
            [widgets.Label("Messages"), self.text_log],
            layout={"width": "100%"},
        )

        self.tabs = widgets.Tab()
        self.tabs.children = [
            tab_data,
            tab_settings,
            tab_log,
        ]
        self.tabs.titles = ["Reduce", "Settings", "Log"]

        def on_tab_change(change):
            old = self.tabs.children[change['old']]
            new = self.tabs.children[change['new']]
            if hasattr(old, 'is_active_tab'):
                old.is_active_tab = False
            if hasattr(new, 'is_active_tab'):
                new.is_active_tab = True

        self.tabs.observe(on_tab_change, names='selected_index')

        self.main = widgets.VBox(
            [
                self.proposal_number_box,
                self.tabs,
            ]
        )

    def load_runs(self):
        self.log("load runs from path")
        metadata = [
            self.read_meta_data(fpath)
            for fpath in glob.glob(os.path.join(self.path, '*.hdf'))
        ]
        return pd.DataFrame(metadata)

    @property
    def widget(self):
        return self.main

    def log_text(self, message):
        out = widgets.Output()
        with out:
            display(message)
        self.text_log.children = (out, *self.text_log.children)

    def log_progress(self, progress):
        self.progress_log.children = (progress,)


class AmorBatchReductionGUI(ReflectometryBatchReductionGUI):
    def __init__(self):
        super().__init__()
        self.nexus_explorer = NexusExplorer(self.runs_table, self.get_filepath_from_run)
        self.detector_display = DetectorView(
            self.runs_table, self.get_filepath_from_run
        )
        self.tabs.children = (
            *self.tabs.children,
            self.nexus_explorer,
            self.detector_display,
        )
        # Empty titles are automatically added for the new children
        self.tabs.titles = [*self.tabs.titles[:-2], "Nexus Explorer", "Detector View"]

    def read_meta_data(self, path):
        with h5py.File(path) as f:
            return {
                "Sample": f['entry1']['sample']['name'][()][0].decode('utf8'),
                "Run": path[-8:-4],
                "Angle": f['entry1']['Amor']['master_parameters']['mu']['value'][0, 0],
            }

    def get_renderers_for_reduction_table(self):
        return {
            'Angle': TextRenderer(text_value=VegaExpr("format(cell.value, ',.3f')"))
        }

    def get_renderers_for_custom_reduction_table(self):
        return {
            'Angle': TextRenderer(text_value=VegaExpr("format(cell.value, ',.3f')"))
        }

    def get_renderers_for_runs_table(self):
        return {
            'Angle': TextRenderer(text_value=VegaExpr("format(cell.value, ',.3f')"))
        }

    @staticmethod
    def _merge_old_and_new_state(new, old, on, how='left'):
        old = old if on in old else old.assign(**{on: None})
        new = new if on in new else new.assign(**{on: None})
        df = new.merge(old, how=how, on=on)
        for right in df.columns:
            if right.endswith("_y"):
                new = right.removesuffix("_y")
                left = new + "_x"
                df[new] = df[right].combine_first(df[left])
                df = df.drop(columns=[left, right])
        return df

    @staticmethod
    def _setdefault(df, col, value):
        df[col] = value if col not in df.columns else df[col].fillna(value)

    @staticmethod
    def _ordercolumns(df, *cols):
        columns = [*cols, *sorted(set(df.columns) - {*cols})]
        return df[columns]

    def sync_runs_table(self, db):
        df = self._merge_old_and_new_state(db["meta"], db["user_runs"], on='Run')
        df = df[db['run_number_min'] <= df['Run'].astype(int)][
            db['run_number_max'] >= df['Run'].astype(int)
        ]
        self._setdefault(df, "Exclude", False)
        self._setdefault(df, "Comment", "")  # Add default empty comment
        df = self._ordercolumns(df, 'Run', 'Sample', 'Angle', 'Exclude', 'Comment')
        return df.sort_values(by='Run')

    def sync_reduction_table(self, db):
        df = db["user_runs"]
        df = (
            df[df["Sample"] != "sm5"][~df["Exclude"]]
            .groupby(["Sample", "Angle"], as_index=False)
            .agg(Runs=("Run", tuple))
            .sort_values(["Sample", "Angle"])
        )
        # We don't want changes to Sample or Angle made
        # in the user_reduction table to persist
        user_reduction = db['user_reduction'].drop(
            columns=["Sample", "Angle"], errors='ignore'
        )
        df = self._merge_old_and_new_state(df, user_reduction, on='Runs')
        self._setdefault(df, "QBins", 391)
        self._setdefault(df, "QStart", 0.01)
        self._setdefault(df, "QStop", 0.3)
        self._setdefault(df, "Scale", 1.0)
        df = self._ordercolumns(df, 'Sample', 'Angle', 'Runs')
        return df.sort_values(["Sample", "Angle"])

    def sync_reference_table(self, db):
        df = db["user_runs"]
        df = (
            df[df["Sample"] == "sm5"][~df["Exclude"]]
            .groupby(["Sample", "Angle"], as_index=False)
            .agg(Runs=("Run", tuple))
            .sort_values(["Sample", "Angle"])
        )
        # We don't want changes to Sample
        # in the user_reference table to persist
        user_reference = db['user_reference'].drop(
            columns=["Sample", "Angle"], errors='ignore'
        )
        df = self._merge_old_and_new_state(df, user_reference, on='Runs')
        self._setdefault(df, "Ymin", 17)
        self._setdefault(df, "Ymax", 47)
        self._setdefault(df, "Zmin", 60)
        self._setdefault(df, "Zmax", 380)
        self._setdefault(df, "Lmin", 3.0)
        self._setdefault(df, "Lmax", 12.5)
        df = self._ordercolumns(df, 'Sample', 'Angle', 'Runs')
        return df.sort_values(["Sample", "Angle"])

    def sync_custom_reduction_table(self):
        df = self.custom_reduction_table.data.copy()
        if 'Runs' in df.columns:
            df['Runs'] = df['Runs'].map(
                lambda x: tuple(x)
                if isinstance(x, tuple | list)
                else tuple(x.split(','))
            )
        self.custom_reduction_table.data = df
        self.set_table_colors(self.custom_reduction_table)

    def display_results(self):
        df = self.get_selected_rows()
        if len(df) == 0:
            self.log('There was nothing to display')
            return
        for _ in range(2):
            results = [
                self.results[key]
                for _, row in df.iterrows()
                if (key := self.get_row_key(row)) in self.results
            ]
            if len(results) == len(df):
                break
            # No results were found for some of the selected rows.
            # It hasn't been computed yet, so compute it and try again.
            self.run_workflow()

        def get_unique_names(df):
            # Create labels with Sample name and runs
            labels = [
                f"{params['Sample']} ({','.join(params['Runs'])})"
                for (_, params) in df.iterrows()
            ]
            duplicated_name_counter = {}
            unique = []
            for i, name in enumerate(labels):
                if name not in labels[:i]:
                    unique.append(name)
                else:
                    duplicated_name_counter.setdefault(name, 0)
                    duplicated_name_counter[name] += 1
                    unique.append(f'{name}_{duplicated_name_counter[name]}')
            return unique

        results = dict(zip(get_unique_names(df), results, strict=True))

        q4toggle = widgets.ToggleButton(value=False, description="R*Q^4")
        plot_box = widgets.VBox(
            [
                pp.plot(
                    results,
                    norm='log',
                    figsize=(12, 6),
                    vmin=1e-6,
                )
            ]
        )
        curve_toggles = [
            widgets.Checkbox(value=True, description=name) for name in results.keys()
        ]

        def make_plot(change):
            plot_box.children[0].ax.clear()
            plot = pp.plot(
                {k: v * sc.midpoints(v.coords['Q']) ** 4 for k, v in results.items()}
                if change['new']
                else results,
                norm='log',
                ax=plot_box.children[0].ax,
            )
            for toggle in curve_toggles:
                toggle_line(toggle.description, toggle.value, plot)
            plot_box.children = (plot,)

        q4toggle.observe(make_plot, names='value')

        remove_button = widgets.Button(icon='trash-alt', layout={'width': '40px'})
        remove_button.unique_id = uuid.uuid4()

        def remove_plot(own):
            self.plot_log.children = tuple(
                box for box in self.plot_log.children if own.unique_id != box.unique_id
            )

        remove_button.on_click(remove_plot)

        def toggle_line(name, value, figure):
            view = figure.view
            for artist in view.artists.values():
                if artist.label == name:
                    artist._line.set_visible(value)
                    artist._mask.set_visible(value)
                    if artist._error is not None:
                        for c in artist._error.get_children():
                            c.set_visible(value)
            view.canvas.draw()

        for toggle in curve_toggles:
            toggle.observe(
                lambda change: toggle_line(
                    change['owner'].description, change['new'], plot_box.children[0]
                ),
                names='value',
            )

        comment_box = widgets.Textarea(
            placeholder='Add comments about this plot here...',
            layout=widgets.Layout(width='75%', height='40px'),
        )
        box = widgets.VBox(
            [
                widgets.HBox([remove_button, q4toggle, comment_box]),
                widgets.HBox(curve_toggles),
                plot_box,
            ]
        )
        box.unique_id = remove_button.unique_id
        self.plot_log.children = (box, *self.plot_log.children)

    def get_filepath_from_run(self, run):
        fname = next(
            name
            for name in os.listdir(self.path)
            if re.match(f'amor\\d{{4}}n{int(run):06d}.hdf', name)
        )
        return os.path.join(self.path, fname)

    def get_row_key(self, row):
        reference_metadata = (
            tuple(self.reference_table.data.iloc[0])
            if len(self.reference_table.data) > 0
            else (None,)
        )
        return (tuple(row), tuple(reference_metadata))

    def get_selected_rows(self):
        chunks = [
            rows
            for table in (self.reduction_table, self.custom_reduction_table)
            if (rows := _get_selected_rows(table)) is not None
        ]
        # Select everything if nothing is selected
        if len(chunks) == 0:
            chunks = [self.reduction_table.data, self.custom_reduction_table.data]
        return pd.concat(chunks)

    def run_workflow(self):
        sample_df = self.get_selected_rows()
        reference_df = self.reference_table.data.iloc[0]

        workflow = amor.AmorWorkflow()
        workflow[SampleSize[SampleRun]] = sc.scalar(10, unit='mm')
        workflow[SampleSize[ReferenceRun]] = sc.scalar(10, unit='mm')

        workflow[ChopperPhase[ReferenceRun]] = sc.scalar(7.5, unit='deg')
        workflow[ChopperPhase[SampleRun]] = sc.scalar(7.5, unit='deg')

        workflow[WavelengthBins] = sc.geomspace(
            'wavelength',
            reference_df['Lmin'],
            reference_df['Lmax'],
            2001,
            unit='angstrom',
        )

        workflow[YIndexLimits] = (
            sc.scalar(reference_df['Ymin']),
            sc.scalar(reference_df['Ymax']),
        )
        workflow[ZIndexLimits] = (
            sc.scalar(reference_df['Zmin']),
            sc.scalar(reference_df['Zmax']),
        )

        progress = widgets.IntProgress(min=0, max=len(sample_df))
        self.log_progress(progress)

        if (key := self.get_row_key(reference_df)) in self.results:
            reference_result = self.results[key]
        else:
            reference_result = with_filenames(
                workflow,
                ReferenceRun,
                list(map(self.get_filepath_from_run, reference_df["Runs"])),
            ).compute(ReducedReference)
            self.set_result(reference_df, reference_result)

        workflow[ReducedReference] = reference_result
        progress.value += 1

        for _, params in sample_df.iterrows():
            if (key := self.get_row_key(params)) in self.results:
                progress.value += 1
                continue

            wf = with_filenames(
                workflow,
                SampleRun,
                list(map(self.get_filepath_from_run, params['Runs'])),
            )
            wf[QBins] = sc.geomspace(
                dim='Q',
                start=params['QStart'],
                stop=params['QStop'],
                num=int(params['QBins']),
                unit='1/angstrom',
            )
            self.set_result(
                params, params["Scale"] * wf.compute(ReflectivityOverQ).hist()
            )
            progress.value += 1
