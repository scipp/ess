# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)

from shutil import copy2

import matplotlib.pyplot as plt
import numpy as np
import pytest

from ess.amor import data
from ess.reflectometry.gui import AmorBatchReductionGUI


@pytest.fixture
def widget_backend():
    previous_backend = plt.get_backend()
    plt.switch_backend('module://ipympl.backend_nbagg')
    try:
        yield
    finally:
        plt.close('all')
        plt.switch_backend(previous_backend)


@pytest.mark.usefixtures('widget_backend')
@pytest.mark.filterwarnings(
    "ignore:.*Invalid transformation, .*missing attribute 'vector':UserWarning"
)
@pytest.mark.filterwarnings(
    r'ignore:Passing unrecognized arguments to super\(Toolbar\):DeprecationWarning'
)
def test_amor_batch_reduction_and_plot(tmp_path):
    for run in (1632, 1634, 1635):
        filename = data.amor_run(run)
        copy2(filename, tmp_path / filename.name)

    gui = AmorBatchReductionGUI()
    gui.proposal_number_box.value = str(tmp_path)
    runs = gui.runs_table.data
    assert set(runs['Run']) == {'1632', '1634', '1635'}

    reference_index = runs.index[runs['Run'] == '1632'][0]
    gui.runs_table.set_cell_value('Reference', reference_index, True)
    assert list(gui.reference_table.data['Runs']) == [('1632',)]
    assert set(gui.reduction_table.data['Runs']) == {('1634',), ('1635',)}

    # No selection reduces both samples, reusing the reference for the second.
    gui.run_workflow()
    for _, row in gui.reduction_table.data.iterrows():
        curve = gui.results[gui.get_row_key(row)]
        assert curve.bins is None
        assert curve.sizes == {'Q': 390}
        assert curve.variances is not None
        # Bins without reference coverage can be non-finite.
        assert (np.isfinite(curve.values) & (curve.values > 0)).any()

    # Error bars for bins without reference coverage contain inf - inf.
    with np.errstate(invalid='ignore'):
        gui.display_results()
    assert len(gui.plot_log.children) == 1
    figure = gui.plot_log.children[0].children[-1].children[0]
    assert {artist.label for artist in figure.view.artists.values()} == {
        'NiTi ml (1634)',
        'NiTi ml (1635)',
    }
    figure.fig.canvas.draw()
    figure.fig.canvas.close()
    figure.close()
    gui.widget.close()
