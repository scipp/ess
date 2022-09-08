# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)
from ._loader import EntryMixin, InstrumentMixin
from ._loader import Fields, Detectors, Monitors, Sample

# What API do we WANT?

# # lazy?
# data = NeutronData(filename)
# # no nesting... at least for the important bits, but where to draw the line?
# data.monitors
# data.sample
# data.source
# data.instrument
# data.entry
# data.detectors
# # How do configure partial det/mon load?
# data = NeutronData(filename, load_detectors=False, load_monitors=False)
# data.detectors['bank1'] = data.detectors['bank1'].select_events['pulse', :1000][()]
# 
# 
# def EventLoader(event_min=None, event_max=None):
# 
#     def _load(nxdetector):
#         return nxdetector.select_events['pulse', event_min:event_max]
# 
#     return _load
# 
# 
# # filter and preprocess?
# data = NeutronData(filename, detectors=EventLoader(pulse_max=1000), monitors=None)
# 
# from nexus.preprocess import skip
# 
# preprocess = {
#     NXdetector: lambda x: x.select_events['pulse', :1000],
#     NXmonitor: lambda x: None  # skip all
#     NXmonitor: skip
# }
