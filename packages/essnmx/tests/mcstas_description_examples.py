# flake8: noqa

no_detectors = """
SPLIT 999 COMPONENT Xtal = Single_crystal(
                order = 1,
                p_transmit=0.001,
        reflections = "Rubredoxin.lau",
        xwidth = XtalSize_width,
                yheight = XtalSize_height,
                zdepth = XtalSize_depth,
                mosaic = XtalMosaicity,
                delta_d_d=1e-4)
   AT (0, 0, deltaz) RELATIVE PREVIOUS
   ROTATED (XtalPhiX,XtalPhiY, XtalPhiZ) RELATIVE armSample
 EXTEND %{
        if (!SCATTERED) {ABSORB;}
  %}

COMPONENT Sphere = PSD_monitor_4PI(
    nx = 360, ny = 360, filename = "4pi", radius = 0.2,
    restore_neutron = 1)

AT (0, 0, deltaz) RELATIVE armSample
"""

two_detectors_two_filenames = """
COMPONENT nD_Mantid_0 = Monitor_nD(
        options ="mantid square x limits=[0 0.512] bins=1280 y limits=[0 0.512] bins=1280, neutron pixel min=1 t, list all neutrons",
    xmin = 0,
    xmax = 0.512,
    ymin = 0,
    ymax = 0.512,
    restore_neutron = 1,
    filename = "bank01_events.dat")
        AT (-0.25, -0.25, 0.29) RELATIVE armSample
        ROTATED (0, 0, 0) RELATIVE armSample

COMPONENT nD_Mantid_1 = Monitor_nD(
        options ="mantid square x limits=[0 0.512] bins=1280 y limits=[0 0.512] bins=1280, neutron pixel min=2000000 t, list all neutrons",
    xmin = 0,
    xmax = 0.512,
    ymin = 0,
    ymax = 0.512,
    restore_neutron = 1,
    filename = "bank02_events.dat")
  AT (-0.29, -0.25, 0.25) RELATIVE armSample
  ROTATED (0, 90, 0) RELATIVE armSample
"""

one_detector_no_filename = """
COMPONENT nD_Mantid_2 = Monitor_nD(
        options ="mantid square x limits=[0 0.512] bins=1280 y limits=[0 0.512] bins=1280, neutron pixel min=2000000 t, list all neutrons",
    xmin = 0,
    xmax = 0.512,
    ymin = 0,
    ymax = 0.512,
    restore_neutron = 1,
  AT (-0.29, -0.25, 0.25) RELATIVE armSample
  ROTATED (0, 90, 0) RELATIVE armSample
"""

two_detectors_same_filename = """
COMPONENT nD_Mantid_0 = Monitor_nD(
        options ="mantid square x limits=[0 0.512] bins=1280 y limits=[0 0.512] bins=1280, neutron pixel min=1 t, list all neutrons",
    xmin = 0,
    xmax = 0.512,
    ymin = 0,
    ymax = 0.512,
    restore_neutron = 1,
    filename = "bank01_events.dat")
        AT (-0.25, -0.25, 0.29) RELATIVE armSample
        ROTATED (0, 0, 0) RELATIVE armSample

COMPONENT nD_Mantid_1 = Monitor_nD(
        options ="mantid square x limits=[0 0.512] bins=1280 y limits=[0 0.512] bins=1280, neutron pixel min=2000000 t, list all neutrons",
    xmin = 0,
    xmax = 0.512,
    ymin = 0,
    ymax = 0.512,
    restore_neutron = 1,
    filename = "bank01_events.dat")
  AT (-0.29, -0.25, 0.25) RELATIVE armSample
  ROTATED (0, 90, 0) RELATIVE armSample
"""
