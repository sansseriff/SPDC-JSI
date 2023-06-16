import csv
import matplotlib.pyplot as plt
from snsphd import viz
import numpy as np
from snsphd.hist import SplineTool
from snsphd.help import prinfo

from scipy.signal import find_peaks

from typing import Tuple

Colors, swatches = viz.phd_style(jupyterStyle=True, svg_mode=True)
from dataclasses import dataclass

from scipy.optimize import leastsq, curve_fit

import json

from utils import load_spectrum_file, gaussian, frequency_and_bandwidth

from spdc_functions import (
    ExtraordinaryIndex,
    ExtraordinaryIndex1Percent,
    OrdinaryIndex,
    Mgo_doped_Linb03_calculate_indexes,
    Linb03_calculate_indexes,
    DetectorParams,
    SpdcParams,
    JointSpectrumParams,
    refractive_index_ppln,
    gaussian2D,
    detector_profile,
    pump_envelope,
    phase_mismatch,
    # sinc2,
    sinc2_monochromatic,
    spdc_profile,
    wl_pump_envelope,
    joint_spectrum,
    wrapper_joint_spectrum,
    GaussianFilterParams,
    single_filter,
    lmfit_wrapper_join_spectrum,
)

from lmfit import Model
from matplotlib.colors import Normalize
