import csv
import numpy as np
from collections import namedtuple


def load_spectrum_file(filename):
    data = []
    with open(filename) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            data.append(row)

    start_idx = None
    for i in range(len(data)):
        if len(data[i]) >= 3:
            if data[i][0] == "Stop":
                start_idx = i + 1
                break
    if start_idx is None:
        print("Could not find start of data")
    else:
        wavelengths = []
        values = []
        for i in range(start_idx, len(data)):
            wavelengths.append(float(data[i][0]))
            values.append(float(data[i][1]))
    return np.array(wavelengths), np.array(values)


def gaussian(x, amplitude, mean, stddev):
    return amplitude * np.exp(-(((x - mean) / stddev) ** 2))


def scale_units(number):
    if number >= 1e3 and number < 1e6:
        return number / 1e3, "K"
    if number >= 1e6 and number < 1e9:
        return number / 1e6, "M"
    if number >= 1e9 and number < 1e12:
        return number / 1e9, "G"
    if number >= 1e12 and number < 1e15:
        return number / 1e12, "T"
    if number >= 1e15 and number < 1e18:
        return number / 1e15, "Y"


def frequency_and_bandwidth(wl, wl_sigma):
    c = 299792458

    wl *= 1e-9
    wl_sigma *= 1e-9

    v_start = c / (wl + (wl_sigma / 2))
    v_end = c / (wl - (wl_sigma / 2))
    sigma_v = v_end - v_start
    freq, freq_unit = scale_units(c / wl)
    sigma_v, sigma_v_unit = scale_units(sigma_v)

    bw = namedtuple("FreqBandwidth", ["freq", "freq_unit", "sigma", "sigma_unit"])
    return bw(
        freq=freq,
        freq_unit=freq_unit + "Hz",
        sigma=sigma_v,
        sigma_unit=sigma_v_unit + "Hz",
    )
