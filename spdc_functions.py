from dataclasses import dataclass
import numpy as np
from snsphd.hist import SplineTool
from scipy.special import erf

from typing import Union

from snsphd.help import prinfo


########### Refractive Index Functions


@dataclass
class ExtraordinaryIndex:
    a1: float = 5.756
    a2: float = 0.0983
    a3: float = 0.2020
    a4: float = 189.32
    a5: float = 12.52
    a6: float = 1.32e-2
    b1: float = 2.86e-6
    b2: float = 4.7e-8
    b3: float = 6.113e-8
    b4: float = 1.516e-4


@dataclass
class ExtraordinaryIndex1Percent(ExtraordinaryIndex):
    a1: float = 5.078
    a2: float = 0.0964
    a3: float = 0.2065
    a4: float = 61.16
    a5: float = 10.55
    a6: float = 1.59e-2
    b1: float = 4.67e-7
    b2: float = 7.822e-8
    b3: float = -2.653e-8
    b4: float = 1.096e-4


@dataclass
class OrdinaryIndex:
    a1: float = 5.653
    a2: float = 0.1185
    a3: float = 0.2091
    a4: float = 89.61
    a5: float = 10.85
    a6: float = 1.97e-2
    b1: float = 7.941e-7
    b2: float = 3.134e-8
    b3: float = -4.641e-9
    b4: float = -2.188e-6


def Mgo_doped_Linb03_calculate_indexes(
    lambda_: float, temp_: float, ei: ExtraordinaryIndex, eo: OrdinaryIndex
):
    # lambda_ *= 1e-3 # Convert for Sellmeir Coefficients

    lamb = lambda_ * 1e-3
    F = (temp_ - 24.5) * (temp_ + 570.82)

    l2 = lamb**2
    nz = (
        ei.a1
        + ei.b1 * F
        + (ei.a2 + ei.b2 * F) / (l2 - (ei.a3 + ei.b3 * F) ** 2)
        + (ei.a4 + ei.b4 * F) / (l2 - ei.a5**2)
        - ei.a6 * l2
    ) ** 0.5

    nx = (
        eo.a1
        + eo.b1 * F
        + (eo.a2 + eo.b2 * F) / (l2 - (eo.a3 + eo.b3 * F) ** 2)
        + (eo.a4 + eo.b4 * F) / (l2 - eo.a5**2)
        - eo.a6 * l2
    ) ** 0.5
    ny = nx

    return nz


def Linb03_calculate_indexes(lambda_: float, temp_: float):
    lamb = lambda_ * 1e-3
    # lambda_ = lambda_*1e-3
    l2 = lamb**2

    nx = (4.9048 - 0.11768 / (0.04750 - l2) - 0.027169 * l2) ** 0.5
    ny = nx
    nz = (4.5820 - 0.099169 / (0.044432 - l2) - 0.021950 * l2) ** 0.5

    dnx = -0.874e-6
    dny = dnx
    dnz = 39.073e-6

    nx = nx + (temp_ - 20.0) * dnx
    ny = ny + (temp_ - 20.0) * dny
    nz = nz + (temp_ - 20.0) * dnz

    return nx


@dataclass
class DetectorParams:
    sigma: float
    w_0: float


@dataclass
class SpdcParams:
    w_central: float
    sigma_p: float
    temp: float
    L: float
    gamma: float
    sellmeier_ordinary: OrdinaryIndex
    sellmeier_extraordinary: ExtraordinaryIndex


@dataclass
class JointSpectrumParams:
    detector: DetectorParams
    spdc: SpdcParams
    A: float


############ SPDC functions


class Dwdm:
    def __init__(self, width_adj=1.0, gaussian=False, transmission=1.0):
        self.guass = gaussian
        self.width_adj = width_adj
        self.dwdm = np.array(
            [
                [1549.2, 0.0],
                [1549.3, 0.1e-9],
                [1549.4, 0.3e-9],
                [1549.5, 0.4e-9],
                [1549.6, 10.7e-9],
                [1549.7, 101.4e-9],
                [1549.8, 1.726e-6],
                [1549.9, 47.3e-6],
                [1550.0, 365.1e-6],
                [1550.1, 0.567e-3],
                [1550.2, 0.587e-3],
                [1550.3, 0.591e-3],
                [1550.4, 0.578e-3],
                [1550.5, 0.543e-3],
                [1550.6, 0.393e-3],
                [1550.7, 68.8e-6],
                [1550.8, 0.656e-6],
                [1550.9, 4e-9],
                [1551.0, 0.9e-9],
                [1551.1, 0.1e-9],
                [1551.2, 0.0],
            ]
        )
        self.norm = np.max(self.dwdm[:, 1])
        print(self.norm)
        self.dwdm_wl = self.dwdm[:, 0]
        self.dwdm_t = (self.dwdm[:, 1] / self.norm) * transmission
        if width_adj != 1:
            self.dwdm_wl = self.dwdm_wl * width_adj
        self.center_of_mass: float = (np.dot(self.dwdm_t, self.dwdm_wl)) / np.sum(
            self.dwdm_t
        )

    def from_array(self, wl_array: np.ndarray, center_wl: float):
        if self.guass:
            gauss = gaussian(wl_array, center_wl, 0.2 * self.width_adj)
            return gauss / np.max(gauss)
        else:
            # return a trasmission array from an input wavelength array
            delta_wl = center_wl - self.center_of_mass
            translated_dwdm_wl = self.dwdm_wl + delta_wl
            t_array = np.interp(wl_array, translated_dwdm_wl, self.dwdm_t)
            return t_array


def dwdm2D(x, y, x_0, y_0, x_trans=0.8, y_trans=0.8, width_adj=1.0, gaussian=False):
    if x_0 == 0:
        # if y_0 is nonzero and x_0 is zero, then apply just the y filter.
        # Used for fitting to singles count rates
        print("found zero x")
        dwdm_y = Dwdm(width_adj=width_adj, gaussian=gaussian, transmission=y_trans)
        return dwdm_y.from_array(y, y_0)

    if y_0 == 0:
        # if x_0 is nonzero and y_0 is zero, then apply just the x filter
        # Used for fitting to singles count rates
        print("found zero y")
        dwdm_x = Dwdm(width_adj=width_adj, gaussian=gaussian, transmission=x_trans)
        return dwdm_x.from_array(x, x_0)

    # take the x and put it through the filter. Find the transmission
    # take the y and put it thrugh the filter. Find the transmission.
    dwdm_x = Dwdm(width_adj=width_adj, gaussian=gaussian, transmission=x_trans)
    dwdm_y = Dwdm(width_adj=width_adj, gaussian=gaussian, transmission=y_trans)

    filtered_x = dwdm_x.from_array(x, x_0)
    filtered_y = dwdm_y.from_array(y, y_0)
    return filtered_x * filtered_y


def refractive_index_ppln(wavelength):
    """
    This is the large n(\lambda) equation in Sam's paper
    """
    w_in_um = 1e-3 * wavelength  # convert nm to um
    w2 = np.square(w_in_um)
    out = 1 + (2.6734 * w2) / (w2 - 0.01764)
    out += (1.2290 * w2) / (w2 - 0.05914)
    out += (12.614 * w2) / (w2 - 474.6)
    return np.sqrt(out)


def gaussian2D(x, y, x_0, y_0, sigma_x, sigma_y):
    # is this missing factors of 2?
    # check normalization!
    aux = ((x - x_0) / sigma_x) ** 2
    aux += ((y - y_0) / sigma_y) ** 2
    norm = 2 * np.pi * sigma_x * sigma_y
    return np.exp(-aux) / norm


def gaussian(x, x_0, sigma_x):
    aux = 0.5 * (((x - x_0) / sigma_x) ** 2)
    norm = sigma_x * np.sqrt(2 * np.pi)
    return np.exp(-aux) / norm


def detector_profile(w_idler, w_signal, params: DetectorParams):
    """
    I need to decide if our why I'm using this.
    Would I also need the loss of all the DWDM channels?
    Well Sam couldn't characterize her narroband filter versus wavelength relationship...

    w0 here is the center of the guassian
    this creates a 2d guassian centered at (w_0, w_0) with different standard deviations
    """
    return gaussian2D(
        w_idler, w_signal, params.w_0, params.w_0, params.sigma, params.sigma
    )


def pump_envelope(omega_i, omega_s, omega_0, sigma_p):
    nu_i = omega_i - omega_0
    nu_s = omega_s - omega_0

    aux = (nu_i + nu_s) / sigma_p
    return np.exp(-0.5 * np.square(aux))


def phase_mismatch(
    w_idler, w_signal, w_pump, gamma, temp, spdc_params: SpdcParams
):  # nanometers
    n_i = Mgo_doped_Linb03_calculate_indexes(
        w_idler,
        temp,
        spdc_params.sellmeier_extraordinary,
        spdc_params.sellmeier_ordinary,
    )
    n_s = Mgo_doped_Linb03_calculate_indexes(
        w_signal,
        temp,
        spdc_params.sellmeier_extraordinary,
        spdc_params.sellmeier_ordinary,
    )
    n_p = Mgo_doped_Linb03_calculate_indexes(
        w_pump,
        temp,
        spdc_params.sellmeier_extraordinary,
        spdc_params.sellmeier_ordinary,
    )

    return 2 * np.pi * ((n_p / w_pump) - (n_i / w_idler) - (n_s / w_signal) - gamma)


def sinc2_monochromatic(w_idler, w_signal, gamma, spdc_params: SpdcParams):
    c = 3.0 * 10**8 * 10**9  # nm/s
    v_idler, v_signal = c / w_idler, c / w_signal
    v_pump = v_idler + v_signal
    w_pump = c / v_pump

    delta_k = phase_mismatch(
        w_idler, w_signal, w_pump, gamma, spdc_params.temp, spdc_params
    )

    return np.sinc(0.5 * spdc_params.L * delta_k / np.pi) ** 2


def spdc_profile(w_idler, w_signal, gamma, spdc_params: SpdcParams):
    c = 3 * 10**8 * 10**9  # nm/s
    omega_i, omega_s, omega_0 = (
        2 * np.pi * c / w_idler,
        2 * np.pi * c / w_signal,
        2 * np.pi * c / spdc_params.w_central,
    )
    return sinc2_monochromatic(w_idler, w_signal, gamma, spdc_params) * pump_envelope(
        omega_i, omega_s, omega_0, spdc_params.sigma_p
    )


def wl_pump_envelope(w_idler, w_signal, params: SpdcParams):
    c = 3 * 10**8 * 10**9  # nm/s
    omega_i, omega_s, omega_0 = (
        2 * np.pi * c / w_idler,
        2 * np.pi * c / w_signal,
        2 * np.pi * c / params.w_central,
    )
    return pump_envelope(omega_i, omega_s, omega_0, params.sigma_p)


def joint_spectrum(w_idler, w_signal, gamma, A, params: JointSpectrumParams):
    out = A * detector_profile(w_idler, w_signal, params.detector)
    out *= spdc_profile(w_idler, w_signal, gamma, params.spdc)
    return out


def joint_spectrum_no_detector(
    w_idler, w_signal, gamma, A, params: JointSpectrumParams
):
    out = A * spdc_profile(w_idler, w_signal, gamma, params.spdc)
    return out


def wrapper_joint_spectrum(params: JointSpectrumParams):
    def _joint_spectrum(M, gamma, A, params: JointSpectrumParams = params):
        x, y = M[:, 0], M[:, 1]
        return joint_spectrum(x, y, gamma, A, params)

    return _joint_spectrum


def lmfit_wrapper_join_spectrum(
    M,
    detector_sigma,
    detector_w_0,
    spdc_w_central,
    spdc_sigma_p,
    spdc_temp,
    spdc_L,
    spdc_gamma,
    A,
):
    detector_params = DetectorParams(sigma=detector_sigma, w_0=detector_w_0)
    spdc_params = SpdcParams(
        w_central=spdc_w_central,
        sigma_p=spdc_sigma_p,
        temp=spdc_temp,
        L=spdc_L,
        gamma=spdc_gamma,
        sellmeier_ordinary=OrdinaryIndex(),
        sellmeier_extraordinary=ExtraordinaryIndex1Percent(),
    )
    params = JointSpectrumParams(detector_params, spdc_params, A)
    x, y = M[:, 0], M[:, 1]
    return joint_spectrum(x, y, params.spdc.gamma, params.A, params)


def create_signal_idler_arrays(
    signal_pts: list[float] | np.ndarray,
    idler_pts: list[float] | np.ndarray,
    number_pts: int = 150,
    df=None,
):
    if df is None:
        df = np.average(np.abs(np.diff(signal_pts)))
    signal_min = min(signal_pts) - df
    signal_max = max(signal_pts) + df
    idler_min = min(idler_pts) - df
    idler_max = max(idler_pts) + df

    signal_wl = np.linspace(signal_min, signal_max, number_pts)
    idler_wl = np.linspace(idler_min, idler_max, number_pts)
    X, Y = np.meshgrid(idler_wl, signal_wl)
    return signal_wl, idler_wl, X, Y


def lmfit_wrapper_join_spectrum_filter_integrate(
    M,
    # X,
    # Y,
    detector_sigma,
    detector_w_0,
    spdc_w_central,
    spdc_sigma_p,
    spdc_temp,
    spdc_L,
    spdc_gamma,
    A,
):
    """
    lmfit_wrapper_join_spectrum assumes the the dwdm filters are narroband enough that
    they pick out singular points of the jsi. In fact, they are wide enough that it's
    more accurate to think of the collected coincidence rates as from an integration of JSI
    times the filter bandwidths. This updated fitting function applies that integration
    """
    detector_params = DetectorParams(sigma=detector_sigma, w_0=detector_w_0)
    spdc_params = SpdcParams(
        w_central=spdc_w_central,
        sigma_p=spdc_sigma_p,
        temp=spdc_temp,
        L=spdc_L,
        gamma=spdc_gamma,
        sellmeier_ordinary=OrdinaryIndex(),
        sellmeier_extraordinary=ExtraordinaryIndex1Percent(),
    )
    params = JointSpectrumParams(detector_params, spdc_params, A)
    y, x = M[:, 0], M[:, 1]

    output = np.zeros(len(x), dtype=float)
    for i, (x_wl, y_wl) in enumerate(zip(x, y)):
        X, Y = create_sub_mesh_grids(x_wl, y_wl, 15)

        filter_dwdm = dwdm2D(X, Y, x_wl, y_wl)
        # prinfo(np.shape(filter_dwdm))

        # prinfo(filter_dwdm)
        # prinfo(np.sum(filter_dwdm))
        filter_dwdm = filter_dwdm / np.sum(filter_dwdm)  # normalize

        # integrate the transmission through the filter over the sub grid
        output[i] = np.sum(
            filter_dwdm * joint_spectrum(X, Y, params.spdc.gamma, params.A, params)
        )
    return output


def transmission_from_wavelength(wavelengths: list[float], transmissions: dict) -> list:
    lookup_dict = {
        1549.32: transmissions["35"],
        1548.51: transmissions["36"],
        1547.72: transmissions["37"],
        1546.92: transmissions["38"],
        1546.12: transmissions["39"],
        1545.32: transmissions["40"],
        1544.53: transmissions["41"],
        1543.73: transmissions["42"],
        1535.82: transmissions["52"],
        1535.04: transmissions["53"],
        1534.25: transmissions["54"],
        1533.47: transmissions["55"],
        1532.68: transmissions["56"],
        1531.90: transmissions["57"],
        1531.12: transmissions["58"],
        1530.33: transmissions["59"],
    }

    output_trans = []
    for wl in wavelengths:
        if wl == 0.0:
            output_trans.append(-1)
            continue
        try:
            output_trans.append(lookup_dict[wl])
        except KeyError:
            print(f"Key {wl} not found in lookup dict")

    return output_trans


# [1530.33, 1531.12, 1531.9, 1532.68, 1533.47, 1534.25, 1535.04, 1535.82]


def lmfit_wrapper_join_spectrum_filter_integrate_cs(
    M,
    # X,
    # Y,
    detector_sigma,
    detector_w_0,
    spdc_w_central,
    spdc_sigma_p,
    spdc_temp,
    spdc_L,
    spdc_gamma,
    A,
    signal_ch_35,
    signal_ch_36,
    signal_ch_37,
    signal_ch_38,
    signal_ch_39,
    signal_ch_40,
    signal_ch_41,
    signal_ch_42,
    idler_ch_52,
    idler_ch_53,
    idler_ch_54,
    idler_ch_55,
    idler_ch_56,
    idler_ch_57,
    idler_ch_58,
    idler_ch_59,
):
    """
    lmfit_wrapper_join_spectrum assumes the the dwdm filters are narroband enough that
    they pick out singular points of the jsi. In fact, they are wide enough that it's
    more accurate to think of the collected coincidence rates as from an integration of JSI
    times the filter bandwidths. This updated fitting function applies that integration
    """

    detector_params = DetectorParams(sigma=detector_sigma, w_0=detector_w_0)
    spdc_params = SpdcParams(
        w_central=spdc_w_central,
        sigma_p=spdc_sigma_p,
        temp=spdc_temp,
        L=spdc_L,
        gamma=spdc_gamma,
        sellmeier_ordinary=OrdinaryIndex(),
        sellmeier_extraordinary=ExtraordinaryIndex1Percent(),
    )
    params = JointSpectrumParams(detector_params, spdc_params, A)
    y, x = M[:, 0], M[:, 1]
    # y array is signal, x array is idler

    transmissions = {
        "35": signal_ch_35,
        "36": signal_ch_36,
        "37": signal_ch_37,
        "38": signal_ch_38,
        "39": signal_ch_39,
        "40": signal_ch_40,
        "41": signal_ch_41,
        "42": signal_ch_42,
        "52": idler_ch_52,
        "53": idler_ch_53,
        "54": idler_ch_54,
        "55": idler_ch_55,
        "56": idler_ch_56,
        "57": idler_ch_57,
        "58": idler_ch_58,
        "59": idler_ch_59,
    }

    y_transmissions = transmission_from_wavelength(y, transmissions)
    x_transmissions = transmission_from_wavelength(x, transmissions)

    print(transmissions)
    print()

    output = np.zeros(len(x), dtype=float)
    for i, (x_wl, y_wl, x_trans, y_trans) in enumerate(
        zip(x, y, x_transmissions, y_transmissions)
    ):
        X, Y = create_sub_mesh_grids(x_wl, y_wl, 30)

        filter_dwdm = dwdm2D(X, Y, x_wl, y_wl, x_trans=x_trans, y_trans=y_trans)
        # filter_dwdm = filter_dwdm / np.sum(filter_dwdm)  # normalize

        # integrate the transmission through the filter over the sub grid
        output[i] = np.sum(
            filter_dwdm * joint_spectrum(X, Y, params.spdc.gamma, params.A, params)
        )
    return output


def create_sub_mesh_grids(x_wl: float, y_wl: float, res: int, span: float = 0.8):
    """create a 2D grid of points a little larger than a filter transmission window.
    Used for integrating the transmission through the filter

    Args:
        x_wl (float): x center of sub-grid
        y_wl (float): y center of sub-grid
        res (int): sub-grid resolution
        span (float): 1/2 width of sub-grid

    Returns:
        _type_: X and Y mesh grids
    """
    if x_wl != 0.0:
        start_x_wl = x_wl - span
        end_x_wl = x_wl + span
        array_x_wl = np.linspace(start_x_wl, end_x_wl, res)
    else:
        # make extra long sub-grid in this dimension for single filter
        start_x_wl = 1540 - 20
        end_x_wl = 1540 + 20
        array_x_wl = np.linspace(start_x_wl, end_x_wl, res)

    if y_wl != 0.0:
        start_y_wl = y_wl - span
        end_y_wl = y_wl + span
        array_y_wl = np.linspace(start_y_wl, end_y_wl, res)
    else:
        # make extra long sub-grid in this dimension for single filter
        start_y_wl = 1540 - 20
        end_y_wl = 1540 + 20
        array_y_wl = np.linspace(start_y_wl, end_y_wl, res)

    # print(array_x_wl)
    # print(array_y_wl)
    X, Y = np.meshgrid(array_x_wl, array_y_wl)
    return X, Y


@dataclass
class GaussianFilterParams:
    center_x: float
    center_y: float
    sigma_x: float
    sigma_y: float


def cwdm_profile(x, y, sigma_x=13.0, sigma_y=13.0):
    # There is no factor 2 to normalize this one? We could
    return gaussian2D(x, y, 1530, 1550, sigma_x, sigma_y) + gaussian2D(
        x, y, 1550, 1530, sigma_x, sigma_y
    )


def single_filter(x, y, params: GaussianFilterParams):
    return gaussian2D(
        x, y, params.center_x, params.center_y, params.sigma_x, params.sigma_y
    )
