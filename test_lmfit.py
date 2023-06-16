# from lmfit import Model
# from numpy import exp, linspace, random
# from dataclasses import dataclass


# @dataclass
# class Stuff:
#     amp: float
#     cen: float
#     wid: float


# # def gaussian(x, **kwargs):
# #     return kwargs["amp"] * exp(-((x - kwargs["cen"]) ** 2) / kwargs["wid"])


# def gaussian(x, stuff: Stuff):
#     return stuff.amp * exp(-((x - stuff.cen) ** 2) / stuff.wid)


# gmodel = Model(gaussian)
# print(f"parameter names: {gmodel.param_names}")
# print(f"independent variables: {gmodel.independent_vars}")

import numpy as np


from lmfit import Model


def fitfun(x, a, b):
    return np.exp(a * (x - b))


# turn this model function into a Model:
mymodel = Model(fitfun)

# create parameters with initial values.  Note that parameters are
# **named** according to the arguments of your model function:
params = mymodel.make_params(a=1, b=10)

# tell the 'b' parameter to not vary during the fit
params["b"].vary = False

# do fit
result = mymodel.fit(ydata, params, x=xdata)
print(result.fit_report())
