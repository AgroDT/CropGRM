from pathlib import Path
import numpy as np
import sympy as sp
import multiprocessing

DATA_TYPE_SPECTRAL = 'spectral'
DATA_TYPE_METEO = 'meteorological'

CHUNK_SIZE = 50_000
N_JOBS = multiprocessing.cpu_count()

FEATURE_CONFIG = {
    "red":   {"pipeline": "mean"},
    "nir":   {"pipeline": "mean"},
    "blue":  {"pipeline": "mean"},
    "swir1": {"pipeline": "mean"},
    "green": {"pipeline": "mean"},
    "swir2": {"pipeline": "mean"},
    "ndyi":  {"pipeline": "extrema"},
    "ndmi":  {"pipeline": "curve"},
    "wrdvi": {"pipeline": "curve"},
}

BOUNDS_CONFIG = {
    'wrdvi': ([-1, -1, 0, 0, 0, 0], [1, 1, 365, 365, 1, 1]),
    'ndmi': ([-0.2, -0.2, 0, 0, 0, 0], [1, 1, 365, 365, 1, 1]),
}

X_VALUES = np.linspace(1, 365, 2000)

# SymPy Derivative Calculations
x = sp.symbols('x')
wNDVI_sym, mNDVI_sym, S_sym, A_sym, mS_sym, mA_sym = sp.symbols('wNDVI mNDVI S A mS mA')

sigmoid1_sym = 1 / (1 + sp.exp(-mS_sym * (x - S_sym)))
sigmoid2_sym = 1 / (1 + sp.exp(mA_sym * (x - A_sym)))
seasonal_term_sym = sigmoid1_sym + sigmoid2_sym - 1
sympy_dlf_template = wNDVI_sym + (mNDVI_sym - wNDVI_sym) * seasonal_term_sym

first_derivative_sym = sp.diff(sympy_dlf_template, x)
second_derivative_sym = sp.diff(first_derivative_sym, x)
third_derivative_sym = sp.diff(second_derivative_sym, x)
fourth_derivative_sym = sp.diff(third_derivative_sym, x)

symbols_for_lambdify = [x, wNDVI_sym, mNDVI_sym, S_sym, A_sym, mS_sym, mA_sym]

FOURTH_DERIVATIVE_LAMBDIFIED = sp.lambdify(symbols_for_lambdify, fourth_derivative_sym, 'numpy')