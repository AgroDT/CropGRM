from functools import lru_cache

from cropgrm._optional import import_optional


@lru_cache(maxsize=1)
def get_fourth_derivative():
    """Build the seasonal curve's fourth derivative once per process on demand."""
    sp = import_optional("sympy", "features")
    x = sp.symbols("x")
    wNDVI, mNDVI, S, A, mS, mA = sp.symbols("wNDVI mNDVI S A mS mA")
    sigmoid1 = 1 / (1 + sp.exp(-mS * (x - S)))
    sigmoid2 = 1 / (1 + sp.exp(mA * (x - A)))
    curve = wNDVI + (mNDVI - wNDVI) * (sigmoid1 + sigmoid2 - 1)
    derivative = sp.diff(curve, x, 4)
    return sp.lambdify([x, wNDVI, mNDVI, S, A, mS, mA], derivative, "numpy")
