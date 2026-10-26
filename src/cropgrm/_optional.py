from importlib import import_module

_DEPENDENCIES = {
    "features": {"joblib", "polars", "scipy", "sympy"},
    "gee": {"ee", "geedim", "geemap", "geopandas", "pycrs", "pyproj", "rasterio"},
    "geo": {"geopandas", "matplotlib", "pyproj", "rasterio", "rioxarray", "xarray"},
}


def import_optional(module_name: str, extra: str):
    """Import an optional capability, preserving unrelated import failures."""
    try:
        return import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name not in _DEPENDENCIES[extra]:
            raise
        raise ModuleNotFoundError(
            f"Missing optional dependency '{exc.name}'. "
            f"Install the '{extra}' extra from the project directory: "
            f'pip install ".[{extra}]"',
            name=exc.name,
        ) from exc
