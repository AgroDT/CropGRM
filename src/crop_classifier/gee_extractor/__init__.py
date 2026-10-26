from crop_classifier._optional import import_optional

_MODULES = {
    "SpectralExtractor": ("data_downloading.spectral", "gee"),
    "MeteoExtractor": ("data_downloading.meteo", "gee"),
    "RasterExporter": ("data_downloading.exporter", "gee"),
    "TiffTransformator": ("processor.tiff_transformer", "geo"),
}

__all__ = ["SpectralExtractor", "MeteoExtractor", "RasterExporter", "TiffTransformator"]


def __getattr__(name):
    """Load extractor classes only when requested, including standalone TIFF tools."""
    if name not in _MODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, extra = _MODULES[name]
    module = import_optional(f"{__name__}.{module_name}", extra)
    return getattr(module, name)
