"""MariSAM core model and command-line entry points."""

__version__ = '0.1.0'


def build_marisam(*args, **kwargs):
    from .model import build_marisam as build
    return build(*args, **kwargs)
