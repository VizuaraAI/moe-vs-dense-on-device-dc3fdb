"""Measurement harness for "Mixture of Experts Is Probably Wrong On Device".

Import policy: this package must stay importable on a machine with no torch and
no model weights, so that `python -m src.cli device` and `python -m src.cli
predict` always work. Heavy dependencies are imported inside the functions that
need them, never at module scope.
"""

__version__ = "0.1.0"
