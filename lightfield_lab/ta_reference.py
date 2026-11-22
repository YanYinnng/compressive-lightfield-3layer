"""Locate the unchanged TA reference in the workspace or a portable bundle."""
import importlib.util
from pathlib import Path


def load_reference():
    root = Path(__file__).resolve().parents[1]
    path = root/'ta_reference.py'
    if not path.is_file():
        path = root/'testing/inference.py'
    spec = importlib.util.spec_from_file_location('ta_optical_reference', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
