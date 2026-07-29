"""ctypes access to the Mojo triangulation kernel."""

from __future__ import annotations

import ctypes
import os
import subprocess

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_LIB = os.path.join(ROOT, "dist", "libmojo-triangle.so")
LIB = os.environ.get("MOJO_TRIANGLE_LIB") or DEFAULT_LIB

I = ctypes.c_int64


class BuildError(RuntimeError):
    pass


def build(force: bool = False) -> str:
    source = os.path.join(ROOT, "src", "triangle.mojo")
    if LIB != DEFAULT_LIB:
        if not os.path.isfile(LIB):
            raise BuildError(f"MOJO_TRIANGLE_LIB does not exist: {LIB}")
        return LIB
    if not force and os.path.exists(LIB) and os.path.getmtime(LIB) >= os.path.getmtime(
        source
    ):
        return LIB
    proc = subprocess.run(
        ["bash", os.path.join(ROOT, "build", "build.sh")],
        capture_output=True,
        text=True,
        timeout=1800,
    )
    if proc.returncode != 0 or not os.path.exists(LIB):
        raise BuildError((proc.stderr or proc.stdout).strip()[:4000])
    return LIB


_library: ctypes.CDLL | None = None


def lib() -> ctypes.CDLL:
    global _library
    if _library is None:
        _library = ctypes.CDLL(build())
        fn = _library.mt_triangulate
        fn.argtypes = [I] * 13
        fn.restype = I
    return _library


def addr(
    array: np.ndarray,
    dtype: np.dtype,
    minimum_elements: int,
    name: str,
) -> int:
    """Validate a native buffer immediately before exposing its address."""
    if array.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}")
    if not array.flags.c_contiguous:
        raise ValueError(f"{name} must be C-contiguous")
    if array.size < minimum_elements:
        raise ValueError(f"{name} is smaller than the native call requires")
    address = int(array.ctypes.data)
    if address == 0:
        raise ValueError(f"{name} has a null data pointer")
    return address
