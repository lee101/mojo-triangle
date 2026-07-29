"""End-to-end benchmarks against the upstream triangle package."""

from __future__ import annotations

import math
import os
import platform
import sys
import time

import numpy as np

sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"),
)

import mojo_triangle
import triangle as upstream


def timeit(function, repeat: int = 5) -> float:
    best = math.inf
    for _ in range(repeat):
        start = time.perf_counter()
        function()
        best = min(best, time.perf_counter() - start)
    return best


def cpu_name() -> str:
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as stream:
            for line in stream:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown CPU"


def constrained_case(n: int):
    boundary_count = max(20, n // 10)
    angles = np.linspace(0, 2 * np.pi, boundary_count, endpoint=False)
    radii = 0.82 + 0.12 * np.sin(5 * angles + 0.2)
    boundary = np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))
    rng = np.random.default_rng(n)
    inner_angles = rng.random(n - boundary_count) * 2 * np.pi
    inner_radii = 0.55 * np.sqrt(rng.random(n - boundary_count))
    interior = np.column_stack(
        (inner_radii * np.cos(inner_angles), inner_radii * np.sin(inner_angles))
    )
    vertices = np.ascontiguousarray(np.vstack((boundary, interior)))
    segments = np.column_stack(
        (
            np.arange(boundary_count, dtype=np.int32),
            np.roll(np.arange(boundary_count, dtype=np.int32), -1),
        )
    )
    return {"vertices": vertices, "segments": segments}


def main() -> None:
    rng = np.random.default_rng(0)
    cases = []
    for n in (250, 1_000, 3_000):
        vertices = np.ascontiguousarray(rng.random((n, 2)))
        cases.append(
            (
                f"delaunay, {n:,} points",
                lambda vertices=vertices: mojo_triangle.delaunay(vertices),
                lambda vertices=vertices: upstream.delaunay(vertices),
            )
        )
    for n in (250, 1_000):
        data = constrained_case(n)
        cases.append(
            (
                f"constrained polygon, {n:,} points",
                lambda data=data: mojo_triangle.triangulate(data, "p"),
                lambda data=data: upstream.triangulate(data, "p"),
            )
        )

    print(f"Machine: {cpu_name()}; {platform.platform()}")
    print()
    print("| case | mojo-triangle | upstream triangle | upstream / Mojo |")
    print("|---|---:|---:|---:|")
    for name, ours, theirs in cases:
        ours()
        theirs()
        mojo_time = timeit(ours)
        upstream_time = timeit(theirs)
        ratio = upstream_time / mojo_time
        print(
            f"| {name} | {mojo_time * 1e3:.3f} ms | "
            f"{upstream_time * 1e3:.3f} ms | {ratio:.3f}x |"
        )


if __name__ == "__main__":
    main()
