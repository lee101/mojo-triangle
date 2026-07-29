"""Python-compatible API backed by the Mojo constrained Delaunay kernel."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from ._lib import addr, lib

_PASSIVE_OPTIONS = frozenset("pce nzQXCiFl".replace(" ", ""))
_UNSUPPORTED_OPTIONS = frozenset("rqaDsSv")


def _integers(value, dtype: np.dtype, name: str) -> np.ndarray:
    """Convert integer-valued input without truncation or wraparound."""
    array = np.asarray(value)
    if array.dtype.kind not in "biuf":
        raise TypeError(f"{name} must contain integers")
    if array.dtype.kind == "f":
        if not np.isfinite(array).all() or np.any(array != np.trunc(array)):
            raise ValueError(f"{name} must contain exact finite integers")
    bounds = np.iinfo(dtype)
    if np.any(array < bounds.min) or np.any(array > bounds.max):
        raise OverflowError(f"{name} values do not fit in {np.dtype(dtype)}")
    return np.ascontiguousarray(array, dtype=dtype)


def _options(opts: str) -> set[str]:
    if not isinstance(opts, str):
        raise TypeError("opts must be a string")
    unsupported = sorted(set(opts) & _UNSUPPORTED_OPTIONS)
    if unsupported:
        raise NotImplementedError(
            "unsupported Triangle option(s): " + ", ".join(unsupported)
        )
    unknown = sorted(
        char
        for char in set(opts)
        if not (char.isdigit() or char in ".+-" or char in _PASSIVE_OPTIONS)
    )
    if unknown:
        raise ValueError("unknown Triangle option(s): " + ", ".join(unknown))
    return set(opts)


def _points(value, name: str = "vertices") -> np.ndarray:
    points = np.ascontiguousarray(value, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"{name} must have shape (n, 2)")
    if not np.isfinite(points).all():
        raise ValueError(f"{name} must contain only finite coordinates")
    if len(np.unique(points, axis=0)) != len(points):
        raise ValueError("duplicate vertices are not supported")
    return points


def _segments(value, n: int) -> np.ndarray:
    segments = _integers(value, np.dtype(np.int64), "segments")
    if segments.size == 0:
        return np.empty((0, 2), dtype=np.int64)
    if segments.ndim != 2 or segments.shape[1] != 2:
        raise ValueError("segments must have shape (m, 2)")
    if np.any(segments < 0) or np.any(segments >= n):
        raise ValueError("segment endpoint is outside the vertex array")
    if np.any(segments[:, 0] == segments[:, 1]):
        raise ValueError("segments must have distinct endpoints")
    canonical = np.sort(segments, axis=1)
    if len(np.unique(canonical, axis=0)) != len(segments):
        raise ValueError("duplicate segments are not supported")
    return segments


def _validate_pslg(vertices: np.ndarray, segments: np.ndarray) -> None:
    if not len(segments):
        return
    extent = np.ptp(vertices, axis=0).max(initial=1.0)
    tolerance = 1e-14 * max(1.0, float(extent)) ** 2
    x_coordinates = vertices[:, 0]
    y_coordinates = vertices[:, 1]
    endpoints_a = vertices[segments[:, 0]]
    endpoints_b = vertices[segments[:, 1]]
    directions = endpoints_b - endpoints_a
    for start in range(0, len(segments), 64):
        stop = min(start + 64, len(segments))
        a = endpoints_a[start:stop]
        b = endpoints_b[start:stop]
        direction = directions[start:stop]
        cross_values = y_coordinates[None, :] - a[:, 1, None]
        cross_values *= direction[:, 0, None]
        scratch = x_coordinates[None, :] - a[:, 0, None]
        cross_values -= direction[:, 1, None] * scratch
        np.abs(cross_values, out=scratch)
        candidates = scratch <= tolerance
        rows = np.arange(stop - start)
        candidates[rows, segments[start:stop, 0]] = False
        candidates[rows, segments[start:stop, 1]] = False
        if np.any(candidates):
            inside = candidates.copy()
            inside &= (
                x_coordinates[None, :]
                >= np.minimum(a[:, 0], b[:, 0])[:, None] - tolerance
            )
            inside &= (
                x_coordinates[None, :]
                <= np.maximum(a[:, 0], b[:, 0])[:, None] + tolerance
            )
            inside &= (
                y_coordinates[None, :]
                >= np.minimum(a[:, 1], b[:, 1])[:, None] - tolerance
            )
            inside &= (
                y_coordinates[None, :]
                <= np.maximum(a[:, 1], b[:, 1])[:, None] + tolerance
            )
            if np.any(inside):
                raise ValueError(
                    "a segment passes through a non-endpoint vertex; split it first"
                )

    orientation_a = (
        directions[:, 0, None]
        * (endpoints_a[None, :, 1] - endpoints_a[:, None, 1])
        - directions[:, 1, None]
        * (endpoints_a[None, :, 0] - endpoints_a[:, None, 0])
    )
    orientation_b = (
        directions[:, 0, None]
        * (endpoints_b[None, :, 1] - endpoints_a[:, None, 1])
        - directions[:, 1, None]
        * (endpoints_b[None, :, 0] - endpoints_a[:, None, 0])
    )
    starts = segments[:, 0]
    ends = segments[:, 1]
    distinct = (
        (starts[:, None] != starts[None, :])
        & (starts[:, None] != ends[None, :])
        & (ends[:, None] != starts[None, :])
        & (ends[:, None] != ends[None, :])
    )
    crossing = (
        np.triu(distinct, 1)
        & (orientation_a * orientation_b < -tolerance)
        & (orientation_a.T * orientation_b.T < -tolerance)
    )
    if np.any(crossing):
        raise ValueError("PSLG segments must not cross")


def _boundary_edges(triangles: np.ndarray) -> np.ndarray:
    if not len(triangles):
        return np.empty((0, 2), dtype=np.int32)
    directed = triangles[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2)
    low = np.minimum(directed[:, 0], directed[:, 1]).astype(np.int64)
    high = np.maximum(directed[:, 0], directed[:, 1]).astype(np.int64)
    keys = low * (int(triangles.max()) + 1) + high
    _, first, counts = np.unique(keys, return_index=True, return_counts=True)
    return directed[first[counts == 1]].astype(np.int32, copy=False)


def _all_edges(triangles: np.ndarray) -> np.ndarray:
    if not len(triangles):
        return np.empty((0, 2), dtype=np.int32)
    directed = triangles[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2)
    low = np.minimum(directed[:, 0], directed[:, 1]).astype(np.int64)
    high = np.maximum(directed[:, 0], directed[:, 1]).astype(np.int64)
    keys = low * (int(triangles.max()) + 1) + high
    _, first = np.unique(keys, return_index=True)
    return directed[first].astype(np.int32, copy=False)


def _neighbors(triangles: np.ndarray) -> np.ndarray:
    neighbors = np.full(triangles.shape, -1, dtype=np.int32)
    owners: dict[tuple[int, int], tuple[int, int]] = {}
    for t, triangle in enumerate(triangles):
        for opposite in range(3):
            a = int(triangle[(opposite + 1) % 3])
            b = int(triangle[(opposite + 2) % 3])
            key = (min(a, b), max(a, b))
            if key in owners:
                other_t, other_opposite = owners[key]
                neighbors[t, opposite] = other_t
                neighbors[other_t, other_opposite] = t
            else:
                owners[key] = (t, opposite)
    return neighbors


def _hull_from_triangles(triangles: np.ndarray) -> np.ndarray:
    boundary = _boundary_edges(triangles)
    return boundary[:, ::-1].copy()


def _run(
    vertices: np.ndarray,
    segments: np.ndarray,
    holes: np.ndarray,
    clip: bool,
) -> np.ndarray:
    n = len(vertices)
    if n < 3:
        return np.empty((0, 3), dtype=np.int32)
    capacity = 4 * n + 16
    triangles = np.empty((capacity, 3), dtype=np.int64)
    marks = np.empty(capacity, dtype=np.float64)
    edge_a = np.empty(3 * capacity, dtype=np.int64)
    edge_b = np.empty(3 * capacity, dtype=np.int64)
    coefficients = np.empty((4, capacity), dtype=np.float64)
    segment_buffer = (
        segments if len(segments) else np.empty((1, 2), dtype=np.int64)
    )
    hole_buffer = holes if len(holes) else np.empty((1, 2), dtype=np.float64)
    count = int(
        lib().mt_triangulate(
            addr(vertices, np.dtype(np.float64), 2 * n, "vertices"),
            n,
            addr(
                segment_buffer,
                np.dtype(np.int64),
                max(1, 2 * len(segments)),
                "segments",
            ),
            len(segments),
            addr(
                hole_buffer,
                np.dtype(np.float64),
                max(1, 2 * len(holes)),
                "holes",
            ),
            len(holes),
            int(clip),
            addr(triangles, np.dtype(np.int64), 3 * capacity, "triangles"),
            capacity,
            addr(marks, np.dtype(np.float64), capacity, "marks"),
            addr(edge_a, np.dtype(np.int64), 3 * capacity, "edge_a"),
            addr(edge_b, np.dtype(np.int64), 3 * capacity, "edge_b"),
            addr(
                coefficients,
                np.dtype(np.float64),
                4 * capacity,
                "coefficients",
            ),
        )
    )
    if count == -1:
        return np.empty((0, 3), dtype=np.int32)
    if count == -2:
        raise RuntimeError("internal triangulation capacity exceeded")
    if count == -3:
        raise ValueError("could not recover every constrained segment")
    if count < 0:
        raise RuntimeError(f"Mojo triangulation failed with status {count}")
    return triangles[:count].astype(np.int32)


def delaunay(pts) -> np.ndarray:
    """Compute the Delaunay triangulation of two-dimensional points."""
    vertices = _points(pts, "pts")
    return _run(
        vertices,
        np.empty((0, 2), dtype=np.int64),
        np.empty((0, 2), dtype=np.float64),
        False,
    )


def convex_hull(pts) -> np.ndarray:
    """Return directed segments enclosing the convex hull of *pts*."""
    vertices = _points(pts, "pts")
    return _hull_from_triangles(delaunay(vertices))


def triangulate(tri: Mapping, opts: str = "") -> dict[str, np.ndarray]:
    """Triangulate a Triangle-style input dictionary."""
    if not isinstance(tri, Mapping):
        raise TypeError("tri must be a mapping")
    if "vertices" not in tri:
        raise ValueError("tri must contain 'vertices'")
    flags = _options(opts)
    vertices = _points(tri["vertices"])
    pslg = "p" in flags
    requested_segments = _segments(
        tri.get("segments", np.empty((0, 2), dtype=np.int64)), len(vertices)
    )
    segments = requested_segments if pslg else np.empty((0, 2), dtype=np.int64)
    if pslg:
        _validate_pslg(vertices, segments)
    holes = _points(tri.get("holes", np.empty((0, 2))), "holes")
    if not pslg:
        holes = np.empty((0, 2), dtype=np.float64)

    if "c" in flags:
        hull = _hull_from_triangles(
            _run(
                vertices,
                np.empty((0, 2), dtype=np.int64),
                np.empty((0, 2), dtype=np.float64),
                False,
            )
        ).astype(np.int64)
        existing = {tuple(sorted(map(int, edge))) for edge in segments}
        extra = [edge for edge in hull if tuple(sorted(map(int, edge))) not in existing]
        if extra:
            segments = np.vstack((segments, np.asarray(extra, dtype=np.int64)))

    triangles = _run(vertices, segments, holes, pslg)
    result: dict[str, np.ndarray] = {"vertices": vertices.copy()}

    if "vertex_markers" in tri:
        markers = _integers(
            tri["vertex_markers"], np.dtype(np.int32), "vertex_markers"
        )
        if markers.size != len(vertices):
            raise ValueError("vertex_markers must have one value per vertex")
        result["vertex_markers"] = markers.reshape(-1, 1).copy()
    else:
        markers = np.zeros((len(vertices), 1), dtype=np.int32)
        boundary = _boundary_edges(triangles)
        if len(boundary):
            markers[np.unique(boundary)] = 1
        if len(segments):
            markers[np.unique(segments)] = 1
        result["vertex_markers"] = markers

    if len(triangles):
        result["triangles"] = triangles

    if pslg or "c" in flags:
        if len(triangles):
            result["segments"] = segments.astype(np.int32)
            base_markers = tri.get("segment_markers")
            if base_markers is None:
                segment_markers = np.ones((len(segments), 1), dtype=np.int32)
            else:
                supplied = _integers(
                    base_markers, np.dtype(np.int32), "segment_markers"
                ).reshape(-1, 1)
                if len(supplied) != len(requested_segments):
                    raise ValueError(
                        "segment_markers must have one value per input segment"
                    )
                segment_markers = np.ones((len(segments), 1), dtype=np.int32)
                segment_markers[: len(supplied)] = supplied
            result["segment_markers"] = segment_markers

    if "n" in flags and len(triangles):
        result["neighbors"] = _neighbors(triangles)
    if "e" in flags and len(triangles):
        edges = _all_edges(triangles)
        constrained = {
            tuple(sorted(map(int, edge))): i for i, edge in enumerate(segments)
        }
        boundary = {
            tuple(sorted(map(int, edge))) for edge in _boundary_edges(triangles)
        }
        edge_markers = np.zeros((len(edges), 1), dtype=np.int32)
        supplied = _integers(
            tri.get("segment_markers", np.ones((len(requested_segments), 1))),
            np.dtype(np.int32),
            "segment_markers",
        ).reshape(-1)
        for i, edge in enumerate(edges):
            key = tuple(sorted(map(int, edge)))
            if key in constrained:
                segment_index = constrained[key]
                edge_markers[i] = (
                    supplied[segment_index]
                    if segment_index < len(supplied)
                    else 1
                )
            elif key in boundary:
                edge_markers[i] = 1
        result["edges"] = edges
        result["edge_markers"] = edge_markers
    return result
