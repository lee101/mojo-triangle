"""Parity and geometric invariant tests against the upstream triangle package."""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

upstream = pytest.importorskip("triangle")

import mojo_triangle as triangle


def triangle_set(values) -> set[tuple[int, int, int]]:
    return {tuple(sorted(map(int, row))) for row in np.asarray(values)}


def edge_set(values) -> set[tuple[int, int]]:
    return {tuple(sorted(map(int, row))) for row in np.asarray(values)}


def mesh_area(vertices, triangles) -> float:
    vertices = np.asarray(vertices)
    triangles = np.asarray(triangles)
    if not len(triangles):
        return 0.0
    ab = vertices[triangles[:, 1]] - vertices[triangles[:, 0]]
    ac = vertices[triangles[:, 2]] - vertices[triangles[:, 0]]
    return float(np.abs(ab[:, 0] * ac[:, 1] - ab[:, 1] * ac[:, 0]).sum() / 2)


@pytest.mark.parametrize("n", [5, 30, 100, 300])
def test_delaunay_matches_upstream_on_general_position_points(n):
    vertices = np.random.default_rng(n).random((n, 2))
    ours = triangle.delaunay(vertices)
    theirs = upstream.delaunay(vertices)
    assert triangle_set(ours) == triangle_set(theirs)
    assert ours.dtype == np.int32


def test_delaunay_simd_tail_matches_upstream():
    vertices = np.random.default_rng(37).normal(size=(37, 2))
    assert triangle_set(triangle.delaunay(vertices)) == triangle_set(
        upstream.delaunay(vertices)
    )


def test_delaunay_matches_published_api_vector():
    vertices = [[0, 0], [0, 1], [0.5, 0.5], [1, 1], [1, 0]]
    expected = [[1, 0, 2], [2, 4, 3], [4, 2, 0], [2, 3, 1]]
    assert triangle_set(triangle.delaunay(vertices)) == triangle_set(expected)


def test_delaunay_is_scale_and_translation_invariant():
    vertices = np.random.default_rng(7).normal(size=(80, 2))
    expected = triangle_set(triangle.delaunay(vertices))
    for transformed in (vertices * 1e-7, vertices * 1e7 + [4e9, -2e9]):
        assert triangle_set(triangle.delaunay(transformed)) == expected


def test_convex_hull_matches_upstream():
    vertices = np.random.default_rng(11).normal(size=(200, 2))
    assert edge_set(triangle.convex_hull(vertices)) == edge_set(
        upstream.convex_hull(vertices)
    )


def test_convex_hull_matches_published_api_vector():
    vertices = [[0, 0], [0, 1], [1, 1], [1, 0]]
    assert edge_set(triangle.convex_hull(vertices)) == edge_set(
        [[3, 0], [2, 3], [1, 2], [0, 1]]
    )


def test_concave_pslg_matches_upstream_triangle_for_triangle():
    vertices = np.array([[0, 0], [3, 0], [3, 2], [1.2, 0.8], [0, 2]], float)
    segments = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 0]])
    data = {"vertices": vertices, "segments": segments}
    ours = triangle.triangulate(data, "p")
    theirs = upstream.triangulate(data, "p")
    assert triangle_set(ours["triangles"]) == triangle_set(theirs["triangles"])
    assert edge_set(ours["segments"]) == edge_set(theirs["segments"])
    assert mesh_area(vertices, ours["triangles"]) == pytest.approx(
        mesh_area(vertices, theirs["triangles"])
    )


def test_random_constrained_polygon_matches_upstream():
    rng = np.random.default_rng(4)
    angles = np.sort(rng.random(24) * 2 * np.pi)
    radii = 0.75 + 0.25 * rng.random(24)
    boundary = np.column_stack((np.cos(angles) * radii, np.sin(angles) * radii))
    inner_angles = rng.random(40) * 2 * np.pi
    inner_radii = 0.2 * np.sqrt(rng.random(40))
    interior = np.column_stack(
        (np.cos(inner_angles) * inner_radii, np.sin(inner_angles) * inner_radii)
    )
    vertices = np.vstack((boundary, interior))
    segments = np.column_stack((np.arange(24), np.roll(np.arange(24), -1)))
    data = {"vertices": vertices, "segments": segments}
    ours = triangle.triangulate(data, "p")
    theirs = upstream.triangulate(data, "p")
    assert triangle_set(ours["triangles"]) == triangle_set(theirs["triangles"])


def test_constrained_validation_batch_tail_matches_upstream():
    boundary_count = 70
    angles = np.linspace(0, 2 * np.pi, boundary_count, endpoint=False)
    radii = 0.85 + 0.08 * np.sin(7 * angles + 0.3)
    boundary = np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))
    rng = np.random.default_rng(70)
    inner_angles = rng.random(30) * 2 * np.pi
    inner_radii = 0.45 * np.sqrt(rng.random(30))
    interior = np.column_stack(
        (inner_radii * np.cos(inner_angles), inner_radii * np.sin(inner_angles))
    )
    vertices = np.vstack((boundary, interior))
    segments = np.column_stack(
        (
            np.arange(boundary_count),
            np.roll(np.arange(boundary_count), -1),
        )
    )
    data = {"vertices": vertices, "segments": segments}
    assert triangle_set(triangle.triangulate(data, "p")["triangles"]) == triangle_set(
        upstream.triangulate(data, "p")["triangles"]
    )


def test_hole_removal_matches_upstream_area_and_constraints():
    vertices = np.array(
        [
            [0.0, 0.0],
            [5.0, 0.2],
            [4.7, 4.4],
            [-0.2, 4.0],
            [1.1, 1.0],
            [3.6, 1.2],
            [3.3, 3.1],
            [0.9, 2.8],
        ]
    )
    segments = np.array(
        [
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 0],
            [4, 5],
            [5, 6],
            [6, 7],
            [7, 4],
        ]
    )
    data = {"vertices": vertices, "segments": segments, "holes": [[2.0, 2.0]]}
    ours = triangle.triangulate(data, "p")
    theirs = upstream.triangulate(data, "p")
    assert mesh_area(vertices, ours["triangles"]) == pytest.approx(
        mesh_area(vertices, theirs["triangles"]), abs=1e-12
    )
    ours_edges = edge_set([
        edge
        for tri in ours["triangles"]
        for edge in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0]))
    ])
    assert edge_set(segments) <= ours_edges
    hole = np.asarray(data["holes"][0])
    for tri in ours["triangles"]:
        polygon = vertices[tri]
        crosses = []
        for i in range(3):
            edge = polygon[(i + 1) % 3] - polygon[i]
            delta = hole - polygon[i]
            crosses.append(edge[0] * delta[1] - edge[1] * delta[0])
        assert not (all(value >= 0 for value in crosses) or all(value <= 0 for value in crosses))


def test_edge_neighbor_and_marker_outputs_match_upstream_semantics():
    vertices = np.array(
        [[0, 0], [3, 0], [3, 2], [1.1, 1.0], [0, 2], [2.2, 0.8]], float
    )
    segments = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 0]])
    markers = np.array([[2], [3], [4], [5], [6]], np.int32)
    data = {
        "vertices": vertices,
        "segments": segments,
        "segment_markers": markers,
    }
    ours = triangle.triangulate(data, "pen")
    theirs = upstream.triangulate(data, "pen")
    assert edge_set(ours["edges"]) == edge_set(theirs["edges"])
    assert Counter(np.count_nonzero(row >= 0) for row in ours["neighbors"]) == Counter(
        np.count_nonzero(row >= 0) for row in theirs["neighbors"]
    )
    marker_by_edge = {
        tuple(sorted(map(int, edge))): int(marker)
        for edge, marker in zip(ours["edges"], ours["edge_markers"].ravel())
    }
    for edge, marker in zip(segments, markers.ravel()):
        assert marker_by_edge[tuple(sorted(map(int, edge)))] == marker


def test_c_option_returns_convex_hull_segments():
    vertices = np.random.default_rng(9).random((50, 2))
    result = triangle.triangulate({"vertices": vertices}, "c")
    assert edge_set(result["segments"]) == edge_set(upstream.convex_hull(vertices))


@pytest.mark.parametrize("option", ["z", "Q", "X", "C", "i", "F", "l"])
def test_documented_compatibility_switches_preserve_the_covered_mesh(option):
    vertices = np.random.default_rng(23).random((20, 2))
    assert triangle_set(
        triangle.triangulate({"vertices": vertices}, option)["triangles"]
    ) == triangle_set(triangle.triangulate({"vertices": vertices})["triangles"])


def test_internal_constrained_edge_is_recovered():
    vertices = np.array([[0, 0], [2, 0], [2, 2], [0, 2]], dtype=float)
    segments = np.array([[0, 1], [1, 2], [2, 3], [3, 0], [0, 2]])
    result = triangle.triangulate(
        {"vertices": vertices, "segments": segments}, "p"
    )
    mesh_edges = edge_set([
        edge
        for tri in result["triangles"]
        for edge in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0]))
    ])
    assert edge_set(segments) <= mesh_edges
    assert triangle_set(result["triangles"]) == triangle_set(
        upstream.triangulate(
            {"vertices": vertices, "segments": segments}, "p"
        )["triangles"]
    )


def test_open_pslg_is_eaten_like_upstream():
    data = {
        "vertices": [[0, 0], [2, 0], [2, 2], [0, 2], [0.7, 0.8]],
        "segments": [[0, 2]],
    }
    assert "triangles" not in triangle.triangulate(data, "p")
    assert "triangles" not in upstream.triangulate(data, "p")


def test_collinear_points_return_no_triangles():
    result = triangle.triangulate({"vertices": [[0, 0], [1, 0], [2, 0]]})
    assert "triangles" not in result


def test_rejects_crossing_segments():
    data = {
        "vertices": [[0, 0], [1, 0], [1, 1], [0, 1]],
        "segments": [[0, 2], [1, 3]],
    }
    with pytest.raises(ValueError, match="must not cross"):
        triangle.triangulate(data, "p")


@pytest.mark.parametrize("opts", ["q", "a0.1", "D", "r"])
def test_unsupported_refinement_options_fail_loudly(opts):
    with pytest.raises(NotImplementedError):
        triangle.triangulate({"vertices": [[0, 0], [1, 0], [0, 1]]}, opts)
