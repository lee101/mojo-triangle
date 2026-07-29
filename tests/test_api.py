import numpy as np
import pytest

import mojo_triangle as triangle


def test_triangulate_preserves_vertices_without_copying_input_mutations():
    vertices = np.array([[0, 0], [1, 0], [0, 1]], float)
    result = triangle.triangulate({"vertices": vertices})
    vertices[0] = 99
    assert np.array_equal(result["vertices"], [[0, 0], [1, 0], [0, 1]])


def test_vertex_markers_are_preserved():
    data = {
        "vertices": [[0, 0], [1, 0], [0, 1]],
        "vertex_markers": [7, 8, 9],
    }
    assert triangle.triangulate(data)["vertex_markers"].ravel().tolist() == [7, 8, 9]


@pytest.mark.parametrize(
    "data, message",
    [
        ({}, "vertices"),
        ({"vertices": [[0, 0, 0]]}, "shape"),
        ({"vertices": [[0, 0], [0, 0], [1, 1]]}, "duplicate"),
        (
            {
                "vertices": [[0, 0], [1, 0], [0, 1]],
                "segments": [[0, 4]],
            },
            "outside",
        ),
    ],
)
def test_input_validation(data, message):
    with pytest.raises(ValueError, match=message):
        triangle.triangulate(data, "p")


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("segments", [[0, 1.5]], ValueError),
        ("segments", [[0, 2**64 - 1]], OverflowError),
        ("vertex_markers", [1, 2, 2**40], OverflowError),
        ("segment_markers", [1.25, 2, 3], ValueError),
    ],
)
def test_integer_inputs_never_narrow_silently(field, value, error):
    data = {
        "vertices": [[0, 0], [1, 0], [0, 1]],
        "segments": [[0, 1], [1, 2], [2, 0]],
        field: value,
    }
    with pytest.raises(error):
        triangle.triangulate(data, "pe")
