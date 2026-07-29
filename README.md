# mojo-triangle

`mojo-triangle` is an independent Mojo implementation of two-dimensional
constrained Delaunay triangulation with a NumPy-facing Python API. The covered
functions deliberately match the useful core of the
[`triangle`](https://rufat.be/triangle/API.html) package:

- `triangulate(tri, opts="")`
- `delaunay(pts)`
- `convex_hull(pts)`

This is a functional port, not a wrapper around Triangle's C library. It is
currently slower than that mature C implementation; the benchmark below makes
the gap explicit.

## Coverage

`triangulate` supports point sets and planar straight-line graphs containing
non-crossing `segments`, including concave boundaries, internal constrained
edges, and `holes`. It emits `vertices`, `vertex_markers`, `triangles`,
`segments`, and `segment_markers` using Triangle-compatible array shapes.
Options `e` and `n` add edge and neighbor arrays, and `c` adds convex-hull
segments. The usual compatibility switches `z`, `Q`, `X`, `C`, `i`, `F`, and
`l` are accepted; indexing is always zero-based and the algorithm remains the
Mojo incremental implementation.

The implementation does not perform quality or area refinement (`q`, `a`),
conforming Delaunay refinement (`D`), mesh refinement (`r`), Steiner-point
insertion or segment splitting (`s`, `S`). It also does not cover regions,
per-triangle attributes, Voronoi output, or Triangle's file-loading and plotting
helpers. Unsupported meshing switches raise `NotImplementedError` instead of
being silently ignored.

Predicates use scaled `Float64` tolerances. Unlike upstream Triangle, this port
does not provide adaptive exact arithmetic. Duplicate points, crossing
segments, and a segment passing through a non-endpoint vertex are rejected
early. Split such segments at the intermediate vertices before calling the
port.

## Install

The repository pins the tested Mojo nightly and supplies all development
dependencies through Pixi:

```bash
pixi install
pixi run build
pixi run test
```

The Pixi environment adds `python/` to `PYTHONPATH`. The upstream `py-triangle`
package is installed only to provide the parity oracle and benchmark
comparison.

## Usage

```python
import numpy as np
import mojo_triangle as triangle

mesh = triangle.triangulate(
    {
        "vertices": np.array(
            [[0, 0], [3, 0], [3, 2], [1.2, 0.8], [0, 2]],
            dtype=np.float64,
        ),
        "segments": np.array(
            [[0, 1], [1, 2], [2, 3], [3, 4], [4, 0]],
            dtype=np.int32,
        ),
    },
    "pen",
)

print(mesh["triangles"])
# [[1 2 3]
#  [0 1 3]
#  [0 3 4]]
```

Save the example as `example.py` and run it from the checkout with
`pixi run python example.py`.

## Benchmarks

Measured with `pixi run bench` on an Intel Xeon E5-2697 v4 at 2.30 GHz,
Linux 6.8.0-136-generic x86-64, using the best of five warm runs. A ratio below
`1.00x` means the Mojo port is slower.

| case | mojo-triangle | upstream triangle | upstream / Mojo |
|---|---:|---:|---:|
| delaunay, 250 points | 0.398 ms | 0.296 ms | 0.744x |
| delaunay, 1,000 points | 3.446 ms | 0.792 ms | 0.230x |
| delaunay, 3,000 points | 14.195 ms | 2.473 ms | 0.174x |
| constrained polygon, 250 points | 2.029 ms | 0.368 ms | 0.181x |
| constrained polygon, 1,000 points | 9.181 ms | 1.367 ms | 0.149x |

Upstream remains faster in every measured case. The Mojo cavity predicate now
uses translation-stable cached coefficients scanned at the native Float64 SIMD
width, with an explicit scalar tail. Cavity removal touches only matched
triangles, unconstrained meshes skip redundant post-construction legalization,
and constrained meshes use a scratch-buffer edge hash with adjacency links
maintained across flips. PSLG validation and edge extraction are batched NumPy
operations instead of Python scalar geometry loops.

No threaded CPU path is enabled. Point insertions mutate the mesh sequentially,
and the independent scan inside each insertion is too short at these sizes to
amortize thousands of thread launches after SIMD. No GPU path is provided
either: the hot geometry loops perform well below two floating-point operations
per byte moved once coefficient, index, and coordinate traffic is counted, so
host/device transfer and launch overhead would dominate. The `max` dependency
is therefore intentionally not added.

## How it works

The Mojo kernel first builds an unconstrained Delaunay mesh with incremental
Bowyer-Watson insertion and a temporary super-triangle. It then builds
adjacency through an open-addressed edge table, recovers each PSLG segment by
flipping intersecting edges, legalizes unconstrained interior edges, and
flood-fills from the convex hull and each hole seed to remove triangles outside
the requested domain.

Python and NumPy own every allocation. C-contiguous `float64` coordinates and
native 64-bit index workspaces cross the C ABI as integer addresses through
`ctypes`; Mojo reconstructs mutable pointers and writes into those buffers.
The final public index arrays are returned as Triangle-compatible `int32`
arrays. One compilation unit produces `dist/libmojo-triangle.so`, so there is
no per-kernel build or allocation overhead.
