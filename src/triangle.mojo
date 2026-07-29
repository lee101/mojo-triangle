"""Incremental constrained Delaunay triangulation over caller-owned buffers."""

from std.sys.info import simd_width_of

comptime FPtr = UnsafePointer[Float64, AnyOrigin[mut=True]]
comptime IPtr = UnsafePointer[Int, AnyOrigin[mut=True]]


def px(points: FPtr, n: Int, vertex: Int, midx: Float64, span: Float64) -> Float64:
    if vertex < n:
        return points[2 * vertex]
    if vertex == n:
        return midx - 4096.0 * span
    if vertex == n + 1:
        return midx + 4096.0 * span
    return midx


def py(points: FPtr, n: Int, vertex: Int, midy: Float64, span: Float64) -> Float64:
    if vertex < n:
        return points[2 * vertex + 1]
    if vertex == n or vertex == n + 1:
        return midy - 2048.0 * span
    return midy + 4096.0 * span


def orient(
    points: FPtr,
    n: Int,
    a: Int,
    b: Int,
    c: Int,
    midx: Float64,
    midy: Float64,
    span: Float64,
) -> Float64:
    return (
        (px(points, n, b, midx, span) - px(points, n, a, midx, span))
        * (py(points, n, c, midy, span) - py(points, n, a, midy, span))
        - (py(points, n, b, midy, span) - py(points, n, a, midy, span))
        * (px(points, n, c, midx, span) - px(points, n, a, midx, span))
    )


def incircle(
    points: FPtr,
    n: Int,
    a: Int,
    b: Int,
    c: Int,
    d: Int,
    midx: Float64,
    midy: Float64,
    span: Float64,
) -> Float64:
    var ax = px(points, n, a, midx, span) - px(points, n, d, midx, span)
    var ay = py(points, n, a, midy, span) - py(points, n, d, midy, span)
    var bx = px(points, n, b, midx, span) - px(points, n, d, midx, span)
    var by = py(points, n, b, midy, span) - py(points, n, d, midy, span)
    var cx = px(points, n, c, midx, span) - px(points, n, d, midx, span)
    var cy = py(points, n, c, midy, span) - py(points, n, d, midy, span)
    return (
        (ax * ax + ay * ay) * (bx * cy - by * cx)
        - (bx * bx + by * by) * (ax * cy - ay * cx)
        + (cx * cx + cy * cy) * (ax * by - ay * bx)
    )


def set_circle_coefficients(
    points: FPtr,
    n: Int,
    triangles: IPtr,
    t: Int,
    coefficients: FPtr,
    capacity: Int,
    midx: Float64,
    midy: Float64,
    span: Float64,
):
    var a = Int(triangles[3 * t])
    var b = Int(triangles[3 * t + 1])
    var c = Int(triangles[3 * t + 2])
    var ax = px(points, n, a, midx, span) - midx
    var ay = py(points, n, a, midy, span) - midy
    var bx = px(points, n, b, midx, span) - midx
    var by = py(points, n, b, midy, span) - midy
    var cx = px(points, n, c, midx, span) - midx
    var cy = py(points, n, c, midy, span) - midy
    var aa = ax * ax + ay * ay
    var bb = bx * bx + by * by
    var cc = cx * cx + cy * cy
    coefficients[t] = (
        ax * (by - cy) + bx * (cy - ay) + cx * (ay - by)
    )
    coefficients[capacity + t] = -(
        ay * (bb - cc) + by * (cc - aa) + cy * (aa - bb)
    )
    coefficients[2 * capacity + t] = (
        ax * (bb - cc) + bx * (cc - aa) + cx * (aa - bb)
    )
    coefficients[3 * capacity + t] = (
        ax * (by * cc - bb * cy)
        - ay * (bx * cc - bb * cx)
        + aa * (bx * cy - by * cx)
    )


def mark_cavity(
    points: FPtr,
    vertex: Int,
    marks: FPtr,
    bad_triangles: IPtr,
    nt: Int,
    coefficients: FPtr,
    capacity: Int,
    midx: Float64,
    midy: Float64,
    circle_eps: Float64,
) -> Int:
    comptime W = simd_width_of[DType.float64]()
    var dx = points[2 * vertex] - midx
    var dy = points[2 * vertex + 1] - midy
    var squared = dx * dx + dy * dy
    var t = 0
    var nbad = 0
    while t + W <= nt:
        var orientation = coefficients.load[width=W](t)
        var linear_x = coefficients.load[width=W](capacity + t)
        var linear_y = coefficients.load[width=W](2 * capacity + t)
        var constant = coefficients.load[width=W](3 * capacity + t)
        var values = (
            constant + linear_x * dx + linear_y * dy - orientation * squared
        )
        var mask = values.gt(circle_eps)
        marks.store(t, mask.cast[DType.float64]())
        if mask.reduce_or():
            for lane in range(W):
                if mask[lane]:
                    bad_triangles[nbad] = t + lane
                    nbad += 1
        t += W
    while t < nt:
        var value = (
            coefficients[3 * capacity + t]
            + coefficients[capacity + t] * dx
            + coefficients[2 * capacity + t] * dy
            - coefficients[t] * squared
        )
        if value > circle_eps:
            marks[t] = 1.0
            bad_triangles[nbad] = t
            nbad += 1
        else:
            marks[t] = 0.0
        t += 1
    return nbad


def remove_cavity(
    triangles: IPtr,
    marks: FPtr,
    nt: Int,
    bad_triangles: IPtr,
    nbad: Int,
    coefficients: FPtr,
    capacity: Int,
) -> Int:
    var last = nt - 1
    for i in range(nbad):
        var t = Int(bad_triangles[i])
        if t > last:
            break
        while last > t and marks[last] != 0.0:
            last -= 1
        if last == t:
            last -= 1
            break
        triangles[3 * t] = triangles[3 * last]
        triangles[3 * t + 1] = triangles[3 * last + 1]
        triangles[3 * t + 2] = triangles[3 * last + 2]
        for k in range(4):
            coefficients[k * capacity + t] = coefficients[k * capacity + last]
        last -= 1
    return last + 1


def tri_has_edge(triangles: IPtr, t: Int, a: Int, b: Int) -> Bool:
    var x = Int(triangles[3 * t])
    var y = Int(triangles[3 * t + 1])
    var z = Int(triangles[3 * t + 2])
    return (
        (x == a and y == b) or (x == b and y == a)
        or (y == a and z == b) or (y == b and z == a)
        or (z == a and x == b) or (z == b and x == a)
    )


def opposite(triangles: IPtr, t: Int, a: Int, b: Int) -> Int:
    for k in range(3):
        var vertex = Int(triangles[3 * t + k])
        if vertex != a and vertex != b:
            return vertex
    return -1


def edge_slot(triangles: IPtr, t: Int, a: Int, b: Int) -> Int:
    for e in range(3):
        var u = Int(triangles[3 * t + e])
        var v = Int(triangles[3 * t + (e + 1) % 3])
        if (u == a and v == b) or (u == b and v == a):
            return e
    return -1


def link_across(
    triangles: IPtr,
    neighbors: IPtr,
    t: Int,
    a: Int,
    b: Int,
    other: Int,
):
    var slot = edge_slot(triangles, t, a, b)
    if slot >= 0:
        neighbors[3 * t + slot] = other
    if other >= 0:
        var other_slot = edge_slot(triangles, other, a, b)
        if other_slot >= 0:
            neighbors[3 * other + other_slot] = t


def build_adjacency(
    triangles: IPtr,
    nt: Int,
    n: Int,
    neighbors: IPtr,
    hash_keys: IPtr,
    hash_values: IPtr,
    hash_size: Int,
):
    for i in range(3 * nt):
        neighbors[i] = -1
    for i in range(hash_size):
        hash_keys[i] = -1
    for t in range(nt):
        for e in range(3):
            var a = Int(triangles[3 * t + e])
            var b = Int(triangles[3 * t + (e + 1) % 3])
            var low = min(a, b)
            var high = max(a, b)
            var key = low * (n + 1) + high
            var slot = key % hash_size
            while hash_keys[slot] >= 0 and hash_keys[slot] != key:
                slot = (slot + 1) % hash_size
            if hash_keys[slot] < 0:
                hash_keys[slot] = key
                hash_values[slot] = 3 * t + e
            else:
                var encoded = Int(hash_values[slot])
                var other = encoded // 3
                var other_edge = encoded % 3
                neighbors[3 * t + e] = other
                neighbors[3 * other + other_edge] = t


def is_constraint(segments: IPtr, m: Int, a: Int, b: Int) -> Bool:
    for s in range(m):
        var u = Int(segments[2 * s])
        var v = Int(segments[2 * s + 1])
        if (u == a and v == b) or (u == b and v == a):
            return True
    return False


def mesh_has_edge(triangles: IPtr, nt: Int, a: Int, b: Int) -> Bool:
    for t in range(nt):
        if tri_has_edge(triangles, t, a, b):
            return True
    return False


def proper_cross(
    points: FPtr,
    n: Int,
    a: Int,
    b: Int,
    c: Int,
    d: Int,
    midx: Float64,
    midy: Float64,
    span: Float64,
    eps: Float64,
) -> Bool:
    var o1 = orient(points, n, a, b, c, midx, midy, span)
    var o2 = orient(points, n, a, b, d, midx, midy, span)
    var o3 = orient(points, n, c, d, a, midx, midy, span)
    var o4 = orient(points, n, c, d, b, midx, midy, span)
    return o1 * o2 < -eps and o3 * o4 < -eps


def set_ccw(
    points: FPtr,
    n: Int,
    triangles: IPtr,
    t: Int,
    a: Int,
    b: Int,
    c: Int,
    midx: Float64,
    midy: Float64,
    span: Float64,
):
    if orient(points, n, a, b, c, midx, midy, span) > 0.0:
        triangles[3 * t] = a
        triangles[3 * t + 1] = b
        triangles[3 * t + 2] = c
    else:
        triangles[3 * t] = b
        triangles[3 * t + 1] = a
        triangles[3 * t + 2] = c


def flip_edge(
    points: FPtr,
    n: Int,
    triangles: IPtr,
    neighbors: IPtr,
    t: Int,
    other: Int,
    a: Int,
    b: Int,
    c: Int,
    d: Int,
    midx: Float64,
    midy: Float64,
    span: Float64,
):
    var neighbor_ac = Int(neighbors[3 * t + edge_slot(triangles, t, a, c)])
    var neighbor_bc = Int(neighbors[3 * t + edge_slot(triangles, t, b, c)])
    var neighbor_ad = Int(
        neighbors[3 * other + edge_slot(triangles, other, a, d)]
    )
    var neighbor_bd = Int(
        neighbors[3 * other + edge_slot(triangles, other, b, d)]
    )
    set_ccw(points, n, triangles, t, c, d, a, midx, midy, span)
    set_ccw(points, n, triangles, other, d, c, b, midx, midy, span)
    for e in range(3):
        neighbors[3 * t + e] = -1
        neighbors[3 * other + e] = -1
    link_across(triangles, neighbors, t, c, d, other)
    link_across(triangles, neighbors, t, a, c, neighbor_ac)
    link_across(triangles, neighbors, t, a, d, neighbor_ad)
    link_across(triangles, neighbors, other, b, c, neighbor_bc)
    link_across(triangles, neighbors, other, b, d, neighbor_bd)


def recover_constraints(
    points: FPtr,
    n: Int,
    segments: IPtr,
    m: Int,
    triangles: IPtr,
    neighbors: IPtr,
    nt: Int,
    midx: Float64,
    midy: Float64,
    span: Float64,
    eps: Float64,
) -> Bool:
    for s in range(m):
        var u = Int(segments[2 * s])
        var v = Int(segments[2 * s + 1])
        var attempts = 0
        while not mesh_has_edge(triangles, nt, u, v):
            var did_flip = False
            for t in range(nt):
                if did_flip:
                    break
                for e in range(3):
                    var a = Int(triangles[3 * t + e])
                    var b = Int(triangles[3 * t + (e + 1) % 3])
                    if not proper_cross(
                        points, n, u, v, a, b, midx, midy, span, eps
                    ):
                        continue
                    var other = Int(neighbors[3 * t + e])
                    if other < 0:
                        continue
                    var c = opposite(triangles, t, a, b)
                    var d = opposite(triangles, other, a, b)
                    if proper_cross(
                        points, n, a, b, c, d, midx, midy, span, eps
                    ):
                        flip_edge(
                            points, n, triangles, neighbors,
                            t, other, a, b, c, d,
                            midx, midy, span,
                        )
                        did_flip = True
                        break
            attempts += 1
            if not did_flip or attempts > 8 * nt * nt:
                return False
    return True


def legalize(
    points: FPtr,
    n: Int,
    segments: IPtr,
    m: Int,
    triangles: IPtr,
    neighbors: IPtr,
    nt: Int,
    midx: Float64,
    midy: Float64,
    span: Float64,
    cross_eps: Float64,
    circle_eps: Float64,
):
    var passes = 0
    while passes < 8 * nt:
        var changed = False
        for t in range(nt):
            if changed:
                break
            for e in range(3):
                var a = Int(triangles[3 * t + e])
                var b = Int(triangles[3 * t + (e + 1) % 3])
                if is_constraint(segments, m, a, b):
                    continue
                var other = Int(neighbors[3 * t + e])
                if other <= t:
                    continue
                var c = Int(triangles[3 * t + (e + 2) % 3])
                var d = opposite(triangles, other, a, b)
                if (
                    proper_cross(
                        points, n, a, b, c, d, midx, midy, span, cross_eps
                    )
                    and incircle(
                        points, n, a, b, c, d, midx, midy, span
                    ) > circle_eps
                ):
                    flip_edge(
                        points, n, triangles, neighbors,
                        t, other, a, b, c, d,
                        midx, midy, span,
                    )
                    changed = True
                    break
        if not changed:
            break
        passes += 1


def mark_connected(
    segments: IPtr,
    m: Int,
    triangles: IPtr,
    neighbors: IPtr,
    nt: Int,
    marks: FPtr,
):
    var changed = True
    while changed:
        changed = False
        for t in range(nt):
            if marks[t] == 0.0:
                continue
            for e in range(3):
                var a = Int(triangles[3 * t + e])
                var b = Int(triangles[3 * t + (e + 1) % 3])
                if is_constraint(segments, m, a, b):
                    continue
                var other = Int(neighbors[3 * t + e])
                if other >= 0 and marks[other] == 0.0:
                    marks[other] = 1.0
                    changed = True


def filter_domain(
    points: FPtr,
    n: Int,
    segments: IPtr,
    m: Int,
    holes: FPtr,
    nholes: Int,
    triangles: IPtr,
    neighbors: IPtr,
    nt: Int,
    marks: FPtr,
    midx: Float64,
    midy: Float64,
    span: Float64,
    eps: Float64,
) -> Int:
    for t in range(nt):
        marks[t] = 0.0
    for t in range(nt):
        for e in range(3):
            var a = Int(triangles[3 * t + e])
            var b = Int(triangles[3 * t + (e + 1) % 3])
            if (
                neighbors[3 * t + e] < 0
                and not is_constraint(segments, m, a, b)
            ):
                marks[t] = 1.0
    mark_connected(segments, m, triangles, neighbors, nt, marks)
    for h in range(nholes):
        var hx = holes[2 * h]
        var hy = holes[2 * h + 1]
        for t in range(nt):
            var a = Int(triangles[3 * t])
            var b = Int(triangles[3 * t + 1])
            var c = Int(triangles[3 * t + 2])
            var ab = (
                (px(points, n, b, midx, span) - px(points, n, a, midx, span))
                * (hy - py(points, n, a, midy, span))
                - (py(points, n, b, midy, span) - py(points, n, a, midy, span))
                * (hx - px(points, n, a, midx, span))
            )
            var bc = (
                (px(points, n, c, midx, span) - px(points, n, b, midx, span))
                * (hy - py(points, n, b, midy, span))
                - (py(points, n, c, midy, span) - py(points, n, b, midy, span))
                * (hx - px(points, n, b, midx, span))
            )
            var ca = (
                (px(points, n, a, midx, span) - px(points, n, c, midx, span))
                * (hy - py(points, n, c, midy, span))
                - (py(points, n, a, midy, span) - py(points, n, c, midy, span))
                * (hx - px(points, n, c, midx, span))
            )
            if ab >= -eps and bc >= -eps and ca >= -eps:
                marks[t] = 1.0
                break
    mark_connected(segments, m, triangles, neighbors, nt, marks)
    var kept = 0
    for t in range(nt):
        if marks[t] == 0.0:
            triangles[3 * kept] = triangles[3 * t]
            triangles[3 * kept + 1] = triangles[3 * t + 1]
            triangles[3 * kept + 2] = triangles[3 * t + 2]
            kept += 1
    return kept


def triangulate_kernel(
    points: FPtr,
    n: Int,
    segments: IPtr,
    m: Int,
    holes: FPtr,
    nholes: Int,
    clip: Bool,
    triangles: IPtr,
    capacity: Int,
    marks: FPtr,
    edge_a: IPtr,
    edge_b: IPtr,
    coefficients: FPtr,
    neighbors: IPtr,
) -> Int:
    if n < 3:
        return 0
    var xmin = points[0]
    var xmax = points[0]
    var ymin = points[1]
    var ymax = points[1]
    for i in range(1, n):
        xmin = min(xmin, points[2 * i])
        xmax = max(xmax, points[2 * i])
        ymin = min(ymin, points[2 * i + 1])
        ymax = max(ymax, points[2 * i + 1])
    var midx = 0.5 * (xmin + xmax)
    var midy = 0.5 * (ymin + ymax)
    var span = max(xmax - xmin, ymax - ymin)
    if span <= 0.0:
        return -1
    var eps = 1.0e-14 * span * span
    var cross_eps = eps * eps
    var circle_eps = eps * span * span
    triangles[0] = n
    triangles[1] = n + 1
    triangles[2] = n + 2
    set_circle_coefficients(
        points, n, triangles, 0, coefficients, capacity, midx, midy, span
    )
    var nt = 1
    for vertex in range(n):
        var nbad = mark_cavity(
            points, vertex, marks, edge_b + capacity, nt, coefficients, capacity,
            midx, midy, circle_eps,
        )
        var ne = 0
        for bad in range(nbad):
            var t = Int(edge_b[capacity + bad])
            for e in range(3):
                var a = Int(triangles[3 * t + e])
                var b = Int(triangles[3 * t + (e + 1) % 3])
                var found = -1
                for j in range(ne):
                    if (
                        edge_a[j] >= 0
                        and (
                            (edge_a[j] == a and edge_b[j] == b)
                            or (edge_a[j] == b and edge_b[j] == a)
                        )
                    ):
                        found = j
                        break
                if found >= 0:
                    edge_a[found] = -1
                else:
                    edge_a[ne] = a
                    edge_b[ne] = b
                    ne += 1
        nt = remove_cavity(
            triangles, marks, nt, edge_b + capacity, nbad,
            coefficients, capacity,
        )
        for e in range(ne):
            if edge_a[e] < 0:
                continue
            if nt >= capacity:
                return -2
            set_ccw(
                points, n, triangles, nt, Int(edge_a[e]), Int(edge_b[e]), vertex,
                midx, midy, span,
            )
            set_circle_coefficients(
                points, n, triangles, nt, coefficients, capacity,
                midx, midy, span,
            )
            nt += 1
    var kept = 0
    for t in range(nt):
        if (
            triangles[3 * t] < n
            and triangles[3 * t + 1] < n
            and triangles[3 * t + 2] < n
        ):
            triangles[3 * kept] = triangles[3 * t]
            triangles[3 * kept + 1] = triangles[3 * t + 1]
            triangles[3 * kept + 2] = triangles[3 * t + 2]
            kept += 1
    nt = kept
    if m > 0 or clip:
        build_adjacency(
            triangles, nt, n, neighbors, edge_a, edge_b, 3 * capacity
        )
    if m > 0:
        if not recover_constraints(
            points, n, segments, m, triangles, neighbors, nt,
            midx, midy, span, cross_eps,
        ):
            return -3
        legalize(
            points, n, segments, m, triangles, neighbors, nt, midx, midy, span,
            cross_eps, circle_eps,
        )
    if clip:
        nt = filter_domain(
            points, n, segments, m, holes, nholes,
            triangles, neighbors, nt, marks,
            midx, midy, span, eps,
        )
    return nt


@export("mt_triangulate")
def mt_triangulate(
    points_addr: Int,
    n: Int,
    segments_addr: Int,
    m: Int,
    holes_addr: Int,
    nholes: Int,
    clip: Int,
    triangles_addr: Int,
    capacity: Int,
    marks_addr: Int,
    edge_a_addr: Int,
    edge_b_addr: Int,
    coefficients_addr: Int,
) abi("C") -> Int:
    var coefficients = FPtr(unsafe_from_address=coefficients_addr)
    var neighbors = IPtr(unsafe_from_address=coefficients_addr)
    return triangulate_kernel(
        FPtr(unsafe_from_address=points_addr),
        n,
        IPtr(unsafe_from_address=segments_addr),
        m,
        FPtr(unsafe_from_address=holes_addr),
        nholes,
        clip != 0,
        IPtr(unsafe_from_address=triangles_addr),
        capacity,
        FPtr(unsafe_from_address=marks_addr),
        IPtr(unsafe_from_address=edge_a_addr),
        IPtr(unsafe_from_address=edge_b_addr),
        coefficients,
        neighbors,
    )
