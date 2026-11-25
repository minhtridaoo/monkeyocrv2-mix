"""Box-frame geometry for learned box refinement (pure numpy, CPU-testable).

A detector quad p0..p3 (cyclic) defines a bilinear map from the unit square:
    P(s, t) = (1-s)(1-t) p0 + s(1-t) p1 + s t p2 + (1-s) t p3
The refined box keeps the quad's frame (no rotation) and moves its four edges:
    s in [0 + d_left, 1 + d_right],  t in [0 + d_top, 1 + d_bottom]
so the offsets are normalized by the box's own side lengths, offset 0 reproduces the quad EXACTLY, and the
result is the image of an axis-aligned rectangle in (s, t) -> never self-intersecting while s1 > s0 and t1 > t0.
"""
import numpy as np

BOUND = .5        # |offset| per edge, in units of that side's length (primary, locked before any A number)
MIN_EXTENT = .1   # refined side must keep >= 10 % of the original side, else fallback to the original quad


def quad(poly):
    q = np.asarray(poly, dtype=float).reshape(-1, 2)
    return q if q.shape == (4, 2) and np.isfinite(q).all() else None


def bilinear(q, s, t):
    s, t = np.asarray(s, float)[..., None], np.asarray(t, float)[..., None]
    return (1 - s)*(1 - t)*q[0] + s*(1 - t)*q[1] + s*t*q[2] + (1 - s)*t*q[3]


def apply_offsets(q, d, bound=BOUND, min_extent=MIN_EXTENT):
    """d = (left, right, top, bottom) in side units. Returns (new 8-list, status). status 'ok' | 'fallback_*'."""
    d = np.asarray(d, dtype=float)
    if q is None:
        return None, 'fallback_not_quad'
    if d.shape != (4,) or not np.isfinite(d).all():
        return q.reshape(-1).tolist(), 'fallback_nonfinite'
    if bound is not None:
        d = np.clip(d, -bound, bound)
    s0, s1, t0, t1 = 0. + d[0], 1. + d[1], 0. + d[2], 1. + d[3]
    if s1 - s0 < min_extent or t1 - t0 < min_extent:
        return q.reshape(-1).tolist(), 'fallback_collapse'
    if not d.any():
        return q.reshape(-1).tolist(), 'ok'
    pts = bilinear(q, [s0, s1, s1, s0], [t0, t0, t1, t1])
    if signed_area(pts)*signed_area(q) <= 0:
        return q.reshape(-1).tolist(), 'fallback_orientation'
    return pts.reshape(-1).tolist(), 'ok'


def signed_area(p):
    x, y = p[:, 0], p[:, 1]
    return .5*float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def to_frame(q, pts):
    """Affine (parallelogram) inverse of the quad frame: image points -> (s, t). Used only for TARGETS."""
    a, b = q[1] - q[0], q[3] - q[0]
    M = np.stack([a, b], 1)
    if abs(np.linalg.det(M)) < 1e-9:
        return None
    return np.linalg.solve(M, (np.asarray(pts, float) - q[0]).T).T


def target_offsets(q, gt_pts):
    """Regression target: GT extent in the box frame -> (left, right, top, bottom) offsets (unclamped)."""
    st = to_frame(q, gt_pts)
    if st is None:
        return None
    return np.array([st[:, 0].min(), st[:, 0].max() - 1., st[:, 1].min(), st[:, 1].max() - 1.])


def uniform_offsets(q, d_px):
    """Unclip-like control: move every edge outward by the same pixel distance d_px."""
    la, lb = np.linalg.norm(q[1] - q[0]), np.linalg.norm(q[3] - q[0])
    return np.array([-d_px/la, d_px/la, -d_px/lb, d_px/lb])
