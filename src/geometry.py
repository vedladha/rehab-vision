import numpy as np

def angle_degrees(a, b, c, width, height):
    pts = np.asarray([a, b, c], dtype=float) * np.array([width, height])
    u, v = pts[0] - pts[1], pts[2] - pts[1]
    if not np.isfinite(pts).all() or np.linalg.norm(u) == 0 or np.linalg.norm(v) == 0:
        return np.nan
    cosine = np.clip(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)), -1, 1)
    return float(np.degrees(np.arccos(cosine)))

def midpoint(a, b):
    return (np.asarray(a, float) + np.asarray(b, float)) / 2

def trunk_angle(pose, width, height):
    s = midpoint(pose[5], pose[6]) * [width, height]
    h = midpoint(pose[11], pose[12]) * [width, height]
    d = s - h
    return float(np.degrees(np.arctan2(abs(d[0]), max(1e-9, -d[1]))))
