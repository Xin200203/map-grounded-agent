"""Pure helpers for multi-view node-centroid fusion (R2).

The scene graph accumulates one point cloud per object; the node center was
the mean over all accumulated points, so a single bad frame (depth bleed at
mask borders, mis-associated detection merged at sim_threshold) drags the
navigation coordinate by meters. R2 keeps one centroid vote per detection
and takes the component-wise median: each frame gets one vote regardless of
how many points it contributed, and a minority of bad views is ignored.
"""


def robust_center_from_views(det_centroids, min_views=2):
    """Component-wise median over per-detection centroid votes.

    Returns [x, y, z] in the votes' coordinate frame, or None when fewer
    than `min_views` well-formed votes exist (caller falls back to the
    accumulated-point-cloud mean).
    """
    votes = []
    for c in det_centroids or []:
        if c is None:
            continue
        try:
            vote = [float(v) for v in list(c)[:3]]
        except (TypeError, ValueError):
            continue
        if len(vote) == 3:
            votes.append(vote)
    if len(votes) < max(int(min_views), 1):
        return None
    center = []
    n = len(votes)
    for axis in range(3):
        ordered = sorted(v[axis] for v in votes)
        mid = n // 2
        if n % 2:
            center.append(ordered[mid])
        else:
            center.append(0.5 * (ordered[mid - 1] + ordered[mid]))
    return center


def view_spread(det_centroids):
    """Max pairwise XY distance between votes (diagnostic; meters in the
    votes' frame). 0.0 when fewer than two well-formed votes."""
    votes = []
    for c in det_centroids or []:
        if c is None:
            continue
        try:
            vote = [float(v) for v in list(c)[:2]]
        except (TypeError, ValueError):
            continue
        if len(vote) == 2:
            votes.append(vote)
    spread = 0.0
    for i in range(len(votes)):
        for j in range(i + 1, len(votes)):
            dx = votes[i][0] - votes[j][0]
            dy = votes[i][1] - votes[j][1]
            spread = max(spread, (dx * dx + dy * dy) ** 0.5)
    return spread
