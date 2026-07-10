"""R2 robust-center fusion pure logic."""

from smoothnav.graph_fusion import robust_center_from_views, view_spread


def test_median_ignores_single_outlier_vote():
    votes = [[1.0, 1.0, 0.5], [1.1, 0.9, 0.5], [1.0, 1.1, 0.5], [9.0, 9.0, 0.5]]
    center = robust_center_from_views(votes, min_views=2)
    assert abs(center[0] - 1.05) < 1e-9
    assert abs(center[1] - 1.05) < 1e-9


def test_mean_would_have_been_dragged():
    votes = [[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [9.0, 9.0, 0.0]]
    center = robust_center_from_views(votes, min_views=2)
    assert center[0] == 1.0 and center[1] == 1.0  # mean would be 3.0


def test_below_min_views_returns_none():
    assert robust_center_from_views([[1.0, 2.0, 3.0]], min_views=2) is None
    assert robust_center_from_views([], min_views=2) is None
    assert robust_center_from_views(None, min_views=2) is None


def test_single_view_allowed_when_min_views_one():
    assert robust_center_from_views([[1.0, 2.0, 3.0]], min_views=1) == [1.0, 2.0, 3.0]


def test_malformed_votes_are_skipped():
    votes = [None, [1.0], "bad", [2.0, 2.0, 2.0], [4.0, 4.0, 4.0]]
    center = robust_center_from_views(votes, min_views=2)
    assert center == [3.0, 3.0, 3.0]


def test_odd_count_takes_middle_vote():
    votes = [[0.0, 0.0, 0.0], [5.0, 5.0, 5.0], [100.0, 100.0, 100.0]]
    assert robust_center_from_views(votes, min_views=2) == [5.0, 5.0, 5.0]


def test_view_spread_diagnostic():
    assert view_spread([[0.0, 0.0, 0.0], [3.0, 4.0, 0.0]]) == 5.0
    assert view_spread([[1.0, 1.0]]) == 0.0
    assert view_spread(None) == 0.0
