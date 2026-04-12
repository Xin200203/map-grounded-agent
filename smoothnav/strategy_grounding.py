"""Strategy grounding helpers decoupled from the heavy runtime entrypoint."""


def apply_strategy(strategy, graph, bev_map, args, global_goals):
    """Ground a semantic strategy into a local-map goal update."""
    graph.set_full_map(bev_map.full_map)
    graph.set_full_pose(bev_map.full_pose)

    bias = strategy.bias_position if strategy else None
    goal = graph.get_goal(goal=bias)

    if goal is None:
        return

    goal = list(goal)
    goal[0] -= bev_map.local_map_boundary[0, 0]
    goal[1] -= bev_map.local_map_boundary[0, 2]
    if 0 <= goal[0] < args.local_width and 0 <= goal[1] < args.local_height:
        global_goals[0] = goal[0]
        global_goals[1] = goal[1]
