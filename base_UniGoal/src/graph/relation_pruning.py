"""Lightweight helpers for pruning graph relation candidates before LLM prompting."""

import math


def node_distance(node1, node2):
    center1 = getattr(node1, "center", None)
    center2 = getattr(node2, "center", None)
    if center1 is None or center2 is None:
        return float("inf")
    try:
        dx = float(center1[0]) - float(center2[0])
        dy = float(center1[1]) - float(center2[1])
    except (TypeError, ValueError, IndexError):
        return float("inf")
    return math.hypot(dx, dy)


def _same_room(node1, node2):
    room1 = getattr(node1, "room_node", None)
    room2 = getattr(node2, "room_node", None)
    return room1 is not None and room2 is not None and room1 is room2


def select_relation_node_pairs(new_nodes, old_nodes, *, same_room_only=False,
                               topk_per_new_node=0, max_pairs=0,
                               max_unknown_room_distance=12.0):
    candidates = []
    seen_pairs = set()
    all_nodes = list(new_nodes)

    for new_node in new_nodes:
        for old_node in old_nodes:
            key = tuple(sorted((id(new_node), id(old_node))))
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            candidates.append((new_node, old_node))
        all_nodes.append(new_node)

    for idx, node1 in enumerate(new_nodes):
        for node2 in new_nodes[idx + 1:]:
            key = tuple(sorted((id(node1), id(node2))))
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            candidates.append((node1, node2))

    initial_pair_count = len(candidates)
    dropped_pairs = []

    if same_room_only:
        kept = []
        for node1, node2 in candidates:
            room1 = getattr(node1, "room_node", None)
            room2 = getattr(node2, "room_node", None)
            if room1 is None or room2 is None:
                if node_distance(node1, node2) > max_unknown_room_distance:
                    dropped_pairs.append((node1, node2))
                    continue
                kept.append((node1, node2))
                continue
            if room1 is not room2:
                dropped_pairs.append((node1, node2))
                continue
            kept.append((node1, node2))
        candidates = kept

    candidates = sorted(
        candidates,
        key=lambda pair: (node_distance(pair[0], pair[1]), id(pair[0]), id(pair[1])),
    )

    if topk_per_new_node:
        counts = {id(node): 0 for node in new_nodes}
        kept = []
        for node1, node2 in candidates:
            incident_new_nodes = [
                node_id
                for node_id in (id(node1), id(node2))
                if node_id in counts
            ]
            if any(counts[node_id] >= topk_per_new_node for node_id in incident_new_nodes):
                dropped_pairs.append((node1, node2))
                continue
            for node_id in incident_new_nodes:
                counts[node_id] += 1
            kept.append((node1, node2))
        candidates = kept

    deferred_pairs = []
    if max_pairs and len(candidates) > max_pairs:
        deferred_pairs = candidates[max_pairs:]
        candidates = candidates[:max_pairs]

    stats = {
        "initial_edge_count": int(initial_pair_count),
        "selected_edge_count": len(candidates),
        "dropped_edge_count": len(dropped_pairs),
        "deferred_edge_count": len(deferred_pairs),
        "same_room_only": bool(same_room_only),
        "topk_per_new_node": int(topk_per_new_node or 0),
        "max_pairs": int(max_pairs or 0),
    }
    return candidates, dropped_pairs, deferred_pairs, stats


def edge_distance(edge):
    return node_distance(getattr(edge, "node1", None), getattr(edge, "node2", None))


def _dedupe_edges(edges):
    seen = set()
    ordered = []
    for edge in edges:
        edge_id = id(edge)
        if edge_id in seen:
            continue
        seen.add(edge_id)
        ordered.append(edge)
    return ordered


def prune_relation_edges(edges, *, new_nodes=None, same_room_only=False,
                         topk_per_new_node=0, max_pairs=0):
    ordered_edges = _dedupe_edges(edges)
    dropped_edges = []

    if same_room_only:
        kept = []
        for edge in ordered_edges:
            node1 = getattr(edge, "node1", None)
            node2 = getattr(edge, "node2", None)
            room1 = getattr(node1, "room_node", None)
            room2 = getattr(node2, "room_node", None)
            if room1 is None or room2 is None:
                if node_distance(node1, node2) > 12.0:
                    dropped_edges.append(edge)
                    continue
            elif room1 is not room2:
                dropped_edges.append(edge)
                continue
            kept.append(edge)
        ordered_edges = kept

    ordered_edges = sorted(
        ordered_edges,
        key=lambda edge: (edge_distance(edge), id(edge)),
    )

    if topk_per_new_node and new_nodes:
        new_node_ids = {id(node) for node in new_nodes}
        counts = {node_id: 0 for node_id in new_node_ids}
        kept = []
        for edge in ordered_edges:
            incident_new_nodes = [
                node_id
                for node_id in (
                    id(getattr(edge, "node1", None)),
                    id(getattr(edge, "node2", None)),
                )
                if node_id in new_node_ids
            ]
            if not incident_new_nodes:
                kept.append(edge)
                continue
            if any(counts[node_id] >= topk_per_new_node for node_id in incident_new_nodes):
                dropped_edges.append(edge)
                continue
            for node_id in incident_new_nodes:
                counts[node_id] += 1
            kept.append(edge)
        ordered_edges = kept

    deferred_edges = []
    if max_pairs and len(ordered_edges) > max_pairs:
        deferred_edges = ordered_edges[max_pairs:]
        ordered_edges = ordered_edges[:max_pairs]

    stats = {
        "initial_edge_count": len(_dedupe_edges(edges)),
        "selected_edge_count": len(ordered_edges),
        "dropped_edge_count": len(dropped_edges),
        "deferred_edge_count": len(deferred_edges),
        "same_room_only": bool(same_room_only),
        "topk_per_new_node": int(topk_per_new_node or 0),
        "max_pairs": int(max_pairs or 0),
    }
    return ordered_edges, dropped_edges, deferred_edges, stats
