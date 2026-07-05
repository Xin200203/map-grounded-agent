"""Generation-2 tests for executor adoption bookkeeping helpers."""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from smoothnav.executor_adoption import (
    compute_adoption_transition,
    resolve_strategy_epoch_transition,
    should_allow_text_visible_temp_goal,
    should_suppress_stuck_override,
)


class ExecutorAdoptionGen2Tests(unittest.TestCase):
    def test_strategy_epoch_change_clears_stale_temp_goal(self):
        transition = resolve_strategy_epoch_transition(
            current_strategy_epoch=2,
            incoming_strategy_epoch=3,
            has_temp_goal=True,
            temp_goal_epoch=2,
        )

        self.assertEqual(transition["next_strategy_epoch"], 3)
        self.assertTrue(transition["stale_temp_goal_cleared"])

    def test_same_epoch_keeps_current_temp_goal(self):
        transition = resolve_strategy_epoch_transition(
            current_strategy_epoch=3,
            incoming_strategy_epoch=3,
            has_temp_goal=True,
            temp_goal_epoch=3,
        )

        self.assertEqual(transition["next_strategy_epoch"], 3)
        self.assertFalse(transition["stale_temp_goal_cleared"])

    def test_strategy_epoch_change_clears_stale_stuck_goal(self):
        transition = resolve_strategy_epoch_transition(
            current_strategy_epoch=2,
            incoming_strategy_epoch=3,
            has_temp_goal=False,
            temp_goal_epoch=None,
            has_stuck_goal=True,
        )

        self.assertEqual(transition["next_strategy_epoch"], 3)
        self.assertTrue(transition["stale_stuck_goal_cleared"])

    def test_adoption_transition_tracks_before_after_and_changed(self):
        first = compute_adoption_transition(
            None,
            source="global_goal",
            goal_summary=[12, 18],
            goal_epoch=4,
        )
        second = compute_adoption_transition(
            first["adopted_after"],
            source="global_goal",
            goal_summary=[12, 18],
            goal_epoch=4,
        )

        self.assertTrue(first["adopted_changed"])
        self.assertFalse(second["adopted_changed"])
        self.assertEqual(second["adopted_before"]["goal"], [12, 18])
        self.assertEqual(second["adopted_after"]["source"], "global_goal")

    def test_controller_stuck_replan_can_suppress_executor_override(self):
        self.assertTrue(
            should_suppress_stuck_override(
                been_stuck=True,
                suppress_stuck_override=True,
            )
        )
        self.assertFalse(
            should_suppress_stuck_override(
                been_stuck=False,
                suppress_stuck_override=True,
            )
        )

    def test_text_visible_temp_goal_is_disabled_for_text_goals(self):
        self.assertFalse(
            should_allow_text_visible_temp_goal(
                goal_type="text",
                current_target_region="unexplored north",
                goal_name="chair",
            )
        )
        self.assertFalse(
            should_allow_text_visible_temp_goal(
                goal_type="text",
                current_target_region="bedroom",
                goal_name="chair",
            )
        )
        self.assertFalse(
            should_allow_text_visible_temp_goal(
                goal_type="text",
                current_target_region="object: chair",
                goal_name="chair",
            )
        )
        self.assertFalse(
            should_allow_text_visible_temp_goal(
                goal_type="text",
                current_target_region="object: windows",
                goal_name="chair",
            )
        )
        self.assertTrue(
            should_allow_text_visible_temp_goal(
                goal_type="ins-image",
                current_target_region="unexplored north",
            )
        )


if __name__ == "__main__":
    unittest.main()
