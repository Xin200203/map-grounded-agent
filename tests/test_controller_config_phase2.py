"""Phase 2 tests for controller profiles and overrides."""

import os
import sys
import unittest
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from smoothnav.controller_config import resolve_controller_config


class ControllerConfigPhase2Tests(unittest.TestCase):
    def test_infers_baseline_periodic_profile_from_mode(self):
        args = SimpleNamespace(
            mode="baseline",
            num_local_steps=40,
            controller_profile=None,
            controller_enable_monitor=None,
            controller_monitor_policy=None,
            controller_enable_prefetch=None,
            controller_replan_policy=None,
            controller_enable_stuck_replan=None,
            controller_fixed_plan_interval_steps=None,
            controller_prefetch_near_threshold=None,
        )

        resolved = resolve_controller_config(args)

        self.assertEqual(resolved.controller_profile, "baseline-periodic")
        self.assertFalse(resolved.controller_enable_monitor)
        self.assertEqual(resolved.controller_replan_policy, "baseline_explore")

    def test_rules_only_profile_selects_rules_monitor(self):
        args = SimpleNamespace(
            mode="smoothnav",
            num_local_steps=40,
            controller_profile="smoothnav-rules-only",
            controller_enable_monitor=None,
            controller_monitor_policy=None,
            controller_enable_prefetch=None,
            controller_replan_policy=None,
            controller_enable_stuck_replan=None,
            controller_fixed_plan_interval_steps=None,
            controller_prefetch_near_threshold=None,
        )

        resolved = resolve_controller_config(args)

        self.assertTrue(resolved.controller_enable_monitor)
        self.assertEqual(resolved.controller_monitor_policy, "rules")
        self.assertEqual(resolved.controller_replan_policy, "event")

    def test_explicit_override_wins_over_profile_default(self):
        args = SimpleNamespace(
            mode="smoothnav",
            num_local_steps=40,
            controller_profile="smoothnav-full",
            controller_enable_monitor=None,
            controller_monitor_policy="off",
            controller_enable_prefetch=False,
            controller_replan_policy=None,
            controller_enable_stuck_replan=None,
            controller_fixed_plan_interval_steps=17,
            controller_prefetch_near_threshold=6.0,
        )

        resolved = resolve_controller_config(args)

        self.assertFalse(resolved.controller_enable_monitor)
        self.assertEqual(resolved.controller_monitor_policy, "off")
        self.assertFalse(resolved.controller_enable_prefetch)
        self.assertEqual(resolved.controller_fixed_plan_interval_steps, 17)
        self.assertEqual(resolved.controller_prefetch_near_threshold, 6.0)


if __name__ == "__main__":
    unittest.main()
