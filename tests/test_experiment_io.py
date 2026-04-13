"""Phase 0 tests for run manifests and environment-based API config."""

import json
import os
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from smoothnav.experiment_io import resolve_api_config, setup_run_environment


class ResolveApiConfigTests(unittest.TestCase):
    @mock.patch.dict(os.environ, {}, clear=True)
    def test_missing_api_key_raises_clear_error(self):
        args = SimpleNamespace(
            api_key_env="TEST_SMOOTHNAV_API_KEY",
            base_url_env="TEST_SMOOTHNAV_BASE_URL",
            api_provider="openai",
            api_protocol="openai-responses",
        )

        with self.assertRaisesRegex(RuntimeError, "TEST_SMOOTHNAV_API_KEY"):
            resolve_api_config(args)

    @mock.patch.dict(
        os.environ,
        {
            "TEST_SMOOTHNAV_API_KEY": "top-secret",
            "TEST_SMOOTHNAV_BASE_URL": "https://example.invalid/v1/",
        },
        clear=True,
    )
    def test_reads_named_env_vars(self):
        args = SimpleNamespace(
            api_key_env="TEST_SMOOTHNAV_API_KEY",
            base_url_env="TEST_SMOOTHNAV_BASE_URL",
            api_provider="openai",
            api_protocol="openai-responses",
        )

        resolved = resolve_api_config(args)

        self.assertEqual(resolved.api_key, "top-secret")
        self.assertEqual(resolved.base_url, "https://example.invalid/v1/")
        self.assertEqual(resolved.api_provider, "openai")
        self.assertEqual(resolved.api_protocol, "openai-responses")

    @mock.patch.dict(
        os.environ,
        {
            "TEST_SMOOTHNAV_API_KEY": "top-secret",
            "TEST_SMOOTHNAV_BASE_URL": "https://example.invalid",
        },
        clear=True,
    )
    def test_defaults_to_anthropic_messages_when_unspecified(self):
        args = SimpleNamespace(
            api_key_env="TEST_SMOOTHNAV_API_KEY",
            base_url_env="TEST_SMOOTHNAV_BASE_URL",
        )

        resolved = resolve_api_config(args)

        self.assertEqual(resolved.api_provider, "anthropic")
        self.assertEqual(resolved.api_protocol, "anthropic-messages")

    @mock.patch.dict(
        os.environ,
        {
            "TEST_SMOOTHNAV_API_KEY": "top-secret",
            "TEST_SMOOTHNAV_BASE_URL": "https://example.invalid/v1/",
        },
        clear=True,
    )
    def test_provider_protocol_mismatch_raises(self):
        args = SimpleNamespace(
            api_key_env="TEST_SMOOTHNAV_API_KEY",
            base_url_env="TEST_SMOOTHNAV_BASE_URL",
            api_provider="anthropic",
            api_protocol="openai-responses",
        )

        with self.assertRaisesRegex(RuntimeError, "mismatch"):
            resolve_api_config(args)


class SetupRunEnvironmentTests(unittest.TestCase):
    def test_creates_isolated_run_bundle_and_redacts_secrets(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            args = SimpleNamespace(
                mode="baseline",
                goal_type="text",
                results_root=tmpdir,
                config_file="base_UniGoal/configs/config_habitat.yaml",
                num_eval=5,
                num_eval_episodes=5,
                llm_model="planner-model",
                llm_model_fast="monitor-model",
                vlm_model="vision-model",
                api_key_env="SMOOTHNAV_API_KEY",
                base_url_env="SMOOTHNAV_BASE_URL",
                api_provider="anthropic",
                api_protocol="anthropic-messages",
                api_key="super-secret-key",
                base_url="https://example.invalid/v1/",
                controller_profile="smoothnav-full",
                controller_enable_monitor=True,
                controller_monitor_policy="llm",
                controller_enable_prefetch=True,
                controller_replan_policy="event",
                controller_enable_stuck_replan=True,
                controller_fixed_plan_interval_steps=40,
                controller_prefetch_near_threshold=10.0,
            )

            with mock.patch(
                "smoothnav.experiment_io.get_repo_root",
                return_value=tmpdir,
            ), mock.patch(
                "smoothnav.experiment_io.get_git_hash",
                return_value="abc1234",
            ):
                configured = setup_run_environment(
                    args,
                    argv=["python", "smoothnav/main.py", "--mode", "baseline"],
                    prompt_versions={"planner": "v1", "monitor": "v1"},
                )

            self.assertTrue(os.path.isdir(configured.run_dir))
            self.assertTrue(os.path.isdir(configured.step_trace_dir))
            self.assertTrue(os.path.isdir(configured.planner_call_dir))
            self.assertTrue(os.path.isdir(configured.monitor_call_dir))

            with open(configured.manifest_path, "r", encoding="utf-8") as handle:
                manifest = json.load(handle)
            self.assertEqual(manifest["git_hash"], "abc1234")
            self.assertEqual(manifest["mode"], "baseline")
            self.assertEqual(manifest["goal_type"], "text")
            self.assertEqual(manifest["api_env"]["api_key_env"], "SMOOTHNAV_API_KEY")
            self.assertEqual(manifest["api_env"]["provider"], "anthropic")
            self.assertEqual(manifest["api_env"]["protocol"], "anthropic-messages")
            self.assertEqual(manifest["controller"]["profile"], "smoothnav-full")

            with open(configured.effective_config_path, "r", encoding="utf-8") as handle:
                effective_config = json.load(handle)
            self.assertEqual(effective_config["api_key"], "<redacted>")
            self.assertEqual(effective_config["base_url"], "https://example.invalid/v1/")


if __name__ == "__main__":
    unittest.main()
