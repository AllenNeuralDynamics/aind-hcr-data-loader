"""Unit tests for aind_hcr_data_loader.codeocean_utils."""

import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from aind_hcr_data_loader.codeocean_utils import (
    AttachResult,
    MouseRecord,
    _find_data_asset_by_name,
    attach_mouse_record_to_capsule,
    attach_mouse_record_to_pipeline,
    create_client_from_env,
    get_capsule_id_from_env,
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

SAMPLE_RECORD_DICT = {
    "schema_version": "1.0.0",
    "mouse_id": "782149",
    "rounds": {
        "R1": "HCR_782149_2025-11-05_13-00-00_processed_2025-11-10_20-37-29",
        "R2": "HCR_782149_2025-11-12_13-00-00_processed_2025-11-13_22-04-32",
    },
    "mouse_metadata": {"species": "mouse", "nickname": "Lemongrass"},
    "derived_assets": {
        "roi_shape_metrics": "HCR_782149_2025-11-05_13-00-00_roi-shape-metrics",
        "cell_typing": "HCR_782149_cell-typing_2026-03-09_12-00-00",
    },
    "notes": ["Some note."],
}


def _make_mock_asset(asset_id: str, name: str) -> MagicMock:
    """Return a mock DataAsset with the given id and name."""
    asset = MagicMock()
    asset.id = asset_id
    asset.name = name
    return asset


def _make_mock_attach_result(asset_id: str) -> MagicMock:
    """Return a mock DataAssetAttachResults."""
    r = MagicMock()
    r.id = asset_id
    r.ready = True
    return r


def _make_client(assets_by_name: dict) -> MagicMock:
    """
    Build a mock CodeOcean client.

    ``assets_by_name`` maps asset name → mock DataAsset (or None to simulate
    a missing asset).  The search iterator yields assets whose name is in the
    dict and is not None.
    """
    client = MagicMock()

    def _search_iter(search_params):
        # Yield all non-None mock assets so _find_data_asset_by_name can pick
        # the exact-name match.
        for name, asset in assets_by_name.items():
            if asset is not None:
                yield asset

    client.data_assets.search_data_assets_iterator.side_effect = _search_iter

    def _attach_capsule(capsule_id, attach_params):
        return [_make_mock_attach_result(p.id) for p in attach_params]

    client.capsules.attach_data_assets.side_effect = _attach_capsule

    def _attach_pipeline(pipeline_id, attach_params):
        return [_make_mock_attach_result(p.id) for p in attach_params]

    client.pipelines.attach_data_assets.side_effect = _attach_pipeline

    return client


# ---------------------------------------------------------------------------
# create_client_from_env tests
# ---------------------------------------------------------------------------


class TestCreateClientFromEnv(unittest.TestCase):
    def test_returns_client_when_token_set(self):
        with patch.dict(os.environ, {"API_TOKEN": "tok-123"}):
            client = create_client_from_env()
            self.assertIsNotNone(client)

    def test_raises_when_token_missing(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("API_TOKEN", None)
            with self.assertRaises(EnvironmentError) as ctx:
                create_client_from_env()
        self.assertIn("API_TOKEN", str(ctx.exception))

    def test_custom_token_env_var_name(self):
        with patch.dict(os.environ, {"MY_TOKEN": "tok-456"}):
            client = create_client_from_env(token_env="MY_TOKEN")
            self.assertIsNotNone(client)


# ---------------------------------------------------------------------------
# get_capsule_id_from_env tests
# ---------------------------------------------------------------------------


class TestGetCapsuleIdFromEnv(unittest.TestCase):
    def test_returns_capsule_id_when_set(self):
        with patch.dict(os.environ, {"CO_CAPSULE_ID": "caps-uuid-abc"}):
            self.assertEqual(get_capsule_id_from_env(), "caps-uuid-abc")

    def test_raises_when_missing(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("CO_CAPSULE_ID", None)
            with self.assertRaises(EnvironmentError) as ctx:
                get_capsule_id_from_env()
        self.assertIn("CO_CAPSULE_ID", str(ctx.exception))

    def test_error_message_mentions_capsule_runtime(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("CO_CAPSULE_ID", None)
            with self.assertRaises(EnvironmentError) as ctx:
                get_capsule_id_from_env()
        self.assertIn("capsule", str(ctx.exception).lower())

    def test_custom_env_var_name(self):
        with patch.dict(os.environ, {"MY_CAPSULE": "caps-uuid-xyz"}):
            self.assertEqual(
                get_capsule_id_from_env(capsule_id_env="MY_CAPSULE"), "caps-uuid-xyz"
            )


# ---------------------------------------------------------------------------
# MouseRecord tests
# ---------------------------------------------------------------------------


class TestMouseRecordFromDict(unittest.TestCase):
    """Tests for MouseRecord.from_dict."""

    def test_basic_fields(self):
        record = MouseRecord.from_dict(SAMPLE_RECORD_DICT)
        self.assertEqual(record.mouse_id, "782149")
        self.assertEqual(record.schema_version, "1.0.0")

    def test_rounds_populated(self):
        record = MouseRecord.from_dict(SAMPLE_RECORD_DICT)
        self.assertEqual(len(record.rounds), 2)
        self.assertIn("R1", record.rounds)
        self.assertEqual(
            record.rounds["R1"],
            "HCR_782149_2025-11-05_13-00-00_processed_2025-11-10_20-37-29",
        )

    def test_derived_assets_populated(self):
        record = MouseRecord.from_dict(SAMPLE_RECORD_DICT)
        self.assertEqual(len(record.derived_assets), 2)
        self.assertIn("cell_typing", record.derived_assets)

    def test_missing_rounds_defaults_to_empty(self):
        record = MouseRecord.from_dict({"mouse_id": "000", "schema_version": "1.0.0"})
        self.assertEqual(record.rounds, {})

    def test_missing_derived_defaults_to_empty(self):
        record = MouseRecord.from_dict({"mouse_id": "000", "schema_version": "1.0.0"})
        self.assertEqual(record.derived_assets, {})

    def test_missing_schema_version_defaults(self):
        record = MouseRecord.from_dict({"mouse_id": "000"})
        self.assertEqual(record.schema_version, "1.0.0")

    def test_extra_fields_ignored(self):
        """Fields not in the dataclass (e.g. mouse_metadata, notes) are silently ignored."""
        record = MouseRecord.from_dict(SAMPLE_RECORD_DICT)
        self.assertFalse(hasattr(record, "notes"))
        self.assertFalse(hasattr(record, "mouse_metadata"))


class TestMouseRecordFromJsonFile(unittest.TestCase):
    """Tests for MouseRecord.from_json_file."""

    def test_loads_file(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as fh:
            json.dump(SAMPLE_RECORD_DICT, fh)
            tmp_path = fh.name

        record = MouseRecord.from_json_file(tmp_path)
        self.assertEqual(record.mouse_id, "782149")
        self.assertEqual(len(record.rounds), 2)

    def test_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            MouseRecord.from_json_file("/nonexistent/path/record.json")


# ---------------------------------------------------------------------------
# AttachResult tests
# ---------------------------------------------------------------------------


class TestAttachResult(unittest.TestCase):
    def test_success_true_when_asset_found_no_error(self):
        asset = _make_mock_asset("abc-123", "some-asset")
        result = AttachResult(label="rounds.R1", asset_name="some-asset", data_asset=asset)
        self.assertTrue(result.success)

    def test_success_false_when_no_asset(self):
        result = AttachResult(label="rounds.R1", asset_name="missing-asset")
        self.assertFalse(result.success)

    def test_success_false_when_error_set(self):
        asset = _make_mock_asset("abc-123", "some-asset")
        result = AttachResult(
            label="rounds.R1",
            asset_name="some-asset",
            data_asset=asset,
            error="Attach failed: HTTP 500",
        )
        self.assertFalse(result.success)


# ---------------------------------------------------------------------------
# _find_data_asset_by_name tests
# ---------------------------------------------------------------------------


class TestFindDataAssetByName(unittest.TestCase):
    def test_returns_matching_asset(self):
        asset = _make_mock_asset("id-1", "my-dataset")
        client = _make_client({"my-dataset": asset})
        found = _find_data_asset_by_name(client, "my-dataset")
        self.assertIs(found, asset)

    def test_returns_none_when_not_found(self):
        client = _make_client({})
        found = _find_data_asset_by_name(client, "does-not-exist")
        self.assertIsNone(found)

    def test_does_not_return_partial_name_match(self):
        """An asset named 'my-dataset-extra' should NOT match 'my-dataset'."""
        asset = _make_mock_asset("id-1", "my-dataset-extra")
        client = _make_client({"my-dataset-extra": asset})
        found = _find_data_asset_by_name(client, "my-dataset")
        self.assertIsNone(found)

    def test_search_called_with_correct_query(self):
        client = _make_client({})
        _find_data_asset_by_name(client, "exact-name")
        client.data_assets.search_data_assets_iterator.assert_called_once()
        call_args = client.data_assets.search_data_assets_iterator.call_args
        search_params = call_args[0][0]
        self.assertIn("exact-name", search_params.query)


# ---------------------------------------------------------------------------
# attach_mouse_record_to_capsule tests
# ---------------------------------------------------------------------------


class TestAttachMouseRecordToCapsule(unittest.TestCase):
    def setUp(self):
        self.record = MouseRecord.from_dict(SAMPLE_RECORD_DICT)
        # Build assets for each name in the record
        all_names = list(self.record.rounds.values()) + list(
            self.record.derived_assets.values()
        )
        self.assets = {
            name: _make_mock_asset(f"id-{i}", name)
            for i, name in enumerate(all_names)
        }
        self.client = _make_client(self.assets)
        self.capsule_id = "capsule-uuid-001"

    def test_returns_one_result_per_entry(self):
        results = attach_mouse_record_to_capsule(
            self.record, capsule_id=self.capsule_id, client=self.client
        )
        expected = len(self.record.rounds) + len(self.record.derived_assets)
        self.assertEqual(len(results), expected)

    def test_all_results_succeed(self):
        results = attach_mouse_record_to_capsule(
            self.record, capsule_id=self.capsule_id, client=self.client
        )
        for r in results:
            self.assertTrue(r.success, msg=f"{r.label} failed: {r.error}")

    def test_labels_have_correct_prefixes(self):
        results = attach_mouse_record_to_capsule(
            self.record, capsule_id=self.capsule_id, client=self.client
        )
        labels = [r.label for r in results]
        round_labels = [l for l in labels if l.startswith("rounds.")]
        derived_labels = [l for l in labels if l.startswith("derived_assets.")]
        self.assertEqual(len(round_labels), len(self.record.rounds))
        self.assertEqual(len(derived_labels), len(self.record.derived_assets))

    def test_attach_called_once_per_asset(self):
        attach_mouse_record_to_capsule(
            self.record, capsule_id=self.capsule_id, client=self.client
        )
        expected_calls = len(self.record.rounds) + len(self.record.derived_assets)
        self.assertEqual(
            self.client.capsules.attach_data_assets.call_count, expected_calls
        )

    def test_attach_called_with_capsule_id(self):
        attach_mouse_record_to_capsule(
            self.record, capsule_id=self.capsule_id, client=self.client
        )
        for c in self.client.capsules.attach_data_assets.call_args_list:
            self.assertEqual(c.kwargs["capsule_id"], self.capsule_id)

    def test_missing_asset_does_not_raise(self):
        missing_name = list(self.record.rounds.values())[0]
        assets_with_gap = {k: v for k, v in self.assets.items() if k != missing_name}
        client = _make_client(assets_with_gap)

        results = attach_mouse_record_to_capsule(
            self.record, capsule_id=self.capsule_id, client=client
        )
        missing = [r for r in results if r.asset_name == missing_name]
        self.assertEqual(len(missing), 1)
        self.assertFalse(missing[0].success)
        self.assertIn("not found", missing[0].error)

    def test_missing_asset_others_still_attached(self):
        missing_name = list(self.record.rounds.values())[0]
        assets_with_gap = {k: v for k, v in self.assets.items() if k != missing_name}
        client = _make_client(assets_with_gap)

        results = attach_mouse_record_to_capsule(
            self.record, capsule_id=self.capsule_id, client=client
        )
        successful = [r for r in results if r.success]
        total = len(self.record.rounds) + len(self.record.derived_assets)
        self.assertEqual(len(successful), total - 1)

    def test_include_rounds_false(self):
        results = attach_mouse_record_to_capsule(
            self.record, capsule_id=self.capsule_id, client=self.client,
            include_rounds=False,
        )
        round_results = [r for r in results if r.label.startswith("rounds.")]
        self.assertEqual(len(round_results), 0)
        derived_results = [r for r in results if r.label.startswith("derived_assets.")]
        self.assertEqual(len(derived_results), len(self.record.derived_assets))

    def test_include_derived_false(self):
        results = attach_mouse_record_to_capsule(
            self.record, capsule_id=self.capsule_id, client=self.client,
            include_derived=False,
        )
        derived_results = [r for r in results if r.label.startswith("derived_assets.")]
        self.assertEqual(len(derived_results), 0)
        round_results = [r for r in results if r.label.startswith("rounds.")]
        self.assertEqual(len(round_results), len(self.record.rounds))

    def test_dry_run_skips_attach(self):
        results = attach_mouse_record_to_capsule(
            self.record, capsule_id=self.capsule_id, client=self.client, dry_run=True
        )
        self.client.capsules.attach_data_assets.assert_not_called()
        for r in results:
            self.assertIsNotNone(r.data_asset)
            self.assertIsNone(r.error)

    def test_mount_prefix_applied(self):
        attach_mouse_record_to_capsule(
            self.record, capsule_id=self.capsule_id, client=self.client,
            mount_prefix="inputs",
        )
        for c in self.client.capsules.attach_data_assets.call_args_list:
            params = c.kwargs["attach_params"]
            for p in params:
                self.assertTrue(
                    p.mount.startswith("inputs/"),
                    msg=f"Expected mount to start with 'inputs/', got '{p.mount}'",
                )

    def test_search_error_recorded_not_raised(self):
        self.client.data_assets.search_data_assets_iterator.side_effect = RuntimeError(
            "network error"
        )
        results = attach_mouse_record_to_capsule(
            self.record, capsule_id=self.capsule_id, client=self.client
        )
        for r in results:
            self.assertFalse(r.success)
            self.assertIn("Search failed", r.error)

    def test_attach_api_error_recorded_not_raised(self):
        self.client.capsules.attach_data_assets.side_effect = RuntimeError("API error")
        results = attach_mouse_record_to_capsule(
            self.record, capsule_id=self.capsule_id, client=self.client
        )
        for r in results:
            self.assertFalse(r.success)
            self.assertIn("Attach failed", r.error)

    def test_empty_record_returns_empty_list(self):
        empty_record = MouseRecord(mouse_id="000")
        results = attach_mouse_record_to_capsule(
            empty_record, capsule_id=self.capsule_id, client=self.client
        )
        self.assertEqual(results, [])

    def test_auto_client_from_env(self):
        """When client=None, create_client_from_env should be called."""
        with patch(
            "aind_hcr_data_loader.codeocean_utils.create_client_from_env",
            return_value=self.client,
        ) as mock_factory:
            attach_mouse_record_to_capsule(
                self.record, capsule_id=self.capsule_id
            )
            mock_factory.assert_called_once()

    def test_auto_capsule_id_from_env(self):
        """When capsule_id=None, get_capsule_id_from_env should be called."""
        with patch(
            "aind_hcr_data_loader.codeocean_utils.get_capsule_id_from_env",
            return_value=self.capsule_id,
        ) as mock_caps:
            attach_mouse_record_to_capsule(self.record, client=self.client)
            mock_caps.assert_called_once()

    def test_fully_automatic_from_env(self):
        """Both client and capsule_id resolved from env when neither is passed."""
        with patch(
            "aind_hcr_data_loader.codeocean_utils.create_client_from_env",
            return_value=self.client,
        ) as mock_client_factory, patch(
            "aind_hcr_data_loader.codeocean_utils.get_capsule_id_from_env",
            return_value=self.capsule_id,
        ) as mock_caps_factory:
            attach_mouse_record_to_capsule(self.record)
            mock_client_factory.assert_called_once()
            mock_caps_factory.assert_called_once()


# ---------------------------------------------------------------------------
# attach_mouse_record_to_pipeline tests
# ---------------------------------------------------------------------------


class TestAttachMouseRecordToPipeline(unittest.TestCase):
    def setUp(self):
        self.record = MouseRecord.from_dict(SAMPLE_RECORD_DICT)
        all_names = list(self.record.rounds.values()) + list(
            self.record.derived_assets.values()
        )
        self.assets = {
            name: _make_mock_asset(f"pid-{i}", name)
            for i, name in enumerate(all_names)
        }
        self.client = _make_client(self.assets)
        self.pipeline_id = "pipeline-uuid-001"

    def test_returns_one_result_per_entry(self):
        results = attach_mouse_record_to_pipeline(
            self.record, pipeline_id=self.pipeline_id, client=self.client
        )
        expected = len(self.record.rounds) + len(self.record.derived_assets)
        self.assertEqual(len(results), expected)

    def test_all_results_succeed(self):
        results = attach_mouse_record_to_pipeline(
            self.record, pipeline_id=self.pipeline_id, client=self.client
        )
        for r in results:
            self.assertTrue(r.success, msg=f"{r.label} failed: {r.error}")

    def test_uses_pipelines_not_capsules(self):
        attach_mouse_record_to_pipeline(
            self.record, pipeline_id=self.pipeline_id, client=self.client
        )
        self.client.pipelines.attach_data_assets.assert_called()
        self.client.capsules.attach_data_assets.assert_not_called()

    def test_attach_called_with_pipeline_id(self):
        attach_mouse_record_to_pipeline(
            self.record, pipeline_id=self.pipeline_id, client=self.client
        )
        for c in self.client.pipelines.attach_data_assets.call_args_list:
            self.assertEqual(c.kwargs["pipeline_id"], self.pipeline_id)

    def test_dry_run_skips_attach(self):
        results = attach_mouse_record_to_pipeline(
            self.record, pipeline_id=self.pipeline_id, client=self.client, dry_run=True
        )
        self.client.pipelines.attach_data_assets.assert_not_called()
        for r in results:
            self.assertIsNotNone(r.data_asset)

    def test_missing_asset_error_not_raised(self):
        missing_name = list(self.record.derived_assets.values())[0]
        assets_with_gap = {k: v for k, v in self.assets.items() if k != missing_name}
        client = _make_client(assets_with_gap)

        results = attach_mouse_record_to_pipeline(
            self.record, pipeline_id=self.pipeline_id, client=client
        )
        missing = [r for r in results if r.asset_name == missing_name]
        self.assertEqual(len(missing), 1)
        self.assertFalse(missing[0].success)

    def test_auto_client_from_env(self):
        """When client=None, create_client_from_env should be called."""
        with patch(
            "aind_hcr_data_loader.codeocean_utils.create_client_from_env",
            return_value=self.client,
        ) as mock_factory:
            attach_mouse_record_to_pipeline(
                self.record, pipeline_id=self.pipeline_id
            )
            mock_factory.assert_called_once()


if __name__ == "__main__":
    unittest.main()
