"""Unit tests for aind_hcr_data_loader.pairwise_dataset."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from aind_hcr_data_loader.pairwise_dataset import create_pairwise_unmixing_dataset


HAS_ANNDATA = importlib.util.find_spec("anndata") is not None


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path)


def _write_round_configs(round_dir: Path) -> None:
    manifest = {
        "spot_channels": ["488"],
        "gene_dict": {"488": {"gene": "GFP"}},
    }
    (round_dir / "ds_config.json").write_text(json.dumps({"manifest": manifest}))
    (round_dir / "unmixing_config.json").write_text(json.dumps({}))


class TestPairwiseDatasetNewAggregatedLayout(unittest.TestCase):
    def test_filtered_default_and_all_spots_option(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            asset_path = Path(tmpdir) / "HCR_782149_pairwise-unmixing_2026-05-23_01-29-46"
            round_dir = asset_path / "782149_R1"
            round_dir.mkdir(parents=True)

            _write_round_configs(round_dir)

            _write_csv(
                asset_path
                / "all_cells_unmixed_filtered"
                / "unmixed_all_cells_filtered.csv",
                pd.DataFrame(
                    {"R1-488-GFP": [3]},
                    index=pd.Index([101], name="cell_id"),
                ),
            )
            _write_csv(
                asset_path
                / "all_cells_unmixed_all_spots"
                / "unmixed_all_cells_all_spots.csv",
                pd.DataFrame(
                    {"R1-488-GFP": [7]},
                    index=pd.Index([101], name="cell_id"),
                ),
            )
            _write_csv(
                asset_path
                / "inhibitory_cells_unmixed_filtered"
                / "unmixed_inhibitory_cells_filtered.csv",
                pd.DataFrame(
                    {"marker_score": [0.2]},
                    index=pd.Index([101], name="cell_id"),
                ),
            )
            _write_csv(
                asset_path
                / "inhibitory_cells_unmixed_all_spots"
                / "unmixed_inhibitory_cells_all_spots.csv",
                pd.DataFrame(
                    {"marker_score": [0.9]},
                    index=pd.Index([101], name="cell_id"),
                ),
            )

            dataset = create_pairwise_unmixing_dataset(
                mouse_id="782149",
                pairwise_asset_path=asset_path,
            )

            filtered_cxg = dataset.load_aggregated_cxg(unmixed=True)
            all_spots_cxg = dataset.load_aggregated_cxg(unmixed=True, all_spots=True)
            filtered_inhibitory = dataset.load_inhibitory_cells(unmixed=True)
            all_spots_inhibitory = dataset.load_inhibitory_cells(
                unmixed=True,
                all_spots=True,
            )

            self.assertEqual(filtered_cxg.loc[101, "R1-488-GFP"], 3)
            self.assertEqual(all_spots_cxg.loc[101, "R1-488-GFP"], 7)
            self.assertEqual(filtered_inhibitory.loc[101, "marker_score"], 0.2)
            self.assertEqual(all_spots_inhibitory.loc[101, "marker_score"], 0.9)

    @unittest.skipUnless(HAS_ANNDATA, "anndata is not installed")
    def test_load_aggregated_cxg_as_anndata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            asset_path = Path(tmpdir) / "HCR_782149_pairwise-unmixing_2026-05-23_01-29-46"
            round_dir = asset_path / "782149_R1"
            round_dir.mkdir(parents=True)

            _write_round_configs(round_dir)

            _write_csv(
                asset_path
                / "all_cells_unmixed_filtered"
                / "unmixed_all_cells_filtered.csv",
                pd.DataFrame(
                    {
                        "R1-488-GFP": [3],
                        "R2-514-Ndnf": [4],
                    },
                    index=pd.Index([101], name="cell_id"),
                ),
            )

            dataset = create_pairwise_unmixing_dataset(
                mouse_id="782149",
                pairwise_asset_path=asset_path,
            )

            adata = dataset.load_aggregated_cxg(unmixed=True, as_anndata=True)

            self.assertEqual(adata.shape, (1, 2))
            self.assertListEqual(adata.var_names.tolist(), ["GFP", "Ndnf"])
            self.assertListEqual(adata.obs_names.tolist(), ["101"])

    @unittest.skipUnless(HAS_ANNDATA, "anndata is not installed")
    def test_load_as_anndata_raises_on_duplicate_gene_names(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            asset_path = Path(tmpdir) / "HCR_782149_pairwise-unmixing_2026-05-23_01-29-46"
            round_dir = asset_path / "782149_R1"
            round_dir.mkdir(parents=True)

            _write_round_configs(round_dir)

            _write_csv(
                asset_path
                / "all_cells_unmixed_filtered"
                / "unmixed_all_cells_filtered.csv",
                pd.DataFrame(
                    {
                        "R1-488-GFP": [3],
                        "R2-514-GFP": [4],
                    },
                    index=pd.Index([101], name="cell_id"),
                ),
            )

            dataset = create_pairwise_unmixing_dataset(
                mouse_id="782149",
                pairwise_asset_path=asset_path,
            )

            with self.assertRaisesRegex(
                ValueError,
                "Duplicate gene names after splitting round/channel prefixes",
            ):
                dataset.load_aggregated_cxg(unmixed=True, as_anndata=True)


class TestPairwiseDatasetLegacyAggregatedFallback(unittest.TestCase):
    def test_legacy_single_output_is_used_for_both_scopes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            asset_path = Path(tmpdir) / "HCR_782149_pairwise-unmixing_2026-04-29_18-02-29"
            round_dir = asset_path / "782149_R1"
            round_dir.mkdir(parents=True)

            _write_round_configs(round_dir)

            _write_csv(
                asset_path / "unmixed_cell_by_gene_all_rounds.csv",
                pd.DataFrame(
                    {"R1-488-GFP": [5]},
                    index=pd.Index([101], name="cell_id"),
                ),
            )

            dataset = create_pairwise_unmixing_dataset(
                mouse_id="782149",
                pairwise_asset_path=asset_path,
            )

            filtered_cxg = dataset.load_aggregated_cxg(unmixed=True)
            all_spots_cxg = dataset.load_aggregated_cxg(unmixed=True, all_spots=True)

            self.assertEqual(filtered_cxg.loc[101, "R1-488-GFP"], 5)
            self.assertEqual(all_spots_cxg.loc[101, "R1-488-GFP"], 5)