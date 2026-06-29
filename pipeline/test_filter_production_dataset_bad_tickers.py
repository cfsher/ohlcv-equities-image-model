import csv
import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np


PIPELINE_DIR = Path(__file__).resolve().parent
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

import filter_production_dataset_bad_tickers as fpdbt


class FilterProductionDatasetBadTickersTests(unittest.TestCase):
    def _write_ticker_csv(self, path: Path, rows: list[tuple[str, float, float, float, float, float]]) -> None:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["date", "open", "high", "low", "close", "volume"])
            for row in rows:
                writer.writerow(row)

    def test_filters_bad_tickers_and_rewrites_artifacts(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            tickers_dir = root / "tickers"
            input_dir = root / "production_datasets" / "sample_ds"
            output_dir = root / "production_datasets" / "sample_ds_filtered"
            shards_dir = input_dir / "shards"
            ticker_view_dir = input_dir / "_ticker_universe_view"
            tickers_dir.mkdir(parents=True)
            shards_dir.mkdir(parents=True)
            ticker_view_dir.mkdir(parents=True)

            good_rows = [
                ("2024-01-01", 10.0, 11.0, 9.0, 10.5, 100.0),
                ("2024-01-02", 10.5, 11.2, 10.0, 10.8, 110.0),
                ("2024-01-03", 10.8, 11.1, 10.4, 10.6, 120.0),
                ("2024-01-04", 10.6, 11.3, 10.2, 11.0, 130.0),
                ("2024-01-05", 11.0, 11.5, 10.8, 11.2, 140.0),
                ("2024-01-06", 11.2, 11.8, 11.0, 11.4, 150.0),
                ("2024-01-07", 11.4, 11.9, 11.1, 11.6, 160.0),
                ("2024-01-08", 11.6, 12.0, 11.3, 11.9, 170.0),
                ("2024-01-09", 11.9, 12.2, 11.5, 12.0, 180.0),
                ("2024-01-10", 12.0, 12.4, 11.8, 12.2, 190.0),
            ]
            bad_rows = [
                ("2024-01-01", 20.0, 20.0, 20.0, 20.0, 100.0),
                ("2024-01-02", 20.0, 21.0, 19.5, 20.5, 100.0),
                ("2024-01-03", 20.5, 21.1, 20.1, 20.7, 100.0),
                ("2024-01-04", 20.7, 21.2, 20.4, 20.9, 100.0),
                ("2024-01-05", 20.9, 21.4, 20.6, 21.0, 100.0),
                ("2024-01-06", 21.0, 21.5, 20.8, 21.1, 100.0),
                ("2024-01-07", 21.1, 21.6, 20.9, 21.3, 100.0),
                ("2024-01-08", 21.3, 21.7, 21.0, 21.4, 100.0),
                ("2024-01-09", 21.4, 21.8, 21.2, 21.6, 100.0),
                ("2024-01-10", 21.6, 22.0, 21.4, 21.8, 0.0),
            ]
            self._write_ticker_csv(tickers_dir / "AAA.csv", good_rows)
            self._write_ticker_csv(tickers_dir / "BBB.csv", bad_rows)
            self._write_ticker_csv(tickers_dir / "CCC.csv", good_rows)

            for ticker in ("AAA", "BBB", "CCC"):
                (ticker_view_dir / f"{ticker}.csv").write_text("date,open,high,low,close,volume\n", encoding="utf-8")

            tickers = np.array(["AAA", "BBB", "CCC"], dtype=object)
            np.save(input_dir / "tickers.npy", tickers)

            np.savez(
                input_dir / "production_dataset.npz",
                X=np.arange(12, dtype=np.float64).reshape(4, 3),
                y_raw=np.arange(8, dtype=np.float64).reshape(4, 2),
                timestamps=np.array(["2024-02-01", "2024-02-02", "2024-02-03", "2024-02-04"], dtype=object),
                ticker_ids=np.array([0, 1, 2, 1], dtype=np.int32),
                tickers=tickers,
                feature_cols=np.array(["f1", "f2", "f3"], dtype=object),
                label_cols=np.array(["y1", "y2"], dtype=object),
            )

            np.savez(
                shards_dir / "shard_000000.npz",
                X_img=np.arange(48, dtype=np.uint8).reshape(4, 1, 3, 4),
                y_raw=np.arange(8, dtype=np.float64).reshape(4, 2),
                timestamps=np.array(["2024-02-01", "2024-02-02", "2024-02-03", "2024-02-04"]),
                sample_indices=np.array([10, 11, 12, 13], dtype=np.int64),
                ticker_ids=np.array([0, 1, 2, 1], dtype=np.int32),
            )

            manifest = {
                "created_utc": "2026-01-01T00:00:00+00:00",
                "subset_start": 0,
                "subset_end": 4,
                "subset_count": 4,
                "tickers_count": 3,
                "decomposition_scales": 1,
                "image_height": 3,
                "image_width": 4,
                "day_width": 3,
                "decomposition_windows": 1,
                "image_width_per_scale": [4],
                "shard_size": 512,
                "shard_save_mode": "uncompressed",
                "num_shards": 1,
                "vix_image_retained_count": 4,
                "shards": [
                    {
                        "file": "shards/shard_000000.npz",
                        "sample_start": 0,
                        "sample_end": 4,
                        "count": 4,
                        "relative_start": 0,
                        "relative_end": 4,
                    }
                ],
            }
            (input_dir / "manifest.json").write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

            summary = fpdbt.run_filter(
                input_dir=input_dir,
                output_dir=output_dir,
                tickers_dir=tickers_dir,
                flat_threshold_pct=5.0,
                zero_volume_threshold_pct=5.0,
                progress_every=0,
                copy_universe_view=True,
            )

            self.assertEqual(summary["removed_tickers_count"], 1)
            self.assertEqual(summary["removed_tickers_sample"], ["BBB"])

            output_tickers = np.load(output_dir / "tickers.npy", allow_pickle=True)
            self.assertEqual(output_tickers.tolist(), ["AAA", "CCC"])

            with np.load(output_dir / "production_dataset.npz", allow_pickle=True) as data:
                self.assertEqual(data["ticker_ids"].tolist(), [0, 1])
                self.assertEqual(data["timestamps"].tolist(), ["2024-02-01", "2024-02-03"])
                self.assertEqual(data["tickers"].tolist(), ["AAA", "CCC"])

            manifest_out = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest_out["subset_count"], 2)
            self.assertEqual(manifest_out["subset_end"], 2)
            self.assertEqual(manifest_out["tickers_count"], 2)
            self.assertEqual(manifest_out["num_shards"], 1)
            self.assertEqual(manifest_out["vix_image_retained_count"], 2)
            self.assertEqual(manifest_out["shards"][0]["count"], 2)
            self.assertEqual(manifest_out["bad_ticker_filter"]["removed_tickers_count"], 1)

            with np.load(output_dir / "shards" / "shard_000000.npz", allow_pickle=True) as shard:
                self.assertEqual(shard["ticker_ids"].tolist(), [0, 1])
                self.assertEqual(shard["sample_indices"].tolist(), [10, 12])
                self.assertEqual(shard["timestamps"].tolist(), ["2024-02-01", "2024-02-03"])

            self.assertTrue((output_dir / "_ticker_universe_view" / "AAA.csv").is_file())
            self.assertTrue((output_dir / "_ticker_universe_view" / "CCC.csv").is_file())
            self.assertFalse((output_dir / "_ticker_universe_view" / "BBB.csv").exists())

            removed_csv = (output_dir / fpdbt.REMOVED_TICKERS_CSV_NAME).read_text(encoding="utf-8")
            self.assertIn("BBB", removed_csv)
            self.assertNotIn("AAA", removed_csv)


if __name__ == "__main__":
    unittest.main()
