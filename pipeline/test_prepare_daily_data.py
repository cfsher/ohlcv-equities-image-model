import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


PIPELINE_DIR = Path(__file__).resolve().parent
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

import prepare_daily_data as pdd


class PrepareDailyDataLabelModeTests(unittest.TestCase):
    def _frame(self) -> pd.DataFrame:
        index = pd.date_range("2024-01-01", periods=4, freq="D")
        return pd.DataFrame(
            {
                "open": [10.0, 11.0, 12.0, 10.0],
                "high": [11.0, 13.0, 14.0, 11.0],
                "low": [9.0, 9.0, 11.0, 9.0],
                "close": [10.0, 12.0, 13.0, 10.0],
                "volume": [100.0, 100.0, 100.0, 100.0],
            },
            index=index,
        )

    def test_resolve_effective_label_horizon_for_next_day_mode(self) -> None:
        self.assertEqual(
            pdd.resolve_effective_label_horizon(
                horizon=5,
                label_mode=pdd.LABEL_MODE_NEXT_DAY_CLOSE_RETURN,
            ),
            1,
        )

    def test_compute_labels_daily_default_mode_preserves_horizon_logic(self) -> None:
        labels = pdd.compute_labels_daily(
            self._frame(),
            horizon=2,
            atr_period=1,
            label_mode=pdd.LABEL_MODE_RANGE_ATR,
        )
        first = labels.iloc[0]
        self.assertAlmostEqual(float(first["mfe"]), 1.5, places=6)
        self.assertAlmostEqual(float(first["mae"]), 1.0, places=6)
        self.assertAlmostEqual(float(first["y_raw"]), 0.5, places=6)
        self.assertAlmostEqual(float(first["ret_atr"]), 1.5, places=6)
        self.assertAlmostEqual(float(first["avg_ret_atr"]), 1.25, places=6)
        self.assertAlmostEqual(float(first["log_avg_ret_atr"]), np.log(1.25) / 0.2, places=6)
        self.assertAlmostEqual(float(first["ret_pct"]), 0.3, places=6)
        self.assertTrue(np.isnan(labels.iloc[-2]["ret_pct"]))

    def test_compute_labels_daily_next_day_close_return_ignores_requested_horizon(self) -> None:
        labels = pdd.compute_labels_daily(
            self._frame(),
            horizon=5,
            atr_period=1,
            label_mode=pdd.LABEL_MODE_NEXT_DAY_CLOSE_RETURN,
        )
        first = labels.iloc[0]
        self.assertAlmostEqual(float(first["mfe"]), 1.5, places=6)
        self.assertAlmostEqual(float(first["mae"]), 0.5, places=6)
        self.assertAlmostEqual(float(first["y_raw"]), 0.2, places=6)
        self.assertAlmostEqual(float(first["ret_atr"]), 1.0, places=6)
        self.assertAlmostEqual(float(first["avg_ret_atr"]), 1.0, places=6)
        self.assertAlmostEqual(float(first["log_avg_ret_atr"]), np.log(1.2) / 0.2, places=6)
        self.assertAlmostEqual(float(first["ret_pct"]), 0.2, places=6)
        self.assertTrue(np.isfinite(labels.iloc[1]["ret_pct"]))
        self.assertTrue(np.isfinite(labels.iloc[2]["ret_pct"]))
        self.assertTrue(np.isnan(labels.iloc[-1]["ret_pct"]))


if __name__ == "__main__":
    unittest.main()
