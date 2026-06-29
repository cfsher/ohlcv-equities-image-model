import sys
import unittest
from datetime import date
from pathlib import Path
from unittest import mock

import pandas as pd


PIPELINE_DIR = Path(__file__).resolve().parent
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

import fetch_tickers as ft


class FetchTickersDownloadTests(unittest.TestCase):
    def test_download_chunk_requests_adjusted_ohlc(self) -> None:
        captured: dict[str, object] = {}

        def fake_download(*, tickers, start, end, interval, group_by, auto_adjust, actions, threads, progress):
            captured.update(
                {
                    "tickers": tickers,
                    "start": start,
                    "end": end,
                    "interval": interval,
                    "group_by": group_by,
                    "auto_adjust": auto_adjust,
                    "actions": actions,
                    "threads": threads,
                    "progress": progress,
                }
            )
            return pd.DataFrame()

        with mock.patch.object(ft.yf, "download", side_effect=fake_download):
            frame, rate_limited = ft._download_chunk(
                tickers=["AAA", "BBB"],
                start_date=date(2024, 1, 2),
                end_date_exclusive=date(2024, 1, 10),
            )

        self.assertTrue(frame.empty)
        self.assertEqual(rate_limited, [])
        self.assertEqual(captured["tickers"], ["AAA", "BBB"])
        self.assertEqual(captured["start"], "2024-01-02")
        self.assertEqual(captured["end"], "2024-01-10")
        self.assertEqual(captured["interval"], "1d")
        self.assertEqual(captured["group_by"], "ticker")
        self.assertTrue(bool(captured["auto_adjust"]))
        self.assertFalse(bool(captured["actions"]))
        self.assertTrue(bool(captured["threads"]))
        self.assertTrue(bool(captured["progress"]))


if __name__ == "__main__":
    unittest.main()
