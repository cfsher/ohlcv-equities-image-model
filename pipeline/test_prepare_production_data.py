import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock


PIPELINE_DIR = Path(__file__).resolve().parent
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

import prepare_production_data as ppd


class PrepareProductionDataRetryTests(unittest.TestCase):
    def test_retries_rate_limited_tickers_until_resolved(self) -> None:
        attempts = []
        responses = [
            {"saved": ["AAA"], "missing": ["BBB", "CCC"], "rate_limited": ["BBB"]},
            {"saved": [], "missing": ["BBB"], "rate_limited": ["BBB"]},
            {"saved": ["BBB"], "missing": [], "rate_limited": []},
        ]

        with TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            def fake_attempt(
                *,
                tickers,
                output_dir,
                start_date_inclusive,
                end_date_inclusive,
                chunk_size,
            ):
                del start_date_inclusive, end_date_inclusive, chunk_size
                attempts.append(list(tickers))
                response = responses[len(attempts) - 1]
                for ticker in response["saved"]:
                    (output_dir / f"{ticker}.csv").write_text(
                        "date,open,high,low,close,volume\n",
                        encoding="utf-8",
                    )
                return (
                    len(response["saved"]),
                    list(response["missing"]),
                    list(response["rate_limited"]),
                )

            with mock.patch.object(ppd, "fetch_ticker_history_attempt", side_effect=fake_attempt):
                with mock.patch.object(ppd.time, "sleep") as sleep_mock:
                    meta_df, failed_df = ppd.fetch_russell_ticker_history(
                        tickers=["AAA", "BBB", "CCC"],
                        output_dir=output_dir,
                        start_date="2024-01-01",
                        end_date="2024-01-10",
                        chunk_size=25,
                    )

            self.assertEqual(attempts, [["AAA", "BBB", "CCC"], ["BBB"], ["BBB"]])
            self.assertEqual(
                sleep_mock.call_args_list,
                [
                    mock.call(ppd.DEFAULT_RATE_LIMIT_RETRY_SLEEP_SECONDS),
                    mock.call(ppd.DEFAULT_RATE_LIMIT_RETRY_SLEEP_SECONDS),
                ],
            )
            self.assertEqual(
                meta_df.loc[meta_df["ticker"] == "BBB", "status"].iloc[0],
                "ok",
            )
            self.assertEqual(failed_df["ticker"].tolist(), ["CCC"])
            self.assertEqual(failed_df["failure_reason"].tolist(), ["missing_ohlc"])
            self.assertEqual(
                (output_dir / ppd.fetch_daily.RATE_LIMIT_FILENAME).read_text(encoding="utf-8"),
                "",
            )

    def test_stops_after_configured_rate_limit_retries(self) -> None:
        attempts = []

        with TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            def fake_attempt(
                *,
                tickers,
                output_dir,
                start_date_inclusive,
                end_date_inclusive,
                chunk_size,
            ):
                del output_dir, start_date_inclusive, end_date_inclusive, chunk_size
                attempts.append(list(tickers))
                return 0, list(tickers), list(tickers)

            with mock.patch.object(ppd, "fetch_ticker_history_attempt", side_effect=fake_attempt):
                with mock.patch.object(ppd.time, "sleep") as sleep_mock:
                    meta_df, failed_df = ppd.fetch_russell_ticker_history(
                        tickers=["BBB"],
                        output_dir=output_dir,
                        start_date="2024-01-01",
                        end_date="2024-01-10",
                        chunk_size=25,
                    )

            expected_attempts = [["BBB"]] * (1 + int(ppd.DEFAULT_RATE_LIMIT_RETRIES))
            self.assertEqual(attempts, expected_attempts)
            self.assertEqual(
                sleep_mock.call_args_list,
                [mock.call(ppd.DEFAULT_RATE_LIMIT_RETRY_SLEEP_SECONDS)]
                * int(ppd.DEFAULT_RATE_LIMIT_RETRIES),
            )
            self.assertEqual(meta_df["failure_reason"].tolist(), ["rate_limited"])
            self.assertEqual(failed_df["ticker"].tolist(), ["BBB"])
            self.assertEqual(
                (output_dir / ppd.fetch_daily.RATE_LIMIT_FILENAME).read_text(encoding="utf-8"),
                "BBB\n",
            )


if __name__ == "__main__":
    unittest.main()
