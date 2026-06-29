#!/usr/bin/env python3
"""Fetch daily adjusted OHLCV candles and resolve yearly additive ticker universes."""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence, Tuple
from urllib.request import Request, urlopen

import pandas as pd
import yfinance as yf

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - optional dependency
    tqdm = None

try:
    import yfinance.shared as yf_shared
except Exception:  # pragma: no cover - yfinance internals changed/unavailable
    yf_shared = None

SP500_WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
SP500_CSV_FALLBACK_URL = (
    "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"
)
NASDAQ_LISTED_URL_TEMPLATES = (
    "ftp://ftp.nasdaqtrader.com/SymbolDirectory/nasdaqlisted.txt",
    "https://ftp.nasdaqtrader.com/SymbolDirectory/nasdaqlisted.txt?date={asof}",
    "https://www.nasdaqtrader.com/SymbolDirectory/nasdaqlisted.txt?date={asof}",
    "https://ftp.nasdaqtrader.com/SymbolDirectory/nasdaqlisted.txt",
    "https://www.nasdaqtrader.com/SymbolDirectory/nasdaqlisted.txt",
)
OTHERLISTED_URL_TEMPLATES = (
    "ftp://ftp.nasdaqtrader.com/SymbolDirectory/otherlisted.txt",
    "https://ftp.nasdaqtrader.com/SymbolDirectory/otherlisted.txt?date={asof}",
    "https://www.nasdaqtrader.com/SymbolDirectory/otherlisted.txt?date={asof}",
    "https://ftp.nasdaqtrader.com/SymbolDirectory/otherlisted.txt",
    "https://www.nasdaqtrader.com/SymbolDirectory/otherlisted.txt",
)
NASDAQ_SCREENER_API_URLS = (
    "https://api.nasdaq.com/api/screener/stocks?tableonly=true&download=true&exchange={exchange}&limit=10000&offset=0",
    "https://api.nasdaq.com/api/screener/stocks?tableonly=true&exchange={exchange}&limit=10000&offset=0",
)
RUSSELL_3000_HOLDINGS_URL_TEMPLATES = (
    (
        "https://www.ishares.com/us/products/239714/"
        "ishares-russell-3000-etf/1467271812596.ajax"
        "?fileType=csv&fileName=IWV_holdings&dataType=fund&asOfDate={asof}"
    ),
    (
        "https://www.ishares.com/us/products/239714/"
        "ishares-russell-3000-etf/1467271812596.ajax"
        "?fileType=csv&fileName=IWV_holdings&asOfDate={asof}"
    ),
)
OUT_DIR = "tickers"
CHUNK_SIZE = 200
REQUIRED_COLUMNS = ("open", "high", "low", "close", "volume")
HTTP_TIMEOUT_SECONDS = 30
START_DATE = "2000-01-01"
HTTP_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    )
}
TICKER_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9\.-]*")
RESOLVED_SP500_FILENAME = "_sp500_tickers.txt"
RESOLVED_NYSE_FILENAME = "_nyse_tickers.txt"
RESOLVED_NASDAQ_FILENAME = "_nasdaq_tickers.txt"
RESOLVED_RUSSELL3000_FILENAME = "_russell_3000_tickers.txt"
RATE_LIMIT_FILENAME = "rate_limit_tickers.txt"
MAX_RUSSELL_ASOF_OFFSET_DAYS = 35
MAX_NASDAQ_TRADER_ASOF_OFFSET_DAYS = 7

FIND_UNIVERSE_LABELS = {
    "sp500": "S&P 500",
    "nyse": "NYSE",
    "nasdaq": "NASDAQ",
    "russell-3000": "Russell 3000",
}
FIND_UNIVERSE_FILENAMES = {
    "sp500": RESOLVED_SP500_FILENAME,
    "nyse": RESOLVED_NYSE_FILENAME,
    "nasdaq": RESOLVED_NASDAQ_FILENAME,
    "russell-3000": RESOLVED_RUSSELL3000_FILENAME,
}
EXCHANGE_FALLBACK_CACHE: Dict[str, List[str]] = {}


def normalize_ticker(value: object) -> str:
    return str(value).strip().upper().replace(".", "-")


def dedupe_keep_order(values: Iterable[object]) -> List[str]:
    out: List[str] = []
    seen = set()
    for value in values:
        ticker = normalize_ticker(value)
        if not ticker or ticker in seen:
            continue
        seen.add(ticker)
        out.append(ticker)
    return out


def parse_iso_date(value: str, arg_name: str) -> date:
    try:
        return date.fromisoformat(str(value))
    except ValueError as exc:
        raise ValueError(f"{arg_name} must be YYYY-MM-DD; got {value!r}") from exc


def parse_args() -> argparse.Namespace:
    default_end_date = date.today().isoformat()
    parser = argparse.ArgumentParser(
        description=(
            "Fetch daily adjusted OHLCV candles for S&P 500 tickers or build yearly additive "
            "ticker lists for S&P 500/NYSE/NASDAQ/Russell 3000."
        )
    )
    parser.add_argument(
        "--start-date",
        default=START_DATE,
        help=f"Inclusive start date in YYYY-MM-DD (default: {START_DATE}).",
    )
    parser.add_argument(
        "--end-date",
        default=default_end_date,
        help=f"Inclusive end date in YYYY-MM-DD (default: {default_end_date}).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=CHUNK_SIZE,
        help=f"yfinance download chunk size (default: {CHUNK_SIZE}).",
    )
    parser.add_argument(
        "--out-dir",
        default=OUT_DIR,
        help=f"Output directory for per-ticker CSV files (default: {OUT_DIR}).",
    )
    parser.add_argument(
        "--find-tickers",
        action="store_true",
        default=False,
        help=(
            "Build an additive yearly ticker list from start year to end year and write "
            "a universe-specific *_tickers.txt file, then exit without fetching OHLCV."
        ),
    )
    parser.add_argument(
        "--nyse",
        action="store_true",
        default=False,
        help="With --find-tickers, resolve NYSE universe (default universe is S&P 500).",
    )
    parser.add_argument(
        "--nasdaq",
        action="store_true",
        default=False,
        help="With --find-tickers, resolve NASDAQ universe (default universe is S&P 500).",
    )
    parser.add_argument(
        "--russell-3000",
        action="store_true",
        default=False,
        dest="russell_3000",
        help="With --find-tickers, resolve Russell 3000 universe (default universe is S&P 500).",
    )
    parser.add_argument(
        "--sp500-source-url",
        default=SP500_WIKI_URL,
        help="URL used to resolve the current S&P 500 ticker list.",
    )
    parser.add_argument(
        "--ticker-file",
        "--sp500-tickers-file",
        dest="ticker_file",
        default="",
        help=(
            "Optional local ticker file (.txt/.lst or .csv with Symbol column) used for "
            "fetching instead of resolving S&P 500 constituents from web sources."
        ),
    )
    parser.add_argument(
        "--sp500-fallback-csv-url",
        default=SP500_CSV_FALLBACK_URL,
        help="Fallback CSV URL used if the primary S&P 500 source fails.",
    )
    parser.add_argument(
        "--http-timeout",
        type=int,
        default=HTTP_TIMEOUT_SECONDS,
        help=f"HTTP timeout in seconds for ticker source downloads (default: {HTTP_TIMEOUT_SECONDS}).",
    )
    return parser.parse_args()


def resolve_find_universe(args: argparse.Namespace) -> str:
    selected = []
    if bool(args.nyse):
        selected.append("nyse")
    if bool(args.nasdaq):
        selected.append("nasdaq")
    if bool(args.russell_3000):
        selected.append("russell-3000")
    if len(selected) > 1:
        raise ValueError("Only one of --nyse, --nasdaq, --russell-3000 can be set at a time.")
    if selected:
        return selected[0]
    return "sp500"


def build_yearly_anchor_dates(start_date: date, end_date_inclusive: date) -> List[date]:
    anchors: List[date] = []
    for year in range(int(start_date.year), int(end_date_inclusive.year) + 1):
        anchors.append(date(int(year), 1, 1))
    return anchors


def build_asof_candidates(
    start_date_inclusive: date,
    anchor_date: date,
    end_date_inclusive: date,
    max_offset_days: int,
) -> List[date]:
    if int(max_offset_days) < 0:
        raise ValueError("max_offset_days must be >= 0")

    lower_bound = start_date_inclusive
    upper_bound = end_date_inclusive
    if upper_bound < lower_bound:
        return []

    out: List[date] = []
    seen = set()

    def _add(candidate: date) -> None:
        if candidate < lower_bound or candidate > upper_bound:
            return
        if candidate in seen:
            return
        seen.add(candidate)
        out.append(candidate)

    _add(anchor_date)
    for delta in range(1, int(max_offset_days) + 1):
        _add(anchor_date - timedelta(days=delta))
        _add(anchor_date + timedelta(days=delta))
    return out


def _preview_failures(failures: Sequence[str], limit: int = 4) -> str:
    if not failures:
        return "no failures recorded"
    preview = "; ".join(list(failures)[: int(limit)])
    suffix = "" if len(failures) <= int(limit) else "; ..."
    return f"{preview}{suffix}"


def _log_ticker_resolution(message: str) -> None:
    print(message, flush=True)


def _is_rate_limit_error_message(message: object) -> bool:
    text = str(message).strip().lower()
    if not text:
        return False
    if "too many requests" in text:
        return True
    if "rate limit" in text or "rate limited" in text:
        return True
    if "yfratelimiterror" in text:
        return True
    if "status code 429" in text or " status=429" in text:
        return True
    if "http 429" in text:
        return True
    return False


def _extract_rate_limit_tickers_from_shared_errors() -> List[str]:
    if yf_shared is None:
        return []
    errors = getattr(yf_shared, "_ERRORS", {})
    if not isinstance(errors, dict) or not errors:
        return []

    matches: List[str] = []
    for ticker, err in errors.items():
        if _is_rate_limit_error_message(err):
            matches.append(normalize_ticker(ticker))
    return dedupe_keep_order(matches)


def download_text(url: str, timeout_seconds: int) -> str:
    request = Request(url, headers=HTTP_HEADERS)
    with urlopen(request, timeout=int(timeout_seconds)) as response:
        content = response.read()
        encoding = response.headers.get_content_charset() or "utf-8"
    return content.decode(encoding, errors="replace")


def _normalize_col_name(value: object) -> str:
    return str(value).strip().lower().replace("\xa0", " ")


def _read_pipe_delimited_text(text: str, source: str) -> pd.DataFrame:
    lower = text.lower()
    if "<html" in lower or "<!doctype html" in lower:
        raise RuntimeError(f"got HTML response from {source}")
    try:
        return pd.read_csv(io.StringIO(text), sep="|", dtype=str)
    except Exception:
        try:
            return pd.read_csv(io.StringIO(text), sep="|", dtype=str, engine="python")
        except Exception as exc:
            raise RuntimeError(f"failed to parse pipe-delimited source from {source}: {exc}") from exc


def _parse_nasdaq_listed_tickers(text: str, source: str) -> List[str]:
    frame = _read_pipe_delimited_text(text, source=source)
    if frame is None or frame.empty:
        raise RuntimeError("nasdaqlisted source returned no rows")

    cols = {_normalize_col_name(col): col for col in frame.columns}
    symbol_col = cols.get("symbol") or cols.get("act symbol")
    test_col = cols.get("test issue")
    if not symbol_col:
        raise RuntimeError("nasdaqlisted source missing Symbol column")

    frame = frame[frame[symbol_col].notna()].copy()
    frame = frame[
        ~frame[symbol_col]
        .astype(str)
        .str.contains("file creation time", case=False, na=False)
    ]
    if test_col:
        frame = frame[frame[test_col].astype(str).str.strip().str.upper().ne("Y")]

    tickers = dedupe_keep_order(frame[symbol_col].astype(str).tolist())
    if not tickers:
        raise RuntimeError("no NASDAQ tickers parsed from source")
    return tickers


def _parse_otherlisted_exchange_tickers(text: str, source: str, exchange_code: str) -> List[str]:
    frame = _read_pipe_delimited_text(text, source=source)
    if frame is None or frame.empty:
        raise RuntimeError("otherlisted source returned no rows")

    cols = {_normalize_col_name(col): col for col in frame.columns}
    symbol_col = cols.get("act symbol") or cols.get("symbol")
    exchange_col = cols.get("exchange")
    test_col = cols.get("test issue")
    if not symbol_col or not exchange_col:
        raise RuntimeError("otherlisted source missing required Symbol/Exchange columns")

    frame = frame[frame[symbol_col].notna()].copy()
    frame = frame[
        ~frame[symbol_col]
        .astype(str)
        .str.contains("file creation time", case=False, na=False)
    ]
    frame = frame[frame[exchange_col].astype(str).str.strip().str.upper().eq(str(exchange_code).upper())]
    if test_col:
        frame = frame[frame[test_col].astype(str).str.strip().str.upper().ne("Y")]

    tickers = dedupe_keep_order(frame[symbol_col].astype(str).tolist())
    if not tickers:
        raise RuntimeError(f"no {exchange_code} tickers parsed from source")
    return tickers


def _coerce_exchange_for_api(value: str) -> str:
    key = str(value).strip().lower()
    if key in {"nyse", "n"}:
        return "nyse"
    if key in {"nasdaq", "q", "nasdaqgm", "nasdaqgs", "nasdaqcm"}:
        return "nasdaq"
    return key


def _parse_nasdaq_screener_tickers(text: str, exchange: str, source: str) -> List[str]:
    exchange_key = _coerce_exchange_for_api(exchange)

    try:
        payload = json.loads(text)
    except Exception:
        payload = None

    if isinstance(payload, dict):
        data = payload.get("data")
        rows = data.get("rows") if isinstance(data, dict) else None
        if isinstance(rows, list):
            symbols: List[str] = []
            for row in rows:
                if not isinstance(row, dict):
                    continue
                raw_symbol = row.get("symbol") or row.get("Symbol") or row.get("ticker")
                if raw_symbol is None:
                    continue
                row_exchange = (
                    row.get("exchange")
                    or row.get("Exchange")
                    or row.get("market")
                    or row.get("Market")
                    or ""
                )
                row_exchange_key = _coerce_exchange_for_api(str(row_exchange))
                if row_exchange_key and exchange_key and row_exchange_key != exchange_key:
                    continue
                symbols.append(str(raw_symbol))
            tickers = dedupe_keep_order(symbols)
            if tickers:
                return tickers

    try:
        frame = pd.read_csv(io.StringIO(text))
    except Exception as exc:
        raise RuntimeError(f"failed to parse Nasdaq API response from {source}: {exc}") from exc

    if frame is None or frame.empty:
        raise RuntimeError(f"Nasdaq API response from {source} returned no rows")

    cols = {str(col).strip().lower(): col for col in frame.columns}
    symbol_col = cols.get("symbol")
    if not symbol_col:
        raise RuntimeError(f"Nasdaq API response from {source} missing Symbol column")

    exchange_col = cols.get("exchange")
    parsed = frame.copy()
    if exchange_col:
        parsed_exchange = parsed[exchange_col].astype(str).map(_coerce_exchange_for_api)
        parsed = parsed[parsed_exchange.eq(exchange_key)]

    tickers = dedupe_keep_order(parsed[symbol_col].astype(str).tolist())
    if not tickers:
        raise RuntimeError(f"no {exchange_key.upper()} tickers parsed from Nasdaq API source")
    return tickers


def resolve_exchange_tickers_from_nasdaq_api(exchange: str, timeout_seconds: int) -> Tuple[List[str], str]:
    exchange_key = _coerce_exchange_for_api(exchange)
    cached = EXCHANGE_FALLBACK_CACHE.get(exchange_key)
    if cached:
        return list(cached), f"cache:nasdaq_api:{exchange_key}"

    failures: List[str] = []
    total_urls = len(NASDAQ_SCREENER_API_URLS)
    for idx, url_template in enumerate(NASDAQ_SCREENER_API_URLS, start=1):
        url = str(url_template).format(exchange=exchange_key)
        _log_ticker_resolution(
            f"    Nasdaq API fallback request {idx}/{total_urls}: {url} "
            f"(timeout={int(timeout_seconds)}s)"
        )
        try:
            text = download_text(url=url, timeout_seconds=timeout_seconds)
            tickers = _parse_nasdaq_screener_tickers(text, exchange=exchange_key, source=url)
            EXCHANGE_FALLBACK_CACHE[exchange_key] = list(tickers)
            return tickers, url
        except Exception as exc:
            failures.append(f"{url} -> {exc}")
            _log_ticker_resolution(f"    Nasdaq API fallback failed {idx}/{total_urls}: {exc}")

    raise RuntimeError(
        f"failed Nasdaq API fallback for {exchange_key.upper()}: {_preview_failures(failures)}"
    )


def parse_ishares_holdings_csv(text: str) -> List[str]:
    lines = text.splitlines()
    header_idx = None
    for idx, line in enumerate(lines):
        token = line.split(",", 1)[0].strip().strip('"').lstrip("\ufeff")
        if token.lower() == "ticker":
            header_idx = idx
            break
    if header_idx is None:
        raise ValueError("could not find 'Ticker' header in holdings CSV")

    reader = csv.DictReader(lines[header_idx:])
    if not reader.fieldnames or "Ticker" not in reader.fieldnames:
        raise ValueError("holdings CSV does not include a Ticker column")

    tickers: List[str] = []
    for row in reader:
        raw = str(row.get("Ticker", "")).strip()
        if not raw:
            continue
        asset_class = str(row.get("Asset Class", "")).strip().lower()
        if asset_class and asset_class != "equity":
            continue
        if not all(ch.isalnum() or ch in ".-" for ch in raw):
            continue
        ticker = normalize_ticker(raw)
        if ticker in {"-", "N/A", "USD", "CASH"}:
            continue
        tickers.append(ticker)

    tickers = dedupe_keep_order(tickers)
    if not tickers:
        raise ValueError("no equity tickers parsed from holdings CSV")
    return tickers


def resolve_russell_3000_tickers_for_anchor(
    start_date_inclusive: date,
    anchor_date: date,
    end_date_inclusive: date,
    timeout_seconds: int,
    max_offset_days: int,
) -> Tuple[List[str], date, str]:
    candidates = build_asof_candidates(
        start_date_inclusive=start_date_inclusive,
        anchor_date=anchor_date,
        end_date_inclusive=end_date_inclusive,
        max_offset_days=max_offset_days,
    )
    if not candidates:
        raise RuntimeError(f"no candidate asOf dates for anchor {anchor_date.isoformat()}")

    failures: List[str] = []
    total_candidates = len(candidates)
    total_urls = len(RUSSELL_3000_HOLDINGS_URL_TEMPLATES)
    _log_ticker_resolution(
        f"  [Russell 3000] anchor={anchor_date.isoformat()} candidates={total_candidates} "
        f"urls_per_candidate={total_urls}"
    )
    for candidate_idx, candidate in enumerate(candidates, start=1):
        asof = candidate.strftime("%Y%m%d")
        _log_ticker_resolution(
            f"  [Russell 3000] candidate {candidate_idx}/{total_candidates} "
            f"asOf={asof} (anchor={anchor_date.isoformat()})"
        )
        for url_idx, url_template in enumerate(RUSSELL_3000_HOLDINGS_URL_TEMPLATES, start=1):
            url = str(url_template).format(asof=asof)
            _log_ticker_resolution(
                f"    request {url_idx}/{total_urls}: {url} "
                f"(timeout={int(timeout_seconds)}s)"
            )
            try:
                text = download_text(url=url, timeout_seconds=timeout_seconds)
                tickers = parse_ishares_holdings_csv(text)
                _log_ticker_resolution(
                    f"    success: parsed {len(tickers)} tickers from asOf={asof}"
                )
                return tickers, candidate, url
            except Exception as exc:
                _log_ticker_resolution(
                    f"    failed: asOf={asof} request {url_idx}/{total_urls} -> {exc}"
                )
                failures.append(f"{asof}: {exc}")

    raise RuntimeError(
        f"failed to resolve Russell 3000 holdings around {anchor_date.isoformat()} "
        f"(candidates={len(candidates)}): {_preview_failures(failures)}"
    )


def resolve_nyse_tickers_for_anchor(
    start_date_inclusive: date,
    anchor_date: date,
    end_date_inclusive: date,
    timeout_seconds: int,
    max_offset_days: int,
) -> Tuple[List[str], date, str]:
    candidates = build_asof_candidates(
        start_date_inclusive=start_date_inclusive,
        anchor_date=anchor_date,
        end_date_inclusive=end_date_inclusive,
        max_offset_days=max_offset_days,
    )
    if not candidates:
        raise RuntimeError(f"no candidate dates for anchor {anchor_date.isoformat()}")

    failures: List[str] = []
    total_candidates = len(candidates)
    total_urls = len(OTHERLISTED_URL_TEMPLATES)
    _log_ticker_resolution(
        f"  [NYSE] anchor={anchor_date.isoformat()} candidates={total_candidates} "
        f"urls_per_candidate={total_urls}"
    )
    for candidate_idx, candidate in enumerate(candidates, start=1):
        asof = candidate.strftime("%Y%m%d")
        _log_ticker_resolution(
            f"  [NYSE] candidate {candidate_idx}/{total_candidates} "
            f"asOf={asof} (anchor={anchor_date.isoformat()})"
        )
        for url_idx, url_template in enumerate(OTHERLISTED_URL_TEMPLATES, start=1):
            url = str(url_template).format(asof=asof)
            _log_ticker_resolution(
                f"    request {url_idx}/{total_urls}: {url} "
                f"(timeout={int(timeout_seconds)}s)"
            )
            try:
                text = download_text(url=url, timeout_seconds=timeout_seconds)
                tickers = _parse_otherlisted_exchange_tickers(text, source=url, exchange_code="N")
                _log_ticker_resolution(
                    f"    success: parsed {len(tickers)} tickers from asOf={asof}"
                )
                return tickers, candidate, url
            except Exception as exc:
                _log_ticker_resolution(
                    f"    failed: asOf={asof} request {url_idx}/{total_urls} -> {exc}"
                )
                failures.append(f"{asof}: {exc}")

    _log_ticker_resolution(
        "  [NYSE] SymbolDirectory candidates exhausted; trying Nasdaq API fallback "
        "(current snapshot, non-historical)."
    )
    try:
        tickers, fallback_source = resolve_exchange_tickers_from_nasdaq_api(
            exchange="nyse",
            timeout_seconds=timeout_seconds,
        )
        _log_ticker_resolution(
            f"  [NYSE] Nasdaq API fallback succeeded: {len(tickers)} tickers from {fallback_source}"
        )
        return tickers, anchor_date, fallback_source
    except Exception as exc:
        _log_ticker_resolution(f"  [NYSE] Nasdaq API fallback failed: {exc}")
        failures.append(f"nasdaq_api -> {exc}")

    raise RuntimeError(
        f"failed to resolve NYSE tickers around {anchor_date.isoformat()} "
        f"(candidates={len(candidates)}): {_preview_failures(failures)}"
    )


def resolve_nasdaq_tickers_for_anchor(
    start_date_inclusive: date,
    anchor_date: date,
    end_date_inclusive: date,
    timeout_seconds: int,
    max_offset_days: int,
) -> Tuple[List[str], date, str]:
    candidates = build_asof_candidates(
        start_date_inclusive=start_date_inclusive,
        anchor_date=anchor_date,
        end_date_inclusive=end_date_inclusive,
        max_offset_days=max_offset_days,
    )
    if not candidates:
        raise RuntimeError(f"no candidate dates for anchor {anchor_date.isoformat()}")

    failures: List[str] = []
    total_candidates = len(candidates)
    total_urls = len(NASDAQ_LISTED_URL_TEMPLATES)
    _log_ticker_resolution(
        f"  [NASDAQ] anchor={anchor_date.isoformat()} candidates={total_candidates} "
        f"urls_per_candidate={total_urls}"
    )
    for candidate_idx, candidate in enumerate(candidates, start=1):
        asof = candidate.strftime("%Y%m%d")
        _log_ticker_resolution(
            f"  [NASDAQ] candidate {candidate_idx}/{total_candidates} "
            f"asOf={asof} (anchor={anchor_date.isoformat()})"
        )
        for url_idx, url_template in enumerate(NASDAQ_LISTED_URL_TEMPLATES, start=1):
            url = str(url_template).format(asof=asof)
            _log_ticker_resolution(
                f"    request {url_idx}/{total_urls}: {url} "
                f"(timeout={int(timeout_seconds)}s)"
            )
            try:
                text = download_text(url=url, timeout_seconds=timeout_seconds)
                tickers = _parse_nasdaq_listed_tickers(text, source=url)
                _log_ticker_resolution(
                    f"    success: parsed {len(tickers)} tickers from asOf={asof}"
                )
                return tickers, candidate, url
            except Exception as exc:
                _log_ticker_resolution(
                    f"    failed: asOf={asof} request {url_idx}/{total_urls} -> {exc}"
                )
                failures.append(f"{asof}: {exc}")

    _log_ticker_resolution(
        "  [NASDAQ] SymbolDirectory candidates exhausted; trying Nasdaq API fallback "
        "(current snapshot, non-historical)."
    )
    try:
        tickers, fallback_source = resolve_exchange_tickers_from_nasdaq_api(
            exchange="nasdaq",
            timeout_seconds=timeout_seconds,
        )
        _log_ticker_resolution(
            f"  [NASDAQ] Nasdaq API fallback succeeded: {len(tickers)} tickers from {fallback_source}"
        )
        return tickers, anchor_date, fallback_source
    except Exception as exc:
        _log_ticker_resolution(f"  [NASDAQ] Nasdaq API fallback failed: {exc}")
        failures.append(f"nasdaq_api -> {exc}")

    raise RuntimeError(
        f"failed to resolve NASDAQ tickers around {anchor_date.isoformat()} "
        f"(candidates={len(candidates)}): {_preview_failures(failures)}"
    )


def build_additive_universe_from_anchors(
    universe_key: str,
    start_date: date,
    end_date_inclusive: date,
    timeout_seconds: int,
    max_offset_days: int,
    resolver: Callable[[date, date, date, int, int], Tuple[List[str], date, str]],
) -> Tuple[List[str], pd.DataFrame, int]:
    anchors = build_yearly_anchor_dates(
        start_date=start_date,
        end_date_inclusive=end_date_inclusive,
    )
    year_window_start = date(int(start_date.year), 1, 1)
    year_window_end = date(int(end_date_inclusive.year), 12, 31)
    rows = []
    universe: List[str] = []
    seen = set()
    errors = 0

    label = FIND_UNIVERSE_LABELS.get(universe_key, universe_key.upper())
    for anchor in anchors:
        print(f"Resolving {label} snapshot around {anchor.isoformat()}...", flush=True)
        try:
            tickers, resolved_asof_date, source_url = resolver(
                year_window_start,
                anchor,
                year_window_end,
                int(timeout_seconds),
                int(max_offset_days),
            )
            new_tickers = [ticker for ticker in tickers if ticker not in seen]
            for ticker in new_tickers:
                seen.add(ticker)
                universe.append(ticker)
            rows.append(
                {
                    "snapshot_year": int(anchor.year),
                    "anchor_date": anchor.isoformat(),
                    "resolved_asof_date": resolved_asof_date.isoformat(),
                    "ticker_count": int(len(tickers)),
                    "new_tickers": int(len(new_tickers)),
                    "cumulative_tickers": int(len(universe)),
                    "status": "ok",
                    "source": source_url,
                    "error": "",
                }
            )
            print(
                f"Resolved {anchor.year}: asOf={resolved_asof_date.isoformat()} "
                f"tickers={len(tickers)} new={len(new_tickers)} cumulative={len(universe)}",
                flush=True,
            )
        except Exception as exc:
            errors += 1
            rows.append(
                {
                    "snapshot_year": int(anchor.year),
                    "anchor_date": anchor.isoformat(),
                    "resolved_asof_date": "",
                    "ticker_count": 0,
                    "new_tickers": 0,
                    "cumulative_tickers": int(len(universe)),
                    "status": "error",
                    "source": "",
                    "error": str(exc),
                }
            )
            print(f"Failed {anchor.year}: {exc}", flush=True)

    return universe, pd.DataFrame(rows), int(errors)


def _parse_symbol_column_from_frame(frame: pd.DataFrame) -> List[str]:
    column_map = {str(col).strip().lower(): col for col in frame.columns}
    if "symbol" not in column_map:
        return []
    symbols = frame[column_map["symbol"]].astype(str).tolist()
    return dedupe_keep_order(symbols)


def _parse_ticker_lines(lines: Iterable[object]) -> List[str]:
    out: List[str] = []
    seen = set()
    for raw_line in lines:
        line = str(raw_line).strip()
        if not line or line.startswith("#"):
            continue
        if "|" in line:
            line = line.split("|", 1)[0].strip()
        if not line:
            continue
        first_token = re.split(r"[,\s;]+", line, maxsplit=1)[0].strip()
        if not first_token:
            continue
        if first_token.lower() in {"symbol", "ticker", "act", "act_symbol", "actsymbol"}:
            continue
        ticker = normalize_ticker(first_token)
        if not ticker or ticker in seen:
            continue
        seen.add(ticker)
        out.append(ticker)
    return out


def read_local_sp500_tickers(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"ticker file not found: {path}")

    suffix = path.suffix.lower()
    if suffix in {".txt", ".lst"}:
        raw = path.read_text(encoding="utf-8", errors="replace").splitlines()
        return _parse_ticker_lines(raw)

    if suffix == ".csv":
        frame = pd.read_csv(path)
        tickers = _parse_symbol_column_from_frame(frame)
        if tickers:
            return tickers
        if frame.shape[1] > 0:
            return dedupe_keep_order(frame.iloc[:, 0].astype(str).tolist())
        return []

    raw = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return _parse_ticker_lines(raw)


def resolve_sp500_tickers(url: str, timeout_seconds: int) -> List[str]:
    text = download_text(url=url, timeout_seconds=timeout_seconds)

    if url.lower().endswith(".csv"):
        try:
            frame = pd.read_csv(io.StringIO(text))
        except Exception as exc:
            raise RuntimeError(f"failed to parse S&P 500 CSV from {url}: {exc}") from exc
        tickers = _parse_symbol_column_from_frame(frame)
        if tickers:
            return tickers
        raise RuntimeError("unable to locate a non-empty 'Symbol' column in CSV source")

    try:
        tables = pd.read_html(io.StringIO(text))
    except Exception as exc:
        raise RuntimeError(f"failed to parse S&P 500 HTML table from {url}: {exc}") from exc

    for table in tables:
        tickers = _parse_symbol_column_from_frame(table)
        if tickers:
            return tickers

    raise RuntimeError("unable to locate a non-empty 'Symbol' column for S&P 500 tickers")


def _flatten_column_name(value: object) -> str:
    if isinstance(value, tuple):
        parts = []
        for part in value:
            token = str(part).strip()
            if not token or token.lower() in {"nan", "none"}:
                continue
            if token.lower().startswith("unnamed"):
                continue
            parts.append(token)
        if parts:
            return " ".join(parts)
        return ""
    token = str(value).strip()
    if token.lower() in {"nan", "none"}:
        return ""
    return token


def _flatten_frame_columns(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None:
        return pd.DataFrame()
    out = frame.copy()
    out.columns = [_flatten_column_name(col) for col in frame.columns]
    return out


def _canonicalize_column_name(name: object) -> str:
    text = str(name).strip().lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _find_column_name(
    columns: Sequence[object],
    required_terms: Sequence[str],
    banned_terms: Sequence[str] = (),
) -> str:
    for column in columns:
        key = _canonicalize_column_name(column)
        if not key:
            continue
        if any(term and term not in key for term in required_terms):
            continue
        if any(term and term in key for term in banned_terms):
            continue
        return str(column)
    return ""


def _extract_ticker_token(value: object) -> str:
    raw = str(value).strip()
    if not raw or raw.lower() in {"nan", "none", "n/a", "na", "-", "—"}:
        return ""
    match = TICKER_TOKEN_RE.search(raw.replace("\xa0", " "))
    if not match:
        return ""
    ticker = normalize_ticker(match.group(0))
    if not ticker or not any(ch.isalpha() for ch in ticker):
        return ""
    return ticker


def _find_added_ticker_column(columns: Sequence[object]) -> str:
    for required in (
        ("added", "ticker"),
        ("addition", "ticker"),
        ("added", "symbol"),
        ("addition", "symbol"),
    ):
        col = _find_column_name(columns, required_terms=required)
        if col:
            return col
    return _find_column_name(
        columns,
        required_terms=("added",),
        banned_terms=("security", "name", "company", "reason", "date"),
    )


def _find_removed_ticker_column(columns: Sequence[object]) -> str:
    for required in (
        ("removed", "ticker"),
        ("deletion", "ticker"),
        ("removed", "symbol"),
        ("deletion", "symbol"),
    ):
        col = _find_column_name(columns, required_terms=required)
        if col:
            return col
    return _find_column_name(
        columns,
        required_terms=("removed",),
        banned_terms=("security", "name", "company", "reason", "date"),
    )


def _parse_sp500_change_events(frame: pd.DataFrame) -> List[Tuple[date, str, str]]:
    flat = _flatten_frame_columns(frame)
    if flat.empty:
        return []

    columns = [str(col) for col in flat.columns]
    date_col = _find_column_name(columns, required_terms=("date",))
    if not date_col:
        return []

    added_col = _find_added_ticker_column(columns)
    removed_col = _find_removed_ticker_column(columns)
    if not added_col and not removed_col:
        return []

    parsed_dates = pd.to_datetime(flat[date_col], errors="coerce")
    events: List[Tuple[date, str, str]] = []
    for idx, timestamp in parsed_dates.items():
        if pd.isna(timestamp):
            continue
        event_date = timestamp.date()
        added_ticker = _extract_ticker_token(flat.at[idx, added_col]) if added_col else ""
        removed_ticker = _extract_ticker_token(flat.at[idx, removed_col]) if removed_col else ""
        if not added_ticker and not removed_ticker:
            continue
        events.append((event_date, added_ticker, removed_ticker))

    return events


def resolve_sp500_tickers_and_changes(
    url: str,
    timeout_seconds: int,
) -> Tuple[List[str], List[Tuple[date, str, str]]]:
    if url.lower().endswith(".csv"):
        raise RuntimeError("URL points to CSV; yearly ticker discovery requires an HTML source")

    _log_ticker_resolution(
        f"  [S&P 500] requesting history source: {url} (timeout={int(timeout_seconds)}s)"
    )
    text = download_text(url=url, timeout_seconds=timeout_seconds)
    try:
        tables = pd.read_html(io.StringIO(text))
    except Exception as exc:
        raise RuntimeError(f"failed to parse S&P 500 HTML table from {url}: {exc}") from exc
    _log_ticker_resolution(f"  [S&P 500] parsed {len(tables)} HTML tables from source")

    current_tickers: List[str] = []
    all_events: List[Tuple[date, str, str]] = []
    for table in tables:
        flat = _flatten_frame_columns(table)
        if not current_tickers:
            current_tickers = _parse_symbol_column_from_frame(flat)
        all_events.extend(_parse_sp500_change_events(flat))

    if not current_tickers:
        raise RuntimeError("unable to locate a non-empty 'Symbol' column for current S&P 500 tickers")
    if not all_events:
        raise RuntimeError("unable to locate S&P 500 additions/removals table in source")

    deduped_events: List[Tuple[date, str, str]] = []
    seen = set()
    for event in sorted(all_events, key=lambda row: row[0], reverse=True):
        key = (event[0].isoformat(), event[1], event[2])
        if key in seen:
            continue
        seen.add(key)
        deduped_events.append(event)

    _log_ticker_resolution(
        f"  [S&P 500] extracted current_tickers={len(current_tickers)} "
        f"changes={len(deduped_events)}"
    )

    return current_tickers, deduped_events


def reconstruct_sp500_membership_for_date(
    current_tickers: Sequence[str],
    change_events_desc: Sequence[Tuple[date, str, str]],
    snapshot_date: date,
) -> List[str]:
    membership = set(dedupe_keep_order(current_tickers))
    for event_date, added_ticker, removed_ticker in change_events_desc:
        if event_date <= snapshot_date:
            continue
        if added_ticker:
            membership.discard(added_ticker)
        if removed_ticker:
            membership.add(removed_ticker)

    ordered: List[str] = []
    for ticker in dedupe_keep_order(current_tickers):
        if ticker in membership:
            ordered.append(ticker)
    ordered_set = set(ordered)
    remaining = sorted(ticker for ticker in membership if ticker not in ordered_set)
    ordered.extend(remaining)
    return ordered


def build_yearly_additive_sp500_tickers(
    start_year: int,
    end_year: int,
    current_tickers: Sequence[str],
    change_events_desc: Sequence[Tuple[date, str, str]],
) -> Tuple[List[str], pd.DataFrame]:
    if start_year > end_year:
        return [], pd.DataFrame()

    universe: List[str] = []
    seen = set()
    rows = []
    for year in range(int(start_year), int(end_year) + 1):
        snapshot_date = date(int(year), 12, 31)
        year_tickers = reconstruct_sp500_membership_for_date(
            current_tickers=current_tickers,
            change_events_desc=change_events_desc,
            snapshot_date=snapshot_date,
        )
        new_tickers = [ticker for ticker in year_tickers if ticker not in seen]
        for ticker in new_tickers:
            seen.add(ticker)
            universe.append(ticker)
        rows.append(
            {
                "year": int(year),
                "snapshot_date": snapshot_date.isoformat(),
                "year_ticker_count": int(len(year_tickers)),
                "new_tickers": int(len(new_tickers)),
                "cumulative_tickers": int(len(universe)),
            }
        )
    return universe, pd.DataFrame(rows)


def write_ticker_file(path: Path, tickers: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    values = dedupe_keep_order(tickers)
    if values:
        path.write_text("\n".join(values) + "\n", encoding="utf-8")
    else:
        path.write_text("", encoding="utf-8")


def _download_chunk(
    tickers: Sequence[str],
    start_date: date,
    end_date_exclusive: date,
) -> Tuple[pd.DataFrame, List[str]]:
    frame = yf.download(
        tickers=list(tickers),
        start=start_date.isoformat(),
        end=end_date_exclusive.isoformat(),
        interval="1d",
        group_by="ticker",
        # Use adjusted OHLC so downstream production runup logic is not distorted by splits.
        auto_adjust=True,
        actions=False,
        threads=True,
        progress=True,
    )
    rate_limited = _extract_rate_limit_tickers_from_shared_errors()
    return frame, rate_limited


def download_all(
    tickers: List[str],
    start_date: date,
    end_date_inclusive: date,
    chunk_size: int,
) -> Tuple[pd.DataFrame, List[str]]:
    end_date_exclusive = end_date_inclusive + timedelta(days=1)
    rate_limited_all: List[str] = []
    rate_limited_seen = set()

    if len(tickers) <= chunk_size:
        df, rate_limited = _download_chunk(
            tickers,
            start_date=start_date,
            end_date_exclusive=end_date_exclusive,
        )
        for ticker in rate_limited:
            if ticker in rate_limited_seen:
                continue
            rate_limited_seen.add(ticker)
            rate_limited_all.append(ticker)
        if df is None or df.empty:
            return pd.DataFrame(), rate_limited_all
        if not isinstance(df.columns, pd.MultiIndex):
            df.columns = pd.MultiIndex.from_product([tickers, df.columns])
        return df, rate_limited_all

    frames: List[pd.DataFrame] = []
    chunks = [tickers[i : i + chunk_size] for i in range(0, len(tickers), chunk_size)]
    total_chunks = len(chunks)
    iterator = tqdm(chunks, desc="Downloading chunks", unit="chunk") if tqdm else chunks

    for idx, chunk in enumerate(iterator, start=1):
        if not tqdm:
            print(f"Downloading chunk {idx}/{total_chunks} ({len(chunk)} tickers)...", flush=True)
        df, rate_limited = _download_chunk(
            chunk,
            start_date=start_date,
            end_date_exclusive=end_date_exclusive,
        )
        if rate_limited:
            for ticker in rate_limited:
                if ticker in rate_limited_seen:
                    continue
                rate_limited_seen.add(ticker)
                rate_limited_all.append(ticker)
            print(
                f"Chunk {idx}/{total_chunks}: rate-limit hits={len(rate_limited)} "
                f"(cumulative={len(rate_limited_all)})",
                flush=True,
            )
        if df is None or df.empty:
            continue
        if not isinstance(df.columns, pd.MultiIndex):
            df.columns = pd.MultiIndex.from_product([chunk, df.columns])
        frames.append(df)

    if not frames:
        return pd.DataFrame(), rate_limited_all
    return pd.concat(frames, axis=1), rate_limited_all


def _normalize_ohlcv_columns(frame: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}
    for col in frame.columns:
        key = str(col).strip().lower().replace(" ", "")
        if key in REQUIRED_COLUMNS:
            rename_map[col] = key
    normalized = frame.rename(columns=rename_map)
    missing = [col for col in REQUIRED_COLUMNS if col not in normalized.columns]
    if missing:
        return pd.DataFrame()
    return normalized.loc[:, list(REQUIRED_COLUMNS)]


def split_and_save(data: pd.DataFrame, tickers: Sequence[str], out_dir: Path) -> Tuple[int, List[str]]:
    out_dir.mkdir(parents=True, exist_ok=True)

    multi_index = isinstance(data.columns, pd.MultiIndex)
    available = set(data.columns.get_level_values(0)) if multi_index else set()

    missing: List[str] = []
    saved = 0

    for ticker in tickers:
        if multi_index:
            if ticker not in available:
                missing.append(ticker)
                continue
            frame = data[ticker].copy()
        else:
            frame = data.copy()

        frame = frame.dropna(how="all")
        if frame.empty:
            missing.append(ticker)
            continue

        ohlcv = _normalize_ohlcv_columns(frame)
        if ohlcv.empty:
            missing.append(ticker)
            continue

        ohlcv.index = pd.to_datetime(ohlcv.index, errors="coerce").tz_localize(None)
        ohlcv = ohlcv[~ohlcv.index.isna()].sort_index()
        ohlcv.index.name = "date"
        if ohlcv.empty:
            missing.append(ticker)
            continue

        out_path = out_dir / f"{ticker}.csv"
        ohlcv.to_csv(out_path)
        saved += 1

    return saved, missing


def main() -> int:
    args = parse_args()

    try:
        start_date = parse_iso_date(args.start_date, "--start-date")
        end_date = parse_iso_date(args.end_date, "--end-date")
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 1

    if start_date > end_date:
        print("--start-date must be <= --end-date.", file=sys.stderr)
        return 1
    if int(args.chunk_size) < 1:
        print("--chunk-size must be >= 1.", file=sys.stderr)
        return 1

    try:
        find_universe = resolve_find_universe(args)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if not bool(args.find_tickers) and find_universe != "sp500":
        print(
            "--nyse/--nasdaq/--russell-3000 can only be used together with --find-tickers.",
            file=sys.stderr,
        )
        return 1

    out_dir = Path(args.out_dir)
    if bool(args.find_tickers):
        if str(args.ticker_file).strip():
            print("--ticker-file is ignored when --find-tickers is set.")

        resolved_path = out_dir / FIND_UNIVERSE_FILENAMES[str(find_universe)]
        universe_label = FIND_UNIVERSE_LABELS[str(find_universe)]

        if str(find_universe) == "sp500":
            failures: List[str] = []
            current_tickers: List[str] = []
            change_events_desc: List[Tuple[date, str, str]] = []
            source_used = ""
            find_sources: List[str] = []
            for candidate in (
                str(args.sp500_source_url).strip(),
                str(args.sp500_fallback_csv_url).strip(),
                SP500_WIKI_URL,
            ):
                if candidate and candidate not in find_sources:
                    find_sources.append(candidate)

            for source_url in find_sources:
                _log_ticker_resolution(f"Trying S&P 500 source: {source_url}")
                try:
                    current_tickers, change_events_desc = resolve_sp500_tickers_and_changes(
                        url=source_url,
                        timeout_seconds=int(args.http_timeout),
                    )
                    source_used = source_url
                    _log_ticker_resolution(
                        f"S&P 500 source succeeded: {source_url} "
                        f"(tickers={len(current_tickers)}, changes={len(change_events_desc)})"
                    )
                    break
                except Exception as exc:
                    _log_ticker_resolution(f"S&P 500 source failed: {source_url} -> {exc}")
                    failures.append(f"{source_url} -> {exc}")

            if not current_tickers:
                print(
                    "Failed to resolve S&P 500 yearly ticker history from configured sources.",
                    file=sys.stderr,
                )
                if failures:
                    print("\n".join(failures), file=sys.stderr)
                return 1

            tickers, yearly_summary = build_yearly_additive_sp500_tickers(
                start_year=int(start_date.year),
                end_year=int(end_date.year),
                current_tickers=current_tickers,
                change_events_desc=change_events_desc,
            )
            if not tickers:
                print("Yearly additive ticker universe is empty.", file=sys.stderr)
                return 1

            write_ticker_file(resolved_path, tickers)

            print(f"Resolved additive {universe_label} ticker universe: {len(tickers)} tickers.")
            print(f"Ticker source: {source_used}")
            print(f"Year range: {int(start_date.year)} -> {int(end_date.year)} (inclusive)")
            if not yearly_summary.empty:
                for _, row in yearly_summary.iterrows():
                    print(
                        f"Year {int(row['year'])}: "
                        f"tickers={int(row['year_ticker_count'])} "
                        f"new={int(row['new_tickers'])} "
                        f"cumulative={int(row['cumulative_tickers'])}"
                    )
            print(f"Resolved ticker list written to '{resolved_path}'.")
            print("Find-tickers mode enabled; skipping OHLCV fetch.")
            return 0

        resolver: Callable[[date, date, date, int, int], Tuple[List[str], date, str]]
        max_offset_days = int(MAX_NASDAQ_TRADER_ASOF_OFFSET_DAYS)
        if str(find_universe) == "nyse":
            resolver = resolve_nyse_tickers_for_anchor
        elif str(find_universe) == "nasdaq":
            resolver = resolve_nasdaq_tickers_for_anchor
        else:
            resolver = resolve_russell_3000_tickers_for_anchor
            max_offset_days = int(MAX_RUSSELL_ASOF_OFFSET_DAYS)

        tickers, yearly_summary, yearly_error_count = build_additive_universe_from_anchors(
            universe_key=str(find_universe),
            start_date=start_date,
            end_date_inclusive=end_date,
            timeout_seconds=int(args.http_timeout),
            max_offset_days=int(max_offset_days),
            resolver=resolver,
        )
        if not tickers:
            print(f"Failed to resolve additive {universe_label} ticker universe.", file=sys.stderr)
            return 1

        write_ticker_file(resolved_path, tickers)
        print(f"Resolved additive {universe_label} ticker universe: {len(tickers)} tickers.")
        print(f"Year range: {int(start_date.year)} -> {int(end_date.year)} (inclusive)")
        if not yearly_summary.empty:
            for _, row in yearly_summary.iterrows():
                if str(row.get("status", "")).strip().lower() == "ok":
                    print(
                        f"Year {int(row['snapshot_year'])}: "
                        f"asOf={row['resolved_asof_date']} "
                        f"tickers={int(row['ticker_count'])} "
                        f"new={int(row['new_tickers'])} "
                        f"cumulative={int(row['cumulative_tickers'])}"
                    )
                else:
                    print(f"Year {int(row['snapshot_year'])}: error={row['error']}")
        if int(yearly_error_count) > 0:
            print(f"Yearly resolution errors: {int(yearly_error_count)}")
        print(f"Resolved ticker list written to '{resolved_path}'.")
        print("Find-tickers mode enabled; skipping OHLCV fetch.")
        return 0

    tickers: List[str] = []
    failures: List[str] = []
    source_used = ""
    resolved_path = out_dir / RESOLVED_SP500_FILENAME
    local_tickers_file = str(args.ticker_file).strip()
    if local_tickers_file:
        local_path = Path(local_tickers_file)
        try:
            tickers = read_local_sp500_tickers(local_path)
            source_used = f"file:{local_path}"
        except Exception as exc:
            print(f"Failed to read --ticker-file: {exc}", file=sys.stderr)
            return 1
    else:
        for source_url in [str(args.sp500_source_url), str(args.sp500_fallback_csv_url)]:
            if not source_url:
                continue
            try:
                tickers = resolve_sp500_tickers(
                    url=source_url,
                    timeout_seconds=int(args.http_timeout),
                )
                source_used = source_url
                break
            except Exception as exc:
                failures.append(f"{source_url} -> {exc}")

    if not tickers:
        print("Failed to resolve S&P 500 tickers from all configured sources.", file=sys.stderr)
        if failures:
            print("\n".join(failures), file=sys.stderr)
        return 1

    write_ticker_file(resolved_path, tickers)
    print(f"Resolved {len(tickers)} tickers.")
    print(f"Ticker source: {source_used}")
    print(f"Requested date range: [{start_date.isoformat()} - {end_date.isoformat()}] (inclusive)")
    print(f"Chunk size: {int(args.chunk_size)}")

    data, rate_limited_tickers = download_all(
        tickers=tickers,
        start_date=start_date,
        end_date_inclusive=end_date,
        chunk_size=int(args.chunk_size),
    )
    rate_limit_path = out_dir / RATE_LIMIT_FILENAME
    write_ticker_file(rate_limit_path, rate_limited_tickers)
    print(f"Rate-limit ticker file written to '{rate_limit_path}' ({len(rate_limited_tickers)} tickers).")
    if data is None or data.empty:
        print("No data returned from yfinance.", file=sys.stderr)
        return 1

    saved, missing = split_and_save(data, tickers=tickers, out_dir=out_dir)

    print(f"Saved {saved} ticker CSV files to '{out_dir}/'.")
    print(f"Resolved ticker list written to '{resolved_path}'.")
    if missing:
        preview = ", ".join(missing[:25])
        suffix = "" if len(missing) <= 25 else ", ..."
        print(f"Missing/empty tickers ({len(missing)}): {preview}{suffix}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
