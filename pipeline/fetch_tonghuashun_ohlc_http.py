#!/usr/bin/env python3
"""Fetch daily OHLC data from Tonghuashun iFinD HTTP API.

Official reference:
- https://ftwc.51ifind.com/gwstatic/static/ds_web/quantapi-web/en/example.html
  (HTTP sample: get_access_token + cmd_history_quotation)

Output format per ticker:
- CSV with columns: date,open,high,low,close

Auth (choose one):
1) Set THS_ACCESS_TOKEN directly.
2) Set THS_REFRESH_TOKEN and this script will call get_access_token first.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd
import requests

DEFAULT_API_BASE_URL = "https://quantapi.51ifind.com/api/v1"
DEFAULT_START_DATE = "2012-01-01"
DEFAULT_TIMEOUT_SECONDS = 30
DEFAULT_RETRIES = 3
DEFAULT_RETRY_SLEEP_SECONDS = 1.0
DEFAULT_REQUEST_SLEEP_SECONDS = 0.15

OHLC_COLS = ("open", "high", "low", "close")
CODE_COL_CANDIDATES = (
    "code",
    "codes",
    "ticker",
    "tickers",
    "symbol",
    "symbols",
    "thscode",
)
TOKEN_ERR_HINTS = (
    "token",
    "access_token",
    "auth",
    "unauthorized",
    "权限",
    "令牌",
)


@dataclass
class FetchResult:
    ticker: str
    ok: bool
    rows: int
    path: Path | None
    error: str | None


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download daily OHLC from Tonghuashun iFinD HTTP API "
            "(cmd_history_quotation) into per-ticker CSV files."
        )
    )
    parser.add_argument(
        "--codes",
        default="",
        help="Comma-separated codes, e.g. '000001.SZ,600000.SH'.",
    )
    parser.add_argument(
        "--codes-file",
        default="",
        help=(
            "Optional code list file (.txt/.lst/.csv). If CSV, tries columns: "
            + ",".join(CODE_COL_CANDIDATES)
        ),
    )
    parser.add_argument(
        "--start-date",
        default=DEFAULT_START_DATE,
        help=f"Inclusive start date YYYY-MM-DD (default: {DEFAULT_START_DATE}).",
    )
    parser.add_argument(
        "--end-date",
        default=pd.Timestamp.now("UTC").date().isoformat(),
        help="Inclusive end date YYYY-MM-DD (default: today UTC).",
    )
    parser.add_argument(
        "--out-dir",
        default="tickers_cn",
        help="Output directory for per-ticker CSV files.",
    )
    parser.add_argument(
        "--api-base-url",
        default=os.getenv("THS_API_BASE_URL", DEFAULT_API_BASE_URL),
        help=f"HTTP API base URL (default: {DEFAULT_API_BASE_URL}).",
    )
    parser.add_argument(
        "--ifind-lang",
        default=os.getenv("THS_IFIND_LANG", "cn"),
        help="Optional ifindlang header value (default: cn).",
    )
    parser.add_argument(
        "--fill-mode",
        default="Blank",
        choices=("Blank", "Previous"),
        help="functionpara Fill mode for history quotes (default: Blank).",
    )
    parser.add_argument(
        "--access-token",
        default=os.getenv("THS_ACCESS_TOKEN", ""),
        help="Optional direct access token (or set THS_ACCESS_TOKEN).",
    )
    parser.add_argument(
        "--refresh-token",
        default=os.getenv("THS_REFRESH_TOKEN", ""),
        help="Optional refresh token (or set THS_REFRESH_TOKEN).",
    )
    parser.add_argument(
        "--token-mode",
        default="current",
        choices=("current", "new"),
        help=(
            "Token endpoint mode: 'current' -> get_access_token, "
            "'new' -> update_access_token."
        ),
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT_SECONDS,
        help=f"HTTP timeout in seconds (default: {DEFAULT_TIMEOUT_SECONDS}).",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=DEFAULT_RETRIES,
        help=f"Retries per request on transient errors (default: {DEFAULT_RETRIES}).",
    )
    parser.add_argument(
        "--retry-sleep",
        type=float,
        default=DEFAULT_RETRY_SLEEP_SECONDS,
        help=(
            "Initial retry sleep seconds (exponential backoff) "
            f"(default: {DEFAULT_RETRY_SLEEP_SECONDS})."
        ),
    )
    parser.add_argument(
        "--request-sleep",
        type=float,
        default=DEFAULT_REQUEST_SLEEP_SECONDS,
        help=(
            "Sleep seconds between ticker requests for rate-limit friendliness "
            f"(default: {DEFAULT_REQUEST_SLEEP_SECONDS})."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing ticker CSVs (default: skip existing).",
    )
    return parser.parse_args(argv)


def parse_date(value: str, arg_name: str) -> str:
    try:
        return pd.Timestamp(value).date().isoformat()
    except Exception as exc:
        raise ValueError(f"{arg_name} must be a valid date (YYYY-MM-DD), got {value!r}") from exc


def dedupe_keep_order(values: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen = set()
    for value in values:
        token = str(value).strip()
        if not token:
            continue
        if token in seen:
            continue
        seen.add(token)
        out.append(token)
    return out


def parse_codes_arg(raw: str) -> list[str]:
    if not raw:
        return []
    parts = [x.strip() for x in str(raw).split(",")]
    return dedupe_keep_order(parts)


def parse_codes_file(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"codes file not found: {path}")
    ext = path.suffix.lower()
    if ext == ".csv":
        df = pd.read_csv(path)
        if df.empty:
            return []
        cols_lower = {str(c).strip().lower(): str(c) for c in df.columns}
        selected = None
        for name in CODE_COL_CANDIDATES:
            if name in cols_lower:
                selected = cols_lower[name]
                break
        if selected is None:
            selected = str(df.columns[0])
        return dedupe_keep_order(df[selected].astype(str).tolist())

    out: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            if "," in s:
                s = s.split(",")[0].strip()
            out.append(s)
    return dedupe_keep_order(out)


def load_codes(codes_arg: str, codes_file_arg: str) -> list[str]:
    from_arg = parse_codes_arg(codes_arg)
    from_file: list[str] = []
    if codes_file_arg:
        from_file = parse_codes_file(Path(codes_file_arg))
    codes = dedupe_keep_order(from_arg + from_file)
    if not codes:
        raise ValueError("no codes provided; use --codes and/or --codes-file")
    return codes


def _request_json_with_retries(
    session: requests.Session,
    *,
    url: str,
    headers: dict[str, str],
    payload: dict[str, object] | None,
    timeout: int,
    retries: int,
    retry_sleep: float,
) -> dict[str, object]:
    sleep_s = max(0.0, float(retry_sleep))
    last_exc: Exception | None = None

    for attempt in range(int(retries) + 1):
        try:
            resp = session.post(url=url, headers=headers, json=payload, timeout=int(timeout))
            status = int(resp.status_code)
            if status >= 500 or status == 429:
                raise RuntimeError(f"HTTP {status}: {resp.text[:300]}")
            if status >= 400:
                raise RuntimeError(f"HTTP {status}: {resp.text[:500]}")
            try:
                data = resp.json()
            except Exception as exc:
                raise RuntimeError(f"non-JSON response: {resp.text[:300]}") from exc
            if not isinstance(data, dict):
                raise RuntimeError(f"unexpected response type: {type(data).__name__}")
            return data
        except Exception as exc:  # network + server + parse
            last_exc = exc
            if attempt >= int(retries):
                break
            time.sleep(sleep_s)
            sleep_s = max(0.1, sleep_s * 2.0)

    if last_exc is None:
        raise RuntimeError("request failed for unknown reason")
    raise RuntimeError(str(last_exc))


def _build_ths_headers(
    *,
    access_token: str,
    ifind_lang: str,
) -> dict[str, str]:
    headers = {
        "Content-Type": "application/json",
        "access_token": access_token,
    }
    if ifind_lang:
        headers["ifindlang"] = ifind_lang
    return headers


def get_access_token(
    session: requests.Session,
    *,
    api_base_url: str,
    refresh_token: str,
    ifind_lang: str,
    timeout: int,
    retries: int,
    retry_sleep: float,
    token_mode: str,
) -> str:
    refresh_token = str(refresh_token).strip()
    if not refresh_token:
        raise ValueError("refresh token is empty")
    api_base = api_base_url.rstrip("/")
    endpoint = "get_access_token" if str(token_mode) == "current" else "update_access_token"
    url = f"{api_base}/{endpoint}"
    headers = {
        "Content-Type": "application/json",
        "refresh_token": refresh_token,
    }
    if ifind_lang:
        headers["ifindlang"] = ifind_lang

    payload = {"refresh_token": refresh_token}
    data = _request_json_with_retries(
        session,
        url=url,
        headers=headers,
        payload=payload,
        timeout=timeout,
        retries=retries,
        retry_sleep=retry_sleep,
    )
    errorcode = int(data.get("errorcode", 0) or 0)
    errmsg = str(data.get("errmsg", ""))
    if errorcode != 0:
        raise RuntimeError(f"token request failed: errorcode={errorcode} errmsg={errmsg}")

    token = ""
    data_obj = data.get("data")
    if isinstance(data_obj, dict):
        token = str(data_obj.get("access_token", "")).strip()
    if not token:
        token = str(data.get("access_token", "")).strip()
    if not token:
        raise RuntimeError(f"token response missing access_token: {json.dumps(data)[:400]}")
    return token


def _coerce_list(value: object) -> list[object]:
    if value is None:
        return []
    if isinstance(value, list):
        return list(value)
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _canonical_col(name: object) -> str:
    return str(name).strip().lower().replace(" ", "_").replace("-", "_")


def _normalize_date_series(values: Sequence[object]) -> pd.Series:
    raw = pd.Series(list(values), dtype=object)
    try:
        dt = pd.to_datetime(raw, errors="coerce", format="mixed")
    except TypeError:
        dt = pd.to_datetime(raw, errors="coerce")
    return dt.dt.strftime("%Y-%m-%d")


def parse_history_quotes_response(
    response: dict[str, object],
    *,
    requested_code: str,
) -> pd.DataFrame:
    errorcode = int(response.get("errorcode", 0) or 0)
    errmsg = str(response.get("errmsg", ""))
    if errorcode != 0:
        raise RuntimeError(
            f"history request failed for {requested_code}: "
            f"errorcode={errorcode} errmsg={errmsg}"
        )

    tables = response.get("tables")
    if not isinstance(tables, list) or not tables:
        raise RuntimeError(f"no tables returned for {requested_code}")

    rows: list[dict[str, object]] = []
    for entry in tables:
        if not isinstance(entry, dict):
            continue
        thscode = str(entry.get("thscode", requested_code)).strip() or requested_code
        times = _coerce_list(entry.get("time"))
        raw_table = entry.get("table")

        col_values: dict[str, list[object]] = {}
        if isinstance(raw_table, dict):
            for k, v in raw_table.items():
                key = _canonical_col(k)
                col_values[key] = _coerce_list(v)
        elif isinstance(raw_table, list):
            # Rare variant: table itself is row list.
            for item in raw_table:
                if isinstance(item, dict):
                    rec = {"thscode": thscode}
                    for rk, rv in item.items():
                        rec[_canonical_col(rk)] = rv
                    rows.append(rec)
            continue

        if not col_values:
            # Fallback for flattened payload variants.
            for k, v in entry.items():
                key = _canonical_col(k)
                if key.startswith("table."):
                    col_values[key.split(".", 1)[1]] = _coerce_list(v)

        n = max(
            len(times),
            *(len(v) for v in col_values.values()),
            0,
        )
        if n <= 0:
            continue

        if len(times) < n:
            times = times + [None] * (n - len(times))

        for key in list(col_values.keys()):
            values = col_values[key]
            if len(values) < n:
                values = values + [None] * (n - len(values))
            col_values[key] = values

        for i in range(n):
            rec: dict[str, object] = {
                "thscode": thscode,
                "date": times[i],
            }
            for col in OHLC_COLS:
                if col in col_values:
                    rec[col] = col_values[col][i]
            rows.append(rec)

    if not rows:
        raise RuntimeError(f"no parsed OHLC rows returned for {requested_code}")

    df = pd.DataFrame(rows)
    if "date" not in df.columns:
        raise RuntimeError(f"response missing date/time for {requested_code}")
    for col in OHLC_COLS:
        if col not in df.columns:
            df[col] = pd.NA

    df["date"] = _normalize_date_series(df["date"])
    for col in OHLC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["date"]).copy()
    df = df.dropna(subset=list(OHLC_COLS), how="all")
    if df.empty:
        raise RuntimeError(f"no usable OHLC rows after parsing for {requested_code}")

    df = df[["date", *OHLC_COLS]].sort_values("date")
    df = df.drop_duplicates(subset=["date"], keep="last")
    return df.reset_index(drop=True)


def _safe_filename_from_code(code: str) -> str:
    raw = str(code).strip()
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", raw)
    safe = safe.strip("._-")
    if not safe:
        safe = "unknown"
    return safe


def _looks_like_token_error(message: str) -> bool:
    text = str(message).strip().lower()
    return any(hint in text for hint in TOKEN_ERR_HINTS)


def fetch_one_ticker(
    session: requests.Session,
    *,
    api_base_url: str,
    headers: dict[str, str],
    ticker: str,
    start_date: str,
    end_date: str,
    fill_mode: str,
    timeout: int,
    retries: int,
    retry_sleep: float,
) -> pd.DataFrame:
    url = f"{api_base_url.rstrip('/')}/cmd_history_quotation"
    payload = {
        "codes": ticker,
        "indicators": ",".join(OHLC_COLS),
        "startdate": start_date,
        "enddate": end_date,
        "functionpara": {
            "Fill": fill_mode,
        },
    }
    data = _request_json_with_retries(
        session,
        url=url,
        headers=headers,
        payload=payload,
        timeout=timeout,
        retries=retries,
        retry_sleep=retry_sleep,
    )
    return parse_history_quotes_response(data, requested_code=ticker)


def write_failed_csv(path: Path, failures: Sequence[FetchResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["ticker", "error"])
        for item in failures:
            w.writerow([item.ticker, item.error or "unknown_error"])


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    start_date = parse_date(args.start_date, "--start-date")
    end_date = parse_date(args.end_date, "--end-date")
    if pd.Timestamp(start_date) > pd.Timestamp(end_date):
        raise ValueError("--start-date must be <= --end-date")

    codes = load_codes(args.codes, args.codes_file)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    access_token = str(args.access_token).strip()
    refresh_token = str(args.refresh_token).strip()

    session = requests.Session()
    if not access_token:
        if not refresh_token:
            raise ValueError(
                "missing auth: provide --access-token/THS_ACCESS_TOKEN or "
                "--refresh-token/THS_REFRESH_TOKEN"
            )
        access_token = get_access_token(
            session,
            api_base_url=str(args.api_base_url),
            refresh_token=refresh_token,
            ifind_lang=str(args.ifind_lang),
            timeout=int(args.timeout),
            retries=int(args.retries),
            retry_sleep=float(args.retry_sleep),
            token_mode=str(args.token_mode),
        )

    headers = _build_ths_headers(
        access_token=access_token,
        ifind_lang=str(args.ifind_lang),
    )

    results: list[FetchResult] = []
    failures: list[FetchResult] = []
    request_sleep = max(0.0, float(args.request_sleep))
    overwrite = bool(args.overwrite)

    for idx, ticker in enumerate(codes, start=1):
        ticker = str(ticker).strip()
        if not ticker:
            continue
        out_path = out_dir / f"{_safe_filename_from_code(ticker)}.csv"
        if out_path.exists() and not overwrite:
            with out_path.open("r", encoding="utf-8") as f:
                existing_rows = max(0, sum(1 for _ in f) - 1)
            res = FetchResult(
                ticker=ticker,
                ok=True,
                rows=int(existing_rows),
                path=out_path,
                error=None,
            )
            results.append(res)
            print(
                f"[{idx}/{len(codes)}] {ticker}: skipped existing ({existing_rows} rows) -> {out_path}"
            )
            continue

        try:
            df = fetch_one_ticker(
                session,
                api_base_url=str(args.api_base_url),
                headers=headers,
                ticker=ticker,
                start_date=start_date,
                end_date=end_date,
                fill_mode=str(args.fill_mode),
                timeout=int(args.timeout),
                retries=int(args.retries),
                retry_sleep=float(args.retry_sleep),
            )
        except Exception as exc:
            # Attempt one access-token refresh on auth-like failure.
            if refresh_token and _looks_like_token_error(str(exc)):
                try:
                    access_token = get_access_token(
                        session,
                        api_base_url=str(args.api_base_url),
                        refresh_token=refresh_token,
                        ifind_lang=str(args.ifind_lang),
                        timeout=int(args.timeout),
                        retries=int(args.retries),
                        retry_sleep=float(args.retry_sleep),
                        token_mode=str(args.token_mode),
                    )
                    headers = _build_ths_headers(
                        access_token=access_token,
                        ifind_lang=str(args.ifind_lang),
                    )
                    df = fetch_one_ticker(
                        session,
                        api_base_url=str(args.api_base_url),
                        headers=headers,
                        ticker=ticker,
                        start_date=start_date,
                        end_date=end_date,
                        fill_mode=str(args.fill_mode),
                        timeout=int(args.timeout),
                        retries=int(args.retries),
                        retry_sleep=float(args.retry_sleep),
                    )
                except Exception as exc2:
                    res = FetchResult(
                        ticker=ticker,
                        ok=False,
                        rows=0,
                        path=None,
                        error=str(exc2),
                    )
                    failures.append(res)
                    results.append(res)
                    print(f"[{idx}/{len(codes)}] {ticker}: FAILED -> {exc2}")
                    if request_sleep > 0:
                        time.sleep(request_sleep)
                    continue
            else:
                res = FetchResult(
                    ticker=ticker,
                    ok=False,
                    rows=0,
                    path=None,
                    error=str(exc),
                )
                failures.append(res)
                results.append(res)
                print(f"[{idx}/{len(codes)}] {ticker}: FAILED -> {exc}")
                if request_sleep > 0:
                    time.sleep(request_sleep)
                continue

        df.to_csv(out_path, index=False)
        res = FetchResult(
            ticker=ticker,
            ok=True,
            rows=int(df.shape[0]),
            path=out_path,
            error=None,
        )
        results.append(res)
        print(f"[{idx}/{len(codes)}] {ticker}: ok rows={df.shape[0]} -> {out_path}")

        if request_sleep > 0:
            time.sleep(request_sleep)

    ok_count = sum(1 for r in results if r.ok)
    fail_count = sum(1 for r in results if not r.ok)
    total_rows = sum(int(r.rows) for r in results if r.ok)

    if failures:
        failed_csv = out_dir / "_failed.csv"
        write_failed_csv(failed_csv, failures)
        print(f"failed detail: {failed_csv}")

    print(
        f"done: tickers_total={len(codes)} ok={ok_count} failed={fail_count} "
        f"rows_written={total_rows}"
    )
    return 1 if fail_count > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
