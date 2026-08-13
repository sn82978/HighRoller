#!/usr/bin/env python3
"""Download minute-level price history for Polymarket's BTC Up/Down 15m series.

The series is `btc-up-or-down-15m` (gamma series id 10192); its individual
events are slugged `btc-updown-15m-<start_ts>`, where `start_ts` is the unix
timestamp of the 15-minute window's open. Slots are exactly 900s apart, so the
whole event list can be generated locally instead of paged out of the API.

Pipeline:
  1. Generate every 900s slot in [anchor - days, anchor].
  2. Batch-resolve slugs -> event/market metadata via gamma  (cached).
  3. Fetch /prices-history at fidelity=1 for each CLOB token (cached, resumable).
  4. Emit raw ticks + an exact per-minute grid.

Usage:
    python data/fetch_btc_updown_15m.py                  # full 3-month pull
    python data/fetch_btc_updown_15m.py --days 1         # quick smoke test
    python data/fetch_btc_updown_15m.py --tokens up      # halve the requests
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

GAMMA = "https://gamma-api.polymarket.com/events"
CLOB = "https://clob.polymarket.com/prices-history"

SERIES_PREFIX = "btc-updown-15m"
SLOT_SECONDS = 900

# Last event of the pull: "Bitcoin Up or Down - February 26, 3:45PM-4:00PM ET".
DEFAULT_ANCHOR_TS = 1772138700

SLUGS_PER_GAMMA_CALL = 50
FETCH_PAD_SECONDS = 180
HEADERS = {"User-Agent": "HighRoller/1.0 (research)", "Accept": "application/json"}

HERE = Path(__file__).resolve().parent


# --------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------

_local = threading.local()


def session() -> requests.Session:
    """One requests.Session per worker thread (Sessions aren't thread-safe)."""
    s = getattr(_local, "s", None)
    if s is None:
        s = requests.Session()
        s.headers.update(HEADERS)
        _local.s = s
    return s


def get_json(url: str, params, tries: int = 6):
    """GET with backoff on 429/5xx and transport errors. None if unrecoverable."""
    for attempt in range(tries):
        try:
            r = session().get(url, params=params, timeout=30)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 404:
                return None
            if r.status_code not in (408, 429, 500, 502, 503, 504):
                r.raise_for_status()
        except (requests.RequestException, ValueError):
            if attempt == tries - 1:
                return None
        time.sleep(min(30.0, 1.5 * 2**attempt) * (0.5 + random.random()))
    return None


# --------------------------------------------------------------------------
# Step 1/2 - event metadata
# --------------------------------------------------------------------------


def slot_timestamps(anchor_ts: int, days: float) -> list[int]:
    """Every 900s slot in (anchor - days, anchor], oldest first."""
    n = int(round(days * 86400 / SLOT_SECONDS))
    return [anchor_ts - SLOT_SECONDS * i for i in range(n)][::-1]


def parse_event(ev: dict) -> dict | None:
    """Flatten a gamma event into the fields we need. None if unusable."""
    markets = ev.get("markets") or []
    if not markets:
        return None
    m = markets[0]

    try:
        token_ids = json.loads(m.get("clobTokenIds") or "[]")
        outcomes = json.loads(m.get("outcomes") or "[]")
    except json.JSONDecodeError:
        return None
    if len(token_ids) != len(outcomes) or not token_ids:
        return None

    try:
        prices = [float(p) for p in json.loads(m.get("outcomePrices") or "[]")]
    except (json.JSONDecodeError, ValueError):
        prices = []
    winner = None
    if m.get("closed") and len(prices) == len(outcomes):
        hits = [o for o, p in zip(outcomes, prices) if p >= 0.99]
        if len(hits) == 1:
            winner = hits[0]

    try:
        start_ts = int(str(ev["slug"]).rsplit("-", 1)[1])
    except (KeyError, IndexError, ValueError):
        return None

    return {
        "event_slug": ev["slug"],
        "condition_id": m.get("conditionId"),
        "start_ts": start_ts,
        "end_ts": start_ts + SLOT_SECONDS,
        "closed": bool(m.get("closed")),
        "winner": winner,
        "volume": float(m.get("volumeNum") or 0.0),
        "token_ids": token_ids,
        "outcomes": outcomes,
    }


def load_events(slots: list[int], cache: Path, refresh: bool) -> list[dict]:
    """Resolve slugs -> metadata, reusing `cache` for slugs already resolved.

    Slots with no event on Polymarket are cached as misses. Without that, the
    ~32 genuine gaps in a 3-month window are re-requested on every single run,
    since a slug that resolves to nothing otherwise never looks 'done'.
    """
    known: dict[str, dict | None] = {}
    if cache.exists() and not refresh:
        with cache.open() as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                slug = rec.get("event_slug")
                if slug:
                    known[slug] = None if rec.get("missing") else rec

    wanted = [f"{SERIES_PREFIX}-{ts}" for ts in slots]
    todo = [s for s in wanted if s not in known]
    print(f"[events] {len(wanted)} slots, {len(wanted) - len(todo)} cached, "
          f"{len(todo)} to fetch")

    if todo:
        batches = [todo[i:i + SLUGS_PER_GAMMA_CALL]
                   for i in range(0, len(todo), SLUGS_PER_GAMMA_CALL)]

        def fetch(batch: list[str]) -> tuple[list[str], list[dict] | None]:
            params = [("slug", s) for s in batch]
            params.append(("limit", str(len(batch))))
            data = get_json(GAMMA, params)
            if not isinstance(data, list):
                return batch, None
            return batch, [r for r in (parse_event(e) for e in data) if r]

        cache.parent.mkdir(parents=True, exist_ok=True)
        done = 0
        with cache.open("a") as fh, ThreadPoolExecutor(4) as pool:
            for batch, recs in pool.map(fetch, batches):
                if recs is None:  # request failed; leave uncached so it retries
                    continue
                for rec in recs:
                    known[rec["event_slug"]] = rec
                    fh.write(json.dumps(rec) + "\n")
                for slug in set(batch) - {r["event_slug"] for r in recs}:
                    known[slug] = None
                    fh.write(json.dumps({"event_slug": slug, "missing": True}) + "\n")
                fh.flush()
                done += 1
                if done % 10 == 0 or done == len(batches):
                    print(f"[events] batch {done}/{len(batches)}", flush=True)

    events = [known[s] for s in wanted if known.get(s)]
    missing = len(wanted) - len(events)
    if missing:
        print(f"[events] {missing} slot(s) have no event on Polymarket (cached)")
    return events


# --------------------------------------------------------------------------
# Step 3 - price history
# --------------------------------------------------------------------------


def build_jobs(events: list[dict], tokens: str, pre_minutes: int) -> list[dict]:
    """One fetch job per (event, outcome) we intend to download.

    Fetches FETCH_PAD_SECONDS earlier than the grid actually needs: the first
    tick lands a few seconds *after* startTs, so without the pad the earliest
    grid point would have no prior observation to fill from and be null for
    every single market.
    """
    jobs = []
    for ev in events:
        for idx, (tok, outcome) in enumerate(zip(ev["token_ids"], ev["outcomes"])):
            if tokens == "up" and idx != 0:
                continue
            jobs.append({
                "event_slug": ev["event_slug"],
                "outcome": outcome,
                "token_id": tok,
                "start_ts": ev["start_ts"] - pre_minutes * 60 - FETCH_PAD_SECONDS,
                "end_ts": ev["end_ts"],
            })
    return jobs


def load_price_cache(cache: Path) -> set[str]:
    """Token ids already fetched successfully.

    Failures are cached too (so the run is auditable), but deliberately do NOT
    count as done — otherwise a re-run would skip exactly the series it should
    be retrying.
    """
    done: set[str] = set()
    if cache.exists():
        with cache.open() as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("ok") and "token_id" in rec:
                    done.add(rec["token_id"])
    return done


def fetch_prices(jobs: list[dict], cache: Path, workers: int) -> None:
    """Download missing histories, appending to the JSONL cache as we go."""
    done = load_price_cache(cache)
    todo = [j for j in jobs if j["token_id"] not in done]
    print(f"[prices] {len(jobs)} series, {len(jobs) - len(todo)} cached, "
          f"{len(todo)} to fetch")
    if not todo:
        return

    def fetch(job: dict) -> dict:
        data = get_json(CLOB, {
            "market": job["token_id"],
            "startTs": job["start_ts"],
            "endTs": job["end_ts"],
            "fidelity": 1,
        })
        history = (data or {}).get("history") or []
        return {
            "token_id": job["token_id"],
            "event_slug": job["event_slug"],
            "outcome": job["outcome"],
            "ok": data is not None,
            "history": [[int(p["t"]), float(p["p"])] for p in history
                        if "t" in p and "p" in p],
        }

    cache.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    failed = empty = 0
    with cache.open("a") as fh, ThreadPoolExecutor(workers) as pool:
        for i, rec in enumerate(pool.map(fetch, todo), 1):
            if not rec["ok"]:
                failed += 1
            elif not rec["history"]:
                empty += 1
            fh.write(json.dumps(rec) + "\n")
            if i % 250 == 0 or i == len(todo):
                fh.flush()
                rate = i / max(time.time() - t0, 1e-9)
                eta = (len(todo) - i) / max(rate, 1e-9)
                print(f"[prices] {i}/{len(todo)}  {rate:.1f} req/s  "
                      f"eta {eta / 60:.1f} min  failed={failed} empty={empty}",
                      flush=True)

    if failed:
        print(f"[prices] {failed} series failed after retries — "
              f"re-run to retry only those")


# --------------------------------------------------------------------------
# Step 4 - assemble
# --------------------------------------------------------------------------


def build_frames(events: list[dict], price_cache: Path, pre_minutes: int):
    """Return (raw ticks, per-minute grid) dataframes."""
    meta = pd.DataFrame([{k: v for k, v in e.items()
                          if k not in ("token_ids", "outcomes")} for e in events])

    # A retry appends a second record for the same token; keep the last good one
    # so a successful retry supersedes the earlier failure.
    best: dict[str, dict] = {}
    with price_cache.open() as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("ok") and "token_id" in rec:
                best[rec["token_id"]] = rec

    rows = []
    for tok, rec in best.items():
        for t, p in rec["history"]:
            rows.append((rec["event_slug"], rec["outcome"], tok, t, p))

    raw = pd.DataFrame(rows, columns=["event_slug", "outcome", "token_id",
                                      "t", "price"])
    raw = raw.merge(meta, on="event_slug", how="inner")
    raw = raw.sort_values(["start_ts", "outcome", "t"], ignore_index=True)
    raw["timestamp"] = pd.to_datetime(raw["t"], unit="s", utc=True)
    raw["minute_index"] = (raw["t"] - raw["start_ts"]) // 60

    grid = build_minute_grid(raw, meta, pre_minutes)
    return raw, grid


def build_minute_grid(raw: pd.DataFrame, meta: pd.DataFrame, pre_minutes: int):
    """Snap ticks onto an exact 60s grid per (event, outcome) via as-of fill.

    The CLOB returns ticks roughly — not exactly — 60s apart, so each grid point
    takes the last observation at or before it. Points before the first tick
    stay null rather than being back-filled.
    """
    if raw.empty:
        return raw.copy()

    pairs = raw[["event_slug", "outcome", "token_id"]].drop_duplicates()
    spine = pairs.merge(meta[["event_slug", "start_ts", "end_ts"]], on="event_slug")
    # Inclusive of both ends: 15m window at pre_minutes lookback -> 16 + pre points.
    spine["t"] = [
        list(range(int(s) - pre_minutes * 60, int(e) + 1, 60))
        for s, e in zip(spine["start_ts"], spine["end_ts"])
    ]
    spine = spine.explode("t", ignore_index=True)
    spine["t"] = spine["t"].astype("int64")

    left = spine.sort_values("t", ignore_index=True)
    right = raw[["event_slug", "outcome", "t", "price"]].sort_values(
        "t", ignore_index=True)

    grid = pd.merge_asof(
        left, right, on="t", by=["event_slug", "outcome"], direction="backward"
    )
    grid = grid.merge(
        meta[["event_slug", "condition_id", "closed", "winner", "volume"]],
        on="event_slug", how="left",
    )
    grid["timestamp"] = pd.to_datetime(grid["t"], unit="s", utc=True)
    grid["minute_index"] = (grid["t"] - grid["start_ts"]) // 60
    return grid.sort_values(["start_ts", "outcome", "t"], ignore_index=True)


def write_frames(df: pd.DataFrame, out_dir: Path, stem: str, fmt: str,
                 cols: list[str], partition: str) -> None:
    """Write `df` as a single file, or one file per UTC day under `out_dir/stem`.

    Days are cut on each market's `start_ts`, not on each row's own timestamp, so
    a market straddling midnight (23:45-00:00, plus its pre-open tape) keeps all
    of its rows in one file instead of being split across two.
    """
    df = df[[c for c in cols if c in df.columns]]

    def dump(frame: pd.DataFrame, base: Path) -> None:
        if fmt in ("parquet", "both"):
            frame.to_parquet(base.with_suffix(".parquet"), index=False)
        if fmt in ("csv", "both"):
            frame.to_csv(base.with_suffix(".csv"), index=False)

    if partition == "single":
        out_dir.mkdir(parents=True, exist_ok=True)
        dump(df, out_dir / stem)
        print(f"[write] {out_dir / stem}.*  ({len(df):,} rows)")
        return

    day_dir = out_dir / stem
    day_dir.mkdir(parents=True, exist_ok=True)
    days = pd.to_datetime(df["start_ts"], unit="s", utc=True).dt.strftime("%Y-%m-%d")
    for day, chunk in df.groupby(days, sort=True):
        dump(chunk.reset_index(drop=True), day_dir / f"{stem}_{day}")
    n = days.nunique()
    print(f"[write] {day_dir}/  ({n} daily files, {len(df):,} rows, "
          f"{days.min()} -> {days.max()})")


def write_out(df: pd.DataFrame, out_dir: Path, stem: str, fmt: str,
              partition: str = "day") -> None:
    write_frames(df, out_dir, stem, fmt,
                 ["event_slug", "condition_id", "outcome", "token_id",
                  "start_ts", "end_ts", "t", "timestamp", "minute_index",
                  "price", "closed", "winner", "volume"], partition)


# --------------------------------------------------------------------------


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--anchor-slug", default=f"{SERIES_PREFIX}-{DEFAULT_ANCHOR_TS}",
                    help="last (most recent) event to include")
    ap.add_argument("--days", type=float, default=90.0,
                    help="how far back from the anchor to pull (default: 90)")
    ap.add_argument("--pre-minutes", type=int, default=10,
                    help="minutes of pre-open book history per market (default: 10)")
    ap.add_argument("--tokens", choices=["both", "up"], default="both",
                    help="fetch both outcomes or only Up (default: both)")
    ap.add_argument("--workers", type=int, default=8,
                    help="concurrent price requests (default: 8)")
    ap.add_argument("--format", choices=["parquet", "csv", "both"], default="both")
    ap.add_argument("--partition", choices=["day", "single"], default="day",
                    help="one file per UTC day (default) or one combined file")
    ap.add_argument("--out-dir", type=Path, default=HERE / "polymarket")
    ap.add_argument("--cache-dir", type=Path, default=HERE / "cache")
    ap.add_argument("--refresh-events", action="store_true",
                    help="re-fetch event metadata instead of using the cache")
    ap.add_argument("--build-only", action="store_true",
                    help="skip downloading; rebuild outputs from the cache")
    args = ap.parse_args(argv)

    try:
        anchor_ts = int(str(args.anchor_slug).rsplit("-", 1)[1])
    except (IndexError, ValueError):
        print(f"error: cannot parse a timestamp out of {args.anchor_slug!r}",
              file=sys.stderr)
        return 2
    if anchor_ts % SLOT_SECONDS:
        print(f"error: {anchor_ts} is not on a 15-minute boundary", file=sys.stderr)
        return 2
    if args.days <= 0 or args.pre_minutes < 0 or args.workers < 1:
        print("error: --days must be > 0, --pre-minutes >= 0, --workers >= 1",
              file=sys.stderr)
        return 2

    slots = slot_timestamps(anchor_ts, args.days)
    fmt = lambda ts: datetime.fromtimestamp(ts, timezone.utc).strftime(
        "%Y-%m-%d %H:%M UTC")
    print(f"[range] {len(slots)} slots  {fmt(slots[0])} -> "
          f"{fmt(slots[-1] + SLOT_SECONDS)}")

    events_cache = args.cache_dir / "events.jsonl"
    prices_cache = args.cache_dir / "prices.jsonl"

    events = load_events(slots, events_cache, args.refresh_events)
    if not events:
        print("error: no events resolved", file=sys.stderr)
        return 1

    jobs = build_jobs(events, args.tokens, args.pre_minutes)
    if not args.build_only:
        fetch_prices(jobs, prices_cache, args.workers)
    if not prices_cache.exists():
        print("error: no price cache to build from", file=sys.stderr)
        return 1

    raw, grid = build_frames(events, prices_cache, args.pre_minutes)
    if raw.empty:
        print("error: no price rows collected", file=sys.stderr)
        return 1

    write_out(raw, args.out_dir, "btc_updown_15m_ticks", args.format, args.partition)
    write_out(grid, args.out_dir, "btc_updown_15m_minute", args.format, args.partition)

    covered = grid["price"].notna().mean() * 100
    print(f"[done] {grid['event_slug'].nunique():,} markets, "
          f"{len(grid):,} minute rows, {covered:.1f}% of grid points priced")
    return 0


if __name__ == "__main__":
    sys.exit(main())
