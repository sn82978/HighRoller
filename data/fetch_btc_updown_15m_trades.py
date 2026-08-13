#!/usr/bin/env python3
"""Build 15-second OHLCV candles from true tick data for the BTC Up/Down 15m series.

`prices-history` (see fetch_btc_updown_15m.py) bottoms out at 1-minute sampled
prices and thins out the final ~54s before resolution — exactly where these
markets decide. `data-api.polymarket.com/trades` instead returns every
individual fill, which this script aggregates into 15s candles.

Both outcome legs are consolidated onto one tape by quoting every trade as an
**Up-equivalent price**: a Down fill at 0.40 is an Up fill at 0.60. So a market's
Up and Down trades reinforce one series rather than being split across two.

Trade payloads are large (~8 GB over 3 months), so trades are aggregated in a
streaming fashion and never cached raw — only the resulting candles are stored.

Usage:
    python data/fetch_btc_updown_15m_trades.py                 # full pull (~25 min)
    python data/fetch_btc_updown_15m_trades.py --days 1        # quick check
    python data/fetch_btc_updown_15m_trades.py --bucket-seconds 5
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from fetch_btc_updown_15m import (
    DEFAULT_ANCHOR_TS,
    SLOT_SECONDS,
    HERE,
    get_json,
    load_events,
    slot_timestamps,
    write_frames,
)

TRADES = "https://data-api.polymarket.com/trades"

TRADES_PER_PAGE = 5000
MAX_PAGES = 20  # safety valve; busiest observed market needed 2

# The API 400s on offset >= 15000 regardless of limit, and exposes no time
# filter, so 15k trades is a hard ceiling per market. Only the ~18 busiest
# markets in 3 months exceed it; they lose part of their *pre-open* tape while
# the live 15-minute window stays complete, so they are kept and flagged
# `truncated` rather than dropped.
OFFSET_CEILING = 15000


# --------------------------------------------------------------------------
# Fetch + aggregate
# --------------------------------------------------------------------------


def fetch_trades(condition_id: str, floor_ts: int) -> tuple[list[dict], bool, bool]:
    """Page trades newest-first. Returns (trades, ok, truncated).

    Stops once a page runs short (market exhausted) or the oldest trade on the
    page predates `floor_ts` — everything older is outside the candle window.
    `truncated` means the offset ceiling was hit while still inside the window,
    so the oldest part of the tape is unreachable.
    """
    out: list[dict] = []
    for page in range(MAX_PAGES):
        offset = page * TRADES_PER_PAGE
        if offset >= OFFSET_CEILING:
            return out, True, True
        data = get_json(TRADES, {
            "market": condition_id,
            "limit": TRADES_PER_PAGE,
            "offset": offset,
        })
        if not isinstance(data, list):
            return out, False, False
        out.extend(data)
        if len(data) < TRADES_PER_PAGE:
            break
        oldest = min((t.get("timestamp", 0) for t in data), default=0)
        if oldest < floor_ts:
            break
    return out, True, False


def up_price(trade: dict) -> float | None:
    """Quote a trade as an Up-equivalent price, or None if unusable.

    Prices outside [0, 1] are impossible for a binary market but do occur: the
    3-month pull turned up one fill priced at 4.5137, which would otherwise sink
    a candle's low to -3.51 and overstate its notional.
    """
    try:
        price = float(trade["price"])
    except (KeyError, TypeError, ValueError):
        return None
    if not 0.0 <= price <= 1.0:
        return None
    idx = trade.get("outcomeIndex")
    if idx is None:
        outcome = str(trade.get("outcome", "")).strip().lower()
        if outcome == "up":
            idx = 0
        elif outcome == "down":
            idx = 1
        else:
            return None
    return price if int(idx) == 0 else 1.0 - price


def make_candles(trades: list[dict], start_ts: int, lo: int, hi: int,
                 bucket: int) -> list[list]:
    """Aggregate trades into [idx, o, h, l, c, vwap, vol_shares, vol_usdc, n].

    Two notionals are tracked deliberately. `vol_usdc` sums size x the price
    actually paid, so it stays true dollars traded. VWAP instead weights the
    *Up-equivalent* price, matching OHLC — summing paid prices there would let a
    Down fill at 0.95 drag the mean of an Up-quoted candle to the wrong end.
    """
    # The API returns newest-first; reversing yields chronological order and,
    # for fills sharing a timestamp, preserves their true intra-second sequence.
    buckets: dict[int, dict] = {}
    fills: set[tuple] = set()
    for tr in reversed(trades):
        ts = tr.get("timestamp")
        if ts is None or not (lo <= ts < hi):
            continue
        up = up_price(tr)
        if up is None:
            continue
        try:
            size = float(tr.get("size") or 0.0)
            paid = float(tr["price"])
        except (TypeError, ValueError):
            continue
        # A zero/negative size would leave a candle flagged as traded while
        # carrying no volume, making vwap undefined on a non-empty bucket.
        if size <= 0:
            continue
        # Offset paging can serve the same fill twice if the tape grows between
        # pages — impossible on resolved markets, but this script also accepts a
        # recent --anchor-slug, where it would double-count volume.
        key = (tr.get("transactionHash"), tr.get("asset"), ts, size, paid)
        if key in fills:
            continue
        fills.add(key)

        idx = (ts - start_ts) // bucket
        b = buckets.get(idx)
        if b is None:
            buckets[idx] = {"o": up, "h": up, "l": up, "c": up,
                            "vs": size, "vu": size * paid, "vwn": size * up,
                            "n": 1}
        else:
            if up > b["h"]:
                b["h"] = up
            if up < b["l"]:
                b["l"] = up
            b["c"] = up
            b["vs"] += size
            b["vu"] += size * paid
            b["vwn"] += size * up
            b["n"] += 1

    out = []
    for i, b in sorted(buckets.items()):
        # 10dp, not 8: at 8 the rounding artifact reaches 5e-9, enough to push
        # vwap a hair outside [low, high] on single-trade candles.
        vwap = round(b["vwn"] / b["vs"], 10) if b["vs"] > 0 else None
        out.append([i, b["o"], b["h"], b["l"], b["c"], vwap,
                    round(b["vs"], 6), round(b["vu"], 6), b["n"]])
    return out


def load_cache(cache: Path) -> set[str]:
    """Markets already fetched successfully.

    Failures are cached too (so the run is auditable), but deliberately do NOT
    count as done — otherwise a re-run would skip exactly the markets it should
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
                if rec.get("ok") and "event_slug" in rec:
                    done.add(rec["event_slug"])
    return done


def run_fetch(events: list[dict], cache: Path, workers: int, pre_minutes: int,
              bucket: int) -> None:
    done = load_cache(cache)
    todo = [e for e in events if e["event_slug"] not in done]
    print(f"[trades] {len(events)} markets, {len(events) - len(todo)} cached, "
          f"{len(todo)} to fetch")
    if not todo:
        return

    def work(ev: dict) -> dict:
        lo = ev["start_ts"] - pre_minutes * 60
        hi = ev["end_ts"]
        trades, ok, truncated = fetch_trades(ev["condition_id"], lo)
        return {
            "event_slug": ev["event_slug"],
            "condition_id": ev["condition_id"],
            "start_ts": ev["start_ts"],
            "ok": ok,
            "truncated": truncated,
            "n_trades": len(trades),
            "candles": make_candles(trades, ev["start_ts"], lo, hi, bucket),
        }

    cache.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    failed = trunc = ntr = 0
    with cache.open("a") as fh, ThreadPoolExecutor(workers) as pool:
        for i, rec in enumerate(pool.map(work, todo), 1):
            if not rec["ok"]:
                failed += 1
            elif rec["truncated"]:
                trunc += 1
            ntr += rec["n_trades"]
            fh.write(json.dumps(rec) + "\n")
            if i % 100 == 0 or i == len(todo):
                fh.flush()
                rate = i / max(time.time() - t0, 1e-9)
                print(f"[trades] {i}/{len(todo)}  {rate:.1f} mkt/s  "
                      f"eta {(len(todo) - i) / max(rate, 1e-9) / 60:.1f} min  "
                      f"{ntr:,} trades  failed={failed} truncated={trunc}",
                      flush=True)
    if failed:
        print(f"[trades] {failed} market(s) failed — re-run to retry only those")
    if trunc:
        print(f"[trades] {trunc} market(s) hit the {OFFSET_CEILING:,}-trade "
              f"ceiling; live window intact, early pre-open tape missing")


# --------------------------------------------------------------------------
# Assemble
# --------------------------------------------------------------------------

CANDLE_COLS = ["candle_index", "open", "high", "low", "close", "vwap",
               "volume_shares", "volume_usdc", "trades"]


def build_frame(events: list[dict], cache: Path, pre_minutes: int, bucket: int,
                dense: bool) -> pd.DataFrame:
    meta = {e["event_slug"]: e for e in events}

    # A retry appends a second record for the same market; keep the last good
    # one so a successful retry supersedes the earlier failure.
    best: dict[str, dict] = {}
    with cache.open() as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            slug = rec.get("event_slug")
            if slug in meta and rec.get("ok"):
                best[slug] = rec

    rows = []
    seen = set(best)
    for slug, rec in best.items():
        for c in rec["candles"]:
            rows.append([slug] + c)
    truncated = {s: bool(r.get("truncated")) for s, r in best.items()}

    df = pd.DataFrame(rows, columns=["event_slug"] + CANDLE_COLS)
    if df.empty:
        return df

    if dense:
        df = densify(df, seen, pre_minutes, bucket)

    m = pd.DataFrame([{k: v for k, v in e.items()
                       if k not in ("token_ids", "outcomes")}
                      for e in events if e["event_slug"] in seen])
    df = df.merge(m, on="event_slug", how="left")
    df["t"] = df["start_ts"] + df["candle_index"] * bucket
    df["timestamp"] = pd.to_datetime(df["t"], unit="s", utc=True)
    df["truncated"] = df["event_slug"].map(truncated).astype(bool)
    return df.sort_values(["start_ts", "candle_index"], ignore_index=True)


def densify(df: pd.DataFrame, slugs: set[str], pre_minutes: int,
            bucket: int) -> pd.DataFrame:
    """Emit every bucket in the window, flat-filling ones with no trades.

    A gap means nothing traded, not that price is unknown, so an empty bucket
    becomes a zero-volume doji at the previous close. Buckets before a market's
    first trade stay null — there is no prior price to carry.
    """
    first = -(pre_minutes * 60) // bucket
    last = SLOT_SECONDS // bucket
    spine = pd.MultiIndex.from_product(
        [sorted(slugs), range(first, last)], names=["event_slug", "candle_index"]
    ).to_frame(index=False)

    out = spine.merge(df, on=["event_slug", "candle_index"], how="left")
    out = out.sort_values(["event_slug", "candle_index"], ignore_index=True)

    out["has_trades"] = out["trades"].notna()
    carry = out.groupby("event_slug")["close"].ffill()
    for col in ("open", "high", "low", "close"):
        out[col] = out[col].fillna(carry)
    for col in ("volume_shares", "volume_usdc", "trades"):
        out[col] = out[col].fillna(0)
    return out


def write_out(df: pd.DataFrame, out_dir: Path, stem: str, fmt: str,
              partition: str) -> None:
    write_frames(df, out_dir, stem, fmt,
                 ["event_slug", "condition_id", "start_ts", "end_ts",
                  "candle_index", "t", "timestamp", "open", "high", "low",
                  "close", "vwap", "volume_shares", "volume_usdc",
                  "trades", "has_trades", "truncated", "winner", "volume"],
                 partition)


# --------------------------------------------------------------------------


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--anchor-slug", default=f"btc-updown-15m-{DEFAULT_ANCHOR_TS}")
    ap.add_argument("--days", type=float, default=90.0)
    ap.add_argument("--bucket-seconds", type=int, default=15,
                    help="candle width in seconds (default: 15)")
    ap.add_argument("--pre-minutes", type=int, default=10,
                    help="minutes of pre-open tape per market (default: 10)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--format", choices=["parquet", "csv", "both"], default="both")
    ap.add_argument("--partition", choices=["day", "single"], default="day",
                    help="one file per UTC day (default) or one combined file")
    ap.add_argument("--out-dir", type=Path, default=HERE / "polymarket")
    ap.add_argument("--cache-dir", type=Path, default=HERE / "cache")
    ap.add_argument("--sparse", action="store_true",
                    help="emit only buckets that had trades")
    ap.add_argument("--build-only", action="store_true",
                    help="skip downloading; rebuild output from the cache")
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
    if args.bucket_seconds < 1 or SLOT_SECONDS % args.bucket_seconds:
        print(f"error: --bucket-seconds must divide {SLOT_SECONDS}", file=sys.stderr)
        return 2
    if args.days <= 0 or args.pre_minutes < 0 or args.workers < 1:
        print("error: --days > 0, --pre-minutes >= 0, --workers >= 1 required",
              file=sys.stderr)
        return 2
    if (args.pre_minutes * 60) % args.bucket_seconds:
        print("error: --pre-minutes must be a whole number of buckets",
              file=sys.stderr)
        return 2

    slots = slot_timestamps(anchor_ts, args.days)
    fmt = lambda ts: datetime.fromtimestamp(ts, timezone.utc).strftime(
        "%Y-%m-%d %H:%M UTC")
    print(f"[range] {len(slots)} slots  {fmt(slots[0])} -> "
          f"{fmt(slots[-1] + SLOT_SECONDS)}  @{args.bucket_seconds}s candles")

    events = load_events(slots, args.cache_dir / "events.jsonl", refresh=False)
    events = [e for e in events if e.get("condition_id")]
    if not events:
        print("error: no events resolved", file=sys.stderr)
        return 1

    cache = args.cache_dir / f"candles_{args.bucket_seconds}s.jsonl"
    if not args.build_only:
        run_fetch(events, cache, args.workers, args.pre_minutes,
                  args.bucket_seconds)
    if not cache.exists():
        print("error: no candle cache to build from", file=sys.stderr)
        return 1

    df = build_frame(events, cache, args.pre_minutes, args.bucket_seconds,
                     dense=not args.sparse)
    if df.empty:
        print("error: no candles built", file=sys.stderr)
        return 1

    write_out(df, args.out_dir, f"btc_updown_15m_candles_{args.bucket_seconds}s",
              args.format, args.partition)

    traded = df["trades"].sum()
    print(f"[done] {df['event_slug'].nunique():,} markets, {len(df):,} candles, "
          f"{int(traded):,} trades aggregated")
    if "has_trades" in df:
        print(f"       {df['has_trades'].mean() * 100:.1f}% of buckets had a trade")
    return 0


if __name__ == "__main__":
    sys.exit(main())
