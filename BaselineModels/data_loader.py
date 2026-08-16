"""Temporal train/validation/test splitting for the BTC Up/Down 15m dataset.

The proposal names lookahead bias as "the single most likely way this project
produces a fake result", and asks that the loader itself enforce the cutoff
rather than relying on us to be careful by hand. So this module is the only
sanctioned way to get data into a model:

* splits are cut on each market's ``start_ts``, never on a row's own timestamp,
  so a market can never straddle a boundary;
* every load asserts the returned markets are disjoint from the other splits;
* the test split refuses to load without an explicit ``allow_test=True``, so
  touching it is always a deliberate act that shows up in a diff.

Usage::

    from BaselineModels.data_loader import load_split
    train = load_split("train")
    val   = load_split("val")
    test  = load_split("test", allow_test=True)   # once, at the very end
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
POLYMARKET_DIR = REPO_ROOT / "data" / "polymarket"

#: Dataset name -> directory under data/polymarket/.
DATASETS = {
    "candles_15s": "btc_updown_15m_candles_15s",
    "minute": "btc_updown_15m_minute",
    "ticks": "btc_updown_15m_ticks",
}

#: Fraction of the *calendar span* given to each split, in time order.
SPLIT_FRACTIONS = (0.70, 0.15, 0.15)
SPLIT_NAMES = ("train", "val", "test")

SECONDS_PER_DAY = 86_400

#: Columns that describe the market's final state and are therefore not
#: available at decision time. `volume` is the killer: it is the market's
#: *total* volume, written identically onto every row including the pre-open
#: ones, so a model that reads it knows how busy the market eventually became.
LEAKY_COLUMNS = ("winner", "closed", "volume", "truncated")


@dataclass(frozen=True)
class SplitBounds:
    """Half-open ``start_ts`` cutoffs, snapped to UTC midnight.

    The ordering invariant is enforced here rather than at the point of use.
    ``_assert_disjoint`` checks a load against the bounds it was given, so it
    is tautological if those bounds are themselves degenerate -- a
    hand-constructed overlap would validate happily against its own definition.
    Making the type unconstructible in that state closes the hole.
    """

    train_start: int
    train_end: int
    val_end: int
    test_end: int

    def __post_init__(self) -> None:
        edges = (self.train_start, self.train_end, self.val_end, self.test_end)
        if not all(a < b for a, b in zip(edges, edges[1:])):
            raise ValueError(
                "split bounds must be strictly increasing "
                f"(train_start < train_end < val_end < test_end), got {edges}. "
                "Equal edges make a split empty and let a neighbour swallow it."
            )

    def range_for(self, split: str) -> tuple[int, int]:
        if split == "train":
            return self.train_start, self.train_end
        if split == "val":
            return self.train_end, self.val_end
        if split == "test":
            return self.val_end, self.test_end
        raise ValueError(f"unknown split {split!r}; expected one of {SPLIT_NAMES}")

    def describe(self) -> str:
        fmt = lambda ts: pd.to_datetime(ts, unit="s", utc=True).strftime("%Y-%m-%d")
        return (
            f"train [{fmt(self.train_start)}, {fmt(self.train_end)})  "
            f"val [{fmt(self.train_end)}, {fmt(self.val_end)})  "
            f"test [{fmt(self.val_end)}, {fmt(self.test_end)})"
        )


def dataset_dir(dataset: str) -> Path:
    try:
        return POLYMARKET_DIR / DATASETS[dataset]
    except KeyError:
        raise ValueError(
            f"unknown dataset {dataset!r}; expected one of {sorted(DATASETS)}"
        ) from None


def dataset_files(dataset: str) -> list[str]:
    d = dataset_dir(dataset)
    files = sorted(glob.glob(os.path.join(str(d), "*.parquet")))
    if not files:
        hint = ""
        if dataset == "candles_15s":
            hint = (
                "\nThe 15s candles are not committed to the repo. Build them with:"
                "\n    python data/fetch_btc_updown_15m_trades.py --format parquet"
            )
        raise FileNotFoundError(f"no parquet files under {d}{hint}")
    return files


def market_universe(dataset: str = "candles_15s") -> pd.DataFrame:
    """One row per market: ``event_slug``, ``start_ts``, ``winner``, ``truncated``.

    Reads only the few columns it needs, so scanning all 91 daily files is cheap.
    """
    import pyarrow.parquet as pq

    files = dataset_files(dataset)
    available = set(pq.read_schema(files[0]).names)
    # minute/ticks datasets have no `truncated` column.
    wanted = [c for c in ("event_slug", "start_ts", "winner", "truncated") if c in available]
    df = pd.concat(
        [pd.read_parquet(f, columns=wanted) for f in files], ignore_index=True
    )
    agg = {"start_ts": "first", "winner": "first"}
    if "truncated" in df.columns:
        agg["truncated"] = "any"
    universe = df.groupby("event_slug", as_index=False).agg(agg)
    return universe.sort_values("start_ts", ignore_index=True)


def _snap_to_utc_day(ts: int) -> int:
    return int(ts) - (int(ts) % SECONDS_PER_DAY)


def compute_bounds(
    universe: pd.DataFrame | None = None,
    dataset: str = "candles_15s",
    fractions: tuple[float, float, float] = SPLIT_FRACTIONS,
) -> SplitBounds:
    """Derive the split cutoffs from the market universe, snapped to UTC days.

    Snapping means a split boundary never lands mid-day, so the daily parquet
    partitioning and the split boundaries agree and the splits stay legible when
    written up.
    """
    if abs(sum(fractions) - 1.0) > 1e-9:
        raise ValueError(f"fractions must sum to 1, got {fractions}")
    if universe is None:
        universe = market_universe(dataset)

    first, last = int(universe.start_ts.min()), int(universe.start_ts.max())
    span = last - first
    train_end = _snap_to_utc_day(first + int(span * fractions[0]))
    val_end = _snap_to_utc_day(first + int(span * (fractions[0] + fractions[1])))
    # SplitBounds.__post_init__ rejects degenerate orderings, so a fractions
    # tuple too skewed to produce three non-empty blocks fails loudly here.
    return SplitBounds(
        train_start=_snap_to_utc_day(first),
        train_end=train_end,
        val_end=val_end,
        test_end=last + 1,
    )


def load_split(
    split: str,
    dataset: str = "candles_15s",
    *,
    allow_test: bool = False,
    bounds: SplitBounds | None = None,
    resolved_only: bool = True,
    drop_truncated_preopen: bool = True,
    columns: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Load one split. This is the only supported entry point for model code.

    Parameters
    ----------
    allow_test:
        Guard on the held-out block. The proposal budgets exactly one look at
        it, so loading it requires saying so out loud.
    drop_truncated_preopen:
        18 markets exceed the trades API's 15,000-row ceiling and lose the
        earliest part of their 10-minute pre-open tape (paging runs
        newest-first). Their live window is intact and they are the highest
        volume markets in the set, so the default keeps them but drops their
        incomplete pre-open candles rather than discarding the markets.
    """
    if split not in SPLIT_NAMES:
        raise ValueError(f"unknown split {split!r}; expected one of {SPLIT_NAMES}")
    if split == "test" and not allow_test:
        raise PermissionError(
            "Refusing to load the test split. Tune on 'val'; pass allow_test=True "
            "only for the single final evaluation reported in the paper."
        )

    universe = market_universe(dataset)
    if bounds is None:
        bounds = compute_bounds(universe, dataset)
    lo, hi = bounds.range_for(split)

    keep = universe[(universe.start_ts >= lo) & (universe.start_ts < hi)]
    if resolved_only:
        keep = keep[keep.winner.isin(("Up", "Down"))]
    slugs = set(keep.event_slug)
    if not slugs:
        raise ValueError(f"split {split!r} selected zero markets over [{lo}, {hi})")

    frames = []
    for f in dataset_files(dataset):
        df = pd.read_parquet(f, columns=list(columns) if columns else None)
        df = df[df.event_slug.isin(slugs)]
        if not df.empty:
            frames.append(df)
    out = pd.concat(frames, ignore_index=True)

    if drop_truncated_preopen and "truncated" in out.columns and "candle_index" in out.columns:
        out = out[(~out.truncated.astype(bool)) | (out.candle_index >= 0)]

    sort_cols = [c for c in ("event_slug", "candle_index", "minute_index", "t") if c in out.columns]
    out = out.sort_values(sort_cols, ignore_index=True)

    _assert_disjoint(split, set(out.event_slug), universe, bounds)
    return out


def _assert_disjoint(
    split: str, slugs: set[str], universe: pd.DataFrame, bounds: SplitBounds
) -> None:
    """Fail loudly if a load ever returns a market belonging to another split."""
    for other in SPLIT_NAMES:
        if other == split:
            continue
        lo, hi = bounds.range_for(other)
        other_slugs = set(
            universe[(universe.start_ts >= lo) & (universe.start_ts < hi)].event_slug
        )
        overlap = slugs & other_slugs
        if overlap:
            raise AssertionError(
                f"split {split!r} leaked {len(overlap)} markets from {other!r}, "
                f"e.g. {sorted(overlap)[:3]}"
            )
    lo, hi = bounds.range_for(split)
    stray = universe[universe.event_slug.isin(slugs)]
    bad = stray[(stray.start_ts < lo) | (stray.start_ts >= hi)]
    if not bad.empty:
        raise AssertionError(
            f"split {split!r} returned {len(bad)} markets outside [{lo}, {hi})"
        )


def split_summary(dataset: str = "candles_15s") -> pd.DataFrame:
    """Market/day counts and base rate per split -- for the writeup's data table."""
    universe = market_universe(dataset)
    bounds = compute_bounds(universe, dataset)
    rows = []
    for split in SPLIT_NAMES:
        lo, hi = bounds.range_for(split)
        sel = universe[(universe.start_ts >= lo) & (universe.start_ts < hi)]
        sel = sel[sel.winner.isin(("Up", "Down"))]
        rows.append(
            {
                "split": split,
                "markets": len(sel),
                "days": round((hi - lo) / SECONDS_PER_DAY, 1),
                "start": pd.to_datetime(lo, unit="s", utc=True).date(),
                "end": pd.to_datetime(hi, unit="s", utc=True).date(),
                "base_rate_up": round((sel.winner == "Up").mean(), 4),
            }
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import sys

    ds = sys.argv[1] if len(sys.argv) > 1 else "candles_15s"
    print(compute_bounds(dataset=ds).describe())
    print(split_summary(ds).to_string(index=False))
