"""No-lookahead feature construction from the 15s candle tape.

One row per ``(event_slug, candle_index)``. Every feature on a row is computed
from candles ``<= candle_index`` only, and the row also carries the *next*
candle's OHLC under ``next_*`` so that any action the row triggers is filled at
a price that was not among its own inputs. That one-candle offset is what keeps
a backtest honest; it is baked into the frame rather than left to the caller.

Two traps in this dataset, both handled here:

* ``volume`` is the market's **final total** volume, stamped identically onto
  every row including the pre-open ones. It is a direct read on how busy the
  market eventually became and must never be a feature. The per-candle flow
  columns (``volume_shares``, ``volume_usdc``, ``trades``) are the honest ones.
* The committed ``minute`` dataset lags the real tape by ~60s. That makes it
  safe as an *input* (it is stale, not clairvoyant) but fatal as an *execution
  price*. Fills therefore always come from the candles.

Models consume :data:`FEATURE_COLUMNS` -- an allowlist. Anything not on it,
including the label and the settlement columns, cannot reach a model by
accident.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

#: Lookbacks in candles. The live window is 60 candles (15 min), so 16 candles
#: is 4 minutes -- about the longest window that still fits inside a market.
RETURN_LAGS = (1, 4, 8, 16)
VOL_WINDOWS = (8, 16)
FLOW_WINDOWS = (4, 16)

#: Candles per minute in the 15s tape.
CANDLES_PER_MINUTE = 4
#: Last live candle index (candle_index runs -40..59).
LAST_CANDLE = 59

#: The allowlist. A tuple, not a list: nothing may append to it at runtime, so
#: what a model is allowed to see is fixed by reading this file.
FEATURE_COLUMNS: tuple[str, ...] = (
    # price level
    "p_mkt",
    "logit_p",
    "vwap_gap",
    "mid_hl_gap",
    # momentum
    *[f"ret_{k}" for k in RETURN_LAGS],
    # volatility / spread
    *[f"vol_{w}" for w in VOL_WINDOWS],
    "hl_range",
    *[f"hl_range_mean_{w}" for w in FLOW_WINDOWS],
    # flow
    "log_volume_shares",
    "log_volume_usdc",
    "trades",
    *[f"log_volume_usdc_sum_{w}" for w in FLOW_WINDOWS],
    *[f"traded_frac_{w}" for w in FLOW_WINDOWS],
    # clock
    "candles_remaining",
    "minutes_remaining",
    "is_live",
)

#: Features that only exist if the caller joined an extra dataset in. They are
#: opt-in per frame rather than global, so one call cannot change what a later
#: unrelated call considers a feature.
OPTIONAL_FEATURE_COLUMNS: tuple[str, ...] = ("book_asym",)

#: Present on the frame but never fed to a model.
META_COLUMNS = [
    "event_slug",
    "start_ts",
    "candle_index",
    "timestamp",
    "next_open",
    "next_high",
    "next_low",
    "can_trade",
    "winner",
    "y",
]

_EPS = 1e-6


def feature_columns(df: pd.DataFrame) -> list[str]:
    """The columns a model may read from ``df``.

    Always prefer this over indexing :data:`FEATURE_COLUMNS` directly: it picks
    up optional joined features like ``book_asym`` when they are present and
    silently omits them when they are not, so the same training code works
    either way.
    """
    allowed = FEATURE_COLUMNS + OPTIONAL_FEATURE_COLUMNS
    return [c for c in allowed if c in df.columns]


def _logit(p: pd.Series) -> pd.Series:
    p = p.clip(_EPS, 1 - _EPS)
    return np.log(p / (1 - p))


def build_features(
    candles: pd.DataFrame,
    *,
    live_only: bool = True,
    drop_warmup: bool = True,
) -> pd.DataFrame:
    """Build the model-ready frame from a candle split.

    Parameters
    ----------
    candles:
        Output of ``data_loader.load_split(..., dataset="candles_15s")``.
    live_only:
        Keep only ``candle_index >= 0``, the tradable 15-minute window. The
        40 pre-open candles still feed the rolling windows before being
        dropped, so momentum at candle 0 is real rather than undefined.
    drop_warmup:
        Drop rows whose longest lookback is not yet full, so no feature is
        silently a shorter window than its name claims.
    """
    required = {"event_slug", "candle_index", "close", "high", "low", "winner"}
    missing = required - set(candles.columns)
    if missing:
        raise ValueError(f"candles frame is missing columns: {sorted(missing)}")

    df = candles.sort_values(["event_slug", "candle_index"], ignore_index=True).copy()
    n_markets_in = df.event_slug.nunique()
    g = df.groupby("event_slug", sort=False)

    # Dense output means a no-trade bucket is a doji flat at the previous close,
    # but buckets before a market's first trade are null throughout. Forward
    # fill only -- a backward fill here would import the future.
    for col in ("open", "high", "low", "close", "vwap"):
        if col in df.columns:
            df[col] = g[col].ffill()
    df = df[df.close.notna()].copy()
    g = df.groupby("event_slug", sort=False)

    # -- price level ----------------------------------------------------
    df["p_mkt"] = df.close
    df["logit_p"] = _logit(df.p_mkt)
    # vwap is null on zero-volume candles; the gap is 0 when there was no trade.
    df["vwap_gap"] = (df.vwap - df.close).fillna(0.0) if "vwap" in df.columns else 0.0
    df["mid_hl_gap"] = (df.high + df.low) / 2.0 - df.close

    # -- momentum -------------------------------------------------------
    # Differences, not log returns: p is a bounded probability, and a 0.02 move
    # means the same thing at 0.10 as at 0.50 for a fee that scales as p(1-p).
    for k in RETURN_LAGS:
        df[f"ret_{k}"] = df.close - g.close.shift(k)

    # -- volatility / spread --------------------------------------------
    r1 = df["ret_1"]
    for w in VOL_WINDOWS:
        df[f"vol_{w}"] = (
            r1.groupby(df.event_slug, sort=False)
            .rolling(w, min_periods=max(2, w // 2))
            .std()
            .reset_index(level=0, drop=True)
        )
    df["hl_range"] = df.high - df.low
    for w in FLOW_WINDOWS:
        df[f"hl_range_mean_{w}"] = (
            df.hl_range.groupby(df.event_slug, sort=False)
            .rolling(w, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )

    # -- flow ------------------------------------------------------------
    # Per-candle only. `volume` (market total) is deliberately never touched.
    for src, dst in (("volume_shares", "log_volume_shares"), ("volume_usdc", "log_volume_usdc")):
        df[dst] = np.log1p(df[src].fillna(0.0)) if src in df.columns else 0.0
    if "trades" not in df.columns:
        df["trades"] = 0.0
    df["trades"] = df.trades.fillna(0.0)
    if "has_trades" in df.columns:
        traded = df.has_trades.astype(float)
    else:
        traded = (df.trades > 0).astype(float)

    for w in FLOW_WINDOWS:
        df[f"log_volume_usdc_sum_{w}"] = (
            df.log_volume_usdc.groupby(df.event_slug, sort=False)
            .rolling(w, min_periods=1)
            .sum()
            .reset_index(level=0, drop=True)
        )
        df[f"traded_frac_{w}"] = (
            traded.groupby(df.event_slug, sort=False)
            .rolling(w, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )

    # -- clock -----------------------------------------------------------
    df["candles_remaining"] = LAST_CANDLE - df.candle_index
    df["minutes_remaining"] = df.candles_remaining / CANDLES_PER_MINUTE
    df["is_live"] = (df.candle_index >= 0).astype(float)

    # -- execution prices: the NEXT candle, never this one ---------------
    for col in ("open", "high", "low"):
        df[f"next_{col}"] = g[col].shift(-1)
    df["can_trade"] = df.next_open.notna() & (df.candle_index < LAST_CANDLE)

    # -- label -------------------------------------------------------------
    df["y"] = (df.winner == "Up").astype(int)

    if drop_warmup:
        warmup = max(max(RETURN_LAGS), max(VOL_WINDOWS))
        first = g.candle_index.transform("min")
        df = df[df.candle_index >= first + warmup]
    if live_only:
        df = df[df.candle_index >= 0]

    keep = [c for c in META_COLUMNS if c in df.columns] + list(FEATURE_COLUMNS)
    out = df[keep].reset_index(drop=True)

    # A market whose tape only starts late has no room for the warmup window and
    # falls out entirely. That is the right call -- 16-candle momentum cannot be
    # built from four candles of history -- but it is a silent cut to the sample,
    # and a silent cut reads as full coverage when someone writes up the results.
    lost = n_markets_in - out.event_slug.nunique()
    if lost:
        warnings.warn(
            f"build_features dropped {lost} of {n_markets_in} markets whose tape "
            f"starts too late to fill the {max(max(RETURN_LAGS), max(VOL_WINDOWS))}"
            "-candle warmup. Report this as a sample cut, do not treat the "
            "remaining markets as the full universe.",
            stacklevel=2,
        )
    return out


def add_book_asymmetry(
    features: pd.DataFrame,
    minute: pd.DataFrame,
    *,
    lag_candles: int = CANDLES_PER_MINUTE,
) -> pd.DataFrame:
    """Join ``Up + Down - 1`` from the minute dataset as an extra feature.

    The two legs are quoted from separate order books, so their sum deviates
    from 1.0 and the README calls that deviation real signal. It cannot come
    from the candles, which consolidate both legs into one Up-equivalent tape.

    The minute file's value stamped at minute *m* reflects the market around
    *m-1*, so it is already stale by a minute; ``lag_candles`` shifts it a
    further minute to guarantee we only ever read a value that was published
    strictly before the candle using it.

    Returns a new frame carrying ``book_asym``; pick it up downstream with
    :func:`feature_columns`, which finds it automatically.
    """
    wide = (
        minute.pivot_table(
            index=["event_slug", "minute_index"], columns="outcome", values="price"
        )
        .reset_index()
    )
    if not {"Up", "Down"} <= set(wide.columns):
        raise ValueError("minute frame must contain both Up and Down outcomes")
    wide["book_asym"] = wide["Up"] + wide["Down"] - 1.0
    wide["candle_index"] = wide.minute_index * CANDLES_PER_MINUTE + lag_candles

    out = features.merge(
        wide[["event_slug", "candle_index", "book_asym"]],
        on=["event_slug", "candle_index"],
        how="left",
    )
    out = out.sort_values(["event_slug", "candle_index"], ignore_index=True)
    # Hold the last published quote between minute boundaries; forward only.
    out["book_asym"] = out.groupby("event_slug", sort=False).book_asym.ffill().fillna(0.0)
    return out


def assert_no_lookahead(
    candles: pd.DataFrame, n_markets: int = 3, probe_indices: tuple[int, ...] = (0, 20, 45)
) -> None:
    """Prove no feature reads the future, by recomputing on truncated history.

    For a handful of markets, rebuild the features from only the candles up to
    ``c`` and check every feature value at ``c`` matches the value computed with
    the market's full history. If any feature peeked ahead -- a centred window,
    a backfill, a groupby that forgot to sort -- the two disagree.

    This is a proof by construction rather than a statistical sniff test, so it
    catches leaks that a suspiciously-good AUC would only hint at.
    """
    slugs = list(pd.unique(candles.event_slug))[:n_markets]
    full = build_features(
        candles[candles.event_slug.isin(slugs)], live_only=False, drop_warmup=False
    ).set_index(["event_slug", "candle_index"])

    for slug in slugs:
        one = candles[candles.event_slug == slug]
        for c in probe_indices:
            if (slug, c) not in full.index:
                continue
            truncated = build_features(
                one[one.candle_index <= c], live_only=False, drop_warmup=False
            ).set_index(["event_slug", "candle_index"])
            if (slug, c) not in truncated.index:
                continue
            for col in feature_columns(full.reset_index()):
                a = full.loc[(slug, c), col]
                b = truncated.loc[(slug, c), col]
                if pd.isna(a) and pd.isna(b):
                    continue
                if not np.isclose(float(a), float(b), rtol=1e-9, atol=1e-12):
                    raise AssertionError(
                        f"LOOKAHEAD in {col!r}: market {slug} candle {c} is {a!r} with "
                        f"full history but {b!r} when history stops at {c}"
                    )
