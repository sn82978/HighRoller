# Polymarket BTC Up/Down 15m — price data

Two fetchers for Polymarket's **BTC Up or Down 15m** series (gamma series id
`10192`, slug `btc-up-or-down-15m`; individual events are slugged
`btc-updown-15m-<start_ts>`):

| script | source | resolution |
|---|---|---|
| `fetch_btc_updown_15m.py` | CLOB `prices-history` | 1-minute sampled quotes |
| `fetch_btc_updown_15m_trades.py` | data-api `trades` | **15s OHLCV from every fill** |

**Prefer the candles for modeling.** The minute file is a quote sample that
[lags the real tape by ~60s](#️-the-minute-file-lags-the-real-tape-by-60s); the
candles are built from individual trades stamped at true trade time.

## Window

The pull ends at **`btc-updown-15m-1772138700`** (2026-02-26 20:45–21:00 UTC /
3:45–4:00 PM ET) and reaches 90 days back to 2025-11-28 — 8,640 consecutive
15-minute markets.

Note that the series itself is still running as of 2026-08-13; the anchor above
is a fixed point ~5.5 months in the past, so "last 3 months" here means the 90
days *ending at that event*. To pull a different window:

```bash
python data/fetch_btc_updown_15m.py --anchor-slug btc-updown-15m-<ts> --days 90
```

## Running it

```bash
python data/fetch_btc_updown_15m.py              # full 90-day pull (~15 min)
python data/fetch_btc_updown_15m.py --days 1     # quick check
python data/fetch_btc_updown_15m.py --tokens up  # Up leg only, half the requests
python data/fetch_btc_updown_15m.py --build-only # rebuild outputs from cache
```

Requires `requests`, `pandas`, `pyarrow`.

Everything is cached in `data/cache/*.jsonl` and the job is **resumable** — if it
dies partway, re-running picks up only what's missing. `--build-only` regenerates
the output files without any network calls.

Useful flags: `--pre-minutes` (pre-open book history, default 10), `--workers`
(concurrency, default 8), `--format {parquet,csv,both}`, `--days`.

## Output — `data/polymarket/`

Each dataset is a **directory of one file per UTC day** (91 days), in both
parquet and csv:

| directory | contents |
|---|---|
| `btc_updown_15m_ticks/` | raw ticks exactly as the CLOB returned them |
| `btc_updown_15m_minute/` | ticks snapped to an exact 60s grid |
| `btc_updown_15m_candles_15s/` | 15s OHLCV candles from true tick data — see below |

Days are cut on each market's `start_ts`, not on each row's own timestamp, so a
market straddling midnight (23:45–00:00, plus its pre-open tape) keeps all its
rows in one file. Pass `--partition single` for one combined file instead.

```python
import pandas as pd, glob
df = pd.concat([pd.read_parquet(f)
                for f in sorted(glob.glob("data/polymarket/btc_updown_15m_minute/*.parquet"))])
```

The tick and minute files are long-format, one row per (market, outcome, timestamp):

| column | meaning |
|---|---|
| `event_slug` | `btc-updown-15m-<start_ts>` |
| `condition_id` | on-chain condition id |
| `outcome` | `Up` or `Down` |
| `token_id` | CLOB ERC-1155 token id for that outcome |
| `start_ts` / `end_ts` | unix bounds of the 15-minute window |
| `t` / `timestamp` | observation time (unix / UTC datetime) |
| `minute_index` | minutes relative to `start_ts`; `-10..0` is pre-open, `0..15` live |
| `price` | outcome price in [0, 1] |
| `closed` | whether the market resolved |
| `winner` | resolved outcome (`Up`/`Down`), null if unresolved |
| `volume` | total market volume in USDC |

### Reading the grid file

The CLOB returns ticks *roughly* — not exactly — 60s apart (observed gaps run
44–76s, averaging 59.96s). The grid file snaps each point to the last
observation at or before it (backward as-of join), giving a clean 26 rows per
market per outcome. Grid points before a market's first tick stay null rather
than being back-filled.

`Up` and `Down` are quoted from separate order books, so they sum to ~1.0 but not
exactly (measured mean 1.0000, 1st–99th pct 0.990–1.010). The spread between them
is real signal, which is why both legs are fetched by default.

### ⚠️ The minute file lags the real tape by ~60s

Measured against the trade tape (below), the minute file's price stamped at
minute *m* best matches the market around minute *m−1* — correlation peaks at
**0.9942 at exactly 60s of lag**, falling off cleanly either side:

| lag | 0s | 15s | 30s | 45s | **60s** | 75s | 90s |
|---|---|---|---|---|---|---|---|
| corr | 0.9411 | 0.9568 | 0.9730 | 0.9880 | **0.9942** | 0.9809 | 0.9661 |

This is inherent to `prices-history`: its buckets are jittered ~60s apart, and
the grid's backward as-of fill adds staleness on top. **On a 15-minute market
that lag is 1/15 of the contract's entire life and will look like predictive
signal in a backtest.** Use the 15s candles for anything timing-sensitive.

### Resolution limits of the minute data

**One minute is the API floor.** `fidelity=0` returns exactly the same points as
`fidelity=1`; there is no sub-minute setting on `prices-history`.

**These are 1-minute sampled prices, not ticks** — one price per bucket, not
every quote change or fill.

**The last ~54s before resolution is only partially captured.** The final tick
lands short of `end_ts`, so the endgame — where these markets actually decide —
is undersampled.

---

## 15-second candles — `fetch_btc_updown_15m_trades.py`

Built from `data-api.polymarket.com/trades`, which returns **every individual
fill**, so there is no sampling floor and no lag.

```bash
python data/fetch_btc_updown_15m_trades.py                    # full pull (~20 min)
python data/fetch_btc_updown_15m_trades.py --days 1           # quick check
python data/fetch_btc_updown_15m_trades.py --bucket-seconds 5 # other widths
python data/fetch_btc_updown_15m_trades.py --build-only       # rebuild from cache
```

It reuses `data/cache/events.jsonl`, so run the price script first (or let this
one populate it).

### One consolidated tape

Both legs are quoted as an **Up-equivalent price** — a Down fill at 0.40 is an Up
fill at 0.60 — so Up and Down trades reinforce one series instead of being split
into two half-depth ones. (Measured Up+Down = 1.0000 mean, so they really are two
views of one price.)

Two notionals are tracked on purpose: `volume_usdc` sums size × the price
*actually paid* (true dollars traded), while `vwap` weights the *Up-equivalent*
price to stay consistent with OHLC.

### Columns

`event_slug`, `condition_id`, `start_ts`, `end_ts`, `candle_index`, `t`,
`timestamp`, `open`, `high`, `low`, `close`, `vwap`, `volume_shares`,
`volume_usdc`, `trades`, `has_trades`, `truncated`, `winner`, `volume`

`candle_index` runs **−40 … 59** — 100 buckets per market: 40 pre-open (10 min)
then 60 covering the live 15 minutes. Index *c* covers `[start_ts + 15c,
start_ts + 15(c+1))`.

### Reading the candles

Output is **dense** — every bucket appears. A bucket with no trades is a
zero-volume doji flat at the previous close, flagged `has_trades = False`
(~15% of buckets); `vwap` is null there since it's undefined at zero volume.
Buckets before a market's first trade are null throughout. Use `--sparse` to emit
only traded buckets.

**Expect wide intra-candle ranges** — median high−low is 0.04 on candles with ≥2
trades, p90 is 0.12. That's bid-ask bounce in a wide-spread book, not noise in
the data; consecutive fills genuinely alternate between bid and ask. If you want
a smoother price, `vwap` or `(high+low)/2` will serve better than `close`.

### `truncated` — 18 markets with a partial pre-open tape

The trades API 400s on `offset >= 15000` regardless of limit, and supports no
time filter, so 15,000 trades is a hard ceiling per market. The 18 busiest
markets in the 3 months exceed it (all >$375k volume).

For those, the **live 15-minute window is intact** — only the earliest part of
the 10-minute pre-open tape is unreachable, since paging runs newest-first. Their
live-window bucket coverage is 92.5% vs 98.2% elsewhere. They're flagged rather
than dropped, because they're the highest-volume markets in the set:

```python
df = df[~df.truncated]                       # exclude them entirely
df = df[(~df.truncated) | (df.candle_index >= 0)]   # or keep their live window
```

### Bad ticks are filtered

Prices outside [0, 1] are impossible for a binary market but do appear: this pull
found one fill priced at **4.5137**, which would have dragged its candle's low to
−3.51. Such trades are dropped at aggregation. Verified across all 860,800
candles: every price sits in [0.001, 0.999] and `low <= open,close,vwap <= high`
holds everywhere.

### Scale

8,608 markets · 860,800 candles · **44,856,900 trades** · $795.3M notional ·
91 daily files.

## How it works

1. Slots are exactly 900s apart, so every event slug is generated locally rather
   than paged out of the API (gamma caps `limit` at 100).
2. Slugs are resolved to market metadata in batches of 50 via
   `gamma-api.polymarket.com/events`.
3. `clob.polymarket.com/prices-history?fidelity=1` is fetched per outcome token,
   concurrently, with exponential backoff on 429/5xx.
4. Ticks are assembled into the raw and gridded frames.

Both APIs reject the default Python user-agent with a 403, so a browser-ish
`User-Agent` header is set on every request.
