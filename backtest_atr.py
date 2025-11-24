"""
Pure backtest script for OANDA FX strategy with brute force parameter + strategy optimization.

Key features:
- Fetches historical candles from OANDA (practice or live, via ENV)
- Caches candles in Parquet files (cache_hist/)
- Regime-aware strategy:
    * Trend: Donchian + ATR breakout + Volume confirmation + Swing detection
    * Range: Bollinger + RSI mean reversion
    * Multi-timeframe confirmation (M5 + M15 alignment, via resample if enabled)
    * Adaptive SL/TP based on volatility, with SL based on ATR multiple
    * Time-of-day liquidity filter
- Brute force over:
    * STRATEGY_MODE ∈ {FULL, TREND_ONLY, RANGE_ONLY, NO_MTF}
    * RISK_PCT, SL_ATR_MULTIPLIER, RR (TP = SL * RR), DAILY_MAX_LOSSES
- (Optional) Logs trades & equity into per-config SQLite DBs under backtest_results/
- Incrementally appends TRAIN results to summary_raw.csv to allow resume
- Produces ranked summary.csv (sorted by SCORE = return_pct / max_dd_pct)
- Train/test split:
    * Optimize on first TRAIN_FRACTION of history
    * Automatically re-run top configs on remaining data as TEST (printed)
"""

import os
import time
import sqlite3
import itertools
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any, Tuple

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

# NEW: parallelism
import multiprocessing
from joblib import Parallel, delayed

# =========================
# Config & Globals
# =========================

load_dotenv()

ENV = os.getenv("OANDA_ENV", "practice")  # "practice" | "live"
TOKEN = os.getenv("OANDA_TOKEN", "")
REVERSE_SIGNALS = os.getenv("REVERSE_SIGNALS", "0").lower() in ("1", "true", "yes", "y")

DEFAULT_INSTRUMENT = os.getenv("INSTRUMENT", "EUR_USD")
DEFAULT_GRAN = os.getenv("GRANULARITY", "M5")

# Strategy feature flags (base)
USE_VOLUME_CONFIRM_BASE = os.getenv("USE_VOLUME_CONFIRM", "1").lower() in ("1", "true", "yes")
USE_TIME_FILTER_BASE = os.getenv("USE_TIME_FILTER", "1").lower() in ("1", "true", "yes")
USE_MULTI_TF_BASE = os.getenv("USE_MULTI_TF", "0").lower() in ("1", "true", "yes")
USE_ADAPTIVE_SLTP_BASE = os.getenv("USE_ADAPTIVE_SLTP", "1").lower() in ("1", "true", "yes")
USE_SWING_DETECT_BASE = os.getenv("USE_SWING_DETECT", "1").lower() in ("1", "true", "yes")

# Disable DB writes by default
WRITE_TO_DB = os.getenv("WRITE_TO_DB", "0").lower() in ("1", "true", "yes")

BASE = (
    "https://api-fxpractice.oanda.com"
    if ENV == "practice"
    else "https://api-fxtrade.oanda.com"
)

DB_PATH = os.getenv("BACKTEST_DB_PATH", "backtest_trades.db")

START_EQUITY = float(os.getenv("START_EQUITY", "10000"))
DAYS_BACK = int(os.getenv("DAYS_BACK", "1095"))

# Approx spread in pips (round-trip) for EURUSD on M5
SPREAD_PIPS = float(os.getenv("SPREAD_PIPS", "1.0"))

# Train/test split fraction (e.g. 0.7 = first 70% train, last 30% test)
TRAIN_FRACTION = float(os.getenv("TRAIN_FRACTION", "0.7"))

# Minimum trades filter when ranking configs
MIN_TRADES = int(os.getenv("MIN_TRADES", "200"))

# Number of top configs (by score) to re-test on out-of-sample data
TOP_N_TEST = int(os.getenv("TOP_N_TEST", "20"))

# Optional: override n_jobs via ENV, else use cores-2
N_JOBS_ENV = os.getenv("N_JOBS", "").strip()

session = requests.Session()
if not TOKEN:
    raise SystemExit("Set OANDA_TOKEN in .env")
session.headers.update({"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"})

CACHE_DIR = "cache_hist"
os.makedirs(CACHE_DIR, exist_ok=True)

RESULTS_DIR = "backtest_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

SUMMARY_RAW_PATH = os.path.join(RESULTS_DIR, "summary_raw.csv")
SUMMARY_RANKED_PATH = os.path.join(RESULTS_DIR, "summary.csv")

# Chunk size for parallel brute-force (for incremental CSV writing)
CHUNK_SIZE = 20

# =========================
# Brute Force Parameter Ranges
# =========================

STRATEGY_MODES = ["FULL"]

# Risk per trade (leave full range, you can narrow later if needed)
RISK_PCT_RANGE = [0.01, 0.02, 0.005, 0.015]

# ATR-based SL multipliers (core + fractional + extended for strong trends)
SL_ATR_MULTIPLIER_RANGE = [
    1.0,
    1.25,
    1.5,
    1.75,
    2.0,
    2.25,
    2.5,
]

# RR: TP = SL * RR
RR_RANGE = [1.5, 1.75, 2.0, 2.5, 3.0]

# Max losing trades per day
DAILY_LOSSES_RANGE = [1, 2, 3]

# Hard floor + ceiling for SL in pips (for safety)
MIN_SL_PIPS = float(os.getenv("MIN_SL_PIPS", "4.0"))
MAX_SL_PIPS = float(os.getenv("MAX_SL_PIPS", "60.0"))

# (strategy_mode, risk_pct, sl_atr_mult, rr, daily_max_losses)
PARAM_CONFIGS = list(
    itertools.product(
        STRATEGY_MODES,
        RISK_PCT_RANGE,
        SL_ATR_MULTIPLIER_RANGE,
        RR_RANGE,
        DAILY_LOSSES_RANGE,
    )
)

print(f"[init] Brute force will test {len(PARAM_CONFIGS)} parameter+strategy combinations")

# =========================
# Cache helpers
# =========================

def _cache_path(instrument: str, granularity: str, days_back: int) -> str:
    fname = f"{instrument}_{granularity}_{days_back}d.parquet"
    return os.path.join(CACHE_DIR, fname)


def load_or_fetch_history(
    instrument: str,
    granularity: str,
    days_back: int = DAYS_BACK,
) -> pd.DataFrame:
    """Load from Parquet cache or fetch from OANDA."""
    path = _cache_path(instrument, granularity, days_back)

    if os.path.exists(path):
        print(f"[cache] loading {path}")
        df = pd.read_parquet(path)
        print(f"[cache] loaded {len(df)} candles from cache")
        return df

    print(f"[cache] no cache found, fetching from OANDA...")
    df = fetch_history_oanda(instrument, granularity, days_back=days_back)

    print(f"[cache] saving history -> {path}")
    df.to_parquet(path, index=False)
    return df


# =========================
# SQLite helpers
# =========================

def db_connect(db_path: str = DB_PATH):
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


def db_init(db_path: str = DB_PATH):
    """Create tables for backtest logs (idempotent)."""
    if not WRITE_TO_DB:
        return

    conn = db_connect(db_path)
    cur = conn.cursor()

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS trades (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          open_ts_utc   TEXT NOT NULL,
          close_ts_utc  TEXT NOT NULL,
          instrument    TEXT NOT NULL,
          granularity   TEXT NOT NULL,
          side          TEXT NOT NULL,
          units         INTEGER NOT NULL,
          entry_price   REAL NOT NULL,
          exit_price    REAL NOT NULL,
          sl_price      REAL NOT NULL,
          tp_price      REAL NOT NULL,
          pl            REAL NOT NULL,
          result        TEXT NOT NULL,
          risk_pct      REAL,
          sl_pips       REAL,
          tp_pips       REAL
        );
    """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS equity (
          ts_utc      TEXT NOT NULL,
          equity      REAL NOT NULL,
          granularity TEXT NOT NULL,
          risk_pct    REAL,
          PRIMARY KEY (ts_utc, risk_pct)
        );
    """
    )

    conn.commit()
    conn.close()


def log_equity(
    equity: float,
    gran: str,
    risk_pct: float,
    ts_utc: str,
    db_path: str = DB_PATH,
):
    """Log equity using bar timestamp."""
    if not WRITE_TO_DB:
        return

    conn = db_connect(db_path)
    conn.execute(
        """
        INSERT OR REPLACE INTO equity (ts_utc, equity, granularity, risk_pct)
        VALUES (?, ?, ?, ?);
    """,
        (ts_utc, equity, gran, risk_pct),
    )
    conn.commit()
    conn.close()


def log_trade_row(
    open_ts: str,
    close_ts: str,
    instrument: str,
    gran: str,
    side: str,
    units: int,
    entry: float,
    exit_px: float,
    sl: float,
    tp: float,
    pl: float,
    result: str,
    risk_pct: float,
    sl_pips: float,
    tp_pips: float,
    db_path: str = DB_PATH,
):
    if not WRITE_TO_DB:
        return

    conn = db_connect(db_path)
    conn.execute(
        """
        INSERT INTO trades (
            open_ts_utc, close_ts_utc, instrument, granularity,
            side, units, entry_price, exit_price,
            sl_price, tp_price, pl, result, risk_pct, sl_pips, tp_pips
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
    """,
        (
            open_ts,
            close_ts,
            instrument,
            gran,
            side,
            units,
            entry,
            exit_px,
            sl,
            tp,
            pl,
            result,
            risk_pct,
            sl_pips,
            tp_pips,
        ),
    )
    conn.commit()
    conn.close()


# =========================
# Utility & pricing
# =========================

def pip_value(instrument: str) -> float:
    """Rough pip size: 0.01 for JPY crosses, 0.0001 otherwise."""
    return 0.01 if instrument.endswith("_JPY") else 0.0001


def _price_decimals(instrument: str) -> int:
    """OANDA typical quoting: 3 decimals for JPY, 5 otherwise."""
    return 3 if instrument.endswith("_JPY") else 5


def price_add_pips(price: float, pips: float, side: str, instrument: str) -> float:
    """Add/subtract pips from price depending on side."""
    delta = pips * pip_value(instrument)
    dec = _price_decimals(instrument)
    return round(price + (delta if side.upper() == "LONG" else -delta), dec)


# =========================
# OANDA history fetcher
# =========================

def _max_chunk_days_for_gran(granularity: str) -> int:
    """OANDA max 5000 candles per request. Estimate safe days-per-chunk."""
    granularity = granularity.upper()
    candles_per_day_map = {
        "M1": 24 * 60,
        "M5": 24 * 12,
        "M15": 24 * 4,
        "M30": 24 * 2,
        "H1": 24,
        "H4": 6,
        "D": 1,
    }
    candles_per_day = candles_per_day_map.get(granularity, 500)
    max_days = max(1, 5000 // candles_per_day)
    return max_days


def fetch_history_oanda(
    instrument: str,
    granularity: str,
    days_back: int = DAYS_BACK,
) -> pd.DataFrame:
    """Pull historical candles from OANDA with automatic chunking."""
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days_back)

    all_rows = []
    max_chunk_days = _max_chunk_days_for_gran(granularity)

    print(
        f"[hist] fetching ~{days_back} days of {instrument} {granularity} "
        f"with chunk size {max_chunk_days} day(s)..."
    )

    cur_start = start
    while cur_start < end:
        cur_end = min(cur_start + timedelta(days=max_chunk_days), end)

        params = {
            "granularity": granularity,
            "from": cur_start.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "to": cur_end.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "price": "M",
        }
        url = f"{BASE}/v3/instruments/{instrument}/candles"

        r = session.get(url, params=params, timeout=30)

        if not r.ok:
            print("[hist:error]", r.status_code, r.text)
            r.raise_for_status()

        js = r.json()
        candles = js.get("candles", [])

        print(f"[hist] {cur_start.date()} → {cur_end.date()} got {len(candles)} candles")

        for c in candles:
            if not c.get("complete", False):
                continue
            all_rows.append(
                {
                    "t": c["time"],
                    "o": float(c["mid"]["o"]),
                    "h": float(c["mid"]["h"]),
                    "l": float(c["mid"]["l"]),
                    "c": float(c["mid"]["c"]),
                    "v": int(c["volume"]),
                }
            )

        cur_start = cur_end
        time.sleep(0.2)

    if not all_rows:
        raise RuntimeError("No candles returned from OANDA.")

    df = (
        pd.DataFrame(all_rows)
        .drop_duplicates(subset=["t"])
        .sort_values("t")
        .reset_index(drop=True)
    )
    print(f"[hist] total candles: {len(df)}")
    return df


# =========================
# Indicators & Enhancements
# =========================

def atr(df: pd.DataFrame, period=14) -> pd.Series:
    """Average True Range (simple mean)."""
    h, l, c = df["h"].values, df["l"].values, df["c"].values
    prev_c = np.r_[np.nan, c[:-1]]
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))
    return pd.Series(tr, index=df.index).rolling(period).mean()


def donchian(df: pd.DataFrame, n=20) -> Tuple[pd.Series, pd.Series]:
    """Upper/lower channel extremes over n bars."""
    return df["h"].rolling(n).max(), df["l"].rolling(n).min()


def bollinger(df: pd.DataFrame, n=20, k=2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Bollinger bands on close."""
    ma = df["c"].rolling(n).mean()
    sd = df["c"].rolling(n).std(ddof=0)
    return ma, ma + k * sd, ma - k * sd


def rsi(df: pd.DataFrame, n=14) -> pd.Series:
    """Classic RSI (simple average)."""
    delta = df["c"].diff()
    up = (delta.clip(lower=0)).rolling(n).mean()
    down = (-delta.clip(upper=0)).rolling(n).mean()
    rs = up / (down.replace(0, np.nan))
    return 100 - (100 / (1 + rs))


def atr_percentile(current_atr: float, atr_series: pd.Series, lookback=500) -> float:
    """Where current ATR ranks vs recent ATR distribution (0..100%)."""
    base = atr_series.tail(min(len(atr_series), lookback)).dropna()
    if len(base) < 10:
        return 50.0
    return float((base <= current_atr).mean() * 100.0)


# ========== Enhancement 1: Volume Confirmation ==========

def volume_spike(df: pd.DataFrame, n=20, mult=1.5) -> bool:
    """Is current volume significantly above average?"""
    if len(df) < n:
        return True
    vol_ma = df["v"].rolling(n).mean()
    current_vol = float(df["v"].iloc[-1])
    vol_ma_val = float(vol_ma.iloc[-1])
    if vol_ma_val == 0:
        return True
    return current_vol > vol_ma_val * mult


# ========== Enhancement 2: Time-of-Day Liquidity Filter ==========

def is_liquid_hour(ts: pd.Timestamp) -> bool:
    """Avoid low-liquidity sessions (Tokyo/Sydney close + late Friday)."""
    hour = ts.hour
    weekday = ts.weekday()

    # Late Friday
    if weekday == 4 and hour >= 19:
        return False

    # Low-liquid Asia-only window (approx)
    if 2 <= hour < 8:
        return False

    return True


# ========== Enhancement 3: Swing Detection ==========

def swing_setup(df: pd.DataFrame, n=3) -> Tuple[bool, bool]:
    """
    up_swing: recent candles making higher lows (uptrend structure)
    dn_swing: recent candles making lower highs (downtrend structure)
    """
    if len(df) < n + 1:
        return False, False

    lows = df["l"].tail(n).values
    highs = df["h"].tail(n).values

    up_swing = all(lows[i] > lows[i - 1] for i in range(1, len(lows)))
    dn_swing = all(highs[i] < highs[i - 1] for i in range(1, len(highs)))

    return up_swing, dn_swing


# ========== Enhancement 4: Adaptive SL/TP ==========

def adaptive_sl_tp(
    atr_pct: float,
    sl_base: float,
    tp_base: float,
    use_adaptive: bool,
) -> Tuple[float, float]:
    """
    Adjust SL/TP based on volatility regime:
    - High volatility (atr_pct > 75): wider stops
    - Low volatility (atr_pct < 25): tighter stops
    - Mid regime: use base values
    """
    if not use_adaptive:
        return sl_base, tp_base

    if atr_pct > 75:
        sl = sl_base * 1.5
        tp = tp_base * 1.5
    elif atr_pct < 25:
        sl = sl_base * 0.7
        tp = tp_base * 0.7
    else:
        sl = sl_base
        tp = tp_base

    return sl, tp


# ========== Enhancement 5: Multi-Timeframe Confirmation ==========

def get_multi_tf_signal(signal_m5: str, signal_m15: str) -> str:
    """Only take trades if M15 + M5 align on direction."""
    if signal_m5 == "FLAT" or signal_m15 == "FLAT":
        return "FLAT"
    if signal_m5 == signal_m15:
        return signal_m5
    return "FLAT"


def _resample_to_tf(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """
    Resample current df to higher TF (e.g. '15T') for multi-timeframe signal.
    Keeps same column structure: t, o, h, l, c, v
    """
    if df.empty:
        return df.copy()

    tmp = df.copy()
    ts = pd.to_datetime(tmp["t"], utc=True)
    tmp = tmp.set_index(ts)

    ohlc = {
        "o": "first",
        "h": "max",
        "l": "min",
        "c": "last",
        "v": "sum",
    }
    out = tmp.resample(rule).agg(ohlc).dropna().reset_index(names="ts")
    out["t"] = out["ts"].dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    out = out[["t", "o", "h", "l", "c", "v"]]
    return out


# =========================
# Core Strategy Signal (single TF)
# =========================

def _compute_signal_single_tf(
    df: pd.DataFrame,
    instrument: str,
    sl_atr_mult: float,
    rr: float,
    strategy_mode: str,
    use_volume_confirm: bool,
    use_time_filter: bool,
    use_adaptive_sltp: bool,
    use_swing_detect: bool,
) -> Tuple[str, Dict[str, Any]]:
    """
    Single timeframe signal logic (trend/range, adaptive SL/TP, filters).
    SL is based on ATR multiple, then clipped by MIN_SL_PIPS/MAX_SL_PIPS.
    Does NOT do multi-TF alignment.
    """
    if len(df) < 100:
        return "FLAT", {"why": "not_enough_bars"}

    df = df.copy()
    df["ATR14"] = atr(df, 14)
    dc_hi, dc_lo = donchian(df, 20)
    ma20, bb_up, bb_lo = bollinger(df, 20, 2.0)
    df["RSI14"] = rsi(df, 14)

    c = df["c"].iloc[-1]
    prev_c = df["c"].iloc[-2]
    atr_now = float(df["ATR14"].iloc[-1])
    atr_pct = atr_percentile(atr_now, df["ATR14"], 500)

    ts = pd.to_datetime(df["t"].iloc[-1], utc=True)
    if use_time_filter and not is_liquid_hour(ts):
        return "FLAT", {"why": "illiquid_hour"}

    # ATR in pips
    atr_pips = atr_now / pip_value(instrument) if atr_now > 0 else 0.0

    trend_regime = atr_pct >= 70
    range_regime = atr_pct <= 30

    # --- Strategy-mode gating ---
    use_trend_block = strategy_mode in ("FULL", "TREND_ONLY", "NO_MTF")
    use_range_block = strategy_mode in ("FULL", "RANGE_ONLY", "NO_MTF")

    # TREND_ONLY: ignore range setups entirely
    if strategy_mode == "TREND_ONLY" and not trend_regime:
        return "FLAT", {"why": "not_trend_regime"}

    # RANGE_ONLY: ignore trend setups entirely
    if strategy_mode == "RANGE_ONLY" and not range_regime:
        return "FLAT", {"why": "not_range_regime"}

    # Helper: build SL/TP in pips from ATR
    def build_sl_tp(atr_pips_val: float) -> Tuple[float, float]:
        if atr_pips_val <= 0:
            return 0.0, 0.0
        sl_pips_raw = atr_pips_val * sl_atr_mult
        sl_pips = max(MIN_SL_PIPS, min(sl_pips_raw, MAX_SL_PIPS))
        tp_pips = sl_pips * rr
        sl_pips, tp_pips = adaptive_sl_tp(atr_pct, sl_pips, tp_pips, use_adaptive_sltp)
        return sl_pips, tp_pips

    # --- Trend regime: Donchian breakout ---
    if use_trend_block and trend_regime:
        up_break = (c > dc_hi.iloc[-2]) and (prev_c <= dc_hi.iloc[-3])
        dn_break = (c < dc_lo.iloc[-2]) and (prev_c >= dc_lo.iloc[-3])

        if up_break:
            if use_volume_confirm and not volume_spike(df, 20, 1.5):
                return "FLAT", {"why": "no_volume_confirm"}

            if use_swing_detect:
                up_swing, _ = swing_setup(df, 3)
                if not up_swing:
                    return "FLAT", {"why": "no_upswing"}

            sl_pips, tp_pips = build_sl_tp(atr_pips)
            if sl_pips <= 0 or tp_pips <= 0:
                return "FLAT", {"why": "invalid_sl_tp"}
            return "LONG", {"mode": "trend", "sl_pips": sl_pips, "tp_pips": tp_pips}

        if dn_break:
            if use_volume_confirm and not volume_spike(df, 20, 1.5):
                return "FLAT", {"why": "no_volume_confirm"}

            if use_swing_detect:
                _, dn_swing = swing_setup(df, 3)
                if not dn_swing:
                    return "FLAT", {"why": "no_dnswing"}

            sl_pips, tp_pips = build_sl_tp(atr_pips)
            if sl_pips <= 0 or tp_pips <= 0:
                return "FLAT", {"why": "invalid_sl_tp"}
            return "SHORT", {"mode": "trend", "sl_pips": sl_pips, "tp_pips": tp_pips}

    # --- Range regime: Bollinger + RSI ---
    if use_range_block and range_regime:
        rsi_val = df["RSI14"].iloc[-1]

        # Long mean-reversion
        if (c < bb_lo.iloc[-1]) and (rsi_val < 30):
            sl_pips, tp_pips = build_sl_tp(atr_pips)
            if sl_pips <= 0 or tp_pips <= 0:
                return "FLAT", {"why": "invalid_sl_tp"}
            return "LONG", {"mode": "range", "sl_pips": sl_pips, "tp_pips": tp_pips}

        # Short mean-reversion
        if (c > bb_up.iloc[-1]) and (rsi_val > 70):
            sl_pips, tp_pips = build_sl_tp(atr_pips)
            if sl_pips <= 0 or tp_pips <= 0:
                return "FLAT", {"why": "invalid_sl_tp"}
            return "SHORT", {"mode": "range", "sl_pips": sl_pips, "tp_pips": tp_pips}

    return "FLAT", {"why": "no_setup"}


def compute_signal(
    df: pd.DataFrame,
    instrument: str,
    sl_atr_mult: float,
    rr: float,
    strategy_mode: str,
) -> Tuple[str, Dict[str, Any]]:
    """
    Regime-aware logic with all enhancements + optional multi-timeframe confirmation.
    strategy_mode: "FULL", "TREND_ONLY", "RANGE_ONLY", "NO_MTF"
    """
    use_volume_confirm = USE_VOLUME_CONFIRM_BASE
    use_time_filter = USE_TIME_FILTER_BASE
    use_adaptive_sltp = USE_ADAPTIVE_SLTP_BASE
    use_swing_detect = USE_SWING_DETECT_BASE

    # NO_MTF disables multi-TF even if env says yes
    use_multi_tf = USE_MULTI_TF_BASE and (strategy_mode != "NO_MTF")

    # Base signal on current timeframe (e.g. M5)
    signal_main, meta_main = _compute_signal_single_tf(
        df,
        instrument,
        sl_atr_mult,
        rr,
        strategy_mode,
        use_volume_confirm,
        use_time_filter,
        use_adaptive_sltp,
        use_swing_detect,
    )

    # If multi-TF disabled or no directional signal, we're done
    if not use_multi_tf or signal_main in ("FLAT",):
        return signal_main, meta_main

    # Multi-TF: resample window to M15 and compute a second signal
    try:
        df_m15 = _resample_to_tf(df, "15T")
    except Exception as e:
        return signal_main, {**meta_main, "mtf_error": str(e)}

    if len(df_m15) < 100:
        return "FLAT", {"why": "not_enough_mtf_bars"}

    signal_m15, meta_m15 = _compute_signal_single_tf(
        df_m15,
        instrument,
        sl_atr_mult,
        rr,
        strategy_mode,
        use_volume_confirm,
        use_time_filter,
        use_adaptive_sltp,
        use_swing_detect,
    )

    final_signal = get_multi_tf_signal(signal_main, signal_m15)

    if final_signal == "FLAT":
        return "FLAT", {
            "why": "multi_tf_mismatch",
            "m5_signal": signal_main,
            "m15_signal": signal_m15,
        }

    # Use SL/TP from main TF meta
    return final_signal, meta_main


# =========================
# Sizing
# =========================

def position_size(nav: float, sl_pips: float, instrument: str, risk_pct: float) -> int:
    """Position sizing based on risk per trade."""
    risk_amount = nav * risk_pct
    if sl_pips <= 0:
        return 0
    units = risk_amount / (sl_pips * pip_value(instrument))
    return max(1, int(units))


# =========================
# Backtest core
# =========================

def simulate_backtest(
    df: pd.DataFrame,
    instrument: str,
    gran: str,
    start_equity: float,
    risk_pct: float,
    sl_atr_mult: float,
    rr: float,
    daily_max_losses: int,
    strategy_mode: str,
    config_name: str,
) -> Dict[str, Any]:
    """Run bar-by-bar backtest with given parameters and strategy mode."""

    # Create unique DB for this config (or disable via WRITE_TO_DB)
    if WRITE_TO_DB:
        safe_name = config_name.replace(".", "_").replace(" ", "_")
        db_path = os.path.join(RESULTS_DIR, f"{safe_name}.db")
        db_init(db_path)
    else:
        db_path = ""

    equity = start_equity
    equity_peak = start_equity
    max_dd_pct = 0.0

    position: Optional[Dict[str, Any]] = None

    trade_count = 0
    win_count = 0
    loss_count = 0
    total_pl = 0.0

    # Daily loss-limit tracking
    current_day = None
    daily_loss_count = 0
    daily_limit_hit = False  # stop new entries for the day once hit

    # For reporting: average SL/TP pips encountered (approx)
    sl_pips_sum = 0.0
    tp_pips_sum = 0.0
    sl_tp_count = 0

    for i in range(100, len(df)):
        bar = df.iloc[i]
        t_bar = pd.to_datetime(bar["t"], utc=True)
        o, h, l, c = float(bar["o"]), float(bar["h"]), float(bar["l"]), float(bar["c"])

        # --- Handle new day boundaries for daily loss logic ---
        day = t_bar.date()
        if current_day is None or day != current_day:
            current_day = day
            daily_loss_count = 0
            daily_limit_hit = False

        # 1) Manage open position
        if position is not None:
            side = position["side"]
            sl = position["sl"]
            tp = position["tp"]

            exit_price = None

            # Explicit worst-case bar logic if both SL and TP inside same bar
            if side == "LONG":
                sl_hit = l <= sl
                tp_hit = h >= tp
                if sl_hit and tp_hit:
                    # worst-case assumption: SL first
                    exit_price = sl
                elif sl_hit:
                    exit_price = sl
                elif tp_hit:
                    exit_price = tp
            else:  # SHORT
                sl_hit = h >= sl
                tp_hit = l <= tp
                if sl_hit and tp_hit:
                    exit_price = sl  # worst-case: SL first
                elif sl_hit:
                    exit_price = sl
                elif tp_hit:
                    exit_price = tp

            if exit_price is not None:
                units = position["units"]
                entry = position["entry"]
                side_mult = 1 if side == "LONG" else -1

                # P&L including spread cost (round-trip)
                spread_cost = SPREAD_PIPS * pip_value(instrument) * units
                pl = (exit_price - entry) * units * side_mult - spread_cost

                equity += pl
                total_pl += pl

                result = "WIN" if pl > 0 else "LOSS" if pl < 0 else "EVEN"

                log_trade_row(
                    open_ts=position["open_ts"],
                    close_ts=t_bar.isoformat(),
                    instrument=instrument,
                    gran=gran,
                    side=side,
                    units=units,
                    entry=entry,
                    exit_px=exit_price,
                    sl=sl,
                    tp=tp,
                    pl=pl,
                    result=result,
                    risk_pct=risk_pct,
                    sl_pips=position["sl_pips_pips"],
                    tp_pips=position["tp_pips_pips"],
                    db_path=db_path,
                )

                if result == "WIN":
                    win_count += 1
                elif result == "LOSS":
                    loss_count += 1
                    daily_loss_count += 1

                position = None

        # 2) Check daily loss limit
        if not daily_limit_hit and daily_loss_count >= daily_max_losses:
            daily_limit_hit = True

        # 3) Track max drawdown
        equity_peak = max(equity_peak, equity)
        if equity_peak > 0:
            dd_pct = (equity_peak - equity) / equity_peak * 100.0
            max_dd_pct = max(max_dd_pct, dd_pct)

        # 4) Log equity using bar time
        log_equity(equity, gran, risk_pct, ts_utc=t_bar.isoformat(), db_path=db_path)

        # 5) Check for new signal (if no open position & daily loss limit not hit)
        if position is None and not daily_limit_hit:
            window = df.iloc[: i + 1].copy()
            signal, meta = compute_signal(window, instrument, sl_atr_mult, rr, strategy_mode)
            if signal not in ("LONG", "SHORT"):
                continue

            if REVERSE_SIGNALS:
                actual_side = "SHORT" if signal == "LONG" else "LONG"
            else:
                actual_side = signal

            sl_pips_calc = float(meta.get("sl_pips", 0.0))
            tp_pips_calc = float(meta.get("tp_pips", 0.0))
            if sl_pips_calc <= 0 or tp_pips_calc <= 0:
                continue

            units = position_size(equity, sl_pips_calc, instrument, risk_pct)
            if units <= 0:
                continue

            entry = c
            sl_price = price_add_pips(
                entry,
                sl_pips_calc,
                "SHORT" if actual_side == "LONG" else "LONG",
                instrument,
            )
            tp_price = price_add_pips(entry, tp_pips_calc, actual_side, instrument)

            position = {
                "open_ts": t_bar.isoformat(),
                "side": actual_side,
                "units": units,
                "entry": entry,
                "sl": sl_price,
                "tp": tp_price,
                "sl_pips_pips": sl_pips_calc,
                "tp_pips_pips": tp_pips_calc,
            }

            trade_count += 1
            sl_pips_sum += sl_pips_calc
            tp_pips_sum += tp_pips_calc
            sl_tp_count += 1

    # 6) Close any remaining open position at last bar close
    if position is not None:
        last_bar = df.iloc[-1]
        t_last = pd.to_datetime(last_bar["t"], utc=True)
        c_last = float(last_bar["c"])
        side = position["side"]
        units = position["units"]
        entry = position["entry"]

        side_mult = 1 if side == "LONG" else -1
        spread_cost = SPREAD_PIPS * pip_value(instrument) * units
        pl = (c_last - entry) * units * side_mult - spread_cost

        equity += pl
        total_pl += pl
        result = "WIN" if pl > 0 else "LOSS" if pl < 0 else "EVEN"

        log_trade_row(
            open_ts=position["open_ts"],
            close_ts=t_last.isoformat(),
            instrument=instrument,
            gran=gran,
            side=side,
            units=units,
            entry=entry,
            exit_px=c_last,
            sl=position["sl"],
            tp=position["tp"],
            pl=pl,
            result=result,
            risk_pct=risk_pct,
            sl_pips=position["sl_pips_pips"],
            tp_pips=position["tp_pips_pips"],
            db_path=db_path,
        )

        if result == "WIN":
            win_count += 1
        elif result == "LOSS":
            loss_count += 1

        # final equity already accounted

    win_rate = (win_count / trade_count * 100) if trade_count > 0 else 0.0
    return_pct = ((equity - start_equity) / start_equity * 100) if start_equity > 0 else 0.0
    pl_per_trade = (total_pl / trade_count) if trade_count > 0 else 0.0

    avg_sl_pips = (sl_pips_sum / sl_tp_count) if sl_tp_count > 0 else 0.0
    avg_tp_pips = (tp_pips_sum / sl_tp_count) if sl_tp_count > 0 else 0.0

    # Simple robustness score: return / max_dd
    if max_dd_pct > 0:
        score = return_pct / max_dd_pct
    else:
        score = return_pct  # no drawdown case

    return {
        "strategy_mode": strategy_mode,
        "config": config_name,
        "total_trades": trade_count,
        "wins": win_count,
        "losses": loss_count,
        "win_rate_pct": round(win_rate, 2),
        "total_pl": round(total_pl, 2),
        "final_equity": round(equity, 2),
        "return_pct": round(return_pct, 2),
        "max_dd_pct": round(max_dd_pct, 2),
        "pl_per_trade": round(pl_per_trade, 4),
        "score": round(score, 4),
        "db_path": db_path,
        "risk_pct": risk_pct,
        "sl_atr_mult": sl_atr_mult,
        "rr": rr,
        "avg_sl_pips": round(avg_sl_pips, 2),
        "avg_tp_pips": round(avg_tp_pips, 2),
        "daily_max_losses": daily_max_losses,
    }


# =========================
# Parallel worker for TRAIN
# =========================

def run_one_train_config(args):
    (
        strategy_mode,
        risk_pct,
        sl_atr_mult,
        rr,
        daily_max_losses,
        instr,
        gran,
        start_equity,
        df_train,
    ) = args

    config_name = (
        f"{strategy_mode}_R{risk_pct:.2f}_ATR{sl_atr_mult}_RR{rr}_DL{daily_max_losses}"
    )
    print(f"[TRAIN] {config_name} ...", flush=True)

    try:
        result = simulate_backtest(
            df_train,
            instrument=instr,
            gran=gran,
            start_equity=start_equity,
            risk_pct=risk_pct,
            sl_atr_mult=sl_atr_mult,
            rr=rr,
            daily_max_losses=daily_max_losses,
            strategy_mode=strategy_mode,
            config_name=config_name,
        )
        return result
    except Exception as e:
        print(f"[ERROR] TRAIN {config_name}: {e}")
        return None


# =========================
# Helper: chunking
# =========================

def chunked(iterable, size):
    """Yield successive chunks of length `size` from `iterable`."""
    for i in range(0, len(iterable), size):
        yield iterable[i : i + size]


# =========================
# Main
# =========================

if __name__ == "__main__":
    instr = DEFAULT_INSTRUMENT
    gran = DEFAULT_GRAN

    print(f"\n[main] Loading historical data: {instr} {gran}...")
    df_hist = load_or_fetch_history(instr, gran, days_back=DAYS_BACK)

    if len(df_hist) < 200:
        raise SystemExit("Not enough candles for train/test.")

    split_idx = int(len(df_hist) * TRAIN_FRACTION)
    df_train = df_hist.iloc[:split_idx].reset_index(drop=True)
    df_test = df_hist.iloc[split_idx:].reset_index(drop=True)

    print(f"[main] Train size: {len(df_train)} bars, Test size: {len(df_test)} bars")

    # --- Load existing partial results to enable resume (TRAIN phase) ---
    done_configs = set()
    if os.path.exists(SUMMARY_RAW_PATH):
        try:
            existing = pd.read_csv(SUMMARY_RAW_PATH)
            if "config" in existing.columns:
                done_configs = set(existing["config"].astype(str))
                print(f"[resume] Found {len(done_configs)} completed TRAIN configs in summary_raw.csv")
        except Exception as e:
            print(f"[resume] Could not read existing summary_raw.csv: {e}")
            done_configs = set()

    print(f"\n[main] Starting TRAIN brute force with {len(PARAM_CONFIGS)} configurations...\n")

    # Build task list for joblib (only configs not done)
    tasks = []
    for (strategy_mode, risk_pct, sl_atr_mult, rr, daily_max_losses) in PARAM_CONFIGS:
        config_name = (
            f"{strategy_mode}_R{risk_pct:.2f}_ATR{sl_atr_mult}_RR{rr}_DL{daily_max_losses}"
        )
        if config_name in done_configs:
            print(f"[skip] Already completed TRAIN: {config_name}")
            continue

        tasks.append(
            (
                strategy_mode,
                risk_pct,
                sl_atr_mult,
                rr,
                daily_max_losses,
                instr,
                gran,
                START_EQUITY,
                df_train,
            )
        )

    if not tasks:
        print("[main] No new TRAIN configs to run.")
        run_results = []
    else:
        CORES = multiprocessing.cpu_count()
        if N_JOBS_ENV:
            try:
                N_JOBS = int(N_JOBS_ENV)
            except ValueError:
                N_JOBS = max(1, CORES - 2)
        else:
            N_JOBS = max(1, CORES - 2)

        print(f"[main] Using {N_JOBS} parallel workers out of {CORES} cores")
        print(f"[main] Will run {len(tasks)} TRAIN configs in parallel (chunk size = {CHUNK_SIZE})\n")

        run_results = []
        total_tasks = len(tasks)
        total_chunks = (total_tasks + CHUNK_SIZE - 1) // CHUNK_SIZE

        # Run TRAIN configs in parallel, chunked, and append to summary_raw.csv incrementally
        for chunk_idx, task_chunk in enumerate(chunked(tasks, CHUNK_SIZE), start=1):
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{now}] Running TRAIN chunk {chunk_idx}/{total_chunks} with {len(task_chunk)} configs")


            results = Parallel(n_jobs=N_JOBS, backend="loky", verbose=10)(
                delayed(run_one_train_config)(args) for args in task_chunk
            )

            for result in results:
                if result is None:
                    continue

                run_results.append(result)

                # Append this result to summary_raw.csv (single-writer, main process only)
                df_row = pd.DataFrame([result])
                write_header = not os.path.exists(SUMMARY_RAW_PATH)
                df_row.to_csv(
                    SUMMARY_RAW_PATH,
                    mode="a",
                    index=False,
                    header=write_header,
                )

                print(
                    f"[TRAIN DONE] {result['config']}: "
                    f"{result['total_trades']} trades | {result['win_rate_pct']}% WR | "
                    f"{result['return_pct']}% return | score={result['score']}"
                )

    # === Build ranked TRAIN summary from summary_raw.csv (all runs, old+new) ===
    if os.path.exists(SUMMARY_RAW_PATH):
        print("\n" + "=" * 120)
        print("TRAIN BRUTE FORCE SUMMARY (ALL COMPLETED CONFIGS)")
        print("=" * 120)

        df_all = pd.read_csv(SUMMARY_RAW_PATH)

        # Drop duplicate configs, keep the last occurrence
        df_all = df_all.drop_duplicates(subset=["config"], keep="last").reset_index(drop=True)

        # Filter: minimum trades + positive return
        df_all = df_all[df_all["total_trades"] >= MIN_TRADES].copy()
        df_all = df_all[df_all["return_pct"] > 0].copy()

        if df_all.empty:
            print("[summary] No configs meet MIN_TRADES and positive return filter.")
        else:
            # Recompute score if missing
            if "score" not in df_all.columns or df_all["score"].isna().any():
                max_dd = df_all["max_dd_pct"].replace(0, np.nan)
                df_all["score"] = df_all["return_pct"] / max_dd
                df_all["score"] = df_all["score"].fillna(df_all["return_pct"])

            # Rank by robustness score
            df_all = df_all.sort_values("score", ascending=False).reset_index(drop=True)
            df_all.insert(0, "rank", df_all.index + 1)

            print(
                df_all[
                    [
                        "rank",
                        "strategy_mode",
                        "config",
                        "total_trades",
                        "wins",
                        "losses",
                        "win_rate_pct",
                        "total_pl",
                        "return_pct",
                        "max_dd_pct",
                        "score",
                        "risk_pct",
                        "sl_atr_mult",
                        "rr",
                        "daily_max_losses",
                        "avg_sl_pips",
                        "avg_tp_pips",
                    ]
                ].to_string(index=False)
            )

            # Save ranked summary
            df_all.to_csv(SUMMARY_RANKED_PATH, index=False)
            print(f"\n[saved] ranked TRAIN summary -> {SUMMARY_RANKED_PATH}")

            # Top 3 (TRAIN)
            print("\n" + "=" * 120)
            print("TOP 3 TRAIN CONFIGURATIONS (by score)")
            print("=" * 120)
            for _, row in df_all.head(3).iterrows():
                print(f"\n#{int(row['rank'])}: {row['config']} ({row['strategy_mode']})")
                print(
                    f"    Trades: {row['total_trades']} | Win Rate: {row['win_rate_pct']}% | "
                    f"Return: {row['return_pct']}% | Max DD: {row['max_dd_pct']}% | Score: {row['score']}"
                )
                print(
                    f"    ATR SL mult: {row['sl_atr_mult']} | RR: {row['rr']} "
                    f"| Avg SL: {row['avg_sl_pips']} pips | Avg TP: {row['avg_tp_pips']} pips"
                )
                print(f"    DB: {row['db_path']}")

            # === Re-test top N configs on TEST set ===
            print("\n" + "=" * 120)
            print(f"OUT-OF-SAMPLE TEST on last {(1-TRAIN_FRACTION)*100:.1f}% of data")
            print("=" * 120)

            top_for_test = df_all.head(min(TOP_N_TEST, len(df_all))).copy()

            test_results = []
            for _, row in top_for_test.iterrows():
                # Recover parameters
                strategy_mode = row["strategy_mode"]
                risk_pct = float(row["risk_pct"])
                sl_atr_mult = float(row["sl_atr_mult"])
                rr = float(row["rr"])
                daily_max_losses = int(row["daily_max_losses"])
                base_config_name = str(row["config"])
                test_config_name = base_config_name + "_TEST"

                print(f"\n[TEST] {test_config_name} ...", end=" ")

                try:
                    res_test = simulate_backtest(
                        df_test,
                        instrument=instr,
                        gran=gran,
                        start_equity=START_EQUITY,
                        risk_pct=risk_pct,
                        sl_atr_mult=sl_atr_mult,
                        rr=rr,
                        daily_max_losses=daily_max_losses,
                        strategy_mode=strategy_mode,
                        config_name=test_config_name,
                    )
                    # attach original train config name so we can merge later
                    res_test["train_config"] = base_config_name

                    test_results.append(res_test)
                    print(
                        f"✓ trades={res_test['total_trades']} | WR={res_test['win_rate_pct']}% | "
                        f"ret={res_test['return_pct']}% | DD={res_test['max_dd_pct']}% | score={res_test['score']}"
                    )
                except Exception as e:
                    print(f"✗ Error: {e}")

            if test_results:
                df_test_res = pd.DataFrame(test_results)

                print("\n" + "=" * 120)
                print("TEST RESULTS (TOP CONFIGS, OUT-OF-SAMPLE)")
                print("=" * 120)

                # Sort TEST results by score (this is score_test)
                df_test_res = df_test_res.sort_values("score", ascending=False).reset_index(drop=True)
                df_test_res.insert(0, "rank", df_test_res.index + 1)  # test_rank

                print(
                    df_test_res[
                        [
                            "rank",            # test rank
                            "strategy_mode",
                            "config",
                            "total_trades",
                            "win_rate_pct",
                            "total_pl",
                            "return_pct",
                            "max_dd_pct",
                            "score",
                            "risk_pct",
                            "sl_atr_mult",
                            "rr",
                            "daily_max_losses",
                        ]
                    ].to_string(index=False)
                )

                # === TRAIN vs TEST comparison + overfit flag + overall ranking ===
                print("\n" + "=" * 120)
                print("TRAIN vs TEST COMPARISON (OVERALL RANKING & OVERFIT CHECK)")
                print("=" * 120)

                # TRAIN metrics (df_all is the TRAIN summary already ranked by score)
                train_cols = [
                    "config",
                    "total_trades",
                    "return_pct",
                    "max_dd_pct",
                    "score",
                    "rank",          # train rank from summary.csv
                    "risk_pct",
                    "sl_atr_mult",
                    "rr",
                    "daily_max_losses",
                ]
                df_train_merge = df_all[train_cols].copy()
                df_train_merge = df_train_merge.rename(
                    columns={
                        "config": "train_config",
                        "total_trades": "total_trades_train",
                        "return_pct": "return_pct_train",
                        "max_dd_pct": "max_dd_pct_train",
                        "score": "score_train",
                        "rank": "train_rank",
                        "risk_pct": "risk_pct_train",
                        "sl_atr_mult": "sl_atr_mult_train",
                        "rr": "rr_train",
                        "daily_max_losses": "daily_max_losses_train",
                    }
                )

                # TEST metrics (train_config added above)
                test_cols = [
                    "train_config",
                    "total_trades",
                    "return_pct",
                    "max_dd_pct",
                    "score",
                    "rank",          # test rank
                ]
                df_test_merge = df_test_res[test_cols].copy()
                df_test_merge = df_test_merge.rename(
                    columns={
                        "total_trades": "total_trades_test",
                        "return_pct": "return_pct_test",
                        "max_dd_pct": "max_dd_pct_test",
                        "score": "score_test",
                        "rank": "test_rank",
                    }
                )

                # Merge TRAIN + TEST on train_config
                df_cmp = pd.merge(df_train_merge, df_test_merge, on="train_config", how="inner")

                if df_cmp.empty:
                    print("[cmp] No TRAIN/TEST pairs to compare.")
                else:
                    # Score difference (%): how much score changed from TRAIN to TEST
                    df_cmp["score_diff_pct"] = np.where(
                        df_cmp["score_train"] != 0,
                        (df_cmp["score_test"] - df_cmp["score_train"]) / df_cmp["score_train"] * 100.0,
                        np.nan,
                    )

                    # Overfit rules
                    cond_score_drop   = df_cmp["score_test"] < df_cmp["score_train"] * 0.5
                    cond_ret_drop     = df_cmp["return_pct_test"] < df_cmp["return_pct_train"] * 0.4
                    cond_dd_worse     = df_cmp["max_dd_pct_test"] > df_cmp["max_dd_pct_train"] * 2.0
                    cond_trades_drop  = df_cmp["total_trades_test"] < df_cmp["total_trades_train"] * 0.5

                    df_cmp["overfit"] = np.where(
                        cond_score_drop | cond_ret_drop | cond_dd_worse | cond_trades_drop,
                        "YES",
                        "NO"
                    )

                    # === Overall score: prioritize TEST, use TRAIN as secondary ===
                    # 70% weight on test_score, 30% on train_score
                    df_cmp["overall_score"] = 0.7 * df_cmp["score_test"] + 0.3 * df_cmp["score_train"]

                    # Rank by overall_score (this is your final "overall" ranking)
                    df_cmp = df_cmp.sort_values("overall_score", ascending=False).reset_index(drop=True)
                    df_cmp.insert(0, "overall_rank", df_cmp.index + 1)

                    print(
                        df_cmp[
                            [
                                "overall_rank",
                                "train_rank",
                                "test_rank",
                                "train_config",
                                "risk_pct_train",
                                "sl_atr_mult_train",
                                "rr_train",
                                "daily_max_losses_train",
                                "total_trades_train",
                                "total_trades_test",
                                "return_pct_train",
                                "return_pct_test",
                                "max_dd_pct_train",
                                "max_dd_pct_test",
                                "score_train",
                                "score_test",
                                "score_diff_pct",
                                "overall_score",
                                "overfit",
                            ]
                        ].to_string(index=False)
                    )

                    # === Save combined summary (this is your summary_combined.csv) ===
                    try:
                        df_cmp.to_csv("summary_combined.csv", index=False)
                        print("\n[saved] summary_combined.csv")
                    except Exception as e:
                        print(f"[error saving summary_combined.csv] {e}")

            print("\n[done] Backtest + train/test evaluation complete.")
