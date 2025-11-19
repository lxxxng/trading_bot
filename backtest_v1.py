"""
Pure backtest script for OANDA FX strategy with brute force parameter + strategy optimization.

- Fetches historical candles from OANDA (practice or live, via ENV)
- Caches candles in Parquet files (cache_hist/)
- Runs regime-aware strategy with enhancements:
    * Trend: Donchian + ATR breakout + Volume confirmation + Swing detection
    * Range: Bollinger + RSI mean reversion
    * Multi-timeframe confirmation (M5 + M15 alignment, via resample if enabled)
    * Adaptive SL/TP based on volatility
    * Time-of-day liquidity filter
- Brute force tests combinations of:
    * STRATEGY_MODE ∈ {FULL, TREND_ONLY, RANGE_ONLY, NO_MTF}
    * RISK_PCT, SL_PIPS, TP_PIPS, DAILY_DD_PCT
- Logs trades & equity into per-config SQLite DBs under backtest_results/
- Incrementally appends results to summary_raw.csv so runs can be resumed
- Produces ranked summary.csv (sorted by win rate)
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

BASE = (
    "https://api-fxpractice.oanda.com"
    if ENV == "practice"
    else "https://api-fxtrade.oanda.com"
)

DB_PATH = os.getenv("BACKTEST_DB_PATH", "backtest_trades.db")

START_EQUITY = 10_000.0
DAYS_BACK = int(os.getenv("DAYS_BACK", "1095"))

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

# =========================
# Brute Force Parameter Ranges
# =========================

STRATEGY_MODES = ["FULL", "TREND_ONLY", "RANGE_ONLY", "NO_MTF"]

RISK_PCT_RANGE = [0.01, 0.02, 0.03, 0.04, 0.05]  # 1% to 5% per trade
SL_PIPS_RANGE = [5, 10, 15, 20]                  # pips
TP_PIPS_RANGE = [10, 15, 20, 25, 30]             # pips
DAILY_DD_PCT_RANGE = [1, 2, 3, 4, 5]             # % max daily drawdown

# (strategy_mode, risk_pct, sl_pips, tp_pips, daily_dd_pct)
PARAM_CONFIGS = list(
    itertools.product(
        STRATEGY_MODES,
        RISK_PCT_RANGE,
        SL_PIPS_RANGE,
        TP_PIPS_RANGE,
        DAILY_DD_PCT_RANGE,
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
    sl_pips_fixed: float,
    tp_pips_fixed: float,
    strategy_mode: str,
    use_volume_confirm: bool,
    use_time_filter: bool,
    use_adaptive_sltp: bool,
    use_swing_detect: bool,
) -> Tuple[str, Dict[str, Any]]:
    """
    Single timeframe signal logic (trend/range, adaptive SL/TP, filters).
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

    trend_regime = atr_pct >= 65
    range_regime = atr_pct <= 35

    # --- Strategy-mode gating ---
    use_trend_block = strategy_mode in ("FULL", "TREND_ONLY", "NO_MTF")
    use_range_block = strategy_mode in ("FULL", "RANGE_ONLY", "NO_MTF")

    # TREND_ONLY: ignore range setups entirely
    if strategy_mode == "TREND_ONLY" and not trend_regime:
        return "FLAT", {"why": "not_trend_regime"}

    # RANGE_ONLY: ignore trend setups entirely
    if strategy_mode == "RANGE_ONLY" and not range_regime:
        return "FLAT", {"why": "not_range_regime"}

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

            sl_pips = max(sl_pips_fixed, 2.0 * (atr_now / pip_value(instrument)))
            tp_pips = max(tp_pips_fixed, 2.0 * sl_pips)
            sl_pips, tp_pips = adaptive_sl_tp(atr_pct, sl_pips, tp_pips, use_adaptive_sltp)

            return "LONG", {"mode": "trend", "sl_pips": sl_pips, "tp_pips": tp_pips}

        if dn_break:
            if use_volume_confirm and not volume_spike(df, 20, 1.5):
                return "FLAT", {"why": "no_volume_confirm"}

            if use_swing_detect:
                _, dn_swing = swing_setup(df, 3)
                if not dn_swing:
                    return "FLAT", {"why": "no_dnswing"}

            sl_pips = max(sl_pips_fixed, 2.0 * (atr_now / pip_value(instrument)))
            tp_pips = max(tp_pips_fixed, 2.0 * sl_pips)
            sl_pips, tp_pips = adaptive_sl_tp(atr_pct, sl_pips, tp_pips, use_adaptive_sltp)

            return "SHORT", {"mode": "trend", "sl_pips": sl_pips, "tp_pips": tp_pips}

    # --- Range regime: Bollinger + RSI ---
    if use_range_block and range_regime:
        rsi_val = df["RSI14"].iloc[-1]

        # Long mean-reversion
        if (c < bb_lo.iloc[-1]) and (rsi_val < 30):
            sl_pips = sl_pips_fixed
            tp_pips = max(tp_pips_fixed, 1.2 * sl_pips_fixed)
            sl_pips, tp_pips = adaptive_sl_tp(atr_pct, sl_pips, tp_pips, use_adaptive_sltp)

            return "LONG", {"mode": "range", "sl_pips": sl_pips, "tp_pips": tp_pips}

        # Short mean-reversion
        if (c > bb_up.iloc[-1]) and (rsi_val > 70):
            sl_pips = sl_pips_fixed
            tp_pips = max(tp_pips_fixed, 1.2 * sl_pips_fixed)
            sl_pips, tp_pips = adaptive_sl_tp(atr_pct, sl_pips, tp_pips, use_adaptive_sltp)

            return "SHORT", {"mode": "range", "sl_pips": sl_pips, "tp_pips": tp_pips}

    return "FLAT", {"why": "no_setup"}


def compute_signal(
    df: pd.DataFrame,
    instrument: str,
    sl_pips_fixed: float,
    tp_pips_fixed: float,
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
        sl_pips_fixed,
        tp_pips_fixed,
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
        sl_pips_fixed,
        tp_pips_fixed,
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
    sl_pips: float,
    tp_pips: float,
    daily_dd_pct: float,
    strategy_mode: str,
    config_name: str,
) -> Dict[str, Any]:
    """Run bar-by-bar backtest with given parameters and strategy mode."""

    # Create unique DB for this config
    safe_name = config_name.replace(".", "_").replace(" ", "_")
    db_path = os.path.join(RESULTS_DIR, f"{safe_name}.db")
    db_init(db_path)

    equity = start_equity
    position: Optional[Dict[str, Any]] = None

    trade_count = 0
    win_count = 0
    loss_count = 0
    total_pl = 0.0

    # Daily drawdown tracking
    current_day = None
    day_start_equity = None
    day_min_equity = None
    daily_dd_hit = False  # stop new entries for the day once hit

    for i in range(100, len(df)):
        bar = df.iloc[i]
        t_bar = pd.to_datetime(bar["t"], utc=True)
        o, h, l, c = float(bar["o"]), float(bar["h"]), float(bar["l"]), float(bar["c"])

        # --- Handle new day boundaries for daily DD logic ---
        day = t_bar.date()
        if current_day is None or day != current_day:
            current_day = day
            day_start_equity = equity
            day_min_equity = equity
            daily_dd_hit = False

        # 1) Manage open position
        if position is not None:
            side = position["side"]
            sl = position["sl"]
            tp = position["tp"]

            exit_price = None

            if side == "LONG":
                if l <= sl:
                    exit_price = sl
                elif h >= tp:
                    exit_price = tp
            else:  # SHORT
                if h >= sl:
                    exit_price = sl
                elif l <= tp:
                    exit_price = tp

            if exit_price is not None:
                units = position["units"]
                entry = position["entry"]
                side_mult = 1 if side == "LONG" else -1
                pl = (exit_price - entry) * units * side_mult
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
                    sl_pips=sl_pips,
                    tp_pips=tp_pips,
                    db_path=db_path,
                )

                if result == "WIN":
                    win_count += 1
                elif result == "LOSS":
                    loss_count += 1

                position = None

        # 2) Update daily DD state (after P&L impact)
        day_min_equity = min(day_min_equity, equity)
        if not daily_dd_hit and day_start_equity is not None:
            dd_abs = day_start_equity - day_min_equity
            dd_limit = day_start_equity * (daily_dd_pct / 100.0)
            if dd_abs >= dd_limit:
                daily_dd_hit = True

        # 3) Log equity using bar time
        log_equity(equity, gran, risk_pct, ts_utc=t_bar.isoformat(), db_path=db_path)

        # 4) Check for new signal (if no open position & daily DD not hit)
        if position is None and not daily_dd_hit:
            window = df.iloc[: i + 1].copy()
            signal, meta = compute_signal(window, instrument, sl_pips, tp_pips, strategy_mode)
            if signal not in ("LONG", "SHORT"):
                continue

            if REVERSE_SIGNALS:
                actual_side = "SHORT" if signal == "LONG" else "LONG"
            else:
                actual_side = signal

            sl_pips_calc = float(meta.get("sl_pips", sl_pips))
            tp_pips_calc = float(meta.get("tp_pips", tp_pips))

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
            }

            trade_count += 1

    # 5) Close any remaining open position at last bar close
    if position is not None:
        last_bar = df.iloc[-1]
        t_last = pd.to_datetime(last_bar["t"], utc=True)
        c_last = float(last_bar["c"])
        side = position["side"]
        units = position["units"]
        entry = position["entry"]

        side_mult = 1 if side == "LONG" else -1
        pl = (c_last - entry) * units * side_mult
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
            sl_pips=sl_pips,
            tp_pips=tp_pips,
            db_path=db_path,
        )

        if result == "WIN":
            win_count += 1
        elif result == "LOSS":
            loss_count += 1

    win_rate = (win_count / trade_count * 100) if trade_count > 0 else 0.0
    return_pct = ((equity - start_equity) / start_equity * 100) if start_equity > 0 else 0.0

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
        "db_path": db_path,
        "risk_pct": risk_pct,
        "sl_pips": sl_pips,
        "tp_pips": tp_pips,
        "daily_dd_pct": daily_dd_pct,
    }


# =========================
# Main
# =========================

if __name__ == "__main__":
    instr = DEFAULT_INSTRUMENT
    gran = DEFAULT_GRAN

    print(f"\n[main] Loading historical data: {instr} {gran}...")
    df_hist = load_or_fetch_history(instr, gran, days_back=DAYS_BACK)

    # --- Load existing partial results to enable resume ---
    done_configs = set()
    if os.path.exists(SUMMARY_RAW_PATH):
        try:
            existing = pd.read_csv(SUMMARY_RAW_PATH)
            if "config" in existing.columns:
                done_configs = set(existing["config"].astype(str))
                print(f"[resume] Found {len(done_configs)} completed configs in summary_raw.csv")
        except Exception as e:
            print(f"[resume] Could not read existing summary_raw.csv: {e}")
            done_configs = set()

    print(f"\n[main] Starting brute force backtest with {len(PARAM_CONFIGS)} configurations...\n")

    # We'll still collect this run's results in memory for printing, but
    # summary_raw.csv is the source of truth across runs.
    run_results = []

    for idx, (strategy_mode, risk_pct, sl_pips, tp_pips, daily_dd_pct) in enumerate(PARAM_CONFIGS, 1):
        config_name = f"{strategy_mode}_R{risk_pct:.2f}_SL{sl_pips}_TP{tp_pips}_DD{daily_dd_pct}"

        if config_name in done_configs:
            print(f"[{idx}/{len(PARAM_CONFIGS)}] Skipping already completed: {config_name}")
            continue

        print(
            f"[{idx}/{len(PARAM_CONFIGS)}] Testing: {config_name}",
            end=" ... ",
            flush=True,
        )

        try:
            result = simulate_backtest(
                df_hist,
                instrument=instr,
                gran=gran,
                start_equity=START_EQUITY,
                risk_pct=risk_pct,
                sl_pips=sl_pips,
                tp_pips=tp_pips,
                daily_dd_pct=daily_dd_pct,
                strategy_mode=strategy_mode,
                config_name=config_name,
            )
            run_results.append(result)

            # --- Append this result to summary_raw.csv immediately ---
            df_row = pd.DataFrame([result])
            write_header = not os.path.exists(SUMMARY_RAW_PATH)
            df_row.to_csv(
                SUMMARY_RAW_PATH,
                mode="a",
                index=False,
                header=write_header,
            )

            print(
                f"✓ {result['total_trades']} trades | {result['win_rate_pct']}% WR | ${result['total_pl']} PL"
            )
        except Exception as e:
            print(f"✗ Error: {e}")

    # === Build ranked summary from summary_raw.csv (all runs, old+new) ===
    if os.path.exists(SUMMARY_RAW_PATH):
        print("\n" + "=" * 120)
        print("BRUTE FORCE BACKTEST SUMMARY (ALL COMPLETED CONFIGS)")
        print("=" * 120)

        df_all = pd.read_csv(SUMMARY_RAW_PATH)

        # Drop duplicate configs, keep the last occurrence
        df_all = df_all.drop_duplicates(subset=["config"], keep="last").reset_index(drop=True)

        # Rank by win_rate_pct descending
        df_all = df_all.sort_values("win_rate_pct", ascending=False).reset_index(drop=True)
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
                ]
            ].to_string(index=False)
        )

        # Save ranked summary
        df_all.to_csv(SUMMARY_RANKED_PATH, index=False)
        print(f"\n[saved] ranked summary -> {SUMMARY_RANKED_PATH}")

        # Top 3
        print("\n" + "=" * 120)
        print("TOP 3 CONFIGURATIONS (by win rate)")
        print("=" * 120)
        for _, row in df_all.head(3).iterrows():
            print(f"\n#{int(row['rank'])}: {row['config']} ({row['strategy_mode']})")
            print(f"    Trades: {row['total_trades']} | Win Rate: {row['win_rate_pct']}%")
            print(f"    P&L: ${row['total_pl']} | Return: {row['return_pct']}%")
            print(f"    DB: {row['db_path']}")
