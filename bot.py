"""
Auto FX bot (OANDA) — Flask + requests + SQLite
- Demo/practice by default
- Regime-aware strategy: Trend breakout (Donchian+ATR) + Range mean-revert (Bollinger+RSI)
- Logs trades, equity snapshots, and per-bar decisions (including FLAT) to SQLite
- Dynamic config via /config endpoint (switch M5/M15 without restart)

IMPORTANT:
- Start on PRACTICE first. Flip to LIVE only when you’re absolutely ready.
- This is an educational template; run small, validate, expand gradually.
"""

import os, time, threading, sqlite3, math, json
from datetime import datetime, timezone, date
from typing import Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
import requests
from flask import Flask, jsonify, request
from dotenv import load_dotenv

# =========================
# Config & Globals
# =========================

load_dotenv()

# --- Core broker config (demo first) ---
ENV = os.getenv("OANDA_ENV", "practice")                # "practice" | "live"
TOKEN = os.getenv("OANDA_TOKEN", "")
ACCOUNT_ID = os.getenv("OANDA_ACCOUNT_ID", "")

# --- Trading config (defaults; can be changed at runtime through /config) ---
DEFAULT_INSTRUMENT = os.getenv("INSTRUMENT", "EUR_USD")
DEFAULT_GRAN = os.getenv("GRANULARITY", "M5")           # "M5" or "M15" (or other supported)
RISK_PCT = float(os.getenv("RISK_PCT", 0.5)) / 100.0    # risk per trade in % of equity
SL_PIPS_FIXED = float(os.getenv("SL_PIPS", 15))         # base SL for range regime; floor for trend regime
TP_PIPS_FIXED = float(os.getenv("TP_PIPS", 20))         # base TP for range regime

# --- Circuit breaker ---
DAILY_DD_PCT = float(os.getenv("DAILY_DD_PCT", 3.0))    # pause for the day if equity drawdown hits this %

# --- Endpoints, DB file, and HTTP session ---
BASE = "https://api-fxpractice.oanda.com" if ENV == "practice" else "https://api-fxtrade.oanda.com"
DB_PATH = os.getenv("DB_PATH", "trades.db")

app = Flask(__name__)
session = requests.Session()
if not TOKEN:
    raise SystemExit("Set OANDA_TOKEN in .env")
session.headers.update({"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"})

# --- Runtime state (protected by a simple lock) ---
_running = False
_loop_thread: Optional[threading.Thread] = None
_state_lock = threading.RLock()

# live state (read under lock)
_state = {
    "instrument": DEFAULT_INSTRUMENT,   # can change at runtime
    "granularity": DEFAULT_GRAN,        # can change at runtime (M5 / M15 / etc.)
    "last_side": None,                  # "LONG" | "SHORT" | None
    "last_equity": None,
    "last_trade_info": {},
    "utc": None
}

# daily breaker state (fixed bug: store date + open equity together)
_day_state: Dict[str, Any] = {"date": None, "open_equity": None}

# =========================
# SQLite helpers
# =========================

def db_connect():
    """Open a SQLite connection for the current thread."""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")  # better concurrent reads
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn

def db_init():
    """Create tables if they don't exist."""
    conn = db_connect()
    cur = conn.cursor()

    # --- main trades table ---
    cur.execute("""
    CREATE TABLE IF NOT EXISTS trades (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      ts_utc TEXT NOT NULL,               -- time trade was opened
      instrument TEXT NOT NULL,
      side TEXT NOT NULL,                 -- BUY or SELL
      units INTEGER NOT NULL,
      entry_est REAL NOT NULL,            -- estimated entry price
      sl REAL NOT NULL,                   -- stop loss
      tp REAL NOT NULL,                   -- take profit
      mode TEXT NOT NULL,                 -- 'trend' or 'range'
      granularity TEXT NOT NULL,
      broker_tx_id TEXT,                  -- OANDA trade ID or lastTransactionID
      realized_pl REAL DEFAULT 0.0,       -- profit/loss once closed
      result TEXT DEFAULT NULL,           -- WIN / LOSS / EVEN
      meta_json TEXT                      -- misc notes
    );
    """)

    # --- equity snapshots ---
    cur.execute("""
    CREATE TABLE IF NOT EXISTS equity (
      ts_utc TEXT PRIMARY KEY,
      equity REAL NOT NULL,
      granularity TEXT NOT NULL
    );
    """)

    # --- per-bar decisions (always log, even when FLAT) ---
    cur.execute("""
    CREATE TABLE IF NOT EXISTS decisions (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      ts_utc TEXT NOT NULL,
      instrument TEXT NOT NULL,
      granularity TEXT NOT NULL,
      signal TEXT NOT NULL,               -- LONG | SHORT | FLAT
      reason TEXT,                        -- 'flat' plus details (no_setup / late_friday / not_enough_bars)
      close REAL,                         -- last close price
      meta_json TEXT
    );
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_decisions_time ON decisions(ts_utc);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_decisions_inst ON decisions(instrument, granularity);")

    conn.commit()
    conn.close()

def log_trade(instrument: str, side: str, units: int, entry_est: float,
              sl: float, tp: float, mode: str, gran: str, tx_id: str, meta_json: str):
    conn = db_connect()
    conn.execute("""
        INSERT INTO trades (ts_utc, instrument, side, units, entry_est, sl, tp, mode, granularity, broker_tx_id, meta_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
    """, (utc_now_iso(), instrument, side, units, entry_est, sl, tp, mode, gran, tx_id, meta_json))
    conn.commit()
    conn.close()

def log_equity(equity: float, gran: str):
    conn = db_connect()
    conn.execute("""
        INSERT OR REPLACE INTO equity (ts_utc, equity, granularity)
        VALUES (?, ?, ?);
    """, (utc_now_iso(), equity, gran))
    conn.commit()
    conn.close()

def log_decision(instrument: str, gran: str, signal: str, reason: str, close: float, meta: Dict[str, Any]):
    conn = db_connect()
    conn.execute("""
        INSERT INTO decisions (ts_utc, instrument, granularity, signal, reason, close, meta_json)
        VALUES (?, ?, ?, ?, ?, ?, ?);
    """, (utc_now_iso(), instrument, gran, signal, reason, close, json.dumps(meta, default=str)))
    conn.commit()
    conn.close()

# =========================
# Utility & Broker helpers
# =========================

def utc_now_iso():
    return datetime.now(timezone.utc).isoformat()

def today_utc() -> date:
    return datetime.now(timezone.utc).date()

def pip_value(instrument: str) -> float:
    """Rough pip size: 0.01 for JPY crosses, 0.0001 otherwise (OK for majors)."""
    return 0.01 if instrument.endswith("_JPY") else 0.0001

def _price_decimals(instrument: str) -> int:
    """OANDA typical quoting: 3 decimals for JPY, 5 otherwise (mid-prices)."""
    return 3 if instrument.endswith("_JPY") else 5

def price_add_pips(price: float, pips: float, side: str, instrument: str) -> float:
    """Add/subtract pips from price depending on order side, with proper rounding."""
    delta = pips * pip_value(instrument)
    dec = _price_decimals(instrument)
    return round(price + (delta if side.upper() == "LONG" else -delta), dec)

def get_candles(instrument: str, gran: str, count=300) -> pd.DataFrame:
    """Pull completed MID-price candles for a given instrument and granularity."""
    params = {"granularity": gran, "count": count, "price": "M"}
    r = session.get(f"{BASE}/v3/instruments/{instrument}/candles", params=params, timeout=30)
    r.raise_for_status()
    js = r.json()
    rows = []
    for c in js["candles"]:
        if not c["complete"]:
            continue
        rows.append({
            "t": c["time"],
            "o": float(c["mid"]["o"]),
            "h": float(c["mid"]["h"]),
            "l": float(c["mid"]["l"]),
            "c": float(c["mid"]["c"]),
            "v": int(c["volume"])
        })
    return pd.DataFrame(rows)

def get_account_nav() -> float:
    r = session.get(f"{BASE}/v3/accounts/{ACCOUNT_ID}/summary", timeout=30)
    r.raise_for_status()
    return float(r.json()["account"]["NAV"])

def market_order(instrument: str, side: str, units: int, sl_price: float,
                 tp_price: float, price_bound: Optional[float]) -> Dict[str, Any]:
    """Place a MARKET order with SL/TP and an optional protective priceBound."""
    body = {
        "order": {
            "type": "MARKET",
            "timeInForce": "FOK",
            "instrument": instrument,
            "units": str(units if side.upper() == "LONG" else -units),
            "positionFill": "DEFAULT",
            "takeProfitOnFill": {"price": f"{tp_price:.5f}"},
            "stopLossOnFill":  {"price": f"{sl_price:.5f}"}
        }
    }
    if price_bound is not None:
        body["order"]["priceBound"] = f"{price_bound:.5f}"
    r = session.post(f"{BASE}/v3/accounts/{ACCOUNT_ID}/orders", json=body, timeout=30)
    if not r.ok:
        raise RuntimeError(f"Order rejected: {r.status_code} {r.text}")
    return r.json()

def update_closed_trades():
    """
    Fetch closed trades from OANDA and update local SQLite log
    with realized profit/loss and result (WIN/LOSS/EVEN).
    """
    try:
        url = f"{BASE}/v3/accounts/{ACCOUNT_ID}/trades?state=CLOSED"
        r = session.get(url, timeout=30)
        if not r.ok:
            print(f"[warn] failed to fetch closed trades: {r.status_code}")
            return

        js = r.json()
        closed = js.get("trades", [])
        if not closed:
            return

        conn = db_connect()
        cur = conn.cursor()

        for t in closed:
            trade_id = t.get("id")
            pl = float(t.get("realizedPL", 0.0))
            outcome = "WIN" if pl > 0 else "LOSS" if pl < 0 else "EVEN"

            # update local DB if trade exists
            cur.execute("""
                UPDATE trades
                SET realized_pl = ?, result = ?
                WHERE broker_tx_id = ?;
            """, (pl, outcome, trade_id))

        conn.commit()
        conn.close()
        if closed:
            print(f"[sync] updated {len(closed)} closed trades")
    except Exception as e:
        print("[error:update_closed_trades]", e)

# =========================
# Indicators & Strategy
# =========================

def atr(df: pd.DataFrame, period=14) -> pd.Series:
    """Average True Range (simple mean for stability)."""
    h, l, c = df["h"].values, df["l"].values, df["c"].values
    prev_c = np.r_[np.nan, c[:-1]]
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))
    return pd.Series(tr).rolling(period).mean()

def donchian(df: pd.DataFrame, n=20) -> Tuple[pd.Series, pd.Series]:
    """Upper/lower channel extremes over n bars."""
    return df["h"].rolling(n).max(), df["l"].rolling(n).min()

def bollinger(df: pd.DataFrame, n=20, k=2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Bollinger bands on close."""
    ma = df["c"].rolling(n).mean()
    sd = df["c"].rolling(n).std(ddof=0)
    return ma, ma + k*sd, ma - k*sd

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

def compute_signal(df: pd.DataFrame, instrument: str) -> Tuple[str, Dict[str, Any]]:
    """
    Regime-aware logic:
      Trend regime (ATR percentile >= 65):
        - Donchian(20) breakout entry
        - SL = max(fixed, 2*ATR pips); TP ≈ 2x SL
      Range regime (ATR percentile <= 35):
        - Fade Bollinger band with RSI confirm
        - SL = fixed; TP ≈ 1.2x SL
    Returns: ("LONG"|"SHORT"|"FLAT", meta)
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

    # Avoid very illiquid late Friday UTC
    ts = pd.to_datetime(df["t"].iloc[-1], utc=True)
    if ts.weekday() == 4 and ts.hour >= 20:
        return "FLAT", {"why": "late_friday"}

    trend_regime = atr_pct >= 65
    range_regime = atr_pct <= 35

    # Trend: Donchian breakout
    if trend_regime:
        up_break = (c > dc_hi.iloc[-2]) and (prev_c <= dc_hi.iloc[-3])
        dn_break = (c < dc_lo.iloc[-2]) and (prev_c >= dc_lo.iloc[-3])
        if up_break:
            sl_pips = max(SL_PIPS_FIXED, 2.0 * (atr_now / pip_value(instrument)))
            tp_pips = max(TP_PIPS_FIXED, 2.0 * sl_pips)
            return "LONG", {"mode": "trend", "sl_pips": sl_pips, "tp_pips": tp_pips}
        if dn_break:
            sl_pips = max(SL_PIPS_FIXED, 2.0 * (atr_now / pip_value(instrument)))
            tp_pips = max(TP_PIPS_FIXED, 2.0 * sl_pips)
            return "SHORT", {"mode": "trend", "sl_pips": sl_pips, "tp_pips": tp_pips}

    # Range: Bollinger + RSI
    if range_regime:
        if (c < bb_lo.iloc[-1]) and (df["RSI14"].iloc[-1] < 30):
            sl_pips = SL_PIPS_FIXED
            tp_pips = max(TP_PIPS_FIXED, 1.2 * SL_PIPS_FIXED)
            return "LONG", {"mode": "range", "sl_pips": sl_pips, "tp_pips": tp_pips}
        if (c > bb_up.iloc[-1]) and (df["RSI14"].iloc[-1] > 70):
            sl_pips = SL_PIPS_FIXED
            tp_pips = max(TP_PIPS_FIXED, 1.2 * SL_PIPS_FIXED)
            return "SHORT", {"mode": "range", "sl_pips": sl_pips, "tp_pips": tp_pips}

    return "FLAT", {"why": "no_setup"}

# =========================
# Sizing, Breakers, and Loop
# =========================

def position_size(nav: float, sl_pips: float, instrument: str) -> int:
    """
    Simple, conservative sizing:
    units ≈ (risk_amount) / (SL_pips * pip)
    This keeps sizing modest across majors. Refine per-instrument if needed.
    """
    risk_amount = nav * RISK_PCT
    units = risk_amount / (sl_pips * pip_value(instrument))
    return max(1, int(units))

def daily_breaker_ok(current_equity: float) -> bool:
    """
    Circuit breaker: if today's equity drawdown hits DAILY_DD_PCT, pause trading
    until the next UTC day (but the loop keeps running & checking).
    """
    today = today_utc()
    if _day_state["date"] != today:
        _day_state["date"] = today
        _day_state["open_equity"] = current_equity
        return True
    if not _day_state["open_equity"]:
        return True
    dd = ((_day_state["open_equity"] - current_equity) / _day_state["open_equity"]) * 100.0
    return dd < DAILY_DD_PCT

def trading_loop():
    global _running
    print("[loop] started")
    while _running:
        try:
            with _state_lock:
                instrument = _state["instrument"]
                gran = _state["granularity"]

            # 1) Data
            df = get_candles(instrument, gran, 300)
            if df.empty:
                time.sleep(30)
                continue

            # 2) Equity & breaker
            equity = get_account_nav()
            log_equity(equity, gran)                 # equity snapshot logging
            with _state_lock:
                _state["last_equity"] = equity
            if not daily_breaker_ok(equity):
                print(f"[breaker] Daily DD ≥ {DAILY_DD_PCT}% → pausing until next UTC day.")
                # also log a flat decision due to breaker
                close_px = float(df["c"].iloc[-1])
                log_decision(instrument, gran, "FLAT", "flat:breaker", close_px, {"why":"daily_drawdown_limit"})
                time.sleep(60)
                continue

            # 3) Signal
            signal, meta = compute_signal(df, instrument)
            close_px = float(df["c"].iloc[-1])

            with _state_lock:
                last_side = _state["last_side"]

            # 3a) Always log the decision (even FLAT)
            reason = "executed" if signal in ("LONG", "SHORT") and signal != last_side else f"flat:{meta.get('why','n/a')}"
            if signal == "FLAT" or signal == last_side:
                # log immediately for non-trade (or same-direction suppression)
                log_decision(instrument, gran, "FLAT", reason, close_px, {"meta": meta})

            # 4) Execute if a *new* actionable signal
            if signal in ("LONG", "SHORT") and signal != last_side:
                sl_pips = float(meta.get("sl_pips", SL_PIPS_FIXED))
                tp_pips = float(meta.get("tp_pips", TP_PIPS_FIXED))
                units = position_size(equity, sl_pips, instrument)

                sl = price_add_pips(close_px, sl_pips, "SHORT" if signal == "LONG" else "LONG", instrument)
                tp = price_add_pips(close_px, tp_pips, signal, instrument)
                bound = price_add_pips(close_px, sl_pips * 0.5, signal, instrument)

                resp = market_order(instrument, signal, units, sl, tp, bound)
                tx_id = str(resp.get("lastTransactionID"))

                # 4a) Log the trade to SQLite
                log_trade(
                    instrument=instrument,
                    side="BUY" if signal == "LONG" else "SELL",
                    units=units,
                    entry_est=close_px,
                    sl=sl,
                    tp=tp,
                    mode=meta.get("mode", "n/a"),
                    gran=gran,
                    tx_id=tx_id,
                    meta_json=json.dumps(meta, default=str)
                )

                # 4b) Log the decision row as executed
                log_decision(instrument, gran, signal, "executed", close_px, {
                    "mode": meta.get("mode", "n/a"),
                    "units": units, "sl": sl, "tp": tp, "tx": tx_id
                })

                with _state_lock:
                    _state["last_side"] = signal
                    _state["last_trade_info"] = {
                        "when": utc_now_iso(),
                        "signal": signal,
                        "mode": meta.get("mode", "n/a"),
                        "close": close_px,
                        "units": units,
                        "sl": sl,
                        "tp": tp,
                        "tx": tx_id
                    }

                print(f"[trade] {signal} {instrument} units={units} @≈{close_px:.5f} SL={sl:.5f} TP={tp:.5f} tx={tx_id}")

            # 5) Sync closed trades
            update_closed_trades()

            # Sleep roughly a minute; strategy reacts on bar closes
            time.sleep(60)

        except Exception as e:
            print("[error]", e)
            time.sleep(10)

    print("[loop] stopped")

# =========================
# Flask Endpoints
# =========================

@app.get("/status")
def status():
    with _state_lock:
        _state["utc"] = utc_now_iso()
        return jsonify(_state)

@app.post("/start")
def start():
    global _running, _loop_thread
    if _running:
        return jsonify({"ok": True, "msg": "already running"})
    if not ACCOUNT_ID:
        return jsonify({"ok": False, "msg": "Set OANDA_ACCOUNT_ID in .env"}), 400

    _running = True
    _loop_thread = threading.Thread(target=trading_loop, daemon=True)
    _loop_thread.start()
    return jsonify({"ok": True, "msg": "started"})

@app.post("/stop")
def stop():
    global _running
    _running = False
    return jsonify({"ok": True, "msg": "stopping"})

@app.post("/config")
def config_update():
    """
    Update runtime config without restart.
    Example:
      curl -X POST http://localhost:8000/config -H "Content-Type: application/json" \
           -d '{"granularity":"M15"}'
      curl -X POST http://localhost:8000/config -H "Content-Type: application/json" \
           -d '{"instrument":"GBP_USD","granularity":"M5"}'
    """
    data = request.get_json(force=True, silent=True) or {}
    updated = {}
    with _state_lock:
        if "instrument" in data and isinstance(data["instrument"], str) and data["instrument"]:
            _state["instrument"] = data["instrument"].upper()
            updated["instrument"] = _state["instrument"]
        if "granularity" in data and isinstance(data["granularity"], str) and data["granularity"]:
            # Accept any OANDA-supported granularity (e.g., M1, M5, M15, H1...)
            _state["granularity"] = data["granularity"].upper()
            updated["granularity"] = _state["granularity"]
    if not updated:
        return jsonify({"ok": False, "msg": "no valid keys (instrument, granularity) supplied"}), 400
    return jsonify({"ok": True, "updated": updated})

# Bootstrapping
if __name__ == "__main__":
    db_init()
    app.run(host="0.0.0.0", port=8000, use_reloader=False)
