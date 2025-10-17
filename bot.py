import os
import io
import math
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, CommandHandler, filters

# Optional: nicer candlestick charts
try:
    import mplfinance as mpf
    HAVE_MPF = True
except Exception:
    HAVE_MPF = False

# ---------------- Config ----------------
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
API_KEY = os.getenv("DATA_API_KEY")  # TwelveData
DEFAULT_INTERVAL = "15min"
CANDLE_THRESHOLD_PERCENT = 0.5  # % movement in last candle considered unusual

# ---------------- Chat state ----------------
chat_states = {}

# ---------------- Helpers ----------------
def normalize_pair(text: str) -> str:
    t = text.strip().upper().replace(" ", "").replace("-", "").replace("\\", "")
    if "/" in t:
        return t
    if len(t) == 6:
        return f"{t[:3]}/{t[3:]}"
    return t

def fetch_forex_series(pair: str, interval: str = DEFAULT_INTERVAL, outputsize: int = 200) -> pd.DataFrame:
    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": pair,
        "interval": interval,
        "outputsize": outputsize,
        "format": "JSON",
        "apikey": API_KEY
    }
    resp = requests.get(url, params=params, timeout=15).json()
    if "status" in resp and resp["status"] == "error":
        raise RuntimeError(f"Data API error: {resp.get('message')}")
    values = resp.get("values", [])
    if not values:
        raise RuntimeError("No data returned from API.")
    df = pd.DataFrame(values).astype(str)
    df = df.set_index(pd.to_datetime(df["datetime"]))
    for col in ("open", "high", "low", "close"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.sort_index()
    return df[["open", "high", "low", "close"]].dropna()

def compute_rsi(series: pd.Series, period: int = 14) -> float:
    delta = series.diff().dropna()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.ewm(alpha=1/period, adjust=False).mean()
    ma_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = ma_up / (ma_down + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return float(rsi.iloc[-1]) if not rsi.empty else float("nan")

def detect_unusual(df: pd.DataFrame) -> dict:
    result = {"unusual": False, "reasons": []}
    closes = df["close"]
    if len(closes) < 3:
        return result
    last = closes.iloc[-1]
    prev = closes.iloc[-2]
    percent_change = (last - prev) / prev * 100.0
    if abs(percent_change) >= CANDLE_THRESHOLD_PERCENT:
        result["unusual"] = True
        result["reasons"].append(f"Large candle: {percent_change:.2f}%")
    rsi = compute_rsi(closes)
    if rsi >= 70:
        result["unusual"] = True
        result["reasons"].append(f"RSI high ({rsi:.0f}) — possible overbought")
    elif rsi <= 30:
        result["unusual"] = True
        result["reasons"].append(f"RSI low ({rsi:.0f}) — possible oversold")
    result["percent_change"] = percent_change
    result["rsi"] = rsi
    return result

def generate_chart_image(df: pd.DataFrame, pair: str, interval: str = DEFAULT_INTERVAL) -> bytes:
    df_plot = df.copy().iloc[-100:]
    title = f"{pair} - {interval} - {datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC"
    buf = io.BytesIO()
    if HAVE_MPF and ("open" in df_plot.columns):
        mpf_df = df_plot.copy()
        mpf_df.index.name = "Date"
        mpf_df.columns = ["Open", "High", "Low", "Close"]
        mpf.plot(mpf_df, type="candle", style="yahoo", volume=False,
                 title=title, savefig=dict(fname=buf, dpi=120, bbox_inches="tight"))
        buf.seek(0)
        return buf.read()
    else:
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(df_plot))
        ax.plot(x, df_plot["close"].values, linewidth=1.2)
        for i, (_, row) in enumerate(df_plot.iterrows()):
            ax.vlines(i, row["low"], row["high"], linewidth=0.8, alpha=0.7)
        ax.set_title(title)
        ax.set_xlabel("Recent candles")
        ax.set_ylabel("Price")
        plt.tight_layout()
        fig.savefig(buf, format="png", dpi=120)
        plt.close(fig)
        buf.seek(0)
        return buf.read()

# ---------------- Handlers ----------------
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    chat_states[chat_id] = {"state": "idle"}
    await update.message.reply_text(
        "Hello — Trading Assistant ready.\n"
        "Type 'I want to trade now' to start an analysis session."
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Steps:\n1) Type 'I want to trade now'\n2) Reply with pair, e.g. EUR/USD\n"
        "3) Use 'send chart', 'send more', 'buy/sell signal'\nCommands: /start, /help"
    )

def ensure_chat_state(chat_id: int):
    if chat_id not in chat_states:
        chat_states[chat_id] = {"state": "idle"}

async def message_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text.strip()
    chat_id = update.effective_chat.id
    ensure_chat_state(chat_id)
    state = chat_states[chat_id].get("state", "idle")
    low = text.lower()

    # Entry trigger
    if low.startswith("i want to trade"):
        chat_states[chat_id]["state"] = "awaiting_pair"
        await update.message.reply_text("Okay, I’m ready. Which pair do you want analyzed? (e.g., EUR/USD)")
        return

    if state == "awaiting_pair":
        pair = normalize_pair(text)
        chat_states[chat_id].update({"state": "analyzing", "pair": pair})
        await update.message.reply_text(f"Fetching data for {pair} — one moment...")
        try:
            df = fetch_forex_series(pair)
            chat_states[chat_id]["last_df"] = df
            last_close = df["close"].iloc[-1]
            prev_close = df["close"].iloc[-2]
            change = (last_close - prev_close) / prev_close * 100.0
            rsi = compute_rsi(df["close"])
            trend = "Uptrend" if last_close > df["close"].rolling(window=20,min_periods=1).mean().iloc[-1] else "Downtrend/Sideways"
            unusual = detect_unusual(df)
            msg = [
                f"💹 {pair} Quick Market Analysis",
                f"📊 Current Price: {last_close:.5f}",
                f"📈 Trend: {trend}",
                f"🧭 RSI: {rsi:.0f}" if not math.isnan(rsi) else "🧭 RSI: N/A",
            ]
            if unusual["unusual"]:
                msg.append("⚠️ Unusual activity detected — possible trap.")
                for r in unusual["reasons"]:
                    msg.append(f"• {r}")
                warn = "⚠️ Unusual activity detected — better WAIT."
                msg.append(f"💡 {warn}")
            else:
                msg.append("✅ Market Stable — you may request more analysis.")
            msg.append("Type 'send more', 'send chart', or 'buy/sell signal'.")
            await update.message.reply_text("\n".join(msg))
            chat_states[chat_id]["state"] = "ready"
        except Exception as e:
            chat_states[chat_id]["state"] = "idle"
            await update.message.reply_text(f"Error fetching/analyzing data: {e}")
        return

    # Ready state: follow-up commands
    if state in ("ready", "analyzing", "detailed"):
        df = chat_states[chat_id].get("last_df")
        pair = chat_states[chat_id].get("pair")
        if low == "send more":
            closes = df["close"]
            rsi = compute_rsi(closes)
            ma20 = closes.rolling(window=20,min_periods=1).mean().iloc[-1]
            ma50 = closes.rolling(window=50,min_periods=1).mean().iloc[-1] if len(closes)>=50 else None
            recent_high = df["high"].iloc[-20:].max()
            recent_low = df["low"].iloc[-20:].min()
            msg = [
                f"📚 {pair} Deeper Analysis",
                f"🧭 RSI(14): {rsi:.1f}",
                f"🔁 MA(20): {ma20:.5f}",
            ]
            if ma50: msg.append(f"🔁 MA(50): {ma50:.5f}")