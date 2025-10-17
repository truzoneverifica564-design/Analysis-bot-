import os
import io
import math
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    ContextTypes,
    MessageHandler,
    CommandHandler,
    filters,
)

try:
    import mplfinance as mpf
    HAVE_MPF = True
except Exception:
    HAVE_MPF = False

# ================== CONFIG ==================
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
API_KEY = os.getenv("DATA_API_KEY")
DEFAULT_INTERVAL = "15min"
CANDLE_THRESHOLD_PERCENT = 0.5

chat_states = {}

# ================== HELPERS ==================
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
        result["reasons"].append(f"RSI high ({rsi:.0f}) — overbought")
    elif rsi <= 30:
        result["unusual"] = True
        result["reasons"].append(f"RSI low ({rsi:.0f}) — oversold")
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

# ================== HANDLERS ==================
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_states[update.effective_chat.id] = {"state": "idle"}
    await update.message.reply_text(
        "🤖 Hello — Trading Assistant ready.\nType 'I want to trade now' to start."
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Steps:\n1) Type 'I want to trade now'\n2) Reply with a pair (e.g., EUR/USD)\n"
        "3) Commands: 'send chart', 'send more', 'buy/sell signal'."
    )

async def message_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text.strip()
    chat_id = update.effective_chat.id
    state = chat_states.get(chat_id, {}).get("state", "idle")
    low = text.lower()

    if low.startswith("i want to trade"):
        chat_states[chat_id] = {"state": "awaiting_pair"}
        await update.message.reply_text("Ok, I’m ready. Which pair? (e.g., EUR/USD)")
        return

    if state == "awaiting_pair":
        pair = normalize_pair(text)
        chat_states[chat_id].update({"state": "analyzing", "pair": pair})
        await update.message.reply_text(f"Fetching data for {pair}...")
        try:
            df = fetch_forex_series(pair)
            chat_states[chat_id]["last_df"] = df
            rsi = compute_rsi(df["close"])
            unusual = detect_unusual(df)
            last_close = df["close"].iloc[-1]

            msg = [f"💹 {pair} Market Summary",
                   f"📊 Price: {last_close:.5f}",
                   f"🧭 RSI: {rsi:.0f}"]

            if unusual["unusual"]:
                msg.append("⚠️ Unusual movement — possible trap.")
                for r in unusual["reasons"]:
                    msg.append(f"• {r}")
            else:
                msg.append("✅ Stable market — no trap detected.")

            msg.append("Type 'send chart', 'send more', or 'buy/sell signal'.")
            await update.message.reply_text("\n".join(msg))
            chat_states[chat_id]["state"] = "ready"
        except Exception as e:
            await update.message.reply_text(f"Error: {e}")
            chat_states[chat_id]["state"] = "idle"
        return

    if state == "ready":
        df = chat_states[chat_id].get("last_df")
        pair = chat_states[chat_id].get("pair")

        if low in ("send chart", "chart"):
            await update.message.reply_text("📈 Generating chart...")
            try:
                img = generate_chart_image(df, pair)
                bio = io.BytesIO(img)
                bio.name = f"{pair.replace('/','')}.png"
                bio.seek(0)
                await update.message.reply_photo(photo=bio, caption=f"{pair} chart")
            except Exception as e:
                await update.message.reply_text(f"Chart error: {e}")

        elif low in ("buy/sell signal", "signal"):
            rsi = compute_rsi(df["close"])
            msg = "💡 BUY signal" if rsi < 30 else "⚠️ SELL signal" if rsi > 70 else "⏳ No clear signal — wait."
            await update.message.reply_text(msg)

# ================== MAIN ==================
if __name__ == "__main__":
    import asyncio

    async def main():
        if not TELEGRAM_TOKEN or not API_KEY:
            raise RuntimeError("Missing TELEGRAM_TOKEN or DATA_API_KEY in environment.")
        app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
        app.add_handler(CommandHandler("start", start_command))
        app.add_handler(CommandHandler("help", help_command))
        app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), message_handler))
        print("✅ Bot running successfully...")
        await app.run_polling()

    asyncio.run(main())