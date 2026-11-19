# -*- coding: utf-8 -*-
"""UDEM | Trading Bot Cuantitativo (RSI + Bandas de Bollinger + Medias Móviles)
+ Random Forest + Gestión de Riesgo VaR/CVaR (SL/TP Dinámico) + Telegram.
"""
import asyncio
import io
import json
import logging
import os
import threading
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import gate_api
from google import genai
from telegram import Update, ReplyKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters
from telegram.request import HTTPXRequest


def _cargar_env():
    """Lee un archivo .env (CLAVE=valor) sin dependencias externas. Las claves NUNCA van en el código."""
    for ruta in (Path(__file__).resolve().parent / '.env', Path.home() / '.env', Path.cwd() / '.env'):
        if ruta.exists():
            for linea in ruta.read_text(encoding='utf-8').splitlines():
                linea = linea.strip()
                if linea and not linea.startswith('#') and '=' in linea:
                    k, v = linea.split('=', 1)
                    os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
            break
_cargar_env()

# =====================================================================
# 1. CREDENCIALES Y CONFIGURACIÓN BASE
# =====================================================================
GATE_API_KEY = os.getenv('GATE_API_KEY', '')
GATE_API_SECRET = os.getenv('GATE_API_SECRET', '')
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '')
TELEGRAM_CHAT_ID = None

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', '')
GEMINI_MODEL = os.getenv('GEMINI_MODEL', 'gemini-3.6-flash')

SETTLE = 'usdt'
DATA_DIR = Path(os.path.expanduser('~/datos_ema_bot'))
DATA_DIR.mkdir(parents=True, exist_ok=True)
STATE_FILE = DATA_DIR / 'estado.json'
TRADES_FILE = DATA_DIR / 'trades.csv'
LOG_FILE = DATA_DIR / 'eventos.csv'
PAIR_FILE = DATA_DIR / 'telegram_chat.json'

if os.getenv('TELEGRAM_CHAT_ID'):
    try:
        TELEGRAM_CHAT_ID = int(os.getenv('TELEGRAM_CHAT_ID'))
    except ValueError:
        pass
elif PAIR_FILE.exists():
    try:
        TELEGRAM_CHAT_ID = int(json.loads(PAIR_FILE.read_text(encoding='utf-8'))['chat_id'])
    except Exception:
        logging.warning('Vinculación anterior no válida; se vinculará con /start')

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                    handlers=[logging.StreamHandler(), logging.FileHandler('trading_bot.log', encoding='utf-8')])
# httpx imprime la URL completa (incluye el token de Telegram): se silencia
for _n in ('httpx', 'httpcore', 'telegram'):
    logging.getLogger(_n).setLevel(logging.WARNING)

@dataclass
class Config:
    symbol: str = 'XRP_USDT'
    interval: str = '5m'
    strategy: str = 'RSI_BB_EMA'    # Principal: RSI + Bollinger + EMA
    leverage: float = 10.0
    position_size: float = 10.0     # 10 XRP (~$20 USDT nocional)
    var_limit_pct: float = 2.5      # Límite VaR 99%
    cvar_limit_pct: float = 3.5     # Límite CVaR 99% (Expected Shortfall)
    rf_trees: int = 150             # Árboles Random Forest
    horizon: int = 12               # Horizonte de predicción (12 velas de 5m = 1h)
    trading_enabled: bool = False

CFG = Config()
LOCK = threading.RLock()
RF_CACHE = {}

MENU = ReplyKeyboardMarkup([
    ['▶️ Activar trading', '🛑 Parar bots'],
    ['⚙️ Estrategia', '⏱️ Temporalidad', '🎯 Elegir Ticker'],
    ['🛡️ VaR/CVaR (Semáforo)', '🔮 Predicción RF'],
    ['📈 Gráfico', '📊 Obtener Excel'],
    ['🤖 Hablar con Gemini', '📋 Estatus', '❌ Cancelar']
], resize_keyboard=True)

MENU_STRAT = ReplyKeyboardMarkup([
    ['RSI + Bollinger + EMA', 'EMA 7/21'],
    ['Golden/Death Cross (50/200)', 'Fibonacci Retracement'],
    ['⬅️ Volver']
], resize_keyboard=True)

MENU_TF = ReplyKeyboardMarkup([
    ['5m', '15m', '1h', '4h'],
    ['⬅️ Volver']
], resize_keyboard=True)

MENU_TICKER = ReplyKeyboardMarkup([
    ['XRP_USDT', 'BTC_USDT', 'ETH_USDT', 'SOL_USDT'],
    ['🔍 Buscar otro ticker', '⬅️ Volver']
], resize_keyboard=True)

def now_iso():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')

def fmt_p(val):
    return f"{val:,.4f}" if abs(val) < 100 else f"{val:,.2f}"

def write_csv(path, record):
    frame = pd.DataFrame([record])
    frame.to_csv(path, mode='a', index=False, header=not path.exists(), encoding='utf-8-sig')

def log_event(event, detail='', candle=None):
    write_csv(LOG_FILE, {'utc': now_iso(), 'evento': event, 'detalle': str(detail), 'vela': candle})

def load_state():
    default = {'position': None, 'last_candle': None, 'trading_enabled': False,
               'realized_pnl_usdt': 0.0, 'last_signal': 'Sin señales',
               'risk_status': 'VERDE', 'last_var': 0.0, 'last_cvar': 0.0}
    if STATE_FILE.exists():
        try:
            loaded = json.loads(STATE_FILE.read_text(encoding='utf-8'))
            default.update({k: loaded[k] for k in default if k in loaded})
        except Exception:
            logging.exception('Estado corrupto; cargando default')
    return default

STATE = load_state()
CFG.trading_enabled = bool(STATE.get('trading_enabled', False))

def save_state():
    tmp = STATE_FILE.with_suffix('.tmp')
    tmp.write_text(json.dumps(STATE, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(STATE_FILE)

# =====================================================================
# 2. CONEXIÓN A GATE.IO E INDICADORES TÉCNICOS (RSI + BOLLINGER + EMAs)
# =====================================================================
def gate_client():
    cfg = gate_api.Configuration(key=GATE_API_KEY, secret=GATE_API_SECRET,
                                 host='https://api-testnet.gateapi.io/api/v4')
    return gate_api.FuturesApi(gate_api.ApiClient(cfg))

def get_candles(symbol=None, interval=None):
    sym = symbol or CFG.symbol
    inter = interval or CFG.interval
    api = gate_client()
    try:
        bars = api.list_futures_candlesticks(settle=SETTLE, contract=sym,
                                             interval=inter, limit=600)
    finally:
        api.api_client.close()

    df = pd.DataFrame([{'t': int(b.t), 'open': float(b.o), 'high': float(b.h),
                        'low': float(b.l), 'close': float(b.c)} for b in bars])
    if df.empty:
        raise RuntimeError(f'Gate.io no devolvió velas para {sym}')

    df = df.sort_values('t').drop_duplicates('t').reset_index(drop=True)
    if len(df) < 220:
        raise RuntimeError('Historial insuficiente (se requieren al menos 220 velas)')

    # Medias Móviles (EMA 9, EMA 21, SMA 50, SMA 200)
    df['ema_fast'] = df.close.ewm(span=9, adjust=False).mean()
    df['ema_slow'] = df.close.ewm(span=21, adjust=False).mean()
    df['sma_50'] = df.close.rolling(50).mean()
    df['sma_200'] = df.close.rolling(200).mean()

    # RSI (14 periodos - Suavizado de Wilder)
    delta = df.close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    df['rsi'] = (100 - (100 / (1 + rs))).fillna(50.0)

    # Bandas de Bollinger (20 periodos, 2 desviaciones estándar)
    df['bb_mid'] = df.close.rolling(20).mean()
    bb_std = df.close.rolling(20).std()
    df['bb_upper'] = df.bb_mid + 2.0 * bb_std
    df['bb_lower'] = df.bb_mid - 2.0 * bb_std
    df['bb_width'] = (df.bb_upper - df.bb_lower) / df.bb_mid
    return df

# =====================================================================
# 3. GESTIÓN DE RIESGO CUANTITATIVO: VaR 99% Y CVaR 99%
# =====================================================================
def compute_var_cvar(df, alpha=0.99):
    returns = df['close'].pct_change().dropna()
    losses = -returns
    var_val = float(np.percentile(losses, alpha * 100))
    tail_losses = losses[losses >= var_val]
    cvar_val = float(tail_losses.mean()) if not tail_losses.empty else var_val

    var_pct = var_val * 100.0
    cvar_pct = cvar_val * 100.0

    blocked = (var_pct > CFG.var_limit_pct) or (cvar_pct > CFG.cvar_limit_pct)
    status = "ROJO (BLOQUEADO)" if blocked else "VERDE (SEGURO)"

    with LOCK:
        STATE['last_var'] = round(var_pct, 3)
        STATE['last_cvar'] = round(cvar_pct, 3)
        STATE['risk_status'] = status
        save_state()

    return {'var_pct': var_pct, 'cvar_pct': cvar_pct, 'blocked': blocked, 'status': status}

# =====================================================================
# 4. RANDOM FOREST REGRESSOR (PREDICCIÓN DE MÁXIMO Y MÍNIMO)
# =====================================================================
def features(df):
    c = df.close
    X = pd.DataFrame(index=df.index)
    for n in (1, 2, 3, 6, 12, 24):
        X[f'ret_{n}'] = c.pct_change(n)
    X['ema_fast_ratio'] = df.ema_fast / c
    X['ema_slow_ratio'] = df.ema_slow / c
    X['rsi'] = df.rsi / 100.0
    X['bb_width'] = df.bb_width
    X['vol20'] = c.pct_change().rolling(20).std()
    X['range20'] = ((df.high - df.low) / c).rolling(20).mean()
    return X.replace([np.inf, -np.inf], np.nan)

def rf_forecast(closed_df, df=None):
    if df is None:
        df = globals().get('_LAST_DF')
    candle = int(df.t.iloc[-1])
    cache_key = f"{CFG.symbol}_{CFG.interval}_{candle}"
    if cache_key in RF_CACHE:
        return RF_CACHE[cache_key]

    h = min(CFG.horizon, 24)
    X = features(df)
    future_high = pd.concat([df.high.shift(-k) for k in range(1, h + 1)], axis=1).max(axis=1)
    future_low = pd.concat([df.low.shift(-k) for k in range(1, h + 1)], axis=1).min(axis=1)
    future_high.iloc[-h:] = np.nan
    future_low.iloc[-h:] = np.nan

    y_hi = np.log(future_high / df.close)
    y_lo = np.log(future_low / df.close)
    y_hi = y_hi.replace([np.inf, -np.inf], np.nan)
    y_lo = y_lo.replace([np.inf, -np.inf], np.nan)
    valid = X.notna().all(axis=1) & y_hi.notna() & y_lo.notna()
    x, hi, lo = X.loc[valid], y_hi.loc[valid], y_lo.loc[valid]

    if len(x) < 120:
        raise RuntimeError('Historial insuficiente para entrenar Random Forest')

    cut = int(len(x) * 0.8)
    models, errors = [], []
    for target in (hi, lo):
        model = RandomForestRegressor(n_estimators=CFG.rf_trees, max_depth=5,
                                      min_samples_leaf=5, random_state=42, n_jobs=-1)
        model.fit(x.iloc[:cut], target.iloc[:cut])
        val_pred = model.predict(x.iloc[cut:])
        errors.append(float(np.mean(np.abs(val_pred - target.iloc[cut:].to_numpy()))) * 100)
        model.fit(x, target)
        models.append(model)

    latest = X.iloc[[-1]]
    spot = float(df.close.iloc[-1])
    predicted_high = spot * np.exp(float(models[0].predict(latest)[0]))
    predicted_low = spot * np.exp(float(models[1].predict(latest)[0]))

    result = {
        'spot': spot, 'high': float(predicted_high), 'low': float(predicted_low),
        'candle': candle, 'mae_hi_pct': errors[0], 'mae_lo_pct': errors[1]
    }
    RF_CACHE.clear()
    RF_CACHE[cache_key] = result
    return result

# =====================================================================
# 5. CONDICIONES DE ENTRADA Y SALIDA (RSI + BOLLINGER + EMA + VaR/CVaR)
# =====================================================================
def make_levels(side, price, forecast):
    """Define SL y TP combinando Random Forest con límites cuantitativos VaR/CVaR."""
    var_unit = max(STATE.get('last_var', 0.5) / 100.0, 0.004)
    cvar_unit = max(STATE.get('last_cvar', 1.0) / 100.0, var_unit * 1.3)

    if side == 'LONG':
        rf_sl_dist = max((price - forecast['low']) / price, 0.0)
        sl_dist = min(max(rf_sl_dist, var_unit), cvar_unit)
        rf_tp_dist = max((forecast['high'] - price) / price, 0.0)
        tp_dist = min(max(rf_tp_dist * 0.5, sl_dist * 0.35), sl_dist * 0.5)
        sl = price * (1.0 - sl_dist)
        tp = price * (1.0 + tp_dist)
    else:  # SHORT
        rf_sl_dist = max((forecast['high'] - price) / price, 0.0)
        sl_dist = min(max(rf_sl_dist, var_unit), cvar_unit)
        rf_tp_dist = max((price - forecast['low']) / price, 0.0)
        tp_dist = min(max(rf_tp_dist * 0.5, sl_dist * 0.35), sl_dist * 0.5)
        sl = price * (1.0 + sl_dist)
        tp = price * (1.0 - tp_dist)

    return {'sl': round(sl, 4), 'tp': round(tp, 4)}

def close_position(price, reason, candle, **_kw):
    pos = STATE['position']
    if not pos:
        return None
    side, entry = pos['side'], float(pos['entry'])
    qty = float(pos['quantity'])
    lev = float(pos.get('leverage', CFG.leverage))

    base_pnl = (price - entry) * qty * (1 if side == 'LONG' else -1)
    leveraged_pnl = base_pnl * lev

    trade = {
        'entrada_utc': pos['entry_utc'], 'salida_utc': now_iso(),
        'symbol': CFG.symbol, 'tipo': side, 'apalancamiento': f'{lev}x',
        'entrada': round(entry, 4), 'salida': round(price, 4), 'cantidad': qty,
        'sl_estimado': pos['sl'], 'tp_estimado': pos['tp'],
        'motivo_salida': reason, 'pnl_neto_usdt': round(leveraged_pnl, 4),
        'vela_salida': candle
    }
    write_csv(TRADES_FILE, trade)
    STATE['realized_pnl_usdt'] += leveraged_pnl
    STATE['position'] = None
    STATE['last_signal'] = f'Cierre {side} ({reason}): {leveraged_pnl:+.4f} USDT ({lev}x)'
    save_state()
    log_event('CIERRE', STATE['last_signal'], candle)
    return STATE['last_signal']

def open_position(side, price, forecast, candle, desc='', **_kw):
    if 'ROJO' in STATE.get('risk_status', ''):
        msg = f'🛑 OPERACIÓN {side} BLOQUEADA POR RIESGO (VaR > {CFG.var_limit_pct}% o CVaR > {CFG.cvar_limit_pct}%)'
        log_event('RIESGO_BLOQUEO', msg, candle)
        STATE['last_signal'] = msg
        save_state()
        return msg

    levels = make_levels(side, price, forecast)
    STATE['position'] = {
        'symbol': CFG.symbol, 'side': side, 'entry': round(price, 4),
        'entry_utc': now_iso(), 'leverage': CFG.leverage,
        'quantity': CFG.position_size, **levels
    }
    msg = (f'⚡ APERTURA {side} ({CFG.symbol}) | {desc}\n'
           f'Entrada: {fmt_p(price)} | 🎯 TP: {fmt_p(levels["tp"])} | 🛡️ SL: {fmt_p(levels["sl"])}')
    STATE['last_signal'] = msg
    save_state()
    log_event('APERTURA', msg, candle)
    return msg

def evaluate_signals(df):
    """Reglas explícitas y reproducibles de entrada."""
    bar = df.iloc[-1]
    prev_bar = df.iloc[-2]
    strat = CFG.strategy

    if strat == 'RSI_BB_EMA':
        recent = df.tail(12)
        touched_low = (recent.low <= recent.bb_mid).any()
        touched_high = (recent.high >= recent.bb_mid).any()
        rsi_low_ok = (recent.rsi.min() <= 48.0) and (bar.rsi < 76.0)
        rsi_high_ok = (recent.rsi.max() >= 52.0) and (bar.rsi > 24.0)
        confirm_bull = (bar.close >= bar.ema_fast) or (bar.rsi >= recent.rsi.min() + 1.5)
        confirm_bear = (bar.close <= bar.ema_fast) or (bar.rsi <= recent.rsi.max() - 1.5)
        bull = touched_low and rsi_low_ok and (bar.close > prev_bar.close) and confirm_bull
        bear = touched_high and rsi_high_ok and (bar.close < prev_bar.close) and confirm_bear
        detail = f"RSI={bar.rsi:.1f} [min12={recent.rsi.min():.1f}/max12={recent.rsi.max():.1f}], Close={fmt_p(bar.close)}, EMA9={fmt_p(bar.ema_fast)}"
        return ('LONG' if bull else ('SHORT' if bear else None)), f'RSI+BB+EMA ({detail})'

    elif strat == 'EMA_7_21':
        bull = (prev_bar.ema_fast <= prev_bar.ema_slow) and (bar.ema_fast > bar.ema_slow)
        bear = (prev_bar.ema_fast >= prev_bar.ema_slow) and (bar.ema_fast < bar.ema_slow)
        return ('LONG' if bull else ('SHORT' if bear else None)), 'Cruce EMA 9/21'

    elif strat == 'GOLDEN_CROSS':
        bull = (prev_bar.sma_50 <= prev_bar.sma_200) and (bar.sma_50 > bar.sma_200)
        bear = (prev_bar.sma_50 >= prev_bar.sma_200) and (bar.sma_50 < bar.sma_200)
        return ('LONG' if bull else ('SHORT' if bear else None)), 'Golden/Death Cross'

    elif strat == 'FIBONACCI':
        window = df.tail(60)
        hh, ll = window.high.max(), window.low.min()
        diff = hh - ll
        fib_618, fib_382 = hh - 0.618 * diff, hh - 0.382 * diff
        price = float(bar.close)
        if abs(price - fib_618) / price < 0.003 and bar.close > prev_bar.close:
            return 'LONG', 'Fibonacci Pullback 0.618'
        elif abs(price - fib_382) / price < 0.003 and bar.close < prev_bar.close:
            return 'SHORT', 'Fibonacci Rechazo 0.382'
        return None, 'Fibonacci'

    return None, 'Desconocida'

def step_strategy(df):
    globals()['_LAST_DF'] = df.iloc[:-1].copy() if df is not None and len(df) > 1 else df
    if df is None or getattr(df, 'empty', True) or len(df) < 25:
        return []
    with LOCK:
        closed_df = df.iloc[:-1].copy()
        live_bar = df.iloc[-1]
        candle = int(closed_df.t.iloc[-1])

        if STATE['last_candle'] is None:
            STATE['last_candle'] = candle - 1
            save_state()

        msgs = []
        pos = STATE['position']
        if pos and CFG.trading_enabled:
            side = pos['side']
            if side == 'LONG':
                if float(live_bar.low) <= pos['sl']:
                    msgs.append(close_position(float(pos['sl']), 'STOP LOSS (VaR/RF)', candle))
                elif float(live_bar.high) >= pos['tp']:
                    msgs.append(close_position(float(pos['tp']), 'TAKE PROFIT (RF)', candle))
            else:
                if float(live_bar.high) >= pos['sl']:
                    msgs.append(close_position(float(pos['sl']), 'STOP LOSS (VaR/RF)', candle))
                elif float(live_bar.low) <= pos['tp']:
                    msgs.append(close_position(float(pos['tp']), 'TAKE PROFIT (RF)', candle))
            if msgs:
                return msgs

        if candle <= STATE['last_candle']:
            return []
        STATE['last_candle'] = candle
        save_state()

        compute_var_cvar(closed_df)
        if not CFG.trading_enabled:
            return []

        bar = closed_df.iloc[-1]
        price = float(bar.close)
        pos = STATE['position']
        signal, desc = evaluate_signals(closed_df)

        if pos:
            side = pos['side']
            if side == 'LONG':
                if float(bar.low) <= pos['sl']:
                    msgs.append(close_position(float(pos['sl']), 'STOP LOSS (VaR/RF)', candle))
                elif float(bar.high) >= pos['tp']:
                    msgs.append(close_position(float(pos['tp']), 'TAKE PROFIT (RF)', candle))
                elif (bar.rsi > 56.0 and bar.close >= bar.bb_mid) or signal == 'SHORT':
                    msgs.append(close_position(price, f'Salida técnica RSI/BB ({bar.rsi:.1f})', candle))
                    if signal == 'SHORT':
                        msgs.append(open_position('SHORT', price, rf_forecast(closed_df), candle, desc))
            else:  # SHORT
                if float(bar.high) >= pos['sl']:
                    msgs.append(close_position(float(pos['sl']), 'STOP LOSS (VaR/RF)', candle))
                elif float(bar.low) <= pos['tp']:
                    msgs.append(close_position(float(pos['tp']), 'TAKE PROFIT (RF)', candle))
                elif (bar.rsi < 44.0 and bar.close <= bar.bb_mid) or signal == 'LONG':
                    msgs.append(close_position(price, f'Salida técnica RSI/BB ({bar.rsi:.1f})', candle))
                    if signal == 'LONG':
                        msgs.append(open_position('LONG', price, rf_forecast(closed_df), candle, desc))
            return msgs

        if signal:
            msgs.append(open_position(signal, price, rf_forecast(closed_df), candle, desc, df=closed_df))
        else:
            STATE['last_signal'] = f"Monitoreando: {desc}"
            save_state()
        return msgs

# =====================================================================
# 6. GRÁFICOS, MÉTRICAS ETAPA 5 Y REPORTES
# =====================================================================
def compute_performance():
    if not TRADES_FILE.exists():
        return {'total': 0, 'wins': 0, 'losses': 0, 'win_rate': 0.0,
                'net_pnl': 0.0, 'profit_factor': 0.0, 'max_dd': 0.0}
    tdf = pd.read_csv(TRADES_FILE)
    if tdf.empty:
        return {'total': 0, 'wins': 0, 'losses': 0, 'win_rate': 0.0,
                'net_pnl': 0.0, 'profit_factor': 0.0, 'max_dd': 0.0}
    pnls = tdf['pnl_neto_usdt'].astype(float)
    wins = int((pnls > 0).sum())
    losses = int((pnls <= 0).sum())
    total = len(pnls)
    win_rate = (wins / total) * 100.0
    gross_profit = float(pnls[pnls > 0].sum())
    gross_loss = abs(float(pnls[pnls < 0].sum()))
    pf = (gross_profit / gross_loss) if gross_loss > 0 else gross_profit
    cum = pnls.cumsum()
    dd = (cum.cummax() - cum).max()
    return {'total': total, 'wins': wins, 'losses': losses, 'win_rate': round(win_rate, 2),
            'net_pnl': round(float(pnls.sum()), 4), 'profit_factor': round(pf, 2),
            'max_dd': round(float(dd), 4)}

def make_chart(df):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    recent = df.tail(100).copy()
    dates = pd.to_datetime(recent.t, unit='s', utc=True)

    ax1.plot(dates, recent.close, label=f'{CFG.symbol} Precio', color='#222222', lw=1.4)
    ax1.plot(dates, recent.ema_fast, label='EMA 9', color='blue', lw=1.1)
    ax1.plot(dates, recent.bb_upper, label='Bollinger Sup', color='teal', ls='--', alpha=0.7)
    ax1.plot(dates, recent.bb_lower, label='Bollinger Inf', color='teal', ls='--', alpha=0.7)
    ax1.fill_between(dates, recent.bb_lower, recent.bb_upper, color='teal', alpha=0.06)

    pos = STATE.get('position')
    if pos:
        ax1.axhline(pos['entry'], color='black', ls='-.', label=f"Entrada ({pos['side']})")
        ax1.axhline(pos['tp'], color='green', ls=':', label='TP')
        ax1.axhline(pos['sl'], color='red', ls=':', label='SL')

    ax1.set_title(f"{CFG.strategy} | {CFG.symbol} ({CFG.interval}) | Riesgo: {STATE.get('risk_status', 'N/A')}")
    ax1.grid(alpha=0.3)
    ax1.legend(loc='upper left', fontsize=8)

    ax2.plot(dates, recent.rsi, label='RSI (14)', color='purple', lw=1.3)
    ax2.axhline(60, color='red', ls='--', alpha=0.6)
    ax2.axhline(40, color='green', ls='--', alpha=0.6)
    ax2.set_ylim(0, 100)
    ax2.grid(alpha=0.3)
    ax2.legend(loc='upper left', fontsize=8)

    fig.autofmt_xdate()
    fig.tight_layout()
    stream = io.BytesIO()
    fig.savefig(stream, format='png', dpi=120)
    plt.close(fig)
    stream.seek(0)
    return stream

def fmt_status():
    with LOCK:
        pos = STATE.get('position')
        perf = compute_performance()
        pos_txt = ('Sin posición activa' if not pos else
                   f"{pos['side']} ({pos['symbol']}) x{pos.get('leverage', CFG.leverage):.0f} | "
                   f"Ent: {fmt_p(pos['entry'])} | TP: {fmt_p(pos['tp'])} | SL: {fmt_p(pos['sl'])}")
        return (
            f"📋 **ESTATUS GENERAL DEL BOT**\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"🤖 Estado: {'🟢 ACTIVO' if CFG.trading_enabled else '🛑 PAUSADO'}\n"
            f"🎯 Ticker: `{CFG.symbol}` | Temporalidad: `{CFG.interval}`\n"
            f"📈 Estrategia: `{CFG.strategy}` | Apalancamiento: `{CFG.leverage:.0f}x`\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"🛡️ **Riesgo**: {STATE.get('risk_status', 'N/A')}\n"
            f"   • VaR 99%: `{STATE.get('last_var', 0.0):.2f}%` (Lím {CFG.var_limit_pct}%)\n"
            f"   • CVaR 99%: `{STATE.get('last_cvar', 0.0):.2f}%` (Lím {CFG.cvar_limit_pct}%)\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"📍 Posición: {pos_txt}\n"
            f"📊 **Desempeño (Etapa 5)**: `{perf['total']}/12` trades cerrados\n"
            f"   • Win Rate: `{perf['win_rate']}%` ({perf['wins']}G / {perf['losses']}P)\n"
            f"   • PnL Neto: `{perf['net_pnl']:+.4f} USDT` | PF: `{perf['profit_factor']}` | DD: `{perf['max_dd']:.4f}`\n"
            f"📢 Última señal: {STATE['last_signal']}"
        )

def fmt_forecast(r):
    return (
        f"🔮 **PREDICCIÓN RANDOM FOREST ({CFG.horizon} velas)**\n"
        f"Ticker: `{CFG.symbol}` | Intervalo: `{CFG.interval}`\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"💵 Spot Actual: `{fmt_p(r['spot'])}`\n"
        f"🟢 Máximo Predicho: `{fmt_p(r['high'])}`\n"
        f"🔴 Mínimo Predicho: `{fmt_p(r['low'])}`\n"
        f"📊 Error MAE: `{r['mae_hi_pct']:.2f}%` / `{r['mae_lo_pct']:.2f}%`"
    )

def ask_gemini(question):
    df = get_candles()
    forecast = rf_forecast(df)
    risk = compute_var_cvar(df)
    context = {
        'symbol': CFG.symbol, 'interval': CFG.interval, 'strategy': CFG.strategy,
        'var_99': f"{risk['var_pct']:.2f}%", 'cvar_99': f"{risk['cvar_pct']:.2f}%",
        'rf_high': forecast['high'], 'rf_low': forecast['low'],
        'spot': forecast['spot'], 'performance': compute_performance()
    }
    prompt = (
        "Eres un asesor cuantitativo de finanzas computacionales. Responde en español.\n"
        f"CONTEXTO: {json.dumps(context, ensure_ascii=False)}\nPREGUNTA: {question}"
    )
    client = genai.Client(api_key=GEMINI_API_KEY)
    for model_name in [GEMINI_MODEL, 'gemini-2.5-flash']:
        try:
            res = client.models.generate_content(model=model_name, contents=prompt)
            if res and res.text:
                return res.text
        except Exception:
            continue
    return "Error al conectar con la API de Gemini."

def make_excel():
    df = get_candles()
    output = DATA_DIR / f'reporte_{CFG.symbol}_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}.xlsx'
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.assign(fecha_utc=pd.to_datetime(df.t, unit='s', utc=True).dt.tz_localize(None))\
          .to_excel(writer, sheet_name='Historico_Velas', index=False)
        (pd.read_csv(TRADES_FILE) if TRADES_FILE.exists() else pd.DataFrame(
            columns=['entrada_utc', 'salida_utc', 'symbol', 'tipo', 'apalancamiento',
                     'entrada', 'salida', 'cantidad', 'sl_estimado', 'tp_estimado',
                     'motivo_salida', 'pnl_neto_usdt'])).to_excel(writer, sheet_name='Trades', index=False)
        pd.DataFrame([compute_performance()]).to_excel(writer, sheet_name='Desempeno_Etapa5', index=False)
        pd.DataFrame([asdict(CFG)]).to_excel(writer, sheet_name='Configuracion', index=False)
    return output

# =====================================================================
# 7. MANEJADORES DE TELEGRAM Y BUCLE MONITOR
# =====================================================================
def authorized(update: Update):
    return TELEGRAM_CHAT_ID is not None and str(update.effective_chat.id) == str(TELEGRAM_CHAT_ID)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global TELEGRAM_CHAT_ID
    if TELEGRAM_CHAT_ID is None:
        if update.effective_chat.type != 'private':
            return
        TELEGRAM_CHAT_ID = update.effective_chat.id
        PAIR_FILE.write_text(json.dumps({'chat_id': TELEGRAM_CHAT_ID}), encoding='utf-8')
    if not authorized(update):
        return await update.message.reply_text('Este bot ya está vinculado a otro chat.')
    context.user_data.clear()
    await update.message.reply_text(fmt_status(), reply_markup=MENU)

async def handle(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not authorized(update):
        return
    txt = (update.message.text or '').strip()

    if txt in ('❌ Cancelar', '⬅️ Volver'):
        context.user_data.clear()
        return await update.message.reply_text('Menú principal:', reply_markup=MENU)

    if txt.startswith(('⚙️', '🎯', '▶️', '🛑', '📋', '🛡️', '🔮', '📈', '📊', '🤖')):
        context.user_data.clear()

    if txt == '⚙️ Estrategia':
        return await update.message.reply_text('Selecciona la estrategia técnica:', reply_markup=MENU_STRAT)
    if txt == '⏱️ Temporalidad':
        context.user_data['choosing_tf'] = True
        return await update.message.reply_text(
            f'Elige o escribe una temporalidad para {CFG.symbol} (ej. 1m, 5m, 15m, 30m, 1h, 4h):',
            reply_markup=MENU_TF)
    if txt == '🎯 Elegir Ticker':
        return await update.message.reply_text('Elige o busca un par de futuros USDT:', reply_markup=MENU_TICKER)

    if txt in ('RSI + Bollinger + EMA', 'EMA 7/21', 'Golden/Death Cross (50/200)', 'Fibonacci Retracement'):
        strat_map = {'RSI + Bollinger + EMA': 'RSI_BB_EMA', 'EMA 7/21': 'EMA_7_21',
                     'Golden/Death Cross (50/200)': 'GOLDEN_CROSS', 'Fibonacci Retracement': 'FIBONACCI'}
        CFG.strategy = strat_map[txt]
        return await update.message.reply_text(f'✅ Estrategia: {txt}', reply_markup=MENU)

    if context.user_data.get('choosing_tf'):
        custom_tf = txt.strip().lower()
        try:
            await asyncio.to_thread(get_candles, CFG.symbol, custom_tf)
            context.user_data.pop('choosing_tf', None)
            CFG.interval = custom_tf
            return await update.message.reply_text(f'✅ Temporalidad validada: `{custom_tf}`', reply_markup=MENU)
        except Exception as e:
            return await update.message.reply_text(f'❌ Temporalidad `{custom_tf}` inválida para `{CFG.symbol}`: {e}', reply_markup=MENU_TF)

    if txt in ('XRP_USDT', 'BTC_USDT', 'ETH_USDT', 'SOL_USDT'):
        CFG.symbol = txt
        return await update.message.reply_text(f'✅ Par activo: {txt}', reply_markup=MENU)

    if txt == '🔍 Buscar otro ticker':
        context.user_data['searching_ticker'] = True
        return await update.message.reply_text('Escribe el símbolo a buscar (ej. ADA, DOGE, SOL):')

    if context.user_data.pop('searching_ticker', False):
        clean_sym = txt.upper().replace('/', '_').replace('-', '_')
        if not clean_sym.endswith('_USDT'):
            clean_sym = f"{clean_sym}_USDT"
        try:
            await asyncio.to_thread(get_candles, clean_sym, CFG.interval)
            CFG.symbol = clean_sym
            return await update.message.reply_text(f'✅ Par validado: `{clean_sym}`', reply_markup=MENU)
        except Exception as e:
            return await update.message.reply_text(f'❌ Error con `{clean_sym}`: {e}', reply_markup=MENU)

    if txt == '▶️ Activar trading':
        with LOCK:
            CFG.trading_enabled = True
            STATE['trading_enabled'] = True
            save_state()
        return await update.message.reply_text(f'▶️ Trading ACTIVADO ({CFG.strategy} | {CFG.symbol} | {CFG.interval}).', reply_markup=MENU)

    if txt == '🛑 Parar bots':
        with LOCK:
            CFG.trading_enabled = False
            STATE['trading_enabled'] = False
            save_state()
        return await update.message.reply_text('🛑 Bot pausado.', reply_markup=MENU)

    if txt == '📋 Estatus':
        return await update.message.reply_text(fmt_status(), reply_markup=MENU)

    if txt == '🛡️ VaR/CVaR (Semáforo)':
        df = await asyncio.to_thread(get_candles)
        risk = await asyncio.to_thread(compute_var_cvar, df)
        return await update.message.reply_text(
            f"🛡️ **RIESGO ({CFG.symbol} {CFG.interval})**\nSemáforo: {risk['status']}\n"
            f"VaR 99%: `{risk['var_pct']:.2f}%` | CVaR 99%: `{risk['cvar_pct']:.2f}%`", reply_markup=MENU)

    if txt == '🔮 Predicción RF':
        df = await asyncio.to_thread(get_candles)
        return await update.message.reply_text(fmt_forecast(await asyncio.to_thread(rf_forecast, df)), reply_markup=MENU)

    if txt == '📈 Gráfico':
        await update.message.reply_text('Generando gráfico RSI + Bollinger + EMA...')
        df = await asyncio.to_thread(get_candles)
        return await update.message.reply_photo(photo=await asyncio.to_thread(make_chart, df), reply_markup=MENU)

    if txt == '📊 Obtener Excel':
        await update.message.reply_text('Generando Excel con pestaña de desempeño...')
        path = await asyncio.to_thread(make_excel)
        with path.open('rb') as f:
            return await update.message.reply_document(document=f, filename=path.name, reply_markup=MENU)

    if txt == '🤖 Hablar con Gemini':
        context.user_data['asking'] = True
        return await update.message.reply_text('Escribe tu consulta para Gemini:')

    if context.user_data.pop('asking', False):
        ans = await asyncio.to_thread(ask_gemini, txt)
        return await update.message.reply_text(ans[:3900], reply_markup=MENU)

    await update.message.reply_text('Selecciona una opción del menú.', reply_markup=MENU)

async def monitor(app: Application):
    await asyncio.sleep(5)
    while True:
        try:
            df = await asyncio.to_thread(get_candles)
            for m in await asyncio.to_thread(step_strategy, df):
                if TELEGRAM_CHAT_ID and m:
                    await app.bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=f"🔔 **ACTUALIZACIÓN**\n{m}")
            await asyncio.sleep(20)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logging.exception('Error en monitor, reintentando...')
            with LOCK:
                STATE['last_candle'] = None
                save_state()
            if TELEGRAM_CHAT_ID:
                try:
                    await app.bot.send_message(chat_id=TELEGRAM_CHAT_ID,
                                               text=f' ⚠ Fallo del ciclo: {type(e).__name__}: {e}')
                except Exception:
                    pass
            with LOCK:
                STATE['last_signal'] = f'⚠️ Error monitor: {type(e).__name__}: {e}'
                save_state()
            await asyncio.sleep(20)

async def post_init(app: Application):
    asyncio.create_task(monitor(app))

def main():
    req = HTTPXRequest(connect_timeout=60.0, read_timeout=60.0, write_timeout=60.0, pool_timeout=60.0)
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).request(req).post_init(post_init).build()
    app.add_handler(CommandHandler('start', start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle))
    app.run_polling(drop_pending_updates=True)

_rf_forecast_raw = rf_forecast

def rf_forecast(df=None, *a, **k):
    # shortcut: si el RF falla, SL/TP salen de VaR/CVaR; revisar si el MAE importa para el reporte
    if df is None:
        df = globals().get('_LAST_DF')
    try:
        return _rf_forecast_raw(df, *a, **k)
    except Exception:
        logging.exception('RF no disponible; niveles por VaR/CVaR')
        spot = float(df.close.iloc[-1]) if df is not None and len(df) else 0.0
        band = max(STATE.get('last_cvar', 1.0) / 100.0, 0.005)
        return {'spot': spot, 'high': spot * (1 + band), 'low': spot * (1 - band),
                'candle': 0, 'mae_hi_pct': float('nan'), 'mae_lo_pct': float('nan')}


# =====================================================================
# 8. EJECUCIÓN REAL DE ÓRDENES EN GATE.IO TESTNET (futuros USDT)
# =====================================================================
import time

def _contract_params():
    api = gate_client()
    try:
        c = api.get_futures_contract(settle=SETTLE, contract=CFG.symbol)
    finally:
        api.api_client.close()
    mult = float(c.quanto_multiplier)
    n = max(int(c.order_size_min), int(round(CFG.position_size / mult)))
    return n, mult

def _set_leverage():
    api = gate_client()
    try:
        api.update_position_leverage(SETTLE, CFG.symbol, str(int(CFG.leverage)))
    except Exception:
        pass
    finally:
        api.api_client.close()

def _send_market(size, close=False):
    api = gate_client()
    try:
        order = gate_api.FuturesOrder(contract=CFG.symbol, size=size, price='0', tif='ioc', close=close)
        res = api.create_futures_order(SETTLE, order)
        time.sleep(1.0)
        res = api.get_futures_order(SETTLE, str(res.id))
    finally:
        api.api_client.close()
    if float(res.fill_price or 0) <= 0:
        raise RuntimeError(f'Orden {res.id} sin ejecutar ({res.status}/{res.finish_as})')
    return res

def _log_order(accion, side, res):
    write_csv(DATA_DIR / 'ordenes_gate.csv', {
        'utc': now_iso(), 'accion': accion, 'lado': side, 'order_id': res.id,
        'contrato': res.contract, 'size': res.size, 'fill_price': res.fill_price,
        'status': res.status, 'finish_as': res.finish_as})

def _fail(texto, candle):
    log_event('ORDEN_ERROR', texto, candle)
    STATE['last_signal'] = texto
    save_state()
    return texto

def open_position(side, price, forecast, candle, desc='', **_kw):
    if 'ROJO' in STATE.get('risk_status', ''):
        msg = f'🛑 OPERACIÓN {side} BLOQUEADA POR RIESGO (VaR > {CFG.var_limit_pct}% o CVaR > {CFG.cvar_limit_pct}%)'
        log_event('RIESGO_BLOQUEO', msg, candle)
        STATE['last_signal'] = msg
        save_state()
        return msg
    try:
        n, mult = _contract_params()
        _set_leverage()
        res = _send_market(n if side == 'LONG' else -n)
        fill = float(res.fill_price)
    except Exception as e:
        return _fail(f'⚠️ No se pudo abrir {side} en Gate: {type(e).__name__}: {str(e)[:150]}', candle)
    _log_order('APERTURA', side, res)
    levels = make_levels(side, fill, forecast)
    STATE['position'] = {
        'symbol': CFG.symbol, 'side': side, 'entry': round(fill, 4), 'entry_utc': now_iso(),
        'leverage': CFG.leverage, 'quantity': n * mult, 'contracts': n,
        'order_open': res.id, **levels}
    msg = (f'⚡ APERTURA {side} REAL ({CFG.symbol}) | {desc}\n'
           f'Orden Gate #{res.id} | {n} contrato(s) | Entrada: {fmt_p(fill)} | '
           f'🎯 TP: {fmt_p(levels["tp"])} | 🛡️ SL: {fmt_p(levels["sl"])}')
    STATE['last_signal'] = msg
    save_state()
    log_event('APERTURA', msg, candle)
    return msg

def close_position(price, reason, candle, **_kw):
    pos = STATE['position']
    if not pos:
        return None
    side, entry, qty = pos['side'], float(pos['entry']), float(pos['quantity'])
    res, exit_p, order_id = None, float(price), 'sin_orden'
    try:
        res = _send_market(0, close=True)
        exit_p, order_id = float(res.fill_price), res.id
    except Exception as e:
        if 'POSITION_EMPTY' not in str(e):
            return _fail(f'⚠️ No se pudo cerrar {side} en Gate: {type(e).__name__}: {str(e)[:150]}', candle)
        reason += ' (sin posición en Gate)'
    gross = (exit_p - entry) * qty * (1 if side == 'LONG' else -1)
    net, fee = gross, None
    try:
        api = gate_client()
        try:
            last = api.list_position_close(SETTLE, contract=CFG.symbol, limit=1)
        finally:
            api.api_client.close()
        if last and time.time() - float(last[0].time) < 60:
            net, fee = float(last[0].pnl), float(last[0].pnl_fee)
    except Exception:
        pass
    if res is not None:
        _log_order('CIERRE', side, res)
    write_csv(TRADES_FILE, {
        'entrada_utc': pos['entry_utc'], 'salida_utc': now_iso(), 'symbol': CFG.symbol,
        'tipo': side, 'apalancamiento': f"{pos.get('leverage', CFG.leverage)}x",
        'entrada': round(entry, 4), 'salida': round(exit_p, 4), 'cantidad': qty,
        'sl_estimado': pos['sl'], 'tp_estimado': pos['tp'], 'motivo_salida': reason,
        'pnl_gross_usdt': round(gross, 4), 'fee_usdt': fee, 'pnl_neto_usdt': round(net, 4),
        'order_id_apertura': pos.get('order_open'), 'order_id_cierre': order_id,
        'vela_salida': candle})
    STATE['realized_pnl_usdt'] += net
    STATE['position'] = None
    STATE['last_signal'] = f'Cierre {side} ({reason}): {net:+.4f} USDT netos (Gate)'
    save_state()
    log_event('CIERRE', STATE['last_signal'], candle)
    return STATE['last_signal']


# =====================================================================
# 9. VERSION 2 "CONSCIENTE DE COSTOS" (hipótesis definidas antes de la prueba)
#    H1: sin salidas técnicas RSI/BB ni inversión de posición -> menos operaciones y menos comisiones
#    H2: relación SL/TP de ~3:1 a 1:1 (TP entre 0.35% y 0.70% según el RF; SL acotado por CVaR)
#    H3: cooldown de 3 velas tras cada cierre y salida por tiempo a las 36 velas (3 h)
#    Tamaño de 100 XRP (~97 USDT de exposición, pensado para una cuenta de ~100 USDT)
# =====================================================================
TP_MIN, TP_MAX, SL_PER_TP, MAX_BARS, COOLDOWN = 0.0035, 0.0070, 1.0, 36, 3
FEE_TAKER = 0.0005
CFG.position_size = 100.0

def make_levels(side, price, forecast):
    var_unit = max(STATE.get('last_var', 0.5) / 100.0, 0.004)
    cvar_unit = max(STATE.get('last_cvar', 1.0) / 100.0, var_unit * 1.3)
    sgn = 1 if side == 'LONG' else -1
    rf_tp = max(sgn * (forecast['high' if side == 'LONG' else 'low'] - price) / price, 0.0)
    tp_dist = min(max(rf_tp, TP_MIN), TP_MAX)
    sl_dist = min(tp_dist * SL_PER_TP, cvar_unit)
    return {'sl': round(price * (1 - sgn * sl_dist), 4), 'tp': round(price * (1 + sgn * tp_dist), 4)}

def _hit(pos, bar):
    if pos['side'] == 'LONG':
        if float(bar.low) <= pos['sl']:
            return float(pos['sl']), 'STOP LOSS (VaR/RF)'
        if float(bar.high) >= pos['tp']:
            return float(pos['tp']), 'TAKE PROFIT (RF)'
    else:
        if float(bar.high) >= pos['sl']:
            return float(pos['sl']), 'STOP LOSS (VaR/RF)'
        if float(bar.low) <= pos['tp']:
            return float(pos['tp']), 'TAKE PROFIT (RF)'
    return None

def _net_and_fee(gross, entry, exit_p, qty):
    for _ in range(3):
        try:
            api = gate_client()
            try:
                last = api.list_position_close(SETTLE, contract=CFG.symbol, limit=1)
            finally:
                api.api_client.close()
            if last and time.time() - float(last[0].time) < 90:
                return float(last[0].pnl), float(last[0].pnl_fee), 'gate'
        except Exception:
            pass
        time.sleep(1.5)
    fee = -FEE_TAKER * (entry + exit_p) * qty
    return gross + fee, fee, 'estimada'

def close_position(price, reason, candle, **_kw):
    pos = STATE['position']
    if not pos:
        return None
    side, entry, qty = pos['side'], float(pos['entry']), float(pos['quantity'])
    res, exit_p, order_id = None, float(price), 'sin_orden'
    try:
        res = _send_market(0, close=True)
        exit_p, order_id = float(res.fill_price), res.id
    except Exception as e:
        if 'POSITION_EMPTY' not in str(e):
            return _fail(f'⚠️ No se pudo cerrar {side} en Gate: {type(e).__name__}: {str(e)[:150]}', candle)
        reason += ' (sin posición en Gate)'
    gross = (exit_p - entry) * qty * (1 if side == 'LONG' else -1)
    net, fee, src = _net_and_fee(gross, entry, exit_p, qty)
    if res is not None:
        _log_order('CIERRE', side, res)
    write_csv(TRADES_FILE, {
        'version': 'v2-costos', 'entrada_utc': pos['entry_utc'], 'salida_utc': now_iso(), 'symbol': CFG.symbol,
        'tipo': side, 'apalancamiento': f"{pos.get('leverage', CFG.leverage)}x",
        'entrada': round(entry, 4), 'salida': round(exit_p, 4), 'cantidad': qty,
        'sl_estimado': pos['sl'], 'tp_estimado': pos['tp'], 'motivo_salida': reason,
        'pnl_gross_usdt': round(gross, 4), 'fee_usdt': round(fee, 5), 'fee_fuente': src,
        'pnl_neto_usdt': round(net, 4), 'retorno_neto_pct': round(net / (entry * qty) * 100, 4),
        'velas_en_posicion': pos.get('bars_held', 0),
        'order_id_apertura': pos.get('order_open'), 'order_id_cierre': order_id, 'vela_salida': candle})
    STATE['realized_pnl_usdt'] += net
    STATE['position'] = None
    STATE['cool'] = COOLDOWN
    STATE['last_signal'] = f'Cierre {side} ({reason}): {net:+.4f} USDT netos (Gate)'
    save_state()
    log_event('CIERRE', STATE['last_signal'], candle)
    return STATE['last_signal']

def step_strategy(df):
    globals()['_LAST_DF'] = df.iloc[:-1].copy() if df is not None and len(df) > 1 else df
    if df is None or getattr(df, 'empty', True) or len(df) < 25:
        return []
    with LOCK:
        closed_df = df.iloc[:-1].copy()
        candle = int(closed_df.t.iloc[-1])
        if STATE['last_candle'] is None:
            STATE['last_candle'] = candle - 1
            save_state()
        msgs, pos = [], STATE['position']
        if pos and CFG.trading_enabled:
            h = _hit(pos, df.iloc[-1])
            if h:
                return [close_position(h[0], h[1], candle)]
        if candle <= STATE['last_candle']:
            return []
        STATE['last_candle'] = candle
        save_state()
        compute_var_cvar(closed_df)
        if not CFG.trading_enabled:
            return []
        price = float(closed_df.close.iloc[-1])
        pos = STATE['position']
        if pos:
            h = _hit(pos, closed_df.iloc[-1])
            if h:
                return [close_position(h[0], h[1], candle)]
            pos['bars_held'] = int(pos.get('bars_held', 0)) + 1
            if pos['bars_held'] >= MAX_BARS:
                return [close_position(price, f'TIME STOP ({MAX_BARS} velas)', candle)]
            save_state()
            return msgs
        if STATE.get('cool', 0) > 0:
            STATE['cool'] -= 1
            save_state()
            return msgs
        signal, desc = evaluate_signals(closed_df)
        if signal:
            msgs.append(open_position(signal, price, rf_forecast(closed_df), candle, desc, df=closed_df))
        else:
            STATE['last_signal'] = f'Monitoreando: {desc}'
            save_state()
        return msgs


if __name__ == '__main__':
    main()
