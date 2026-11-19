#!/usr/bin/env python3
"""Simulación de hiperparámetros del Random Forest + test Diebold-Mariano.
Activo XRP_USDT (Gate.io), velas 5m. Mismos features y objetivos que el bot:
predecir el máximo y el mínimo de las próximas H velas (en log-retorno sobre el cierre).
Validación temporal 60/20/20 (train/validación/test) con hueco H para evitar fuga de datos."""
import itertools, os
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import explained_variance_score

SYMBOL = os.getenv('SYMBOL', 'XRP_USDT')
INTERVAL = os.getenv('INTERVAL', '5m')
N_CANDLES = int(os.getenv('N_CANDLES', '4000'))
HOST = os.getenv('GATE_HOST', 'https://api-testnet.gateapi.io/api/v4')
H = 12
OUT = Path(os.path.expanduser('~/resultados_rf'))
GRID = {'n_estimators': [50, 150, 300], 'max_depth': [3, 5, 8], 'min_samples_leaf': [5, 20]}
BOT_CFG = dict(n_estimators=150, max_depth=5, min_samples_leaf=5)   # configuración actual del bot

# ---------- 1. Datos ----------
def fetch_history(n=N_CANDLES):
    import gate_api
    api = gate_api.FuturesApi(gate_api.ApiClient(gate_api.Configuration(host=HOST)))
    rows, to = [], None
    while len(rows) < n:
        kw = dict(settle='usdt', contract=SYMBOL, interval=INTERVAL, limit=min(2000, n - len(rows)))
        if to:
            kw['to'] = to
        bars = api.list_futures_candlesticks(**kw)
        if not bars:
            break
        rows += [dict(t=int(b.t), open=float(b.o), high=float(b.h), low=float(b.l), close=float(b.c)) for b in bars]
        new_to = min(int(b.t) for b in bars) - 1
        if to is not None and new_to >= to:
            break
        to = new_to
    return pd.DataFrame(rows).sort_values('t').drop_duplicates('t').reset_index(drop=True)

def add_indicators(df):
    c = df.close
    df['ema_fast'] = c.ewm(span=9, adjust=False).mean()
    df['ema_slow'] = c.ewm(span=21, adjust=False).mean()
    d = c.diff()
    g = d.clip(lower=0).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    l = (-d.clip(upper=0)).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    df['rsi'] = (100 - 100 / (1 + g / l.replace(0, np.nan))).fillna(50.0)
    mid, sd = c.rolling(20).mean(), c.rolling(20).std()
    df['bb_width'] = 4.0 * sd / mid
    return df

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

def build_dataset(df):
    fh = pd.concat([df.high.shift(-k) for k in range(1, H + 1)], axis=1).max(axis=1)
    fl = pd.concat([df.low.shift(-k) for k in range(1, H + 1)], axis=1).min(axis=1)
    X = features(df)
    Y = pd.DataFrame({'hi': np.log(fh / df.close), 'lo': np.log(fl / df.close)})
    ok = X.notna().all(axis=1) & Y.notna().all(axis=1)
    return X[ok].reset_index(drop=True), Y[ok].reset_index(drop=True), df.close[ok].reset_index(drop=True)

# ---------- 2. Métricas y test Diebold-Mariano ----------
def metrics(close, y, p):
    pt, pp = close * np.exp(y), close * np.exp(p)
    e = pt - pp
    return dict(RMSE=float(np.sqrt(np.mean(e**2))), MAE=float(np.mean(np.abs(e))),
                MAPE=float(np.mean(np.abs(e) / pt) * 100),
                SMAPE=float(np.mean(2 * np.abs(e) / (np.abs(pt) + np.abs(pp))) * 100),
                VarScore=float(explained_variance_score(y, p)))

def dm_test(e1, e2, h=H):
    """Diebold-Mariano (pérdida cuadrática) con corrección Harvey-Leybourne-Newbold.
    Estadístico < 0 => el modelo 1 tiene menor error que el modelo 2."""
    d = np.asarray(e1)**2 - np.asarray(e2)**2
    n, dbar = len(d), d.mean()
    dc = d - dbar
    lrv = np.sum(dc * dc) / n + 2 * sum(np.sum(dc[k:] * dc[:-k]) / n for k in range(1, h))
    if lrv <= 0:
        lrv = np.sum(dc * dc) / n
    dm = dbar / np.sqrt(lrv / n)
    dm *= np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    return float(dm), float(2 * (1 - stats.t.cdf(abs(dm), df=n - 1)))

# ---------- 3. Modelos ----------
def make_model(kind, **kw):
    if kind == 'rf':
        return RandomForestRegressor(random_state=42, n_jobs=2, **kw)
    if kind == 'knn':
        return make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=15))
    return DummyRegressor(strategy='mean')

def fit_predict(kind, kw, X_tr, Y_tr, X_te):
    return {t: make_model(kind, **kw).fit(X_tr, Y_tr[t]).predict(X_te) for t in ('hi', 'lo')}

def score(close, Y, P):
    m = [metrics(close, Y[t].to_numpy(), P[t]) for t in ('hi', 'lo')]
    return {k: (m[0][k] + m[1][k]) / 2 for k in m[0]}

def save_table(df, path, title, hl=None):
    fig, ax = plt.subplots(figsize=(12, 0.38 * len(df) + 1.4)); ax.axis('off')
    ax.set_title(title, fontsize=11, pad=12)
    tb = ax.table(cellText=df.values, colLabels=list(df.columns), loc='center', cellLoc='center')
    tb.auto_set_font_size(False); tb.set_fontsize(8); tb.scale(1, 1.3)
    for j in range(len(df.columns)):
        tb[0, j].set_facecolor('#2c3e50'); tb[0, j].set_text_props(color='white', weight='bold')
        if hl is not None:
            tb[hl + 1, j].set_facecolor('#d5f5e3')
    fig.savefig(path, dpi=160, bbox_inches='tight'); plt.close(fig)

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = add_indicators(fetch_history())
    X, Y, close = build_dataset(df)
    n = len(X)
    print(f'Velas descargadas: {len(df)} | muestras utilizables: {n} | {SYMBOL} {INTERVAL} | H={H}')
    if n < 1000:
        print('AVISO: pocas muestras; los resultados serán poco estables.')
    i1, i2 = int(n * .6), int(n * .8)
    tr, va = slice(0, i1 - H), slice(i1, i2 - H)
    trva, te = slice(0, i2 - H), slice(i2, n)

    # --- Rejilla de hiperparámetros evaluada en validación ---
    rows = []
    for ne, md, ml in itertools.product(*GRID.values()):
        kw = dict(n_estimators=ne, max_depth=md, min_samples_leaf=ml)
        P = fit_predict('rf', kw, X.iloc[tr], Y.iloc[tr], X.iloc[va])
        rows.append({'Modelo': f'RF n={ne} prof={md} hoja={ml}', 'kw': kw, **score(close.iloc[va], Y.iloc[va], P)})
    for name, kind in (('KNN k=15', 'knn'), ('Base: media histórica', 'base')):
        P = fit_predict(kind, {}, X.iloc[tr], Y.iloc[tr], X.iloc[va])
        rows.append({'Modelo': name, 'kw': None, **score(close.iloc[va], Y.iloc[va], P)})
    grid = pd.DataFrame(rows).sort_values('RMSE').reset_index(drop=True)
    best_kw = grid[grid.kw.notna()].iloc[0].kw
    shown = grid.drop(columns='kw').round({'RMSE': 5, 'MAE': 5, 'MAPE': 4, 'SMAPE': 4, 'VarScore': 4})
    shown.to_csv(OUT / 'tabla_hiperparametros.csv', index=False)
    save_table(shown, OUT / 'tabla_hiperparametros.png',
               f'Simulación de hiperparámetros (validación, {SYMBOL} {INTERVAL}). Métricas promedio máx/mín; RMSE/MAE en USDT, MAPE/SMAPE en %',
               hl=int(grid[grid.kw.notna()].index[0]))
    print('\nMejor configuración RF (validación):', best_kw)

    # --- Evaluación final en test + Diebold-Mariano ---
    cands = {'RF mejor': ('rf', best_kw), 'RF bot actual': ('rf', BOT_CFG),
             'KNN k=15': ('knn', {}), 'Base: media histórica': ('base', {})}
    preds = {k: fit_predict(kind, kw, X.iloc[trva], Y.iloc[trva], X.iloc[te]) for k, (kind, kw) in cands.items()}
    Yte, cte = Y.iloc[te], close.iloc[te]
    test = pd.DataFrame([{'Modelo': k, **score(cte, Yte, P)} for k, P in preds.items()]).round(5)
    test.to_csv(OUT / 'resultados_test.csv', index=False)
    print('\nResultados en TEST:\n', test.to_string(index=False))

    err = {k: {t: Yte[t].to_numpy() - P[t] for t in ('hi', 'lo')} for k, P in preds.items()}
    pairs = [('RF mejor', 'Base: media histórica'), ('RF mejor', 'KNN k=15'),
             ('RF mejor', 'RF bot actual'), ('RF bot actual', 'Base: media histórica')]
    dm_rows = []
    for a, b in pairs:
        row = {'Modelo 1': a, 'Modelo 2': b}
        for t, lab in (('hi', 'Máximo'), ('lo', 'Mínimo')):
            s, p = dm_test(err[a][t], err[b][t])
            row[f'DM {lab}'], row[f'p {lab}'] = round(s, 3), round(p, 4)
        row['Conclusión'] = ('Modelo 1 mejor (p<0.05)' if max(row['p Máximo'], row['p Mínimo']) < 0.05 and row['DM Máximo'] < 0 and row['DM Mínimo'] < 0
                             else 'Sin diferencia significativa / mixto')
        dm_rows.append(row)
    dm = pd.DataFrame(dm_rows)
    dm.to_csv(OUT / 'dm_test.csv', index=False)
    save_table(dm, OUT / 'dm_test.png', f'Test Diebold-Mariano en TEST (pérdida cuadrática, HLN, h={H}). DM<0: gana el Modelo 1')
    print('\nDiebold-Mariano:\n', dm.to_string(index=False))
    print('\nArchivos en', OUT)

if __name__ == '__main__':
    main()
