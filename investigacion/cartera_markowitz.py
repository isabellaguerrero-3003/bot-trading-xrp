#!/usr/bin/env python3
"""Cartera multi-cripto: correlación de Pearson + Markowitz para ponderar el capital.
Activos: los del menú del bot (XRP, BTC, ETH, SOL). Datos diarios de Gate.io (futuros USDT, datos públicos).
Validación fuera de muestra: pesos estimados en el 70% inicial y evaluados en el 30% final."""
import json, os
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

TICKERS = os.getenv('TICKERS', 'XRP_USDT,BTC_USDT,ETH_USDT,SOL_USDT').split(',')
HOST = os.getenv('GATE_HOST', 'https://api.gateio.ws/api/v4')
N_DAYS, CAPITAL, W_MAX, PPY = int(os.getenv('N_DAYS', '730')), float(os.getenv('CAPITAL', '100')), float(os.getenv('W_MAX', '0.6')), 365
OUT = Path(os.path.expanduser('~/resultados_cartera'))

def fetch_close(sym):
    import gate_api
    api = gate_api.FuturesApi(gate_api.ApiClient(gate_api.Configuration(host=HOST)))
    bars = api.list_futures_candlesticks(settle='usdt', contract=sym, interval='1d', limit=N_DAYS)
    s = pd.Series({pd.to_datetime(int(b.t), unit='s').normalize(): float(b.c) for b in bars}).sort_index()
    return s.iloc[:-1]            # descarta la vela del día en curso (incompleta)

def get_returns():
    px = pd.concat({t: fetch_close(t) for t in TICKERS}, axis=1).dropna()
    return px.pct_change().dropna()

ESCALA = 1e4  # evita que SLSQP se detenga en el punto inicial por varianzas ~1e-3
def optimize(kind, mu, cov):
    mu, cov = mu * 100, cov * ESCALA
    n = len(mu)
    f = (lambda w: w @ cov @ w) if kind == 'minvar' else (lambda w: -(w @ mu) / np.sqrt(w @ cov @ w))
    r = minimize(f, np.full(n, 1 / n), method='SLSQP', bounds=[(0, W_MAX)] * n,
                 constraints=({'type': 'eq', 'fun': lambda w: w.sum() - 1},))
    return r.x

def portfolios(R):
    mu, cov = R.mean().to_numpy(), LedoitWolf().fit(R).covariance_
    inv = 1 / np.sqrt(np.diag(cov))
    return {'Mínima varianza': optimize('minvar', mu, cov), 'Máximo Sharpe': optimize('maxsharpe', mu, cov),
            'Paridad de riesgo (1/σ)': inv / inv.sum(), 'Igual ponderación (1/N)': np.full(len(mu), 1 / len(mu))}, mu, cov

def perf(r):
    r = pd.Series(r)
    cum = (1 + r).cumprod()
    return {'Ret_anual_%': ((cum.iloc[-1]) ** (PPY / len(r)) - 1) * 100, 'Vol_anual_%': r.std() * np.sqrt(PPY) * 100,
            'Sharpe': r.mean() / r.std() * np.sqrt(PPY),
            'Sortino': r.mean() / np.sqrt((np.minimum(r, 0) ** 2).mean()) * np.sqrt(PPY),
            'MaxDD_%': (cum / cum.cummax() - 1).min() * 100}

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    R = get_returns()
    print(f'Activos: {TICKERS} | días en común: {len(R)} | {R.index[0].date()} a {R.index[-1].date()}')
    # 1) Correlación de Pearson
    C = R.corr()
    C.round(3).to_csv(OUT / 'correlacion.csv')
    off = C.to_numpy()[np.triu_indices(len(C), 1)]
    print('\nCorrelación de Pearson (retornos diarios):\n', C.round(2).to_string())
    print(f'Correlación promedio entre pares: {off.mean():.2f} (mín {off.min():.2f}, máx {off.max():.2f})')
    fig, ax = plt.subplots(figsize=(5.5, 4.6)); im = ax.imshow(C, vmin=0, vmax=1, cmap='YlOrRd')
    ax.set_xticks(range(len(C))); ax.set_xticklabels([c.split('_')[0] for c in C], rotation=0)
    ax.set_yticks(range(len(C))); ax.set_yticklabels([c.split('_')[0] for c in C])
    for i in range(len(C)):
        for j in range(len(C)):
            ax.text(j, i, f'{C.iloc[i, j]:.2f}', ha='center', va='center', fontsize=9)
    ax.set_title('Correlación de Pearson (retornos diarios)'); fig.colorbar(im); fig.savefig(OUT / 'correlacion.png', dpi=160, bbox_inches='tight'); plt.close(fig)

    # 2) Validación fuera de muestra: pesos con 70% inicial, evaluación en el 30% final
    cut = int(len(R) * .7)
    W_tr, _, _ = portfolios(R.iloc[:cut])
    oos = {k: R.iloc[cut:].to_numpy() @ w for k, w in W_tr.items()}
    tab = pd.DataFrame({k: perf(v) for k, v in oos.items()}).T.round(2)
    tab.to_csv(OUT / 'oos_resultados.csv')
    print(f'\nFUERA DE MUESTRA (pesos del {cut} primeros días, evaluados en los {len(R) - cut} siguientes):\n', tab.to_string())
    fig, ax = plt.subplots(figsize=(8, 4))
    for k, v in oos.items():
        ax.plot(R.index[cut:], (1 + pd.Series(v)).cumprod().to_numpy(), label=k)
    ax.set_title('Valor acumulado fuera de muestra (base 1)'); ax.legend(fontsize=8); fig.savefig(OUT / 'oos_acumulado.png', dpi=160, bbox_inches='tight'); plt.close(fig)

    # 3) Pesos para desplegar (muestra completa) y frontera eficiente
    W, mu, cov = portfolios(R)
    wtab = pd.DataFrame(W, index=TICKERS).T.round(3)
    wtab.to_csv(OUT / 'pesos_cartera.csv')
    print('\nPESOS (muestra completa, tope por activo =', W_MAX, '):\n', wtab.to_string())
    print(f'\nReparto de {CAPITAL:.0f} USDT (máximo Sharpe):', {t: round(float(w) * CAPITAL, 1) for t, w in zip(TICKERS, W['Máximo Sharpe'])})
    json.dump({'capital_usdt': CAPITAL, 'pesos': {k: dict(zip(TICKERS, map(float, w))) for k, w in W.items()}}, open(OUT / 'pesos_cartera.json', 'w'), indent=2)
    pts = []
    for tgt in np.linspace(mu.min(), mu.max(), 40):
        r = minimize(lambda w: (w @ cov @ w) * ESCALA, np.full(len(mu), 1 / len(mu)), method='SLSQP', bounds=[(0, W_MAX)] * len(mu),
                     constraints=({'type': 'eq', 'fun': lambda w: w.sum() - 1}, {'type': 'eq', 'fun': lambda w, t=tgt: (w @ mu - t) * 100}))
        if r.success:
            pts.append((np.sqrt(r.x @ cov @ r.x) * np.sqrt(PPY) * 100, tgt * PPY * 100))
    fig, ax = plt.subplots(figsize=(7, 4.6))
    if pts:
        ax.plot(*zip(*pts), color='gray', label='Frontera eficiente')
    ax.scatter(np.sqrt(np.diag(cov)) * np.sqrt(PPY) * 100, mu * PPY * 100, c='tab:blue', label='Activos')
    for t, x, y in zip(TICKERS, np.sqrt(np.diag(cov)) * np.sqrt(PPY) * 100, mu * PPY * 100):
        ax.annotate(t.split('_')[0], (x, y), textcoords='offset points', xytext=(5, 5), fontsize=8)
    for k, w in W.items():
        ax.scatter(np.sqrt(w @ cov @ w) * np.sqrt(PPY) * 100, w @ mu * PPY * 100, marker='*', s=140, label=k)
    ax.set_xlabel('Volatilidad anual (%)'); ax.set_ylabel('Retorno anual aritmético (%)'); ax.legend(fontsize=7)
    ax.set_title('Frontera eficiente long-only (covarianza Ledoit-Wolf)'); fig.savefig(OUT / 'frontera_eficiente.png', dpi=160, bbox_inches='tight'); plt.close(fig)
    print('\nArchivos en', OUT)

if __name__ == '__main__':
    main()
