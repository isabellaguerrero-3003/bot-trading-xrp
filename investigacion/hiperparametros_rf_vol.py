#!/usr/bin/env python3
"""Versión 2: el Random Forest predice el máximo/mínimo ESTANDARIZADO por la volatilidad reciente (vol20).
Los árboles no extrapolan: si la volatilidad de test sale del rango visto en train, fallan.
Aquí el objetivo es z = log(max_futuro/cierre) / vol20 y la predicción se reescala: pred = z_hat * vol20.
Requiere hiperparametros_rf.py en la misma carpeta."""
import itertools
import pandas as pd
import hiperparametros_rf as m

OUT = m.OUT

def fp_norm(kind, kw, Xtr, Ztr, Xte, Ste):
    return {t: m.make_model(kind, **kw).fit(Xtr, Ztr[t]).predict(Xte) * Ste.to_numpy() for t in ('hi', 'lo')}

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = m.add_indicators(m.fetch_history())
    X, Y, close = m.build_dataset(df)
    S = X['vol20'].clip(lower=1e-5)
    Z = Y.div(S, axis=0)
    n, H = len(X), m.H
    i1, i2 = int(n * .6), int(n * .8)
    tr, va, trva, te = slice(0, i1 - H), slice(i1, i2 - H), slice(0, i2 - H), slice(i2, n)
    print(f'Muestras utilizables: {n} | objetivo estandarizado por vol20')

    rows = []
    for ne, md, ml in itertools.product(*m.GRID.values()):
        kw = dict(n_estimators=ne, max_depth=md, min_samples_leaf=ml)
        P = fp_norm('rf', kw, X.iloc[tr], Z.iloc[tr], X.iloc[va], S.iloc[va])
        rows.append({'Modelo': f'RF-vol n={ne} prof={md} hoja={ml}', 'kw': kw, **m.score(close.iloc[va], Y.iloc[va], P)})
    P = fp_norm('base', {}, X.iloc[tr], Z.iloc[tr], X.iloc[va], S.iloc[va])
    rows.append({'Modelo': 'Base-vol: media(z) x vol20', 'kw': None, **m.score(close.iloc[va], Y.iloc[va], P)})
    P = m.fit_predict('base', {}, X.iloc[tr], Y.iloc[tr], X.iloc[va])
    rows.append({'Modelo': 'Base: media histórica', 'kw': None, **m.score(close.iloc[va], Y.iloc[va], P)})
    grid = pd.DataFrame(rows).sort_values('RMSE').reset_index(drop=True)
    best_kw = grid[grid.kw.notna()].iloc[0].kw
    shown = grid.drop(columns='kw').round({'RMSE': 5, 'MAE': 5, 'MAPE': 4, 'SMAPE': 4, 'VarScore': 4})
    shown.to_csv(OUT / 'tabla_hiperparametros_vol.csv', index=False)
    m.save_table(shown, OUT / 'tabla_hiperparametros_vol.png',
                 'Hiperparámetros con objetivo estandarizado por volatilidad (validación)',
                 hl=int(grid[grid.kw.notna()].index[0]))
    print('Mejor configuración RF-vol (validación):', best_kw)

    Yte, cte, Ste = Y.iloc[te], close.iloc[te], S.iloc[te]
    preds = {'RF-vol mejor': fp_norm('rf', best_kw, X.iloc[trva], Z.iloc[trva], X.iloc[te], Ste),
             'RF-vol bot (150,5,5)': fp_norm('rf', m.BOT_CFG, X.iloc[trva], Z.iloc[trva], X.iloc[te], Ste),
             'Base-vol: media(z) x vol20': fp_norm('base', {}, X.iloc[trva], Z.iloc[trva], X.iloc[te], Ste),
             'Base: media histórica': m.fit_predict('base', {}, X.iloc[trva], Y.iloc[trva], X.iloc[te])}
    test = pd.DataFrame([{'Modelo': k, **m.score(cte, Yte, P)} for k, P in preds.items()]).round(5)
    test.to_csv(OUT / 'resultados_test_vol.csv', index=False)
    print('\nResultados en TEST:\n', test.to_string(index=False))

    err = {k: {t: Yte[t].to_numpy() - P[t] for t in ('hi', 'lo')} for k, P in preds.items()}
    pairs = [('RF-vol mejor', 'Base: media histórica'), ('RF-vol mejor', 'Base-vol: media(z) x vol20'),
             ('Base-vol: media(z) x vol20', 'Base: media histórica')]
    out = []
    for a, b in pairs:
        row = {'Modelo 1': a, 'Modelo 2': b}
        for t, lab in (('hi', 'Máximo'), ('lo', 'Mínimo')):
            s, p = m.dm_test(err[a][t], err[b][t])
            row[f'DM {lab}'], row[f'p {lab}'] = round(s, 3), round(p, 4)
        ok = max(row['p Máximo'], row['p Mínimo']) < 0.05 and row['DM Máximo'] < 0 and row['DM Mínimo'] < 0
        row['Conclusión'] = 'Modelo 1 mejor (p<0.05)' if ok else 'Sin diferencia significativa / mixto'
        out.append(row)
    dm = pd.DataFrame(out)
    dm.to_csv(OUT / 'dm_test_vol.csv', index=False)
    m.save_table(dm, OUT / 'dm_test_vol.png', 'Diebold-Mariano en TEST con objetivo estandarizado por volatilidad (DM<0: gana Modelo 1)')
    print('\nDiebold-Mariano:\n', dm.to_string(index=False))
    print('\nArchivos en', OUT)

if __name__ == '__main__':
    main()
