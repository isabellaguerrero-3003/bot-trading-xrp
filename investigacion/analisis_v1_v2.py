import pandas as pd, numpy as np, glob, json
from scipy import stats
pd.set_option('display.width',250); pd.set_option('display.max_columns',40)
v1 = pd.read_csv('datos/trades_v1.csv'); v2 = pd.read_csv('datos/trades_v2_muestra_oficial.csv')
def prep(d, version):
    d = d.copy(); d['version']=version
    d['entrada_utc']=pd.to_datetime(d.entrada_utc); d['salida_utc']=pd.to_datetime(d.salida_utc)
    fee_est = -0.0005*(d.entrada+d.salida)*d.cantidad
    d['fee_fix'] = d.fee_usdt.fillna(fee_est)
    d['net_fix'] = d.pnl_gross_usdt + d.fee_fix
    d['notional'] = d.entrada*d.cantidad
    d['ret_net'] = d.net_fix/d.notional*100; d['ret_gross']=d.pnl_gross_usdt/d.notional*100; d['ret_fee']=d.fee_fix/d.notional*100
    d['up'] = d.salida>d.entrada; d['pred_up']=d.tipo=='LONG'
    d['motivo']=d.motivo_salida.str.split(' ').str[0].replace({'STOP':'STOP LOSS','TAKE':'TAKE PROFIT','Salida':'Salida técnica RSI/BB','TIME':'TIME STOP'})
    return d
v1=prep(v1,'v1'); v2=prep(v2,'v2')
print('v1 filas con fee NaN corregidas:', int(v1.fee_usdt.isna().sum()), '| v2:', int(v2.fee_usdt.isna().sum()), v2.fee_fuente.value_counts().to_dict())
def resumen(d, lab):
    g=d.net_fix>0
    t_n=stats.ttest_1samp(d.ret_net,0) if len(d)>2 else None
    t_g=stats.ttest_1samp(d.ret_gross,0) if len(d)>2 else None
    acc=(d.up==d.pred_up).mean()
    bt=stats.binomtest(int((d.up==d.pred_up).sum()), len(d), 0.5).pvalue
    return dict(muestra=lab,n=len(d),WR_neto=round(g.mean()*100,1),WR_bruto=round((d.pnl_gross_usdt>0).mean()*100,1),
        bruto_usdt=round(d.pnl_gross_usdt.sum(),3),fees_usdt=round(d.fee_fix.sum(),3),neto_usdt=round(d.net_fix.sum(),3),
        ret_bruto_pct=round(d.ret_gross.mean(),3),ret_fee_pct=round(d.ret_fee.mean(),3),ret_neto_pct=round(d.ret_net.mean(),3),
        p_neto=round(t_n.pvalue,3) if t_n else None,p_bruto=round(t_g.pvalue,3) if t_g else None,
        gan_media=round(d.net_fix[g].mean(),4) if g.any() else None,perd_media=round(d.net_fix[~g].mean(),4),
        exactitud_dir=round(acc*100,1),p_binom=round(bt,3))
tabla = pd.DataFrame([resumen(v1.iloc[:12],'v1 primeras 12'),resumen(v1.iloc[:27],'v1 primeras 27 (muestra diagnóstico)'),
   resumen(v1.iloc[27:],'v1 operaciones 28-48 (datos nuevos)'),resumen(v1,'v1 completa'),
   resumen(v2[v2.entrada_utc<=pd.Timestamp('2026-11-19T15:00:00',tz='UTC')],'v2 hasta corte 9:00'),resumen(v2,'v2 todas')])
print(tabla.T.to_string())
tabla.to_csv('resultados/resumen_v1_v2.csv',index=False)
for lab,d in (('v1',v1),('v2',v2)):
    print('\n',lab,'por motivo:\n', d.groupby('motivo').agg(n=('net_fix','size'),WR=('net_fix',lambda s:round((s>0).mean()*100,1)),bruto=('pnl_gross_usdt','sum'),neto=('net_fix','sum')).round(3))
    print(lab,'por lado:\n', d.groupby('tipo').agg(n=('net_fix','size'),WR=('net_fix',lambda s:round((s>0).mean()*100,1)),bruto=('pnl_gross_usdt','sum'),neto=('net_fix','sum')).round(3))
    cm=pd.crosstab(d.tipo.map({'LONG':'Señal LONG','SHORT':'Señal SHORT'}), d.up.map({True:'Subió',False:'Bajó'})); print(cm)
# Horas activas
h1 = (pd.Timestamp('2026-11-18T01:25:00',tz='UTC')-pd.Timestamp('2026-11-18T00:45:00',tz='UTC')).total_seconds()/3600 + (pd.Timestamp('2026-11-19T06:20:21',tz='UTC')-pd.Timestamp('2026-11-18T22:00:24',tz='UTC')).total_seconds()/3600
h2 = (v2.salida_utc.max()-pd.Timestamp('2026-11-19T06:21:00',tz='UTC')).total_seconds()/3600
print('\nhoras activas v1 ~%.1f, v2 ~%.1f'%(h1,h2))
print('v1 ritmo %.1f ops/h ; v2 %.2f ops/h'%(len(v1)/h1, len(v2)/h2))
print('v1 comisiones/hora a 100 XRP: %.3f ; v2: %.3f'%(-v1.fee_fix.sum()*10/h1, -v2.fee_fix.sum()/h2))
print('v1 neto/hora a 100 XRP: %.3f ; v2: %.3f'%(v1.net_fix.sum()*10/h1, v2.net_fix.sum()/h2))
# DM: todas las pruebas
rows=[]
for tf in ('5m','15m','1h'):
    for f in ('dm_test','dm_test_vol'):
        x=pd.read_csv(f'resultados/rf_{tf}/{f}.csv'); x['tf']=tf; x['version']='RF' if f=='dm_test' else 'RF-vol'; rows.append(x)
dm=pd.concat(rows); P=pd.concat([dm['p Máximo'],dm['p Mínimo']])
print('\npruebas DM (p-values individuales):',len(P),'| p<0.05:',int((P<0.05).sum()),'| esperados por azar ~',round(len(P)*0.05,1),'| p<0.05/%d (Bonferroni):'%len(P),int((P<0.05/len(P)).sum()))
print(dm[(dm['p Máximo']<0.05)|(dm['p Mínimo']<0.05)][['tf','version','Modelo 1','Modelo 2','DM Máximo','p Máximo','DM Mínimo','p Mínimo']].to_string())
# resumen por temporalidad (RF mejor vs base)
for tf in ('5m','15m','1h'):
    t=pd.read_csv(f'resultados/rf_{tf}/resultados_test.csv'); d=pd.read_csv(f'resultados/rf_{tf}/dm_test.csv').iloc[0]
    print(tf, 'RMSE RF mejor %.5f vs base %.5f | VarScore %.3f | DM max p=%.3f min p=%.3f'%(t.iloc[0].RMSE,t.iloc[3].RMSE,t.iloc[0].VarScore,d['p Máximo'],d['p Mínimo']))
