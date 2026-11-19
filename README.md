# Bot de trading automático XRP_USDT (Gate.io Testnet) — Proyecto Final Integrador

Proyecto final de **Finanzas Computacionales (UDEM)**, Opción 3: bot de trading automático en una máquina virtual de Google Cloud, con Random Forest, indicadores técnicos (RSI, Bandas de Bollinger, EMA), gestión de riesgo VaR/CVaR y control por Telegram. Opera con **órdenes reales en Gate.io Testnet** (futuros USDT, sin dinero real).

> **Resultado honesto:** el sistema funciona de punta a punta, pero **no encontramos una ventaja predictiva demostrable** (prueba de Diebold-Mariano, PnL bruto de v1 y matriz de confusión coinciden). Las comisiones (0.1% por operación) convierten un resultado bruto cercano a cero en pérdida neta. El informe explica el diagnóstico y la versión 2 (consciente de costos).

## Resultados principales

| Evidencia | Resultado |
| --- | --- |
| Bot v1 (48 operaciones reales) | Bruto -0.091 USDT · comisiones -0.464 · neto **-0.555** USDT · dirección acertada 52.1% (p = 0.885) |
| Bot v2 (9 operaciones, 100 XRP) | Neto **-2.08** USDT · 3 Take Profit, 6 Stop Loss · comisiones por hora ~5× menores que v1 |
| Diebold-Mariano (42 pruebas, 5m/15m/1h) | 2 con p < 0.05 (por azar se esperaban ~2.1); 0 sobreviven a Bonferroni |
| Cartera Markowitz (XRP, BTC, ETH, SOL) | Correlación promedio 0.78; todas las carteras perdieron fuera de muestra (mercado a la baja) |

## Estructura

```
bot/                     Código del bot (servicio en la VM)
  bot_ema_paper.py       Versión 2 (activa): órdenes reales + estrategia consciente de costos
  bot_ema_paper_v1.py    Versión 1: órdenes reales con la lógica original
investigacion/           Scripts de validación y análisis
  hiperparametros_rf.py      Rejilla de hiperparámetros, métricas y prueba Diebold-Mariano
  hiperparametros_rf_vol.py  Variante con retornos estandarizados por volatilidad
  cartera_markowitz.py       Correlación, Markowitz (Ledoit-Wolf) y evaluación fuera de muestra
  analisis_v1_v2.py          Estadística de las operaciones reales (v1 vs v2)
resultados/              Tablas CSV y gráficas generadas (rf_5m, rf_15m, rf_1h, cartera)
datos/                   Operaciones reales: trades, eventos y órdenes de Gate.io (con ID)
docs/                    Informe técnico (PDF)
```

## Cómo ejecutarlo

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # llena tus claves; .env está en .gitignore
python bot/bot_ema_paper.py # el bot escribe trading_bot.log
```

Investigación (usa datos públicos de Gate.io, no requiere claves):

```bash
SYMBOL=XRP_USDT INTERVAL=5m python investigacion/hiperparametros_rf.py
SYMBOL=XRP_USDT INTERVAL=5m python investigacion/hiperparametros_rf_vol.py
TICKERS="XRP_USDT,BTC_USDT,ETH_USDT,SOL_USDT" python investigacion/cartera_markowitz.py
python investigacion/analisis_v1_v2.py   # desde la raíz del repositorio
```

## Seguridad de credenciales

Las claves de Gate.io, Telegram y Gemini se leen **solo** de variables de entorno o del archivo `.env` (nunca están en el código ni en el repositorio). El bot silencia el registro HTTP para no imprimir el token de Telegram en `trading_bot.log`.

## Originalidad y uso de herramientas

La estrategia, el diseño de las versiones v1 y v2, el análisis estadístico y la cartera son trabajo propio de este proyecto. Se usó asistencia de IA (Claude y Gemini) como herramienta de desarrollo y análisis; el equipo revisó, ejecutó y validó los resultados con datos reales de Gate.io Testnet. Bibliotecas de terceros: scikit-learn, pandas, NumPy, SciPy, matplotlib, gate-api, python-telegram-bot, google-genai.

## Limitaciones

Muestras pequeñas (48 y 9 operaciones, pocas horas, un solo régimen de mercado), testnet sin liquidez real, tres comisiones de v1 estimadas (marcadas en el análisis) y código del bot en un solo archivo (modularizarlo es trabajo futuro).
