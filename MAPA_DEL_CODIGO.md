# Mapa del código

Guía de los bloques lógicos y funciones del proyecto, para revisar el código sin tener que leerlo completo.

## Bot: `bot/bot_ema_paper.py` (versión 2, la que corre en la VM)

El archivo está organizado en 9 bloques, en el orden en que se ejecutan en el bot.

| Bloque | Funciones principales | Qué hace |
| --- | --- | --- |
| 0. Carga de credenciales | `_cargar_env` | Lee el archivo `.env` (claves de Gate.io, Telegram y Gemini). Las claves nunca están en el código |
| 1. Configuración y estado | `Config`, `load_state`, `save_state`, `write_csv`, `log_event` | Parámetros del bot (5m, apalancamiento 10x, 150 árboles, horizonte 12) y registro de operaciones y eventos en CSV |
| 2. Datos e indicadores | `gate_client`, `get_candles` | Descarga velas de Gate.io y calcula RSI (14, Wilder), Bandas de Bollinger (20, 2σ) y EMA 9 |
| 3. Riesgo | `compute_var_cvar` | VaR y CVaR al 99% sobre retornos recientes; acotan la distancia del Stop Loss |
| 4. Random Forest | `features`, `rf_forecast` | Dos regresores que predicen el máximo y el mínimo de las próximas 12 velas |
| 5. Señales y operación | `evaluate_signals`, `make_levels`, `open_position`, `close_position`, `step_strategy` | Regla de entrada (LONG o SHORT por rebote), niveles de Take Profit y Stop Loss, apertura y cierre, salidas |
| 6. Reportes | `compute_performance`, `make_chart`, `fmt_status`, `fmt_forecast`, `ask_gemini`, `make_excel` | Gráfico, desempeño, estatus, consulta a Gemini y Excel de entrega |
| 7. Telegram y monitor | `authorized`, `start`, `handle`, `monitor`, `post_init`, `main` | Menú de Telegram y bucle que revisa el mercado cada 20 segundos |
| 8. Órdenes reales | `_contract_params`, `_set_leverage`, `_send_market`, `_log_order`, `_fail`, y las versiones de `open_position` y `close_position` | Envía órdenes de mercado reales a Gate.io Testnet, guarda su ID en `ordenes_gate.csv` y toma el PnL neto de comisiones |
| 9. Versión 2 "consciente de costos" | `make_levels`, `_hit`, `_net_and_fee`, `close_position`, `step_strategy` | Take Profit y Stop Loss 1 a 1, sin salidas técnicas, enfriamiento de 3 velas y salida por tiempo a las 36 velas |

Nota: en Python, una función definida más abajo reemplaza a la anterior con el mismo nombre. Por eso los bloques 8 y 9 sobrescriben a propósito `open_position`, `close_position`, `make_levels` y `step_strategy` de los bloques 5, y así se construyen las versiones 1 y 2 sin duplicar el resto del bot.

`bot/bot_ema_paper_v1.py` es la versión 1: lo mismo, pero sin el bloque 9.

## Investigación: `investigacion/`

| Script | Qué hace | Salida |
| --- | --- | --- |
| `hiperparametros_rf.py` | Rejilla de 18 combinaciones del Random Forest (árboles, profundidad, hoja mínima), partición temporal 60/20/20 con separación de 12 velas, métricas RMSE, MAE, MAPE, SMAPE y Variance Score, y prueba de Diebold-Mariano contra la media histórica, KNN y el RF del bot | `resultados/rf_<temporalidad>/` |
| `hiperparametros_rf_vol.py` | Variante con retornos estandarizados por volatilidad (hipótesis que resultó negativa) | Mismas carpetas, archivos `*_vol` |
| `cartera_markowitz.py` | Correlación de Pearson, covarianza de Ledoit-Wolf, carteras de mínima varianza, máximo Sharpe, paridad de riesgo e igual ponderación, con evaluación fuera de muestra | `resultados/cartera/` |
| `analisis_v1_v2.py` | Estadística de las operaciones reales: PnL bruto y neto, comisiones, aciertos, matriz de confusión, ROI y conteo de pruebas Diebold-Mariano | `resultados/resumen_v1_v2.csv` |

## Datos: `datos/`

- `trades_v1.csv`, `eventos_v1.csv`: operaciones y eventos de la versión 1 (48 operaciones).
- `ordenes_gate_v1.csv`: 96 órdenes reales con su ID, precio de ejecución y estado.
- `trades_v2_muestra_oficial.csv`: muestra congelada de la versión 2 (9 operaciones).
