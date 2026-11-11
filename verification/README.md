# Verificación formal del risk gate

> Los LLMs proponen, el código dispone. Aquí esa frase deja de ser una promesa
> y pasa a ser un teorema verificado por Lean.

El README del proyecto dice que `risk.py` tiene "hard limits no prompt can override".
Esta carpeta convierte esa afirmación en enunciados precisos y los demuestra.
Al hacerlo aparecieron dos lugares donde la afirmación era falsa. Ya están corregidos.

## 1. Qué se encontró

| # | Falla | Ejemplo | Por qué los tests no la veían |
|---|-------|---------|-------------------------------|
| 1 | `max_positions` y `max_gross_exposure` se comparaban contra `account`, la foto del bróker, que no cambia durante el ciclo. Varias compras válidas una por una podían violar los topes juntas. | 7 posiciones al 68,6 % de exposición (topes: 8 y 80 %). La puerta vieja aprueba BUY A y BUY B y quedan **9 posiciones al 88,6 %**. Con los topes por defecto se llega al 105,8 %, es decir, apalancamiento en una cuenta de margen. | `test_risk.py` evalúa una decisión a la vez, y el bug solo existe a nivel de ciclo. |
| 2 | `check_stops` recalculaba el stop con el ATR **del día**. Si la volatilidad sube, el stop baja, así que el "1 % de riesgo por operación" no acotaba nada. | Stop de entrada en 96 (riesgo de $400). Con ATR 6, el precio cae a 89 sin que el stop se dispare: pérdida latente de $1.100. | Los tests usaban el mismo ATR al comprar y al revisar. |
| 3 (menor) | `round(close - dist, 2)` puede redondear el stop hacia **abajo** y agregar hasta $0,005 por acción al riesgo. | Riesgo real ligeramente mayor que el presupuesto. | Diferencia de centavos. Solo aparece al intentar demostrar la cota. |

El contraejemplo de la falla 1 está **verificado en Lean** (`old_gate_unsafe`), no solo en Python.

## 2. El arreglo (en `risk.py`, `memory.py` y `pipeline.py`)

- **Cartera sombra.** `CycleContext` ahora lleva `held_notional` (exposición por símbolo) y `gross`. Se inicializa con la foto del bróker en la primera llamada (`seed`) y se actualiza con cada orden aprobada. Todos los topes se comparan contra la cartera *como va a quedar*.
- **Stop congelado.** Al aprobar un BUY, el stop se guarda en la tabla `stops` del journal y `check_stops` usa ese valor. Las posiciones abiertas antes del cambio siguen con la regla vieja hasta que se cierren.
- **Stop redondeado hacia arriba** (`_stop_price`). La distancia real al stop nunca supera la que se usó para dimensionar.
- Una SELL ahora también exige que el símbolo siga en la cartera sombra, así que no se puede vender dos veces en el mismo ciclo.

## 3. Los teoremas (`lean/VelveteenRisk.lean`)

El modelo traduce `evaluate` línea por línea, con aritmética racional exacta. `Proposal` es lo que llega a la puerta: `sym`, `act` y `conv` vienen del LLM, y `close`, `atr`, `heldQty` y `vetoed` vienen del mercado y del bróker. **Todos los teoremas cuantifican sobre todos los valores de todos los campos.**

| Teorema | Dice | Código |
|---------|------|--------|
| `cycle_safe` | **Para cualquier lista de propuestas**, si la cartera empieza dentro de sus topes, termina dentro de sus topes. Es decir, ninguna salida del LLM puede violar los límites. | todo `evaluate` + `CycleContext` |
| `velveteen_cycle_safe` | Lo mismo, instanciado con el redondeo real `_round_qty`. | `_round_qty` |
| `step_preserves` | Una sola decisión preserva el invariante `Inv`. | `evaluate` |
| `buySize_caps` | El tamaño final de un BUY respeta a la vez los topes de posición, bruto y sector. | bloque de sizing |
| `trim_spec` | Cada tope solo puede **achicar** la cantidad y deja el resultado bajo ese tope. Es la razón por la que los tres topes se componen. | cada `if … > cap: qty = _round_qty(…)` |
| `buy_risk_le_budget` | `qty × distancia_stop ≤ equity × per_trade_risk_pct × conviction/5`. | `risk_budget`, `stop_distance` |
| `stop_loss_le_budget` | Si se sale en el stop congelado o por encima, la pérdida queda dentro del presupuesto. | `_stop_price`, `check_stops` |
| `sell_only_closes` | Una SELL aprobada cierra una posición existente por exactamente la cantidad del bróker. No hay cortos. | rama SELL |
| `hold_never_orders` | HOLD nunca produce orden. | rama HOLD |
| `roundQty_spec`, `stopPrice_ge` | El redondeo de cantidades es hacia abajo y el del stop es hacia arriba. | `_round_qty`, `_stop_price` |
| `old_gate_unsafe` | La puerta anterior al arreglo **no** cumple `cycle_safe`. Contraejemplo concreto verificado. | versión anterior |

`Inv` dice que hay como mucho `max_positions` posiciones, que la exposición bruta, el turnover, cada símbolo y cada sector están bajo sus topes, y que las exposiciones son no negativas.

Solo se usa el núcleo de Lean (`Rat`, `grind`, `omega`), sin Mathlib, así que compila en segundos con cualquier toolchain reciente.

## 4. Qué NO está demostrado

Esto hay que decirlo explícitamente en cualquier artículo o charla:

1. **Floats.** Python usa dobles IEEE-754 y Lean usa ℚ. Para cubrir esa brecha están los tests `tests/test_risk_cycle.py`, que corren los mismos invariantes sobre el código real en 300 ciclos aleatorios, con tolerancia relativa de 1e-9. El paso siguiente sería portar la puerta a `Decimal` o demostrar cotas de error de punto flotante.
2. **Fidelidad del modelo.** El modelo Lean está escrito a mano a partir de `risk.py`, y la correspondencia se establece por inspección y por los tests diferenciales. No hay extracción mecánica. Ese es el eslabón débil, y es justo el problema de *fidelidad del enunciado* de la tesis.
3. **Ejecución.** Se supone que las órdenes se llenan al cierre de referencia. Un gap que atraviese el stop, el slippage o la diferencia entre el cierre y el precio de llenado al abrir quedan fuera de `stop_loss_le_budget`.
4. **Filtros que solo rechazan.** El halt por drawdown y el filtro de CVaR se abstraen como el booleano `vetoed`. Como solo pueden rechazar, no afectan los invariantes.
5. **Rentabilidad.** Nada de esto dice que la estrategia gane dinero. Dice que, gane o pierda, lo hace dentro de los límites declarados.

## 5. Cómo correrlo

```bash
# Python: invariantes de ciclo + fuzz (no requiere red ni claves)
python -m pytest tests/test_risk.py tests/test_risk_cycle.py -q

# Lean (cualquier instalación con elan)
cd verification/lean
lake build            # o: lean VelveteenRisk.lean
```

`lean-toolchain` fija `leanprover/lean4:v4.34.1`. Estado verificado el 8 de octubre de 2026: 0 errores, 0 warnings, ningún `sorry` ni `native_decide`. `#print axioms` sobre `cycle_safe`, `old_gate_unsafe` y los demás muestra solo los tres axiomas estándar de Lean (`propext`, `Classical.choice` y `Quot.sound`). El contraejemplo concreto se comprueba en el kernel (`decide +kernel`).

Nota: la verificación se hizo con el binario stage0 de Lean (snapshot 4.34.0) más la librería `Init` de las fuentes de v4.34.1, compilada desde el código fuente porque no había descarga directa. Conviene correr `lake build` una vez con el toolchain oficial para confirmarlo.

## 6. Cómo encaja con la tesis (rl_metodos)

- **La misma arquitectura.** En VelveteenTrade, el LLM propone y el código determinista dispone. En la tesis, el agente propone una prueba y el verificador formal dispone. Este proyecto es el caso aplicado de la misma idea: el verificador no evalúa al LLM, sino la *jaula* donde el LLM opera.
- **Un caso real de fidelidad del enunciado.** Los tests de `test_risk.py` eran una especificación débil que pasaba mientras la propiedad importante, el invariante de ciclo, era falsa. Es exactamente el modo de falla que la tesis estudia en los verificadores: pasar el chequeo sin cumplir lo que se quería.
- **Un benchmark para tus agentes.** `step_preserves` y `cycle_safe` son buenos objetivos para los agentes de rl_metodos. Si borras las pruebas, les dejas los enunciados y mides cuántos logran demostrar. También sirve medir si algún agente "gana" debilitando `Inv` o agregando hipótesis, lo que muestra en concreto el reward hacking por cambio de enunciado.

## 7. Hacia el artículo y el portafolio

Título tentativo: *Machine-checked guardrails for LLM trading agents: a case study in statement fidelity*.

El argumento: (i) el patrón "LLM propone, código determinista dispone" es común en agentes. (ii) Su garantía suele quedarse en la palabra del autor. (iii) Al formalizarla en un sistema real encontramos violaciones que los tests unitarios no veían. (iv) Se demuestra la versión corregida para cualquier salida del LLM. (v) Las brechas restantes (floats, fidelidad del modelo, ejecución) se declaran con honestidad.

Para fortalecerlo falta lo siguiente:
- tests diferenciales que ejecuten el modelo Lean (compilado) contra `risk.py` con las mismas entradas;
- portar a `Decimal` o acotar el error de punto flotante;
- el experimento con los agentes de rl_metodos descrito arriba.

## 8. Actualización del 24 oct 2026: la prueba era correcta sobre el objeto equivocado

La auditoría del 24 oct (`parches_2026-10-24/INFORME_brechas.md`, fuera del repo) encontró que `cycle_safe` es un teorema correcto **sobre la cartera sombra que construye `evaluate`**, pero que la cuenta real podía alejarse de ella. Hubo 10 violaciones, cada una reproducida con un test. Esta sección no reemplaza nada de lo anterior: el teorema original y su historia siguen tal cual.

Arreglos en la rama `fix/b2-b8-sombra` (un commit por bug; los tests de reproducción de `tests/test_brechas.py` quedaron invertidos como regresión):

| Bug | Qué separaba la sombra de la cuenta | Arreglo |
|---|---|---|
| B1 | la puerta juzgaba un símbolo y la orden salía con otro | rama `fix/b1-auth` |
| B2 | las órdenes en cola no entraban en la sombra; el turnover era por ciclo | se siembra con las pendientes y el turnover se lleva por sesión (P3) |
| R2b | dos ciclos simultáneos partían de la misma foto | candado de ciclo |
| B3 | un NaN apagaba stops y topes, porque toda comparación con NaN es False | se falla cerrado en el snapshot, en la puerta y en el barrido de stops (P5) |
| B4 | una compra adicional aflojaba el stop congelado del primer lote | el stop solo se aprieta (P4) |
| B5, D3 | segunda orden sobre un símbolo del ciclo (doble venta; BUY y luego SELL) | `traded` en la puerta, que inicia el barrido (P2, P7) |
| B6–B8 | una sola cifra de capital para dimensionar, para el freno y para el track record | S / V / E en `capital.py` (P8) |
| D5 | `RoundDown` y `stopPrice_ge` eran falsas en floats | `Decimal` exacto: las hipótesis valen para la función desplegada |

Sobre el §5.1 («Floats»): D5 cierra las dos hipótesis de redondeo y B3 cierra los NaN. Queda la aritmética interna de `evaluate` en floats.

Lean: `lake build` con el toolchain oficial v4.34.1 confirma la verificación del 8 oct (ver `lake_build_2026-10-24.txt` y `SHA256SUMS`). Los enunciados de la ronda 2, ajustados a estos arreglos, están propuestos y demostrados fuera del repo; entran a `lean/` solo cuando se aprueben.
