# Actividad 3 · Monitoreo, drift y tests — Retiro de asociados

Sobre el sistema de la actividad 2 (entrenamiento de 3 modelos, champion en MLflow, inferencia y API) se añade:

1. **Drift emulado en la inferencia** (`inferencia.nivel_drift`, 0.0 a 1.0): el lote sale de la parte de prueba (`test2_con_y.csv`), se desplazan `round(nivel × 11)` variables numéricas +2 desviaciones típicas y la tasa de retiro sube `nivel × 10` puntos. Cada lote se guarda en `lotes/`.
2. **Pipelines de drift** que comparan el lote con train:
   - `drift-features`: Evidently (`DataDriftPreset`) → `reports/drift_features_<lote>.html/.json/_columnas.csv`; **ALERTA** si la fracción de variables con drift > `monitoreo.umbral_drift` (0.40).
   - `drift-evento`: test chi-cuadrado de la tasa de retiro (Y) → `reports/drift_evento_<lote>.json`; **ALERTA** si p-valor < `monitoreo.alfa_evento` (0.05). Rechaza lotes sin Y.
3. **Tests** (`poetry run pytest`): datos, configuración, modelo, inferencia, API (modelo falso, sin MLflow) y drift. CI en `.github/workflows/ci.yml`.

## Resultados (lote de 2.000 asociados de test2, champion XGBoost)

| Nivel | Variables desplazadas | Drift features (Evidently) | Alerta features | Tasa de retiro | Alerta evento | AUC contra la Y |
|---|---|---|---|---|---|---|
| 0.0 | 0 | 1 de 13 (8 %, SALARIO: drift natural) | no | 2,30 % | no (p = 0,89) | 0,829 |
| 0.3 | 3 | 4 de 13 (31 %) | no | 5,30 % | sí | 0,545 |
| 0.6 | 7 | 8 de 13 (62 %) | **sí** | 8,30 % | sí | 0,536 |

## Comandos (desde `actividad-3`)

```powershell
poetry install
poetry run pytest
poetry run python infraestructure/flows/inferencia_flow.py          # con el nivel de config.yaml
poetry run python infraestructure/flows/drift_features_flow.py      # último lote (o pasa la ruta de un lote)
poetry run python infraestructure/flows/drift_evento_flow.py
poetry run python infraestructure/flows/desplegar.py                # 4 despliegues en Prefect
poetry run uvicorn infraestructure.api.main:app --port 8000
```

## Estructura (lo nuevo respecto a la actividad 2)

```
domain/monitoreo.py            emulación de drift, lectura de Evidently, alerta, drift de target (sin servicios)
domain/inferencia.py           preparar lote, predecir y métricas (sin servicios)
scripts/validacion.py          tests de datos: esquema, rangos, categorías, nulos
scripts/monitoreo_io.py        referencia, lotes y registro opcional en MLflow
infraestructure/api/servicio.py  API sin MLflow (crear_app); main.py solo carga el champion
infraestructure/flows/drift_features_flow.py, drift_evento_flow.py
tests/                         test_datos, test_config, test_modelo, test_inferencia, test_api, test_monitoreo
```
