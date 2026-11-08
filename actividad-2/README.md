# Actividad 2 · Pipeline y empaquetado — Retiro de asociados

Pipeline de **entrenamiento** y de **inferencia** en Prefect, conectadas por el alias `champion` del Model Registry de MLflow, y una **API** con FastAPI que sirve ese modelo. Se predice si un asociado se retira (`Y = 1`, ~2 % de los casos).

Se comparan tres modelos (regresión logística, random forest y XGBoost), entrenados con `train` y evaluados en dos conjuntos que **no** se usan para entrenar:

| Conjunto | Periodo | Filas | % retiros | Papel |
|---|---|---|---|---|
| `train.csv` | sep-2023 a dic-2025 (28 cortes) | 264.400 | 2,21 % | entrenar |
| `test1_con_y.csv` | ene-2026 a jun-2026 (6 cortes) | 57.317 | 2,01 % | elegir el mejor modelo y decidir el champion |
| `test2_con_y.csv` | jul-2026 a sep-2026 (3 cortes) | 28.595 | 2,04 % | confirmar con retiros reales |
| `test2_sin_y.csv` | jul-2026 a sep-2026 | 28.595 | — | lote de la inferencia diaria |

Cada ejecución del entrenamiento genera **3 modelos × 2 conjuntos = 6 experimentos** (runs) en MLflow.

## Estructura

```
actividad-2/
├── configs/
│   ├── config.yaml          # datos, candidatos e hiperparámetros, reglas de champion, inferencia, despliegue
│   └── features.yaml        # selección de variables: ALL | NUMERIC | CUSTOM
├── data/                    # train.csv, test1_con_y/sin_y.csv, test2_con_y/sin_y.csv (no se versionan)
├── domain/modelo.py         # variables, pipelines de los 3 modelos, métricas y regla de promoción
├── scripts/
│   ├── config.py            # lectura de los YAML
│   └── load_data.py         # carga y validación de los CSV
├── infraestructure/
│   ├── flows/
│   │   ├── entrenamiento_flow.py   # ingesta → entrenamiento → evaluación (test1, test2) → registro
│   │   ├── inferencia_flow.py      # champion → lote → predicción → comparación con retiros reales → MLflow
│   │   └── desplegar.py            # 2 despliegues: entrenamiento manual + inferencia diaria
│   └── api/main.py          # FastAPI: /health, /modelo, /predict, /predict/lote
├── predicciones/            # prediccion_AAAA-MM-DD.csv (no se versiona)
├── tests/test_modelo.py
└── pyproject.toml / poetry.lock
```

## Puesta en marcha

Todo se ejecuta **desde la carpeta `actividad-2`**, con los servicios de `../services` levantados y en cada terminal:

```powershell
$env:PREFECT_API_URL = "http://localhost:4200/api"
$env:MLFLOW_TRACKING_URI = "http://localhost:5000"
$env:PYTHONUTF8 = "1"
```

| Paso | Comando |
|---|---|
| Instalar entorno | `poetry install` |
| Revisar datos | `poetry run python scripts/load_data.py` |
| Pruebas | `poetry run pytest -q` |
| Entrenar | `poetry run python infraestructure/flows/entrenamiento_flow.py` |
| Inferir | `poetry run python infraestructure/flows/inferencia_flow.py` |
| Desplegar (dejar abierta) | `poetry run python infraestructure/flows/desplegar.py` |
| API | `poetry run uvicorn infraestructure.api.main:app --port 8000` |

## Reglas para ser champion (`registro` en config.yaml)

El mejor candidato (mayor AUC en test1 entre los que cumplen `recall_minimo`) se registra como versión nueva y recibe el alias `champion` solo si:

1. `auc_test1 >= umbral_auc` (0.75)
2. `recall_test1 >= recall_minimo` (0.50)
3. `auc_test2 >= auc_minimo_confirmacion` (0.70): el resultado se sostiene con retiros reales
4. `auc_test1 >= auc_test1_del_champion + mejora_minima` (0.001)

Si no, los runs quedan en MLflow y el alias no cambia.

## Resultados (features = ALL, 13 variables)

| Modelo | AUC test1 | Recall test1 | AUC test2 | Recall test2 |
|---|---|---|---|---|
| XGBoost | 0.8075 | 0.609 | 0.7942 | 0.565 |
| Random forest | 0.7974 | 0.556 | 0.7825 | 0.517 |
| Regresión logística | 0.6522 | 0.688 | 0.6561 | 0.699 |

## Qué queda en MLflow (experimento `retiro-asociados-actividad2`)

- Run padre `entrenamiento-AAAAMMDD-HHMM` (tag `fase = entrenamiento`): tabla de los 6 experimentos, ROC comparativa por conjunto, mejor modelo y decisión.
- 6 runs hijos `<modelo>-test1` / `<modelo>-test2` (tag `fase = entrenamiento`, `conjunto_evaluacion`): parámetros (`features`, `n_features`, hiperparámetros), métricas (`auc`, `accuracy`, `precision`, `recall`, `f1`), ROC y matriz de confusión. El modelo se guarda en el run de test1.
- Run `inferencia-AAAA-MM-DD` (tag `fase = inferencia`, `modelo_version`, `modelo_run_id`) con el CSV de predicciones y las métricas contra retiros reales (`real_auc`, `real_recall`, `retiros_detectados`…).
- Model Registry: `retiro-asociados-modelo` con alias `champion`.

## Despliegues en Prefect

- `entrenamiento-retiro/entrenamiento-manual`: sin horario.
- `inferencia-retiro/inferencia-diaria`: cron `0 6 * * *`, zona `America/Bogota`.

```powershell
poetry run prefect deployment run "entrenamiento-retiro/entrenamiento-manual"
poetry run prefect deployment run "inferencia-retiro/inferencia-diaria"
```

## Predicción de ejemplo (API)

```powershell
$asociado = @{
    EDAD = 25; GENERO = "MASCULINO"; ANTIGUEDAD = 0.5; ESTADO_CIVIL = "Soltero"; SALARIO = 1300000
    SALDO_APORTE_SOCIAL = 50000; SALDO_AHORRO_PERMANENTE = 0; SALDO_AHO_VISTA = 0
    SALDO_AHORRO_PROGRAMADO = 0; SALDO_CARTERA = 0; CANTIDAD_CREDITOS = 0
    CANT_SERV_COMPLEMENT = 0; CANT_TRANSACCIONES = 0
} | ConvertTo-Json

Invoke-RestMethod -Uri http://localhost:8000/predict -Method Post -ContentType "application/json" -Body $asociado
```

Respuesta (champion versión 2, XGBoost con `ALL`):

```json
{"probabilidad_retiro": 0.61, "prediccion": 1, "modelo": "retiro-asociados-modelo", "version_modelo": "2"}
```

Si el entrenamiento cambia el champion, reinicia la API (Ctrl+C y volver a lanzar uvicorn).
