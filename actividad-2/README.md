# Actividad 2 · Pipeline y empaquetado

Pipeline de **entrenamiento** y de **inferencia** en Prefect, conectadas por el alias `champion` del Model Registry de MLflow, y una **API** con FastAPI que sirve ese modelo. Se comparan tres modelos —regresión logística, random forest y XGBoost— y solo el de mejor AUC ROC compite por ser champion.

## Estructura

```
.
├── configs/
│   ├── config.yaml          # datos, candidatos e hiperparámetros, reglas de champion, inferencia, despliegue
│   └── features.yaml        # selección de variables: ALL | NUMERIC | CUSTOM
├── data/                    # train.csv y test.csv
├── domain/
│   └── modelo.py            # variables, pipelines de los 3 modelos, métricas y regla de promoción (sin Prefect/MLflow)
├── scripts/
│   ├── config.py            # lectura de los YAML
│   └── load_data.py         # carga y validación de train/test
├── infraestructure/
│   ├── flows/
│   │   ├── entrenamiento_flow.py   # ingesta → entrenamiento → evaluación → registro
│   │   ├── inferencia_flow.py      # champion → lote → predicción → CSV con fecha → MLflow
│   │   └── desplegar.py            # 2 despliegues: entrenamiento manual + inferencia diaria
│   └── api/
│       └── main.py          # FastAPI: /health, /modelo, /predict, /predict/lote
├── predicciones/            # prediccion_AAAA-MM-DD.csv (no se versiona)
├── tests/test_modelo.py
├── pyproject.toml / poetry.lock
└── README.md
```

## Puesta en marcha

Todo se ejecuta **desde la raíz del repositorio**, con los servicios de `services/` levantados y en cada terminal:

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

El mejor candidato (mayor AUC entre los que cumplen `recall_minimo`) se registra como versión nueva y recibe el alias `champion` solo si:

1. `auc >= umbral_auc` (0.80)
2. `recall >= recall_minimo` (0.50)
3. `auc >= auc_champion_actual + mejora_minima` (0.001)

Si no, los runs quedan en MLflow y el alias no cambia.

## Qué queda en MLflow (experimento `churn-actividad2`)

- Run padre `entrenamiento-AAAAMMDD-HHMM` (tag `fase = entrenamiento`): tabla comparativa, ROC comparativa, mejor modelo y decisión.
- Un run hijo por modelo (tag `fase = entrenamiento`): parámetros (`features`, `n_features`, hiperparámetros), métricas (`auc`, `accuracy`, `precision`, `recall`, `f1`), ROC, matriz de confusión y el modelo.
- Run `inferencia-AAAA-MM-DD` (tag `fase = inferencia`, `modelo_version`, `modelo_run_id`) con el CSV de predicciones.
- Model Registry: `churn-modelo` con alias `champion`.

## Despliegues en Prefect

- `entrenamiento-churn/entrenamiento-manual`: sin horario.
- `inferencia-churn/inferencia-diaria`: cron `0 6 * * *`, zona `America/Bogota`.

```powershell
poetry run prefect deployment run "entrenamiento-churn/entrenamiento-manual"
poetry run prefect deployment run "inferencia-churn/inferencia-diaria"
```

## Predicción de ejemplo (API)

```powershell
$cliente = @{
    CreditScore = 619; Geography = "France"; Gender = "Female"; Age = 42; Tenure = 2
    Balance = 0.0; NumOfProducts = 1; HasCrCard = 1; IsActiveMember = 1
    EstimatedSalary = 101348.88; Complain = 1; "Satisfaction Score" = 2
    "Card Type" = "DIAMOND"; "Point Earned" = 464
} | ConvertTo-Json

Invoke-RestMethod -Uri http://localhost:8000/predict -Method Post -ContentType "application/json" -Body $cliente
```

Respuesta (con el champion de la versión 2, random forest con `ALL`):

```json
{"probabilidad": 0.9375, "prediccion": 1, "modelo": "churn-modelo", "version_modelo": "2"}
```

Si el entrenamiento cambia el champion, reinicia la API (Ctrl+C y volver a lanzar uvicorn).
