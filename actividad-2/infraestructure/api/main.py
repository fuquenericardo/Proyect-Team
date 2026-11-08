"""Actividad 2: API de inferencia con FastAPI. Sirve el modelo con el alias `champion` de MLflow.

Ejecución (desde la carpeta actividad-2, con MLFLOW_TRACKING_URI definida):
    poetry run uvicorn infraestructure.api.main:app --port 8000

Documentación interactiva (Swagger): http://localhost:8000/docs

El modelo se carga al arrancar. Si el entrenamiento promueve un champion nuevo,
reinicia la API (Ctrl+C y volver a lanzar uvicorn) para que lo use.

El esquema de entrada (/docs) se construye a partir de la firma (signature) que el
flow de entrenamiento guardó en MLflow: siempre coincide con las columnas del champion.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import re  # noqa: E402

import mlflow  # noqa: E402
import pandas as pd  # noqa: E402
from fastapi import FastAPI, HTTPException  # noqa: E402
from mlflow.tracking import MlflowClient  # noqa: E402
from pydantic import BaseModel, ConfigDict, Field, create_model  # noqa: E402

from scripts.config import cargar_config, tracking_uri  # noqa: E402

# ---------------------------------------------------------------------------
# Carga del champion (una vez, al arrancar)
# ---------------------------------------------------------------------------
cfg = cargar_config()
NOMBRE = cfg["registro"]["nombre_modelo"]
ALIAS = cfg["registro"].get("alias", "champion")
UMBRAL = cfg["metricas"]["umbral"]
URI = f"models:/{NOMBRE}@{ALIAS}"

mlflow.set_tracking_uri(tracking_uri(cfg))
client = MlflowClient()
version_info = client.get_model_version_by_alias(NOMBRE, ALIAS)
modelo = mlflow.sklearn.load_model(URI)
COLUMNAS = list(modelo.feature_names_in_)
METRICAS = client.get_run(version_info.run_id).data.metrics
esquema = mlflow.models.get_model_info(URI).signature.inputs

# ---------------------------------------------------------------------------
# Esquema de entrada dinámico (a partir de la firma del modelo)
# ---------------------------------------------------------------------------
TIPOS = {"long": int, "integer": int, "double": float, "float": float, "boolean": bool, "string": str}


def _ejemplo():
    """Primer asociado del lote de inferencia como ejemplo para /docs (si el archivo existe)."""
    try:
        fila = pd.read_csv(cfg["inferencia"]["fuente"], sep=cfg["datos"].get("separador", ","), nrows=1)
        fila = fila[COLUMNAS].iloc[0]
        return {c: (v.item() if hasattr(v, "item") else v) for c, v in fila.items()}
    except Exception:
        return None


EJEMPLO = _ejemplo()
campos = {}
for col in esquema.inputs:
    tipo = TIPOS.get(str(col.type).split(".")[-1].lower(), str)
    nombre_py = re.sub(r"\W+", "_", col.name).strip("_") or "col"
    # Obligatorio pero admite null: un dato faltante (p. ej. ESTADO_CIVIL vacío) lo imputa el pipeline.
    campos[nombre_py] = (tipo | None, Field(alias=col.name))

Asociado = create_model(
    "Asociado",
    __config__=ConfigDict(populate_by_name=True, extra="ignore", json_schema_extra={"example": EJEMPLO}),
    **campos,
)


class Prediccion(BaseModel):
    probabilidad_retiro: float
    prediccion: int
    modelo: str
    version_modelo: str


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
app = FastAPI(
    title=cfg["api"]["titulo"],
    description=f"Predicción con `{URI}` (versión {version_info.version}). Reinicia la API si cambia el champion.",
    version="1.0.0",
)


def _predecir(registros):
    datos = pd.DataFrame(registros)[COLUMNAS]
    proba = modelo.predict_proba(datos)[:, 1]
    return [
        Prediccion(
            probabilidad_retiro=round(float(p), 4),
            prediccion=int(p >= UMBRAL),
            modelo=NOMBRE,
            version_modelo=str(version_info.version),
        )
        for p in proba
    ]


@app.get("/health")
def health():
    return {"estado": "ok", "modelo": NOMBRE, "alias": ALIAS, "version": version_info.version}


@app.get("/modelo")
def info_modelo():
    """Versión, run de entrenamiento, tipo, variables y métricas del champion servido."""
    return {
        "uri": URI,
        "version": version_info.version,
        "run_id": version_info.run_id,
        "tags": version_info.tags,
        "variables": COLUMNAS,
        "metricas_test": {k: round(v, 4) for k, v in METRICAS.items()},
        "umbral": UMBRAL,
    }


@app.post("/predict", response_model=Prediccion)
def predict(asociado: Asociado):
    """Predice el riesgo de retiro de un asociado. Recibe las columnas originales del CSV (ver ejemplo)."""
    try:
        return _predecir([asociado.model_dump(by_alias=True)])[0]
    except Exception as e:  # datos con formato inesperado
        raise HTTPException(status_code=422, detail=str(e)) from e


@app.post("/predict/lote", response_model=list[Prediccion])
def predict_lote(asociados: list[Asociado]):
    """Predice para varios asociados en una sola llamada."""
    if not asociados:
        raise HTTPException(status_code=422, detail="La lista de asociados está vacía")
    return _predecir([a.model_dump(by_alias=True) for a in asociados])
