"""Actividad 3: API de inferencia. Carga el champion desde MLflow y arma la API con `servicio.crear_app`.

Ejecución (desde la carpeta actividad-3, con MLFLOW_TRACKING_URI definida):
    poetry run uvicorn infraestructure.api.main:app --port 8000

Documentación interactiva (Swagger): http://localhost:8000/docs  (http://localhost:8000/ redirige ahí)

El modelo se carga al arrancar: si el entrenamiento promueve un champion nuevo, reinicia la API.
La lógica de la API (endpoints, esquema, predicción) está en servicio.py y se prueba sin MLflow.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import mlflow  # noqa: E402
import pandas as pd  # noqa: E402
from mlflow.tracking import MlflowClient  # noqa: E402

from infraestructure.api.servicio import InfoModelo, crear_app  # noqa: E402
from scripts.config import cargar_config, tracking_uri  # noqa: E402

TIPOS_MLFLOW = {"long": int, "integer": int, "double": float, "float": float, "boolean": bool, "string": str}


def cargar_champion(cfg):
    """Lee de MLflow el modelo con el alias champion y lo que la API necesita saber de él."""
    nombre = cfg["registro"]["nombre_modelo"]
    alias = cfg["registro"].get("alias", "champion")
    uri = f"models:/{nombre}@{alias}"
    mlflow.set_tracking_uri(tracking_uri(cfg))
    client = MlflowClient()
    version = client.get_model_version_by_alias(nombre, alias)
    modelo = mlflow.sklearn.load_model(uri)
    firma = mlflow.models.get_model_info(uri).signature
    tipos = {c.name: TIPOS_MLFLOW.get(str(c.type).split(".")[-1].lower(), str) for c in firma.inputs.inputs}
    info = InfoModelo(
        nombre=nombre,
        alias=alias,
        version=str(version.version),
        run_id=version.run_id,
        tags=dict(version.tags),
        metricas=client.get_run(version.run_id).data.metrics,
        tipos=tipos,
        ejemplo=ejemplo(cfg, list(modelo.feature_names_in_)),
    )
    return modelo, info


def ejemplo(cfg, columnas):
    """Primer asociado del archivo de inferencia como ejemplo para /docs (si existe)."""
    try:
        fila = pd.read_csv(cfg["inferencia"]["fuente"], sep=cfg["datos"].get("separador", ","), nrows=1)[columnas]
        return {c: (v.item() if hasattr(v, "item") else v) for c, v in fila.iloc[0].items()}
    except Exception:
        return None


cfg = cargar_config()
modelo, info = cargar_champion(cfg)
app = crear_app(modelo, info, umbral=cfg["metricas"]["umbral"], titulo=cfg["api"]["titulo"])
