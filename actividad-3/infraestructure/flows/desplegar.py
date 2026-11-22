"""Actividad 3: despliegue de las pipelines en Prefect.

- entrenamiento-retiro/entrenamiento-manual  : sin horario, se lanza a mano.
- inferencia-retiro-drift/inferencia-diaria  : todos los días 06:00 (puntúa un lote con drift emulado).
- drift-features/drift-features-diario       : todos los días 06:15 (reporte Evidently del último lote).
- drift-evento/drift-evento-diario           : todos los días 06:20 (tasa de retiro del último lote).

Horarios y zona en la sección `despliegue` de configs/config.yaml.

Ejecución (desde la carpeta actividad-3, con PREFECT_API_URL y MLFLOW_TRACKING_URI definidas):
    poetry run python infraestructure/flows/desplegar.py

Deja esta terminal abierta: `serve` es el proceso que ejecuta los flows cuando toca.
"""
import sys
from pathlib import Path

from prefect import serve
from prefect.schedules import Cron

# Carpeta de los flows y carpeta actividad-3 en el path, para que los imports funcionen siempre.
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from drift_evento_flow import drift_evento_flow  # noqa: E402
from drift_features_flow import drift_features_flow  # noqa: E402
from entrenamiento_flow import entrenamiento_flow  # noqa: E402
from inferencia_flow import inferencia_flow  # noqa: E402

from scripts.config import cargar_config  # noqa: E402

if __name__ == "__main__":
    cfg = cargar_config()["despliegue"]
    zona = cfg["zona_horaria"]
    despliegues = [
        entrenamiento_flow.to_deployment(
            name=cfg["entrenamiento"], tags=["actividad-3", "entrenamiento"],
            description="Entrena LR, RF y XGBoost (6 experimentos) y promueve el mejor si supera al champion.",
        ),
        inferencia_flow.to_deployment(
            name=cfg["inferencia"], schedule=Cron(cfg["cron_inferencia"], timezone=zona),
            tags=["actividad-3", "inferencia"],
            description="Puntúa un lote de la parte de prueba con drift emulado (inferencia.nivel_drift).",
        ),
        drift_features_flow.to_deployment(
            name=cfg["drift_features"], schedule=Cron(cfg["cron_drift_features"], timezone=zona),
            tags=["actividad-3", "monitoreo"],
            description="Drift de features del último lote con Evidently; alerta si supera monitoreo.umbral_drift.",
        ),
        drift_evento_flow.to_deployment(
            name=cfg["drift_evento"], schedule=Cron(cfg["cron_drift_evento"], timezone=zona),
            tags=["actividad-3", "monitoreo"],
            description="Drift de la tasa de retiro (Y) del último lote; alerta si p-valor < monitoreo.alfa_evento.",
        ),
    ]
    for d in despliegues:
        print(f"Despliegue: {d.flow_name}/{d.name}")
    serve(*despliegues)
