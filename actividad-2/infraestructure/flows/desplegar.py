"""Actividad 2: despliegue de las dos pipelines en Prefect.

- entrenamiento-retiro/entrenamiento-manual : sin horario, se lanza a mano.
- inferencia-retiro/inferencia-diaria       : todos los días según `despliegue.cron_inferencia`
                                             (por defecto 06:00 hora de Bogotá).

Ejecución (desde la carpeta actividad-2, con PREFECT_API_URL y MLFLOW_TRACKING_URI definidas):
    poetry run python infraestructure/flows/desplegar.py

Deja esta terminal abierta: `serve` es el proceso que ejecuta los flows cuando toca.
"""
import sys
from pathlib import Path

from prefect import serve
from prefect.schedules import Cron

# Carpeta de los flows y carpeta actividad-2 en el path, para que los imports funcionen siempre.
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from entrenamiento_flow import entrenamiento_flow  # noqa: E402
from inferencia_flow import inferencia_flow  # noqa: E402

from scripts.config import cargar_config  # noqa: E402

if __name__ == "__main__":
    cfg = cargar_config()["despliegue"]

    despliegue_entrenamiento = entrenamiento_flow.to_deployment(
        name=cfg["entrenamiento"],
        description="Entrena LR, RF y XGBoost, los evalúa en test1 y test2 (6 experimentos) y promueve el mejor si supera al champion.",
        tags=["actividad-2", "entrenamiento"],
    )
    despliegue_inferencia = inferencia_flow.to_deployment(
        name=cfg["inferencia"],
        schedule=Cron(cfg["cron_inferencia"], timezone=cfg["zona_horaria"]),
        description="Puntúa el lote diario de asociados con el champion y lo compara con los retiros reales.",
        tags=["actividad-2", "inferencia"],
    )
    print(f"Inferencia programada: '{cfg['cron_inferencia']}' ({cfg['zona_horaria']})")
    serve(despliegue_entrenamiento, despliegue_inferencia)
