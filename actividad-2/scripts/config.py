"""Lectura de la configuración (configs/config.yaml y configs/features.yaml).

Las rutas se resuelven desde la carpeta donde ejecutas el comando:
ejecuta siempre desde la carpeta actividad-2.
"""
import os

import yaml

RUTA_CONFIG = "configs/config.yaml"


def cargar_config(ruta=RUTA_CONFIG):
    """Devuelve la configuración como diccionario, con la selección de features incluida en cfg['features']."""
    with open(ruta, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    with open(cfg["features"]["archivo"], encoding="utf-8") as f:
        features = yaml.safe_load(f)
    cfg["features"].update(
        {
            "seleccion": str(features.get("seleccion", "ALL")).upper(),
            "custom": features.get("custom") or [],
        }
    )
    return cfg


def tracking_uri(cfg):
    """MLFLOW_TRACKING_URI (variable de entorno) tiene prioridad sobre el YAML."""
    return os.environ.get("MLFLOW_TRACKING_URI", cfg["mlflow"]["tracking_uri"])
