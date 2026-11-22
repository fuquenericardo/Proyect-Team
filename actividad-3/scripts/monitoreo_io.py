"""Entrada/salida de los flows de drift: referencia, lotes y registro opcional en MLflow.

Separado de la lógica (domain/monitoreo.py) para que esa se pruebe sin archivos ni servicios.
"""
from pathlib import Path

from scripts.load_data import cargar_csv


def cargar_referencia(cfg):
    """Datos de entrenamiento (sin columnas id) como referencia. Muestra con semilla para que sea rápido y estable."""
    train = cargar_csv(cfg["datos"]["train"], cfg)
    n = cfg["monitoreo"].get("referencia_muestra")
    if n and n < len(train):
        train = train.sample(n=n, random_state=cfg["inferencia"].get("semilla", 42))
    return train


def resolver_lote(cfg, ruta=None):
    """Qué lote analizar: la ruta indicada; si no, la copia más reciente en carpeta_lotes; si no, lote_actual."""
    if ruta:
        return Path(ruta)
    copias = sorted(Path(cfg["monitoreo"]["carpeta_lotes"]).glob("lote_*_drift-*.csv"))
    if copias:
        return copias[-1]
    return Path(cfg["monitoreo"]["lote_actual"])


def cargar_lote(ruta, cfg):
    """Lee un lote guardado por la inferencia (con columnas id y, si las trae, la Y)."""
    ruta = Path(ruta)
    if not ruta.exists():
        raise FileNotFoundError(f"No existe el lote {ruta}: ejecuta antes la inferencia (inferencia_flow.py)")
    return cargar_csv(ruta, cfg, quitar_ids=True)


def registrar_en_mlflow(cfg, nombre_run, tags, metricas, artefactos=()):
    """Run de MLflow con tag fase = monitoreo. Si MLflow no responde, avisa y sigue (el drift no depende de él)."""
    if not cfg["monitoreo"].get("registrar_en_mlflow", False):
        return None
    import os

    # Si MLflow está apagado, que falle rápido en vez de reintentar durante minutos.
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "1")
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "10")
    try:
        import mlflow

        from scripts.config import tracking_uri

        mlflow.set_tracking_uri(tracking_uri(cfg))
        mlflow.set_experiment(cfg["mlflow"]["experimento"])
        with mlflow.start_run(run_name=nombre_run) as run:
            mlflow.set_tags({"fase": "monitoreo", **tags})
            mlflow.log_metrics(metricas)
            for ruta in artefactos:
                mlflow.log_artifact(str(ruta), artifact_path="reportes")
        print(f"Run de monitoreo en MLflow: {run.info.run_id}")
        return run.info.run_id
    except Exception as e:  # el monitoreo no debe caerse si MLflow está apagado
        print(f"Aviso: no se pudo registrar en MLflow ({type(e).__name__}: {e}). El reporte queda en disco.")
        return None
