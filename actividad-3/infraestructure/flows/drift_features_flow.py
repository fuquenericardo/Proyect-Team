"""Actividad 3: pipeline de drift de FEATURES con Evidently (Tarea 2).

Pasos:  referencia (train) → lote (copia que dejó la inferencia) → reporte Evidently por lote → alerta → registro

- Compara la distribución de cada variable de entrada del lote con la de entrenamiento (DataDriftPreset).
- Guarda por lote: reports/drift_features_<lote>.html (para abrir en el navegador), .json y
  reports/drift_features_<lote>_columnas.csv (variable, método, puntuación, drift).
- ALERTA si la fracción de variables con drift supera `monitoreo.umbral_drift`.
- No necesita etiquetas. MLflow es opcional (`monitoreo.registrar_en_mlflow`).

Ejecución (desde la carpeta actividad-3, después de la inferencia):
    poetry run python infraestructure/flows/drift_features_flow.py                       # último lote
    poetry run python infraestructure/flows/drift_features_flow.py lotes/<archivo>.csv   # un lote concreto
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import warnings  # noqa: E402

import pandas as pd  # noqa: E402
from evidently import Report  # noqa: E402
from evidently.presets import DataDriftPreset  # noqa: E402
from prefect import flow, get_run_logger, task  # noqa: E402
from prefect.cache_policies import NO_CACHE  # noqa: E402

from domain.inferencia import nivel_desde_nombre  # noqa: E402
from domain.monitoreo import (  # noqa: E402
    columnas_desde_evidently,
    fraccion_desde_evidently,
    hay_alerta,
    mensaje_alerta,
)
from scripts.config import cargar_config  # noqa: E402
from scripts.monitoreo_io import cargar_lote, cargar_referencia, registrar_en_mlflow, resolver_lote  # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)


@task(name="referencia-entrenamiento", cache_policy=NO_CACHE)
def referencia(cfg):
    ref = cargar_referencia(cfg)
    print(f"Referencia: {len(ref)} filas de {cfg['datos']['train']}")
    return ref


@task(name="cargar-lote", cache_policy=NO_CACHE)
def lote_a_comparar(cfg, ruta):
    lote = cargar_lote(ruta, cfg)
    print(f"Lote: {ruta} ({len(lote)} filas, nivel de drift {nivel_desde_nombre(ruta)})")
    return lote


@task(name="reporte-evidently", cache_policy=NO_CACHE)
def calcular_drift(ref, lote, cfg, ruta_lote):
    """Ejecuta DataDriftPreset sobre las variables de entrada y guarda el reporte del lote."""
    objetivo = cfg["datos"]["objetivo"]
    variables = [c for c in ref.columns if c != objetivo]
    snapshot = Report([DataDriftPreset()]).run(reference_data=ref[variables], current_data=lote[variables])
    resultado = snapshot.dict()

    carpeta = Path(cfg["monitoreo"]["carpeta_reportes"])
    carpeta.mkdir(parents=True, exist_ok=True)
    base = carpeta / f"drift_features_{Path(ruta_lote).stem}"
    snapshot.save_html(f"{base}.html")
    snapshot.save_json(f"{base}.json")
    columnas = pd.DataFrame(columnas_desde_evidently(resultado)).sort_values("puntuacion", ascending=False)
    columnas.to_csv(f"{base}_columnas.csv", index=False)

    fraccion = fraccion_desde_evidently(resultado)
    con_drift = columnas.loc[columnas["drift"], "columna"].tolist()
    print(f"Variables con drift ({len(con_drift)} de {len(columnas)}): {con_drift or 'ninguna'}")
    print(f"Reporte: {base}.html")
    return fraccion, con_drift, [f"{base}.html", f"{base}_columnas.csv"]


@task(name="alerta", cache_policy=NO_CACHE)
def alerta(fraccion, umbral):
    activa = hay_alerta(fraccion, umbral)
    texto = mensaje_alerta("features", activa, f"{fraccion:.0%} de las variables (umbral {umbral:.0%})")
    if activa:
        get_run_logger().warning(texto)  # ALERTA: se ve en amarillo en los logs de Prefect
    else:
        print(texto)
    return activa


@flow(name="drift-features", log_prints=True)
def drift_features_flow(ruta_lote: str | None = None):
    cfg = cargar_config()
    ruta = resolver_lote(cfg, ruta_lote)
    fraccion, con_drift, reportes = calcular_drift(referencia(cfg), lote_a_comparar(cfg, ruta), cfg, ruta)
    umbral = cfg["monitoreo"]["umbral_drift"]
    activa = alerta(fraccion, umbral)
    registrar_en_mlflow(
        cfg,
        f"drift-features-{ruta.stem}",
        {"tipo_drift": "features", "lote": str(ruta), "nivel_drift": str(nivel_desde_nombre(ruta)),
         "alerta": str(activa), "variables_con_drift": ",".join(con_drift) or "ninguna"},
        {"fraccion_drift": fraccion, "n_variables_drift": len(con_drift), "umbral_drift": umbral, "alerta": int(activa)},
        reportes,
    )
    return {"lote": str(ruta), "fraccion": fraccion, "alerta": activa}


if __name__ == "__main__":
    drift_features_flow(sys.argv[1] if len(sys.argv) > 1 else None)
