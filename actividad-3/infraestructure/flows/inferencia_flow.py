"""Actividad 3: pipeline de inferencia CON DRIFT EMULADO (Tarea 1).

Pasos:  cargar champion → ingesta del lote (parte de prueba + drift emulado, copia en lotes/)
        → predicción → guardar CSV → registrar en MLflow

- El lote sale de `inferencia.fuente` (test2, con la Y real) y se le aplica un drift controlado por
  `inferencia.nivel_drift` (0.0 a 1.0) en configs/config.yaml. El nivel NO se pasa al ejecutar.
- Se guarda una copia del lote puntuado en lotes/lote_<fecha-hora>_drift-<nivel>.csv y en
  `monitoreo.lote_actual`, que leen los flows de drift.
- El modelo se carga por su alias `champion` (models:/<nombre>@champion).

Toda la lógica de datos está en domain/inferencia.py y domain/monitoreo.py (sin MLflow ni Prefect).

Ejecución (desde la carpeta actividad-3, con MLflow y Prefect levantados):
    poetry run python infraestructure/flows/inferencia_flow.py
"""
import sys
from pathlib import Path

# Carpeta actividad-3 en el path: los imports funcionan aunque Poetry no instale el paquete.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from datetime import date, datetime  # noqa: E402

import mlflow  # noqa: E402
from mlflow.tracking import MlflowClient  # noqa: E402
from prefect import flow, task  # noqa: E402
from prefect.cache_policies import NO_CACHE  # noqa: E402

from domain.inferencia import metricas_lote, nombre_lote, preparar_lote, predecir  # noqa: E402
from domain.modelo import resolver_features  # noqa: E402
from scripts.config import cargar_config, tracking_uri  # noqa: E402
from scripts.load_data import cargar_csv  # noqa: E402


@task(name="cargar-champion", cache_policy=NO_CACHE)
def cargar_champion(cfg):
    """Carga el modelo con el alias champion. Devuelve (modelo, versión, run_id de entrenamiento)."""
    nombre, alias = cfg["registro"]["nombre_modelo"], cfg["registro"].get("alias", "champion")
    version = MlflowClient().get_model_version_by_alias(nombre, alias)
    modelo = mlflow.sklearn.load_model(f"models:/{nombre}@{alias}")
    print(f"Modelo cargado: models:/{nombre}@{alias} -> versión {version.version}")
    return modelo, version.version, version.run_id


@task(name="ingesta-lote-con-drift", cache_policy=NO_CACHE)
def ingesta(cfg, marca):
    """Toma un lote de la parte de prueba, le aplica el drift emulado y guarda una copia para el monitoreo."""
    inf, d = cfg["inferencia"], cfg["datos"]
    fuente = cargar_csv(inf["fuente"], cfg, quitar_ids=False)
    numericas, _ = resolver_features(fuente.drop(columns=d.get("columnas_id", [])), d["objetivo"], "ALL")
    lote, desplazadas = preparar_lote(
        fuente, inf["tamano_lote"], inf["semilla"], inf["nivel_drift"], numericas, d["objetivo"], cfg["drift_emulado"]
    )

    sep = d.get("separador", ",")
    carpeta = Path(cfg["monitoreo"]["carpeta_lotes"])
    carpeta.mkdir(parents=True, exist_ok=True)
    ruta_copia = carpeta / nombre_lote(inf["nivel_drift"], marca)
    lote.to_csv(ruta_copia, index=False, sep=sep)
    lote.to_csv(cfg["monitoreo"]["lote_actual"], index=False, sep=sep)

    print(f"Lote de {len(lote)} asociados de {inf['fuente']} | nivel de drift {inf['nivel_drift']}")
    print(f"Variables desplazadas ({len(desplazadas)}): {desplazadas or 'ninguna'}")
    print(f"Tasa de retiro del lote: {lote[d['objetivo']].mean():.2%}")
    print(f"Copia del lote: {ruta_copia} (y {cfg['monitoreo']['lote_actual']})")
    return lote, desplazadas, str(ruta_copia)


@task(name="prediccion", cache_policy=NO_CACHE)
def prediccion(modelo, lote, cfg, marca):
    """Predice con el champion y guarda predicciones/prediccion_<fecha-hora>_drift-<nivel>.csv."""
    resultado = predecir(modelo, lote, cfg["metricas"]["umbral"])
    metricas = metricas_lote(resultado, cfg["datos"]["objetivo"], cfg["metricas"]["umbral"])

    carpeta = Path(cfg["inferencia"]["carpeta_salida"])
    carpeta.mkdir(parents=True, exist_ok=True)
    ruta = carpeta / f"prediccion_{marca}_drift-{cfg['inferencia']['nivel_drift']:.1f}.csv"
    resultado.to_csv(ruta, index=False, sep=cfg["datos"].get("separador", ","))

    print(f"Predicción: {int(resultado['prediccion'].sum())} de {len(resultado)} asociados con riesgo de retiro")
    if "real_auc" in metricas:
        print(
            f"Contra la Y del lote: AUC {metricas['real_auc']:.4f} | detectados {metricas['retiros_detectados']} "
            f"de {metricas['retiros_reales']} retiros (recall {metricas['real_recall']:.2%})"
        )
    print(f"Predicciones guardadas en {ruta}")
    return str(ruta), metricas


@task(name="registrar-inferencia", cache_policy=NO_CACHE)
def registrar(cfg, version, run_id_entrenamiento, desplazadas, ruta_lote, ruta_pred, metricas):
    """Run de MLflow con tag fase = inferencia, el nivel de drift y las variables desplazadas."""
    nivel = cfg["inferencia"]["nivel_drift"]
    with mlflow.start_run(run_name=f"inferencia-{date.today().isoformat()}-drift-{nivel:.1f}") as run:
        mlflow.set_tags(
            {
                "fase": "inferencia",
                "modelo_nombre": cfg["registro"]["nombre_modelo"],
                "modelo_version": version,
                "modelo_run_id": run_id_entrenamiento,
                "nivel_drift": nivel,
                "variables_desplazadas": ",".join(desplazadas) or "ninguna",
                "lote": ruta_lote,
            }
        )
        mlflow.log_params({"nivel_drift": nivel, "tamano_lote": metricas["filas_predichas"],
                           "n_variables_desplazadas": len(desplazadas)})
        mlflow.log_metrics(metricas)
        mlflow.log_artifact(ruta_lote, artifact_path="lote")
        mlflow.log_artifact(ruta_pred, artifact_path="predicciones")
    print(f"Run de inferencia en MLflow: {run.info.run_id}")


@flow(name="inferencia-retiro-drift", log_prints=True)
def inferencia_flow():
    cfg = cargar_config()
    mlflow.set_tracking_uri(tracking_uri(cfg))
    mlflow.set_experiment(cfg["mlflow"]["experimento"])
    marca = datetime.now().strftime("%Y%m%d-%H%M%S")

    modelo, version, run_id_entrenamiento = cargar_champion(cfg)
    lote, desplazadas, ruta_lote = ingesta(cfg, marca)
    ruta_pred, metricas = prediccion(modelo, lote, cfg, marca)
    registrar(cfg, version, run_id_entrenamiento, desplazadas, ruta_lote, ruta_pred, metricas)
    return ruta_lote


if __name__ == "__main__":
    inferencia_flow()
