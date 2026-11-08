"""Actividad 2: pipeline de inferencia (programada a diario, ver desplegar.py).

Pasos:  cargar champion → ingesta del lote → predicción → guardar CSV con fecha → registrar en MLflow

El modelo se carga por su ALIAS (models:/<nombre>@champion), nunca por número de versión.
Cuando el entrenamiento promueve un champion nuevo, la inferencia lo usa sin cambiar código.

Requisitos: un champion registrado (ejecuta antes el entrenamiento) y, en la terminal:
    $env:PREFECT_API_URL = "http://localhost:4200/api"
    $env:MLFLOW_TRACKING_URI = "http://localhost:5000"

Ejecución de prueba (desde la raíz del repositorio):
    poetry run python infraestructure/flows/inferencia_flow.py
"""
import sys
from pathlib import Path

# Raíz del repositorio en el path: los imports funcionan aunque Poetry no instale el paquete.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from datetime import date  # noqa: E402

import mlflow
from mlflow.tracking import MlflowClient
from prefect import flow, task
from prefect.cache_policies import NO_CACHE

from scripts.config import cargar_config, tracking_uri
from scripts.load_data import cargar_csv


@task(name="cargar-champion", cache_policy=NO_CACHE)
def cargar_champion(cfg):
    """Carga el modelo con el alias champion. Devuelve (modelo, versión, run_id de entrenamiento)."""
    nombre, alias = cfg["registro"]["nombre_modelo"], cfg["registro"].get("alias", "champion")
    version = MlflowClient().get_model_version_by_alias(nombre, alias)
    modelo = mlflow.sklearn.load_model(f"models:/{nombre}@{alias}")
    print(f"Modelo cargado: models:/{nombre}@{alias} -> versión {version.version}")
    return modelo, version.version, version.run_id


@task(name="ingesta-lote", cache_policy=NO_CACHE)
def ingesta(cfg):
    """Toma un lote de 'clientes nuevos'. Se simula con una muestra aleatoria del CSV de `inferencia.fuente`."""
    inf, d = cfg["inferencia"], cfg["datos"]
    df = cargar_csv(inf["fuente"], d["objetivo"], d.get("columnas_excluir"), d.get("clase_positiva", 1))
    n = min(inf["tamano_lote"], len(df))
    lote = df.sample(n=n, random_state=inf.get("semilla"))
    print(f"Lote de {len(lote)} clientes tomado de {inf['fuente']}")
    return lote


@task(name="prediccion", cache_policy=NO_CACHE)
def prediccion(modelo, lote, cfg):
    """Calcula probabilidad y clase, y guarda el resultado en predicciones/prediccion_<fecha>.csv."""
    columnas = list(modelo.feature_names_in_)  # las mismas columnas con las que se entrenó el champion
    proba = modelo.predict_proba(lote[columnas])[:, 1]

    resultado = lote.copy()
    resultado["probabilidad"] = proba.round(6)
    resultado["prediccion"] = (proba >= cfg["metricas"]["umbral"]).astype(int)
    resultado["fecha_ejecucion"] = date.today().isoformat()

    carpeta = Path(cfg["inferencia"]["carpeta_salida"])
    carpeta.mkdir(parents=True, exist_ok=True)
    ruta = carpeta / f"prediccion_{date.today().isoformat()}.csv"
    resultado.to_csv(ruta, index=False)
    print(f"Predicciones guardadas en {ruta} | positivas: {int(resultado['prediccion'].sum())}/{len(resultado)}")
    return str(ruta), len(resultado), float(proba.mean()), float(resultado["prediccion"].mean())


@task(name="registrar-inferencia", cache_policy=NO_CACHE)
def registrar_inferencia(cfg, version, run_id_entrenamiento, ruta, filas, proba_media, tasa_positivos):
    """Run de MLflow con tag fase = inferencia que apunta a la versión del modelo usada."""
    with mlflow.start_run(run_name=f"inferencia-{date.today().isoformat()}") as run:
        mlflow.set_tags(
            {
                "fase": "inferencia",
                "modelo_nombre": cfg["registro"]["nombre_modelo"],
                "modelo_version": version,
                "modelo_run_id": run_id_entrenamiento,
            }
        )
        mlflow.log_param("tamano_lote", filas)
        mlflow.log_metrics(
            {"filas_predichas": filas, "probabilidad_media": proba_media, "tasa_positivos": tasa_positivos}
        )
        mlflow.log_artifact(ruta, artifact_path="predicciones")
    print(f"Run de inferencia en MLflow: {run.info.run_id}")


@flow(name="inferencia-churn", log_prints=True)
def inferencia_flow():
    cfg = cargar_config()
    mlflow.set_tracking_uri(tracking_uri(cfg))
    mlflow.set_experiment(cfg["mlflow"]["experimento"])

    modelo, version, run_id_entrenamiento = cargar_champion(cfg)
    lote = ingesta(cfg)
    ruta, filas, proba_media, tasa = prediccion(modelo, lote, cfg)
    registrar_inferencia(cfg, version, run_id_entrenamiento, ruta, filas, proba_media, tasa)
    return ruta


if __name__ == "__main__":
    inferencia_flow()
