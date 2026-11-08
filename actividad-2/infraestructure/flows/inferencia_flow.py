"""Actividad 2: pipeline de inferencia (programada a diario, ver desplegar.py).

Pasos:  cargar champion → ingesta del lote → predicción → guardar CSV con fecha
        → comparar con retiros reales (si hay etiquetas) → registrar en MLflow

El modelo se carga por su ALIAS (models:/<nombre>@champion), nunca por número de versión.
Cuando el entrenamiento promueve un champion nuevo, la inferencia lo usa sin cambiar código.

El lote son los asociados de `inferencia.fuente` (test2 SIN la Y, como si fueran nuevos).
Si `inferencia.etiquetas` apunta al mismo periodo CON la Y, se comparan las predicciones con los
retiros reales (AUC, recall, precisión) y se guardan en el run de inferencia.

Requisitos: un champion registrado (ejecuta antes el entrenamiento) y, en la terminal:
    $env:PREFECT_API_URL = "http://localhost:4200/api"
    $env:MLFLOW_TRACKING_URI = "http://localhost:5000"

Ejecución de prueba (desde la carpeta actividad-2):
    poetry run python infraestructure/flows/inferencia_flow.py
"""
import sys
from pathlib import Path

# Carpeta actividad-2 en el path: los imports funcionan aunque Poetry no instale el paquete.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from datetime import date  # noqa: E402

import mlflow  # noqa: E402
from mlflow.tracking import MlflowClient  # noqa: E402
from prefect import flow, task  # noqa: E402
from prefect.cache_policies import NO_CACHE  # noqa: E402

from domain.modelo import calcular_metricas  # noqa: E402
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


@task(name="ingesta-lote", cache_policy=NO_CACHE)
def ingesta(cfg):
    """Toma el lote de asociados a puntuar (sin la Y). Conserva las columnas id para identificarlos."""
    inf = cfg["inferencia"]
    df = cargar_csv(inf["fuente"], cfg, quitar_ids=False)
    df = df.drop(columns=[cfg["datos"]["objetivo"]], errors="ignore")  # por si la fuente trae la Y
    if inf.get("tamano_lote"):
        df = df.sample(n=min(inf["tamano_lote"], len(df)), random_state=inf.get("semilla"))
    print(f"Lote de {len(df)} asociados tomado de {inf['fuente']}")
    return df


@task(name="prediccion", cache_policy=NO_CACHE)
def prediccion(modelo, lote, cfg):
    """Calcula probabilidad y clase, y guarda el resultado en predicciones/prediccion_<fecha>.csv."""
    columnas = list(modelo.feature_names_in_)  # las mismas columnas con las que se entrenó el champion
    proba = modelo.predict_proba(lote[columnas])[:, 1]

    resultado = lote.copy()
    resultado["probabilidad_retiro"] = proba.round(6)
    resultado["prediccion"] = (proba >= cfg["metricas"]["umbral"]).astype(int)
    resultado["fecha_ejecucion"] = date.today().isoformat()
    print(f"Predicción: {int(resultado['prediccion'].sum())} de {len(resultado)} asociados con riesgo de retiro")
    return resultado


@task(name="comparar-retiros-reales", cache_policy=NO_CACHE)
def comparar_con_reales(resultado, cfg):
    """Une las predicciones con la Y real (inferencia.etiquetas) por las columnas id y calcula métricas."""
    ruta = cfg["inferencia"].get("etiquetas")
    if not ruta:
        print("Sin etiquetas: no se comparan con retiros reales")
        return resultado, {}

    ids, objetivo = cfg["datos"]["columnas_id"], cfg["datos"]["objetivo"]
    reales = cargar_csv(ruta, cfg, quitar_ids=False)[ids + [objetivo]].rename(columns={objetivo: "retiro_real"})
    resultado = resultado.merge(reales, on=ids, how="left", validate="one_to_one")
    con_y = resultado.dropna(subset=["retiro_real"])
    if con_y.empty or con_y["retiro_real"].nunique() < 2:
        print("Las etiquetas no cubren el lote: no se calculan métricas reales")
        return resultado, {}

    m = calcular_metricas(con_y["retiro_real"].astype(int), con_y["probabilidad_retiro"], cfg["metricas"]["umbral"])
    metricas = {f"real_{k}": v for k, v in m.items()}
    metricas["retiros_reales"] = int(con_y["retiro_real"].sum())
    metricas["retiros_detectados"] = int(((con_y["prediccion"] == 1) & (con_y["retiro_real"] == 1)).sum())
    print(
        f"Contra retiros reales: AUC {m['auc']:.4f} | detectados {metricas['retiros_detectados']} "
        f"de {metricas['retiros_reales']} retiros (recall {m['recall']:.2%}) | precisión {m['precision']:.2%}"
    )
    return resultado, metricas


@task(name="guardar-y-registrar", cache_policy=NO_CACHE)
def guardar_y_registrar(resultado, metricas_reales, cfg, version, run_id_entrenamiento):
    """Guarda el CSV con fecha y crea el run de MLflow con tag fase = inferencia."""
    carpeta = Path(cfg["inferencia"]["carpeta_salida"])
    carpeta.mkdir(parents=True, exist_ok=True)
    ruta = carpeta / f"prediccion_{date.today().isoformat()}.csv"
    resultado.to_csv(ruta, index=False, sep=cfg["datos"].get("separador", ","))
    print(f"Predicciones guardadas en {ruta}")

    with mlflow.start_run(run_name=f"inferencia-{date.today().isoformat()}") as run:
        mlflow.set_tags(
            {
                "fase": "inferencia",
                "modelo_nombre": cfg["registro"]["nombre_modelo"],
                "modelo_version": version,
                "modelo_run_id": run_id_entrenamiento,
                "fuente": cfg["inferencia"]["fuente"],
            }
        )
        mlflow.log_param("tamano_lote", len(resultado))
        mlflow.log_metrics(
            {
                "filas_predichas": len(resultado),
                "probabilidad_media": float(resultado["probabilidad_retiro"].mean()),
                "tasa_predicha": float(resultado["prediccion"].mean()),
                **metricas_reales,
            }
        )
        mlflow.log_artifact(str(ruta), artifact_path="predicciones")
    print(f"Run de inferencia en MLflow: {run.info.run_id}")
    return str(ruta)


@flow(name="inferencia-retiro", log_prints=True)
def inferencia_flow():
    cfg = cargar_config()
    mlflow.set_tracking_uri(tracking_uri(cfg))
    mlflow.set_experiment(cfg["mlflow"]["experimento"])

    modelo, version, run_id_entrenamiento = cargar_champion(cfg)
    lote = ingesta(cfg)
    resultado = prediccion(modelo, lote, cfg)
    resultado, metricas_reales = comparar_con_reales(resultado, cfg)
    return guardar_y_registrar(resultado, metricas_reales, cfg, version, run_id_entrenamiento)


if __name__ == "__main__":
    inferencia_flow()
