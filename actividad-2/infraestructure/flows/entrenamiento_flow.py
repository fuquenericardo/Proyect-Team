"""Actividad 2: pipeline de entrenamiento.

Pasos:  ingesta → entrenamiento → evaluación → registro

- ingesta:        lee train/test (configs/config.yaml) y elige las variables (configs/features.yaml).
- entrenamiento:  entrena cada candidato activo (regresión logística, random forest, xgboost).
- evaluación:     calcula AUC, accuracy, precision, recall y f1 sobre test y lo registra en MLflow
                  (un run hijo por modelo, todos con el tag fase = entrenamiento).
- registro:       toma el modelo con mayor AUC y lo compara con el champion vigente. Solo si
                  cumple las reglas de `registro` en el YAML se crea una versión nueva en el
                  Model Registry y recibe el alias `champion`. Si no, los runs quedan, el alias no cambia.

La lógica del modelo está en domain/modelo.py; este archivo solo orquesta y registra.

Requisitos: servicios levantados y, en la terminal:
    $env:PREFECT_API_URL = "http://localhost:4200/api"
    $env:MLFLOW_TRACKING_URI = "http://localhost:5000"

Ejecución (desde la raíz del repositorio):
    poetry run python infraestructure/flows/entrenamiento_flow.py
"""
import sys
from pathlib import Path

# Raíz del repositorio en el path: los imports funcionan aunque Poetry no instale el paquete.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import time
import warnings
from datetime import datetime

import mlflow
import pandas as pd
import skops.io as sio
from mlflow.exceptions import MlflowException
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from prefect import flow, task
from prefect.artifacts import create_markdown_artifact
from prefect.cache_policies import NO_CACHE

from domain.modelo import (
    calcular_metricas,
    construir_modelo,
    decidir_promocion,
    elegir_mejor,
    figura_matriz_confusion,
    figura_roc,
    resolver_features,
    tabla_markdown,
)
from scripts.config import cargar_config, tracking_uri
from scripts.load_data import cargar_train_test


warnings.filterwarnings("ignore", message=".*Integer columns in Python cannot represent missing values.*")


def tipos_confiables(modelo):
    """Tipos del modelo que skops no conoce (p. ej. sklearn.tree._tree.Tree o xgboost.core.Booster)."""
    return sio.get_untrusted_types(data=sio.dumps(modelo))


# ---------------------------------------------------------------------------
# 1. Ingesta
# ---------------------------------------------------------------------------
@task(name="ingesta", cache_policy=NO_CACHE)
def ingesta(cfg):
    """Carga train y test, valida columnas y resuelve la selección de variables."""
    train, test = cargar_train_test(cfg)
    objetivo = cfg["datos"]["objetivo"]
    sel = cfg["features"]["seleccion"]
    numericas, categoricas = resolver_features(train, objetivo, sel, cfg["features"]["custom"])

    print(f"Train: {len(train)} filas | Test: {len(test)} filas | % positivos train: {train[objetivo].mean():.2%}")
    print(f"Selección de features: {sel} -> {len(numericas) + len(categoricas)} variables")
    print(f"  Numéricas ({len(numericas)}): {numericas}")
    print(f"  Categóricas ({len(categoricas)}): {categoricas}")
    return train, test, numericas, categoricas


# ---------------------------------------------------------------------------
# 2. Entrenamiento
# ---------------------------------------------------------------------------
@task(name="entrenamiento", task_run_name="entrenamiento-{candidato[nombre]}", cache_policy=NO_CACHE)
def entrenamiento(train, candidato, numericas, categoricas, objetivo):
    """Entrena un candidato (preprocesado + estimador). Devuelve (modelo, segundos)."""
    print(f"Entrenando {candidato['nombre']} ({candidato['tipo']}) con {candidato.get('parametros')}")
    modelo = construir_modelo(candidato["tipo"], candidato.get("parametros"), numericas, categoricas)
    inicio = time.perf_counter()
    modelo.fit(train[numericas + categoricas], train[objetivo])
    segundos = time.perf_counter() - inicio
    print(f"{candidato['nombre']} entrenado en {segundos:.1f} s")
    return modelo, segundos


# ---------------------------------------------------------------------------
# 3. Evaluación (+ run en MLflow)
# ---------------------------------------------------------------------------
@task(name="evaluacion", task_run_name="evaluacion-{candidato[nombre]}", cache_policy=NO_CACHE)
def evaluacion(modelo, segundos, test, candidato, numericas, categoricas, cfg, padre_id, n_train):
    """Evalúa sobre test y registra un run hijo en MLflow con params, métricas, gráficas y el modelo."""
    objetivo = cfg["datos"]["objetivo"]
    umbral = cfg["metricas"]["umbral"]
    variables = numericas + categoricas
    X_test, y_test = test[variables], test[objetivo]

    proba = modelo.predict_proba(X_test)[:, 1]
    metricas = calcular_metricas(y_test, proba, umbral)
    metricas["segundos_entrenamiento"] = round(segundos, 3)
    print(f"{candidato['nombre']}: " + " | ".join(f"{k}={v:.4f}" for k, v in metricas.items()))

    tags = {"fase": "entrenamiento", "modelo_tipo": candidato["tipo"], "mlflow.parentRunId": padre_id}
    with mlflow.start_run(run_name=candidato["nombre"], tags=tags) as run:
        mlflow.log_params(
            {
                "modelo": candidato["tipo"],
                **{f"hp_{k}": v for k, v in (candidato.get("parametros") or {}).items()},
                "features": cfg["features"]["seleccion"],
                "n_features": len(variables),
                "umbral": umbral,
                "n_train": n_train,
                "n_test": len(test),
            }
        )
        mlflow.log_metrics(metricas)
        mlflow.log_figure(figura_roc(y_test, {candidato["nombre"]: proba}), "graficas/roc.png")
        mlflow.log_figure(
            figura_matriz_confusion(y_test, proba, umbral, f"Matriz de confusión - {candidato['nombre']}"),
            "graficas/matriz_confusion.png",
        )
        info = mlflow.sklearn.log_model(
            modelo,
            name="model",
            signature=infer_signature(X_test, proba),
            input_example=X_test.head(3),
            # MLflow guarda con skops (formato seguro). Los árboles de random forest y el Booster de
            # xgboost no son tipos "confiables" por defecto: como son modelos entrenados aquí, se confía
            # solo en los tipos que contiene este modelo.
            skops_trusted_types=tipos_confiables(modelo),
        )
    print(f"Run de MLflow: {run.info.run_id}")
    return {
        "nombre": candidato["nombre"],
        "tipo": candidato["tipo"],
        "run_id": run.info.run_id,
        "model_uri": info.model_uri,
        "metricas": metricas,
        "proba": proba,
    }


# ---------------------------------------------------------------------------
# 4. Registro (comparación con el champion + Model Registry)
# ---------------------------------------------------------------------------
@task(name="registro", cache_policy=NO_CACHE)
def registro(resultados, test, cfg, padre_id):
    """Elige el mejor por AUC y lo promueve a champion solo si mejora al vigente."""
    client = MlflowClient()
    reglas = cfg["registro"]
    nombre, alias = reglas["nombre_modelo"], reglas.get("alias", "champion")
    objetivo = cfg["datos"]["objetivo"]

    # Tabla comparativa de los candidatos
    tabla = pd.DataFrame([{"modelo": r["nombre"], **r["metricas"]} for r in resultados])
    tabla = tabla.sort_values("auc", ascending=False).reset_index(drop=True)
    print("Comparativa de candidatos:\n" + tabla.round(4).to_string(index=False))

    # Mejor candidato = mayor AUC entre los que cumplen el recall mínimo (si ninguno lo cumple,
    # se toma el de mayor AUC y la regla de promoción lo rechazará).
    mejor = elegir_mejor(resultados, reglas.get("recall_minimo"))
    print(f"Mejor candidato: {mejor['nombre']} (AUC {mejor['metricas']['auc']:.4f})")

    # Champion vigente
    try:
        actual = client.get_model_version_by_alias(nombre, alias)
        auc_champion = client.get_run(actual.run_id).data.metrics["auc"]
        print(f"Champion vigente: versión {actual.version} (AUC {auc_champion:.4f})")
    except MlflowException:
        actual, auc_champion = None, None
        print("Todavía no hay champion")

    promover, motivo = decidir_promocion(mejor["metricas"], auc_champion, reglas)
    version = None
    if promover:
        mv = mlflow.register_model(mejor["model_uri"], nombre)
        version = mv.version
        client.set_registered_model_alias(nombre, alias, version)
        client.set_model_version_tag(nombre, version, "estado", "aprobado")
        client.set_model_version_tag(nombre, version, "modelo_tipo", mejor["tipo"])
        client.set_model_version_tag(nombre, version, "features", cfg["features"]["seleccion"])
        client.set_model_version_tag(nombre, version, "auc", f"{mejor['metricas']['auc']:.4f}")
        resultado = f"Versión {version} ({mejor['nombre']}) es el nuevo {alias}: {motivo}"
    else:
        resultado = f"El {alias} no cambia: {motivo}"
    print(resultado)

    # Evidencias en el run padre de MLflow
    with mlflow.start_run(run_id=padre_id):
        mlflow.log_params({"mejor_modelo": mejor["nombre"], "candidatos": len(resultados)})
        mlflow.log_metric("auc_mejor", mejor["metricas"]["auc"])
        if auc_champion is not None:
            mlflow.log_metric("auc_champion_anterior", auc_champion)
        mlflow.set_tags({"promovido": str(promover), "decision": resultado[:250]})
        if version is not None:
            mlflow.set_tag("version_registrada", version)
        mlflow.log_table(tabla, "comparativa_modelos.json")
        mlflow.log_text(tabla_markdown(tabla), "comparativa_modelos.md")
        fig = figura_roc(test[objetivo], {r["nombre"]: r["proba"] for r in resultados})
        mlflow.log_figure(fig, "graficas/roc_comparativa.png")

    # Evidencia en Prefect (pestaña Artifacts del flow run)
    create_markdown_artifact(
        key="comparativa-modelos",
        markdown=(
            "## Comparativa de modelos (test)\n\n"
            f"{tabla_markdown(tabla)}\n\n"
            f"**Mejor:** {mejor['nombre']}  \n"
            f"**Champion anterior (AUC):** {auc_champion if auc_champion is None else round(auc_champion, 4)}  \n"
            f"**Decisión:** {resultado}\n"
        ),
        description="Resultado del entrenamiento y decisión sobre el champion",
    )
    return {"promovido": promover, "version": version, "mejor": mejor["nombre"], "motivo": motivo}


# ---------------------------------------------------------------------------
# Flow
# ---------------------------------------------------------------------------
@flow(name="entrenamiento-churn", log_prints=True)
def entrenamiento_flow():
    cfg = cargar_config()
    mlflow.set_tracking_uri(tracking_uri(cfg))
    experimento = mlflow.set_experiment(cfg["mlflow"]["experimento"])
    print(f"MLflow: {mlflow.get_tracking_uri()} | experimento: {cfg['mlflow']['experimento']}")

    candidatos = [c for c in cfg["candidatos"] if c.get("activo", True)]
    if not candidatos:
        raise ValueError("No hay candidatos activos en configs/config.yaml")

    # Run padre: agrupa la ejecución completa. Cada modelo es un run hijo.
    client = MlflowClient()
    padre = client.create_run(
        experimento.experiment_id,
        run_name=f"entrenamiento-{datetime.now():%Y%m%d-%H%M}",
        tags={"fase": "entrenamiento", "tipo_run": "ejecucion"},
    )
    padre_id = padre.info.run_id

    try:
        train, test, numericas, categoricas = ingesta(cfg)
        with mlflow.start_run(run_id=padre_id):
            mlflow.log_params(
                {
                    "features": cfg["features"]["seleccion"],
                    "n_features": len(numericas) + len(categoricas),
                    "n_train": len(train),
                    "n_test": len(test),
                }
            )

        resultados = []
        for candidato in candidatos:
            modelo, segundos = entrenamiento(train, candidato, numericas, categoricas, cfg["datos"]["objetivo"])
            resultados.append(
                evaluacion(modelo, segundos, test, candidato, numericas, categoricas, cfg, padre_id, len(train))
            )

        decision = registro(resultados, test, cfg, padre_id)
        client.set_terminated(padre_id, "FINISHED")
        return decision
    except Exception:
        client.set_terminated(padre_id, "FAILED")
        raise


if __name__ == "__main__":
    entrenamiento_flow()
