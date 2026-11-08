"""Actividad 2: pipeline de entrenamiento.

Pasos:  ingesta → entrenamiento → evaluación → registro

- ingesta:        lee train y los conjuntos de evaluación (test1, test2) y aplica configs/features.yaml.
- entrenamiento:  entrena cada candidato activo con train (regresión logística, random forest, xgboost).
- evaluación:     evalúa cada modelo en test1 y en test2 (datos NO usados para entrenar).
                  Cada combinación modelo x conjunto es un run hijo en MLflow con tag fase = entrenamiento:
                  3 modelos x 2 conjuntos = 6 experimentos por ejecución.
- registro:       el mejor por AUC en test1 se compara con el champion vigente y debe confirmarse en
                  test2 (retiros reales). Solo si cumple las reglas de `registro` en el YAML se crea una
                  versión nueva en el Model Registry con el alias `champion`. Si no, el alias no cambia.

La lógica del modelo está en domain/modelo.py; este archivo solo orquesta y registra.

Requisitos: servicios levantados y, en la terminal:
    $env:PREFECT_API_URL = "http://localhost:4200/api"
    $env:MLFLOW_TRACKING_URI = "http://localhost:5000"

Ejecución (desde la carpeta actividad-2):
    poetry run python infraestructure/flows/entrenamiento_flow.py
"""
import sys
from pathlib import Path

# Carpeta actividad-2 en el path: los imports funcionan aunque Poetry no instale el paquete.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import time  # noqa: E402
import warnings  # noqa: E402
from datetime import datetime  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import pandas as pd  # noqa: E402
import skops.io as sio  # noqa: E402
from mlflow.exceptions import MlflowException  # noqa: E402
from mlflow.models import infer_signature  # noqa: E402
from mlflow.tracking import MlflowClient  # noqa: E402
from prefect import flow, task  # noqa: E402
from prefect.artifacts import create_markdown_artifact  # noqa: E402
from prefect.cache_policies import NO_CACHE  # noqa: E402

from domain.modelo import (  # noqa: E402
    calcular_metricas,
    construir_modelo,
    decidir_promocion,
    elegir_mejor,
    figura_matriz_confusion,
    figura_roc,
    resolver_features,
    tabla_markdown,
)
from scripts.config import cargar_config, tracking_uri  # noqa: E402
from scripts.load_data import cargar_entrenamiento_y_evaluacion  # noqa: E402

warnings.filterwarnings("ignore", message=".*Integer columns in Python cannot represent missing values.*")


def tipos_confiables(modelo):
    """Tipos del modelo que skops no conoce (p. ej. sklearn.tree._tree.Tree o xgboost.core.Booster)."""
    return sio.get_untrusted_types(data=sio.dumps(modelo))


# ---------------------------------------------------------------------------
# 1. Ingesta
# ---------------------------------------------------------------------------
@task(name="ingesta", cache_policy=NO_CACHE)
def ingesta(cfg):
    """Carga train y los conjuntos de evaluación, valida columnas y resuelve la selección de variables."""
    train, evaluaciones = cargar_entrenamiento_y_evaluacion(cfg)
    objetivo = cfg["datos"]["objetivo"]
    sel = cfg["features"]["seleccion"]
    numericas, categoricas = resolver_features(train, objetivo, sel, cfg["features"]["custom"])

    print(f"Train: {len(train)} filas | % retiros: {train[objetivo].mean():.2%}")
    for nombre, df in evaluaciones.items():
        print(f"{nombre}: {len(df)} filas | % retiros: {df[objetivo].mean():.2%} (no se usa para entrenar)")
    print(f"Selección de features: {sel} -> {len(numericas) + len(categoricas)} variables")
    print(f"  Numéricas ({len(numericas)}): {numericas}")
    print(f"  Categóricas ({len(categoricas)}): {categoricas}")
    return train, evaluaciones, numericas, categoricas


# ---------------------------------------------------------------------------
# 2. Entrenamiento
# ---------------------------------------------------------------------------
@task(name="entrenamiento", task_run_name="entrenamiento-{candidato[nombre]}", cache_policy=NO_CACHE)
def entrenamiento(train, candidato, numericas, categoricas, objetivo):
    """Entrena un candidato (preprocesado + estimador) solo con train. Devuelve (modelo, segundos)."""
    print(f"Entrenando {candidato['nombre']} ({candidato['tipo']}) con {candidato.get('parametros')}")
    modelo = construir_modelo(candidato["tipo"], candidato.get("parametros"), numericas, categoricas)
    inicio = time.perf_counter()
    modelo.fit(train[numericas + categoricas], train[objetivo])
    segundos = time.perf_counter() - inicio
    print(f"{candidato['nombre']} entrenado en {segundos:.1f} s")
    return modelo, segundos


# ---------------------------------------------------------------------------
# 3. Evaluación (un run de MLflow por conjunto de evaluación)
# ---------------------------------------------------------------------------
@task(name="evaluacion", task_run_name="evaluacion-{candidato[nombre]}", cache_policy=NO_CACHE)
def evaluacion(modelo, segundos, evaluaciones, candidato, numericas, categoricas, cfg, padre_id, n_train):
    """Evalúa el modelo en cada conjunto (test1, test2) y registra un run hijo por conjunto.

    El modelo se guarda como artefacto en el run del conjunto de selección (test1): ese run es
    el que se registra en el Model Registry si el modelo pasa a champion.
    """
    objetivo = cfg["datos"]["objetivo"]
    umbral = cfg["metricas"]["umbral"]
    seleccion = cfg["datos"]["conjunto_seleccion"]
    variables = numericas + categoricas
    # El conjunto de selección va primero para que su run (con el modelo) exista antes que los demás.
    orden = [seleccion] + [n for n in evaluaciones if n != seleccion]

    por_conjunto = {}
    for nombre_conjunto in orden:
        df = evaluaciones[nombre_conjunto]
        X, y = df[variables], df[objetivo]
        proba = modelo.predict_proba(X)[:, 1]
        metricas = calcular_metricas(y, proba, umbral)
        print(
            f"{candidato['nombre']} en {nombre_conjunto}: "
            + " | ".join(f"{k}={v:.4f}" for k, v in metricas.items())
        )

        tags = {
            "fase": "entrenamiento",
            "modelo_tipo": candidato["tipo"],
            "conjunto_evaluacion": nombre_conjunto,
            "rol_conjunto": "seleccion" if nombre_conjunto == seleccion else "confirmacion",
            "mlflow.parentRunId": padre_id,
        }
        if nombre_conjunto != seleccion:
            tags["modelo_run_id"] = por_conjunto[seleccion]["run_id"]

        with mlflow.start_run(run_name=f"{candidato['nombre']}-{nombre_conjunto}", tags=tags) as run:
            mlflow.log_params(
                {
                    "modelo": candidato["tipo"],
                    **{f"hp_{k}": v for k, v in (candidato.get("parametros") or {}).items()},
                    "features": cfg["features"]["seleccion"],
                    "n_features": len(variables),
                    "umbral": umbral,
                    "conjunto_evaluacion": nombre_conjunto,
                    "n_train": n_train,
                    "n_evaluacion": len(df),
                }
            )
            mlflow.log_metrics({**metricas, "segundos_entrenamiento": round(segundos, 3)})
            mlflow.log_figure(
                figura_roc(y, {candidato["nombre"]: proba}, f"Curva ROC - {nombre_conjunto}"), "graficas/roc.png"
            )
            mlflow.log_figure(
                figura_matriz_confusion(
                    y, proba, umbral, f"Matriz de confusión - {candidato['nombre']} ({nombre_conjunto})"
                ),
                "graficas/matriz_confusion.png",
            )
            model_uri = None
            if nombre_conjunto == seleccion:
                info = mlflow.sklearn.log_model(
                    modelo,
                    name="model",
                    signature=infer_signature(X, proba),
                    input_example=X.head(3),
                    # MLflow guarda con skops (formato seguro). Los árboles de random forest y el Booster de
                    # xgboost no son tipos "confiables" por defecto: como son modelos entrenados aquí, se confía
                    # solo en los tipos que contiene este modelo.
                    skops_trusted_types=tipos_confiables(modelo),
                )
                model_uri = info.model_uri
        por_conjunto[nombre_conjunto] = {
            "run_id": run.info.run_id,
            "model_uri": model_uri,
            "metricas": metricas,
            "proba": proba,
        }

    plt.close("all")
    sel = por_conjunto[seleccion]
    return {
        "nombre": candidato["nombre"],
        "tipo": candidato["tipo"],
        "run_id": sel["run_id"],
        "model_uri": sel["model_uri"],
        "metricas": sel["metricas"],  # las del conjunto de selección: deciden el mejor y el champion
        "por_conjunto": por_conjunto,
    }


# ---------------------------------------------------------------------------
# 4. Registro (comparación con el champion + Model Registry)
# ---------------------------------------------------------------------------
@task(name="registro", cache_policy=NO_CACHE)
def registro(resultados, evaluaciones, cfg, padre_id):
    """Elige el mejor por AUC en test1, lo confirma en test2 y lo promueve si mejora al champion."""
    client = MlflowClient()
    reglas = cfg["registro"]
    nombre, alias = reglas["nombre_modelo"], reglas.get("alias", "champion")
    objetivo = cfg["datos"]["objetivo"]
    seleccion = cfg["datos"]["conjunto_seleccion"]
    confirmacion = cfg["datos"].get("conjunto_confirmacion")

    # Tabla de los 6 experimentos (modelo x conjunto)
    filas = [
        {"modelo": r["nombre"], "conjunto": c, **{k: v for k, v in pc["metricas"].items()}}
        for r in resultados
        for c, pc in r["por_conjunto"].items()
    ]
    tabla = pd.DataFrame(filas).sort_values(["conjunto", "auc"], ascending=[True, False]).reset_index(drop=True)
    print(f"Experimentos evaluados: {len(tabla)}\n" + tabla.round(4).to_string(index=False))

    # Mejor candidato = mayor AUC en test1 entre los que cumplen el recall mínimo.
    mejor = elegir_mejor(resultados, reglas.get("recall_minimo"))
    met_conf = mejor["por_conjunto"][confirmacion]["metricas"] if confirmacion else None
    print(
        f"Mejor candidato: {mejor['nombre']} (AUC {seleccion} {mejor['metricas']['auc']:.4f}"
        + (f", AUC {confirmacion} {met_conf['auc']:.4f})" if met_conf else ")")
    )

    # Champion vigente
    try:
        actual = client.get_model_version_by_alias(nombre, alias)
        auc_champion = client.get_run(actual.run_id).data.metrics["auc"]
        print(f"Champion vigente: versión {actual.version} (AUC {seleccion} {auc_champion:.4f})")
    except MlflowException:
        actual, auc_champion = None, None
        print("Todavía no hay champion")

    promover, motivo = decidir_promocion(mejor["metricas"], auc_champion, reglas, met_conf)
    version = None
    if promover:
        mv = mlflow.register_model(mejor["model_uri"], nombre)
        version = mv.version
        client.set_registered_model_alias(nombre, alias, version)
        etiquetas = {
            "estado": "aprobado",
            "modelo_tipo": mejor["tipo"],
            "features": cfg["features"]["seleccion"],
            f"auc_{seleccion}": f"{mejor['metricas']['auc']:.4f}",
        }
        if met_conf:
            etiquetas[f"auc_{confirmacion}"] = f"{met_conf['auc']:.4f}"
        for k, v in etiquetas.items():
            client.set_model_version_tag(nombre, version, k, v)
        resultado = f"Versión {version} ({mejor['nombre']}) es el nuevo {alias}: {motivo}"
    else:
        resultado = f"El {alias} no cambia: {motivo}"
    print(resultado)

    # Evidencias en el run padre de MLflow
    with mlflow.start_run(run_id=padre_id):
        mlflow.log_params({"mejor_modelo": mejor["nombre"], "experimentos": len(tabla)})
        mlflow.log_metric(f"auc_mejor_{seleccion}", mejor["metricas"]["auc"])
        if met_conf:
            mlflow.log_metric(f"auc_mejor_{confirmacion}", met_conf["auc"])
        if auc_champion is not None:
            mlflow.log_metric("auc_champion_anterior", auc_champion)
        mlflow.set_tags({"promovido": str(promover), "decision": resultado[:250]})
        if version is not None:
            mlflow.set_tag("version_registrada", version)
        mlflow.log_table(tabla, "comparativa_experimentos.json")
        mlflow.log_text(tabla_markdown(tabla), "comparativa_experimentos.md")
        for c, df in evaluaciones.items():
            fig = figura_roc(
                df[objetivo], {r["nombre"]: r["por_conjunto"][c]["proba"] for r in resultados}, f"Curva ROC - {c}"
            )
            mlflow.log_figure(fig, f"graficas/roc_comparativa_{c}.png")
    plt.close("all")

    # Evidencia en Prefect (pestaña Artifacts del flow run)
    create_markdown_artifact(
        key="comparativa-experimentos",
        markdown=(
            f"## {len(tabla)} experimentos (modelo x conjunto de evaluación)\n\n"
            f"{tabla_markdown(tabla)}\n\n"
            f"**Mejor (AUC en {seleccion}):** {mejor['nombre']}  \n"
            f"**Champion anterior (AUC {seleccion}):** "
            f"{'ninguno' if auc_champion is None else round(auc_champion, 4)}  \n"
            f"**Decisión:** {resultado}\n"
        ),
        description="Resultado del entrenamiento y decisión sobre el champion",
    )
    return {"promovido": promover, "version": version, "mejor": mejor["nombre"], "motivo": motivo}


# ---------------------------------------------------------------------------
# Flow
# ---------------------------------------------------------------------------
@flow(name="entrenamiento-retiro", log_prints=True)
def entrenamiento_flow():
    cfg = cargar_config()
    mlflow.set_tracking_uri(tracking_uri(cfg))
    experimento = mlflow.set_experiment(cfg["mlflow"]["experimento"])
    print(f"MLflow: {mlflow.get_tracking_uri()} | experimento: {cfg['mlflow']['experimento']}")

    candidatos = [c for c in cfg["candidatos"] if c.get("activo", True)]
    if not candidatos:
        raise ValueError("No hay candidatos activos en configs/config.yaml")

    # Run padre: agrupa la ejecución completa. Cada modelo x conjunto es un run hijo.
    client = MlflowClient()
    padre = client.create_run(
        experimento.experiment_id,
        run_name=f"entrenamiento-{datetime.now():%Y%m%d-%H%M}",
        tags={"fase": "entrenamiento", "tipo_run": "ejecucion"},
    )
    padre_id = padre.info.run_id

    try:
        train, evaluaciones, numericas, categoricas = ingesta(cfg)
        with mlflow.start_run(run_id=padre_id):
            mlflow.log_params(
                {
                    "features": cfg["features"]["seleccion"],
                    "n_features": len(numericas) + len(categoricas),
                    "n_train": len(train),
                    "conjuntos_evaluacion": ",".join(evaluaciones),
                    "conjunto_seleccion": cfg["datos"]["conjunto_seleccion"],
                    "modelos": ",".join(c["nombre"] for c in candidatos),
                }
            )

        resultados = []
        for candidato in candidatos:
            modelo, segundos = entrenamiento(train, candidato, numericas, categoricas, cfg["datos"]["objetivo"])
            resultados.append(
                evaluacion(modelo, segundos, evaluaciones, candidato, numericas, categoricas, cfg, padre_id, len(train))
            )

        decision = registro(resultados, evaluaciones, cfg, padre_id)
        client.set_terminated(padre_id, "FINISHED")
        return decision
    except Exception:
        client.set_terminated(padre_id, "FAILED")
        raise


if __name__ == "__main__":
    entrenamiento_flow()
