"""Lógica de la inferencia con drift emulado, sin MLflow ni Prefect (se prueba con un modelo falso).

El flow (infraestructure/flows/inferencia_flow.py) solo carga el champion, lee archivos y registra;
todo lo que transforma datos vive aquí.
"""
import pandas as pd

from domain.modelo import calcular_metricas
from domain.monitoreo import aplicar_drift


def preparar_lote(fuente, tamano, semilla, nivel, numericas, objetivo, drift_cfg):
    """Toma `tamano` filas de la parte de prueba y les aplica el drift emulado. Devuelve (lote, columnas_desplazadas).

    Con la misma semilla el lote base es siempre el mismo: así las ejecuciones con 0.0, 0.3 y 0.6
    solo se diferencian por el drift.
    """
    base = fuente.sample(n=min(tamano, len(fuente)), random_state=semilla) if tamano else fuente.copy()
    return aplicar_drift(
        base,
        nivel,
        numericas,
        objetivo,
        desviaciones=drift_cfg.get("desviaciones", 2.0),
        subida_tasa_maxima=drift_cfg.get("subida_tasa_maxima", 0.10),
        semilla=semilla,
    )


def predecir(modelo, lote, umbral=0.5):
    """Añade probabilidad_retiro y prediccion al lote, usando solo las columnas con las que se entrenó el modelo."""
    columnas = list(modelo.feature_names_in_)
    faltan = [c for c in columnas if c not in lote.columns]
    if faltan:
        raise ValueError(f"Al lote le faltan columnas que usa el modelo: {faltan}")
    proba = modelo.predict_proba(lote[columnas])[:, 1]
    resultado = lote.copy()
    resultado["probabilidad_retiro"] = proba.round(6)
    resultado["prediccion"] = (proba >= umbral).astype(int)
    return resultado


def metricas_lote(resultado, objetivo, umbral=0.5):
    """Métricas de la predicción frente a la Y real del lote (si la trae y tiene las dos clases)."""
    base = {
        "filas_predichas": len(resultado),
        "probabilidad_media": float(resultado["probabilidad_retiro"].mean()),
        "tasa_predicha": float(resultado["prediccion"].mean()),
    }
    if objetivo not in resultado.columns or resultado[objetivo].nunique() < 2:
        return base
    y = resultado[objetivo].astype(int)
    m = calcular_metricas(y, resultado["probabilidad_retiro"], umbral)
    return {
        **base,
        **{f"real_{k}": v for k, v in m.items()},
        "tasa_real": float(y.mean()),
        "retiros_reales": int(y.sum()),
        "retiros_detectados": int(((resultado["prediccion"] == 1) & (y == 1)).sum()),
    }


def nombre_lote(nivel, marca):
    """Nombre de archivo de un lote: lote_<marca>_drift-<nivel>.csv (p. ej. lote_20261122-101500_drift-0.6.csv)."""
    return f"lote_{marca}_drift-{nivel:.1f}.csv"


def leer_lote(ruta, separador=";"):
    return pd.read_csv(ruta, sep=separador)


def nivel_desde_nombre(nombre):
    """Lee el nivel de drift del nombre del lote (…_drift-0.6.csv -> 0.6). None si no lo trae."""
    import re

    m = re.search(r"_drift-(\d+(?:\.\d+)?)\.csv$", str(nombre))
    return float(m.group(1)) if m else None
