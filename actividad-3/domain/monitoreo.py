"""Lógica de monitoreo: emulación de drift y detección de drift de features y de evento (target).

No usa Prefect ni MLflow: se prueba sin servicios levantados (ver tests/test_monitoreo.py).
"""
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency


# ---------------------------------------------------------------------------
# Emulación de drift (Tarea 1)
# ---------------------------------------------------------------------------
def columnas_a_desplazar(numericas, nivel, semilla=42):
    """Elige qué variables numéricas se desplazan: round(nivel x n.º de numéricas), al azar con semilla fija."""
    validar_nivel(nivel)
    n = round(nivel * len(numericas))
    if n == 0:
        return []
    rng = np.random.default_rng(semilla)
    return sorted(rng.choice(list(numericas), size=n, replace=False).tolist())


def validar_nivel(nivel):
    if not 0 <= nivel <= 1:
        raise ValueError(f"El nivel de drift debe estar entre 0 y 1, no {nivel}")


def aplicar_drift(df, nivel, numericas, objetivo=None, desviaciones=2.0, subida_tasa_maxima=0.10, semilla=42):
    """Emula drift en un lote. Devuelve (lote_con_drift, columnas_desplazadas).

    1. Features: desplaza `round(nivel x n.º de numéricas)` variables numéricas sumándoles
       `desviaciones` desviaciones típicas (calculadas en el propio lote). Las demás no cambian.
    2. Target: sube la tasa de `objetivo` en `nivel x subida_tasa_maxima` puntos, convirtiendo
       ceros en unos al azar (p. ej. nivel 0.6 y máximo 0.10 -> +6 puntos de tasa de retiro).

    Con nivel 0.0 el lote sale idéntico al de entrada.
    """
    validar_nivel(nivel)
    lote = df.copy()
    desplazadas = columnas_a_desplazar(numericas, nivel, semilla)
    for columna in desplazadas:
        std = lote[columna].std()
        if pd.notna(std) and std > 0:
            lote[columna] = lote[columna] + desviaciones * std

    if objetivo is not None and objetivo in lote.columns and nivel > 0:
        lote[objetivo] = subir_tasa(lote[objetivo], nivel * subida_tasa_maxima, semilla)
    return lote, desplazadas


def subir_tasa(objetivo, subida, semilla=42):
    """Convierte ceros en unos al azar hasta que la tasa de unos sube `subida` puntos (máximo 100 %)."""
    y = objetivo.astype(int)
    meta = round(min(y.mean() + subida, 1.0) * len(y))
    faltan = min(meta - int(y.sum()), int((y == 0).sum()))
    if faltan <= 0:
        return y
    rng = np.random.default_rng(semilla)
    a_cambiar = rng.choice(y.index[y == 0], size=faltan, replace=False)
    nuevo = y.copy()
    nuevo.loc[a_cambiar] = 1
    return nuevo


# ---------------------------------------------------------------------------
# Drift de features (Tarea 2) — lectura del resultado de Evidently
# ---------------------------------------------------------------------------
def fraccion_desde_evidently(resultado):
    """Fracción de variables con drift a partir de `snapshot.dict()` de Evidently (métrica DriftedColumnsCount)."""
    for metrica in resultado["metrics"]:
        if metrica["metric_name"].startswith("DriftedColumnsCount"):
            return float(metrica["value"]["share"])
    raise ValueError("El resultado de Evidently no incluye DriftedColumnsCount")


def columnas_desde_evidently(resultado):
    """Detalle por variable a partir de las métricas ValueDrift de Evidently.

    Devuelve una lista de dicts {columna, metodo, umbral, puntuacion, drift}. En los métodos de
    distancia (Wasserstein, Jensen-Shannon…) hay drift si puntuación >= umbral; en los tests
    estadísticos (p-valor: K-S, chi-cuadrado, Z) hay drift si puntuación < umbral.
    """
    filas = []
    for metrica in resultado["metrics"]:
        nombre = metrica["metric_name"]
        if not nombre.startswith("ValueDrift("):
            continue
        partes = dict(p.split("=", 1) for p in nombre[len("ValueDrift("):-1].split(",") if "=" in p)
        metodo = partes.get("method", "")
        umbral = float(partes.get("threshold", "nan"))
        puntuacion = float(metrica["value"])
        es_p_valor = any(k in metodo.lower() for k in ("p_value", "test", "k-s", "chi-square"))
        drift = puntuacion < umbral if es_p_valor else puntuacion >= umbral
        filas.append(
            {"columna": partes.get("column"), "metodo": metodo, "umbral": umbral,
             "puntuacion": puntuacion, "drift": bool(drift)}
        )
    return filas


def hay_alerta(fraccion, umbral):
    """Alerta si la fracción de variables con drift SUPERA el umbral (estrictamente mayor)."""
    return fraccion > umbral


def mensaje_alerta(tipo, alerta, detalle):
    estado = "ALERTA" if alerta else "Sin alerta"
    return f"{estado}: drift de {tipo} — {detalle}"


# ---------------------------------------------------------------------------
# Drift de evento / target (Tarea 2)
# ---------------------------------------------------------------------------
def validar_lote(lote, objetivo):
    """Falla con un mensaje claro si el lote no trae las etiquetas reales del objetivo."""
    if objetivo not in lote.columns:
        raise ValueError(
            f"El lote no tiene la columna '{objetivo}': sin etiquetas reales no se puede calcular el drift de "
            "evento. En producción las etiquetas llegan con retraso: vuelve a ejecutar cuando estén disponibles."
        )
    if lote[objetivo].isna().any():
        faltan = int(lote[objetivo].isna().sum())
        raise ValueError(f"El lote tiene {faltan} filas con '{objetivo}' vacío: faltan etiquetas reales")
    valores = set(pd.unique(lote[objetivo]))
    if not valores <= {0, 1}:
        raise ValueError(f"'{objetivo}' debe ser 0/1 y tiene {sorted(map(str, valores))}")


def drift_de_target(referencia, lote, alfa=0.05):
    """Compara la tasa del objetivo (proporción de 1) entre referencia y lote con un test chi-cuadrado."""
    ref, cur = referencia.astype(int), lote.astype(int)
    pos_ref, pos_cur = int(ref.sum()), int(cur.sum())
    tabla = [[pos_ref, len(ref) - pos_ref], [pos_cur, len(cur) - pos_cur]]
    _, p_valor, _, _ = chi2_contingency(tabla)
    return {
        "tasa_referencia": pos_ref / len(ref),
        "tasa_lote": pos_cur / len(cur),
        "diferencia": pos_cur / len(cur) - pos_ref / len(ref),
        "p_valor": float(p_valor),
        "drift": bool(p_valor < alfa),
    }
