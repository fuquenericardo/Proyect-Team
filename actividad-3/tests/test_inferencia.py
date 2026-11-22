"""Tests de la inferencia con drift emulado (domain/inferencia.py) con un modelo falso: sin MLflow."""
import numpy as np
import pandas as pd
import pytest

from domain.inferencia import metricas_lote, nivel_desde_nombre, nombre_lote, preparar_lote, predecir
from tests.conftest import NUMERICAS

DRIFT = {"desviaciones": 2.0, "subida_tasa_maxima": 0.10}


class ModeloFalso:
    """Imita al champion: usa dos columnas y da más riesgo a quien tiene pocas transacciones."""

    feature_names_in_ = np.array(["EDAD", "CANT_TRANSACCIONES"])

    def predict_proba(self, X):
        p = np.where(X["CANT_TRANSACCIONES"] < 4, 0.8, 0.1)
        return np.column_stack([1 - p, p])


def test_lote_tiene_el_tamano_pedido_y_es_reproducible(asociados):
    a, _ = preparar_lote(asociados, 200, 42, 0.0, NUMERICAS, "Y", DRIFT)
    b, _ = preparar_lote(asociados, 200, 42, 0.0, NUMERICAS, "Y", DRIFT)
    assert len(a) == 200
    pd.testing.assert_frame_equal(a, b)


def test_mismo_lote_base_para_todos_los_niveles(asociados):
    """0.0 y 0.6 parten de las mismas filas: solo cambia el drift."""
    sin, _ = preparar_lote(asociados, 200, 42, 0.0, NUMERICAS, "Y", DRIFT)
    con, desplazadas = preparar_lote(asociados, 200, 42, 0.6, NUMERICAS, "Y", DRIFT)
    assert list(sin.index) == list(con.index)
    intactas = [c for c in NUMERICAS if c not in desplazadas]
    pd.testing.assert_frame_equal(sin[intactas], con[intactas])


def test_predecir_agrega_probabilidad_y_clase(asociados):
    resultado = predecir(ModeloFalso(), asociados, umbral=0.5)
    assert {"probabilidad_retiro", "prediccion"} <= set(resultado.columns)
    assert resultado["probabilidad_retiro"].between(0, 1).all()
    assert set(resultado["prediccion"].unique()) <= {0, 1}
    assert len(resultado) == len(asociados)


def test_predecir_falla_si_faltan_columnas_del_modelo(asociados):
    with pytest.raises(ValueError, match="CANT_TRANSACCIONES"):
        predecir(ModeloFalso(), asociados.drop(columns=["CANT_TRANSACCIONES"]))


def test_metricas_contra_la_y_real(asociados):
    m = metricas_lote(predecir(ModeloFalso(), asociados), "Y")
    assert m["filas_predichas"] == len(asociados)
    assert 0.5 < m["real_auc"] <= 1
    assert m["retiros_detectados"] <= m["retiros_reales"]


def test_metricas_sin_y_no_fallan(asociados):
    m = metricas_lote(predecir(ModeloFalso(), asociados.drop(columns=["Y"])), "Y")
    assert "real_auc" not in m and m["filas_predichas"] == len(asociados)


def test_nombre_del_lote_guarda_el_nivel():
    nombre = nombre_lote(0.6, "20261122-101500")
    assert nombre == "lote_20261122-101500_drift-0.6.csv"
    assert nivel_desde_nombre(f"lotes/{nombre}") == 0.6
    assert nivel_desde_nombre("lotes/lote_actual.csv") is None
