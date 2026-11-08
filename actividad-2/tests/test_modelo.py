"""Pruebas de la lógica del modelo (domain/modelo.py). No necesitan MLflow ni Prefect.

Ejecución: poetry run pytest -q
"""
import numpy as np
import pandas as pd
import pytest

from domain.modelo import (
    calcular_metricas,
    construir_modelo,
    decidir_promocion,
    elegir_mejor,
    resolver_features,
)

REGLAS = {"umbral_auc": 0.80, "mejora_minima": 0.001, "recall_minimo": 0.5}


@pytest.fixture
def df():
    rng = np.random.default_rng(0)
    n = 300
    x = rng.normal(size=n)
    return pd.DataFrame(
        {
            "edad": rng.integers(18, 80, n),
            "saldo": x * 1000,
            "pais": rng.choice(["A", "B", "C"], n),
            "y": (x + rng.normal(scale=0.5, size=n) > 0).astype(int),
        }
    )


@pytest.mark.parametrize(
    "seleccion, esperado",
    [("ALL", (["edad", "saldo"], ["pais"])), ("NUMERIC", (["edad", "saldo"], [])), ("CUSTOM", (["saldo"], ["pais"]))],
)
def test_resolver_features(df, seleccion, esperado):
    assert resolver_features(df, "y", seleccion, ["saldo", "pais"]) == esperado


def test_custom_con_columna_inexistente(df):
    with pytest.raises(ValueError):
        resolver_features(df, "y", "CUSTOM", ["no_existe"])


@pytest.mark.parametrize("tipo", ["logistic_regression", "random_forest", "xgboost"])
def test_los_tres_modelos_entrenan_y_predicen(df, tipo):
    modelo = construir_modelo(tipo, {}, ["edad", "saldo"], ["pais"])
    modelo.fit(df[["edad", "saldo", "pais"]], df["y"])
    proba = modelo.predict_proba(df[["edad", "saldo", "pais"]])[:, 1]
    metricas = calcular_metricas(df["y"], proba)
    assert 0.5 < metricas["auc"] <= 1.0
    assert set(metricas) == {"auc", "accuracy", "precision", "recall", "f1"}


def test_promocion_sin_champion():
    assert decidir_promocion({"auc": 0.85, "recall": 0.7}, None, REGLAS)[0]


def test_no_promueve_bajo_el_minimo():
    assert not decidir_promocion({"auc": 0.75, "recall": 0.7}, None, REGLAS)[0]


def test_no_promueve_si_no_mejora_al_champion():
    assert not decidir_promocion({"auc": 0.8705, "recall": 0.7}, 0.87, REGLAS)[0]
    assert decidir_promocion({"auc": 0.88, "recall": 0.7}, 0.87, REGLAS)[0]


def test_elegir_mejor_respeta_recall_minimo():
    resultados = [
        {"nombre": "a", "metricas": {"auc": 0.90, "recall": 0.40}},
        {"nombre": "b", "metricas": {"auc": 0.88, "recall": 0.70}},
    ]
    assert elegir_mejor(resultados, 0.5)["nombre"] == "b"
    assert elegir_mejor(resultados, None)["nombre"] == "a"
