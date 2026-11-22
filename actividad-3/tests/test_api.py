"""Tests de la API con un modelo falso: sin MLflow (Tarea 3).

1. `crear_app` (servicio.py) con un modelo falso: endpoints, validación y errores.
2. `main.py` completo con MLflow simulado (mocks): comprueba que carga el champion por su alias.
"""
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from infraestructure.api.servicio import InfoModelo, crear_app

ASOCIADO = {
    "EDAD": 47, "GENERO": "FEMENINO", "ANTIGUEDAD": 17.87, "ESTADO_CIVIL": "Casado", "SALARIO": 1750905,
    "SALDO_APORTE_SOCIAL": 3732613.26, "SALDO_AHORRO_PERMANENTE": 20791482.8, "SALDO_AHO_VISTA": 0,
    "SALDO_AHORRO_PROGRAMADO": 10, "SALDO_CARTERA": 98200, "CANTIDAD_CREDITOS": 1,
    "CANT_SERV_COMPLEMENT": 2, "CANT_TRANSACCIONES": 5,
}
COLUMNAS = list(ASOCIADO)
TIPOS = {c: (str if c in ("GENERO", "ESTADO_CIVIL") else float) for c in COLUMNAS}


class ModeloFalso:
    feature_names_in_ = np.array(COLUMNAS)

    def predict_proba(self, X):
        return np.tile([0.3, 0.7], (len(X), 1))


@pytest.fixture
def client():
    info = InfoModelo(nombre="retiro-asociados-modelo", alias="champion", version="7", tipos=TIPOS,
                      metricas={"auc": 0.81234})
    return TestClient(crear_app(ModeloFalso(), info, umbral=0.5))


def test_health(client):
    assert client.get("/health").json() == {
        "estado": "ok", "modelo": "retiro-asociados-modelo", "alias": "champion", "version": "7"}


def test_predict_un_asociado(client):
    assert client.post("/predict", json=ASOCIADO).json() == {
        "probabilidad_retiro": 0.7, "prediccion": 1, "modelo": "retiro-asociados-modelo", "version_modelo": "7"}


def test_predict_lote(client):
    r = client.post("/predict/lote", json=[ASOCIADO, ASOCIADO])
    assert r.status_code == 200 and len(r.json()) == 2


def test_predict_acepta_nulos_e_ignora_columnas_extra(client):
    cuerpo = {**ASOCIADO, "ESTADO_CIVIL": None, "CODIGO_ASOCIADO": 16775}
    assert client.post("/predict", json=cuerpo).status_code == 200


def test_predict_rechaza_asociado_incompleto(client):
    assert client.post("/predict", json={"EDAD": 30}).status_code == 422


def test_lote_vacio_da_422(client):
    assert client.post("/predict/lote", json=[]).status_code == 422


def test_modelo_y_raiz(client):
    assert client.get("/modelo").json()["metricas_test"] == {"auc": 0.8123}
    assert client.get("/", follow_redirects=False).headers["location"] == "/docs"


def test_main_carga_el_champion_por_alias(monkeypatch, cfg):
    """main.py con MLflow simulado: debe pedir el alias champion y servir esa versión."""
    import mlflow
    import mlflow.models
    import mlflow.sklearn
    import mlflow.tracking

    pedidos = []
    firma = SimpleNamespace(inputs=SimpleNamespace(inputs=[
        SimpleNamespace(name=c, type="string" if TIPOS[c] is str else "double") for c in COLUMNAS]))
    cliente = SimpleNamespace(
        get_model_version_by_alias=lambda nombre, alias: pedidos.append((nombre, alias))
        or SimpleNamespace(version="3", run_id="abc", tags={"estado": "aprobado"}),
        get_run=lambda run_id: SimpleNamespace(data=SimpleNamespace(metrics={"auc": 0.8})),
    )
    monkeypatch.setattr(mlflow.sklearn, "load_model", lambda uri: ModeloFalso())
    monkeypatch.setattr(mlflow.tracking, "MlflowClient", lambda *a, **k: cliente)
    monkeypatch.setattr(mlflow.models, "get_model_info", lambda uri: SimpleNamespace(signature=firma))
    sys.modules.pop("infraestructure.api.main", None)

    import infraestructure.api.main as main

    c = TestClient(main.app)
    assert pedidos == [(cfg["registro"]["nombre_modelo"], "champion")]
    assert c.get("/health").json()["version"] == "3"
    assert c.post("/predict", json=ASOCIADO).json()["version_modelo"] == "3"
