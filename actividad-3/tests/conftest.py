"""Fixtures comunes. Los tests se ejecutan desde la carpeta actividad-3 (`poetry run pytest`).

Ningún test necesita MLflow ni Prefect levantados. Los datos reales (data/*.csv) no están en Git:
los tests que los usan se saltan si no existen (por ejemplo en el CI de GitHub); el resto usa
datos sintéticos con las mismas columnas.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from scripts.config import cargar_config  # noqa: E402

NUMERICAS = [
    "EDAD", "ANTIGUEDAD", "SALARIO", "SALDO_APORTE_SOCIAL", "SALDO_AHORRO_PERMANENTE", "SALDO_AHO_VISTA",
    "SALDO_AHORRO_PROGRAMADO", "SALDO_CARTERA", "CANTIDAD_CREDITOS", "CANT_SERV_COMPLEMENT", "CANT_TRANSACCIONES",
]
CATEGORICAS = ["GENERO", "ESTADO_CIVIL"]


@pytest.fixture(scope="session")
def cfg():
    import os

    os.chdir(RAIZ)  # las rutas del YAML son relativas a actividad-3
    return cargar_config()


def generar_asociados(n=600, semilla=0):
    """Asociados sintéticos con el mismo esquema que los CSV reales (Y ~ 10 %, ligada a pocas transacciones)."""
    rng = np.random.default_rng(semilla)
    df = pd.DataFrame(
        {
            "FECHA_CORTE": "31/07/2026",
            "CODIGO_ASOCIADO": np.arange(1, n + 1),
            "EDAD": rng.integers(18, 80, n),
            "GENERO": rng.choice(["FEMENINO", "MASCULINO"], n),
            "ANTIGUEDAD": rng.uniform(0, 40, n).round(2),
            "ESTADO_CIVIL": rng.choice(["Soltero", "Casado", "Union libre", "Viudo"], n),
            "SALARIO": rng.lognormal(14.5, 0.6, n).round(0),
            "SALDO_APORTE_SOCIAL": rng.lognormal(14, 1, n).round(2),
            "SALDO_AHORRO_PERMANENTE": rng.lognormal(15, 1, n).round(2),
            "SALDO_AHO_VISTA": rng.lognormal(13, 1.5, n).round(0),
            "SALDO_AHORRO_PROGRAMADO": rng.choice([0, 500000, 2000000], n),
            "SALDO_CARTERA": rng.choice([0, 1000000, 5000000], n),
            "CANTIDAD_CREDITOS": rng.integers(0, 4, n),
            "CANT_SERV_COMPLEMENT": rng.integers(0, 5, n),
            "CANT_TRANSACCIONES": rng.integers(0, 30, n),
        }
    )
    df["Y"] = ((df["CANT_TRANSACCIONES"] < 4) & (rng.random(n) < 0.6)).astype(int)
    return df


@pytest.fixture
def asociados():
    return generar_asociados()


def ruta_datos(cfg, clave):
    return RAIZ / cfg["datos"][clave] if clave == "train" else RAIZ / cfg["datos"]["evaluacion"][clave]
