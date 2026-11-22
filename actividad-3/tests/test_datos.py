"""Tests de datos: esquema, rangos, categorías y nulos (scripts/validacion.py)."""
import numpy as np
import pytest

from scripts.load_data import cargar_csv
from scripts.validacion import (
    validar_categorias,
    validar_datos,
    validar_esquema,
    validar_nulos,
    validar_objetivo,
    validar_rangos,
)
from tests.conftest import RAIZ


# --- Reglas, con datos sintéticos (corren siempre, también en el CI) ---------------------------
def test_datos_validos_no_tienen_problemas(asociados, cfg):
    assert validar_datos(asociados, cfg["validacion"], "Y") == []


def test_esquema_detecta_columna_faltante(asociados, cfg):
    problemas = validar_esquema(asociados.drop(columns=["SALARIO"]), cfg["validacion"]["columnas"])
    assert problemas and "SALARIO" in problemas[0]


def test_rangos_detectan_edad_imposible_y_saldo_negativo(asociados, cfg):
    asociados.loc[0, "EDAD"] = 150
    asociados.loc[1, "SALDO_APORTE_SOCIAL"] = -10
    problemas = validar_rangos(asociados, cfg["validacion"]["rangos"])
    assert any("EDAD" in p for p in problemas)
    assert any("SALDO_APORTE_SOCIAL" in p for p in problemas)


def test_categorias_detectan_valor_no_permitido(asociados, cfg):
    asociados.loc[0, "GENERO"] = "OTRO"
    assert validar_categorias(asociados, cfg["validacion"]["categorias"])


def test_nulos_respetan_el_maximo_por_columna(asociados):
    asociados.loc[:4, "ESTADO_CIVIL"] = np.nan  # 5 de 600 = 0,8 %
    assert validar_nulos(asociados, {"ESTADO_CIVIL": 0.02}) == []
    assert validar_nulos(asociados, {"ESTADO_CIVIL": 0.001})
    asociados.loc[0, "CANT_TRANSACCIONES"] = np.nan  # columna sin vacíos permitidos
    assert any("CANT_TRANSACCIONES" in p for p in validar_nulos(asociados, {}))


def test_objetivo_debe_ser_binario_y_completo(asociados):
    assert validar_objetivo(asociados, "Y") == []
    asociados.loc[0, "Y"] = 2
    assert validar_objetivo(asociados, "Y")
    assert validar_objetivo(asociados.drop(columns=["Y"]), "Y")


# --- Datos reales: se saltan si los CSV no están (no se versionan en Git) ----------------------
ARCHIVOS = [("train", True), ("test1", True), ("test2", True)]


@pytest.mark.parametrize("clave, con_y", ARCHIVOS)
def test_csv_reales_cumplen_las_reglas(cfg, clave, con_y):
    ruta = RAIZ / (cfg["datos"]["train"] if clave == "train" else cfg["datos"]["evaluacion"][clave])
    if not ruta.exists():
        pytest.skip(f"{ruta.name} no está en data/ (los datos reales no se versionan)")
    df = cargar_csv(ruta, cfg, quitar_ids=False)
    assert validar_datos(df, cfg["validacion"], cfg["datos"]["objetivo"] if con_y else None) == []


def test_lote_de_inferencia_real_cumple_el_esquema(cfg):
    ruta = RAIZ / cfg["inferencia"]["fuente"]
    if not ruta.exists():
        pytest.skip(f"{ruta.name} no está en data/")
    df = cargar_csv(ruta, cfg, quitar_ids=False)
    assert validar_esquema(df, cfg["validacion"]["columnas"], cfg["datos"]["objetivo"]) == []
