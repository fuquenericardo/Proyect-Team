"""Tests de la lógica de drift (domain/monitoreo.py). Sin MLflow ni Prefect; Evidently corre en memoria."""
import numpy as np
import pandas as pd
import pytest

from domain.monitoreo import (
    aplicar_drift,
    columnas_a_desplazar,
    columnas_desde_evidently,
    drift_de_target,
    fraccion_desde_evidently,
    hay_alerta,
    mensaje_alerta,
    subir_tasa,
    validar_lote,
)
from tests.conftest import CATEGORICAS, NUMERICAS, generar_asociados


# --- Emulación (Tarea 1) -----------------------------------------------------------------------
def test_nivel_cero_no_cambia_el_lote(asociados):
    lote, desplazadas = aplicar_drift(asociados, 0.0, NUMERICAS, "Y")
    assert desplazadas == []
    pd.testing.assert_frame_equal(lote, asociados)


@pytest.mark.parametrize("nivel, esperadas", [(0.3, 3), (0.6, 7), (1.0, 11)])
def test_desplaza_la_fraccion_de_variables_que_indica_el_nivel(asociados, nivel, esperadas):
    lote, desplazadas = aplicar_drift(asociados, nivel, NUMERICAS, "Y")
    cambiadas = [c for c in NUMERICAS if not np.allclose(lote[c], asociados[c])]
    assert len(desplazadas) == esperadas
    assert sorted(cambiadas) == desplazadas


def test_las_variables_elegidas_crecen_con_el_nivel():
    """Con la misma semilla, las de 0.3 están incluidas en las de 0.6: el drift es acumulativo y comparable."""
    assert len(columnas_a_desplazar(NUMERICAS, 0.3)) < len(columnas_a_desplazar(NUMERICAS, 0.6))


def test_el_drift_sube_la_media_dos_desviaciones(asociados):
    lote, desplazadas = aplicar_drift(asociados, 0.3, NUMERICAS, "Y", desviaciones=2.0)
    c = desplazadas[0]
    assert lote[c].mean() == pytest.approx(asociados[c].mean() + 2 * asociados[c].std())


def test_el_drift_sube_la_tasa_del_objetivo_en_proporcion_al_nivel(asociados):
    tasa = asociados["Y"].mean()
    lote, _ = aplicar_drift(asociados, 0.6, NUMERICAS, "Y", subida_tasa_maxima=0.10)
    assert lote["Y"].mean() == pytest.approx(tasa + 0.06, abs=0.005)


def test_subir_tasa_no_pasa_de_100():
    assert subir_tasa(pd.Series([0, 1, 1, 1]), 0.9).mean() == 1.0


@pytest.mark.parametrize("nivel", [-0.1, 1.5])
def test_nivel_fuera_de_rango_falla(asociados, nivel):
    with pytest.raises(ValueError, match="entre 0 y 1"):
        aplicar_drift(asociados, nivel, NUMERICAS, "Y")


# --- Drift de features: lectura de Evidently y alerta ------------------------------------------
RESULTADO = {"metrics": [
    {"metric_name": "DriftedColumnsCount(drift_share=0.5)", "value": {"count": 2.0, "share": 0.5}},
    {"metric_name": "ValueDrift(column=EDAD,method=Wasserstein distance (normed),threshold=0.1)", "value": 0.8},
    {"metric_name": "ValueDrift(column=SALARIO,method=Wasserstein distance (normed),threshold=0.1)", "value": 0.02},
    {"metric_name": "ValueDrift(column=GENERO,method=chi-square p_value,threshold=0.05)", "value": 0.001},
    {"metric_name": "ValueDrift(column=ESTADO_CIVIL,method=K-S p_value,threshold=0.05)", "value": 0.4},
]}


def test_fraccion_desde_evidently():
    assert fraccion_desde_evidently(RESULTADO) == 0.5


def test_fraccion_sin_metrica_falla():
    with pytest.raises(ValueError):
        fraccion_desde_evidently({"metrics": []})


def test_detalle_por_columna_distingue_distancias_y_p_valores():
    drift = {f["columna"]: f["drift"] for f in columnas_desde_evidently(RESULTADO)}
    assert drift == {"EDAD": True, "SALARIO": False, "GENERO": True, "ESTADO_CIVIL": False}


@pytest.mark.parametrize("fraccion, alerta", [(0.077, False), (0.40, False), (0.54, True)])
def test_alerta_solo_si_supera_el_umbral(fraccion, alerta):
    assert hay_alerta(fraccion, 0.40) is alerta
    assert mensaje_alerta("features", alerta, "x").startswith("ALERTA" if alerta else "Sin alerta")


def test_evidently_real_0_frente_a_0_6():
    """Criterio de éxito con Evidently de verdad: el lote 0.0 no alerta y el 0.6 sí, con reportes distintos."""
    from evidently import Report
    from evidently.presets import DataDriftPreset

    referencia = generar_asociados(3000, semilla=1)
    base = generar_asociados(1500, semilla=2)
    variables = NUMERICAS + CATEGORICAS
    fracciones = {}
    for nivel in (0.0, 0.6):
        lote, _ = aplicar_drift(base, nivel, NUMERICAS, "Y")
        r = Report([DataDriftPreset()]).run(reference_data=referencia[variables], current_data=lote[variables]).dict()
        fracciones[nivel] = fraccion_desde_evidently(r)
    assert fracciones[0.0] != fracciones[0.6]
    assert not hay_alerta(fracciones[0.0], 0.40)
    assert hay_alerta(fracciones[0.6], 0.40)


# --- Drift de evento (target) ------------------------------------------------------------------
def test_validar_lote_rechaza_lote_sin_etiquetas(asociados):
    with pytest.raises(ValueError, match="sin etiquetas reales"):
        validar_lote(asociados.drop(columns=["Y"]), "Y")


def test_validar_lote_rechaza_etiquetas_vacias(asociados):
    asociados["Y"] = asociados["Y"].astype(float)
    asociados.loc[0, "Y"] = np.nan
    with pytest.raises(ValueError, match="vacío"):
        validar_lote(asociados, "Y")


def test_validar_lote_acepta_lote_completo(asociados):
    validar_lote(asociados, "Y")


def test_drift_de_target_detecta_subida_de_la_tasa():
    referencia = pd.Series([1] * 220 + [0] * 9780)  # 2,2 % (como train)
    lote = pd.Series([1] * 166 + [0] * 1834)        # 8,3 % (nivel 0.6)
    r = drift_de_target(referencia, lote, alfa=0.05)
    assert r["drift"] is True and r["diferencia"] > 0.05


def test_drift_de_target_no_alerta_con_la_misma_tasa():
    referencia = pd.Series([1] * 220 + [0] * 9780)
    lote = pd.Series([1] * 45 + [0] * 1955)         # 2,25 %
    assert drift_de_target(referencia, lote, alfa=0.05)["drift"] is False
