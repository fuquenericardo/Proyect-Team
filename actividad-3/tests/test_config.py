"""Tests de la configuración: lo que cambia sin tocar código debe tener valores válidos."""


def test_nivel_de_drift_valido(cfg):
    assert 0.0 <= cfg["inferencia"]["nivel_drift"] <= 1.0


def test_umbrales_de_monitoreo_validos(cfg):
    assert 0 < cfg["monitoreo"]["umbral_drift"] < 1
    assert 0 < cfg["monitoreo"]["alfa_evento"] < 1


def test_la_inferencia_usa_un_archivo_con_etiquetas(cfg):
    """El drift de evento necesita la Y: la fuente del lote debe ser un archivo _con_y."""
    assert "con_y" in cfg["inferencia"]["fuente"]


def test_features_validas(cfg):
    assert cfg["features"]["seleccion"] in {"ALL", "NUMERIC", "CUSTOM"}
