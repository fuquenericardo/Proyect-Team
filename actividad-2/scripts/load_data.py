"""Carga de los CSV (train, conjuntos de evaluación y lotes de inferencia), con validación.

Ejecución (desde la carpeta actividad-2) para revisar los datos antes de entrenar:
    poetry run python scripts/load_data.py

Muestra filas, periodos, nulos y % de retiros de cada archivo, y qué columnas son
numéricas o categóricas (útil para llenar `custom` en configs/features.yaml).
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # carpeta actividad-2, para `python scripts/load_data.py`

from scripts.config import cargar_config  # noqa: E402


def cargar_csv(ruta, cfg, quitar_ids=True):
    """Lee un CSV con el separador del YAML y deja el objetivo (si existe) como 0/1.

    quitar_ids=True  -> quita las columnas_id (para entrenar/evaluar).
    quitar_ids=False -> las conserva (para guardar predicciones identificables).
    """
    d = cfg["datos"]
    df = pd.read_csv(ruta, sep=d.get("separador", ","))
    df.columns = df.columns.str.strip()
    if df.empty:
        raise ValueError(f"El CSV {ruta} está vacío")

    if quitar_ids:
        df = df.drop(columns=[c for c in d.get("columnas_id", []) if c in df.columns])

    objetivo, positiva = d["objetivo"], d.get("clase_positiva", 1)
    if objetivo in df.columns:
        valores = set(df[objetivo].dropna().unique())
        if len(valores) != 2:
            raise ValueError(f"'{objetivo}' en {ruta} debe ser binaria y tiene {sorted(map(str, valores))}")
        if pd.api.types.is_numeric_dtype(df[objetivo]) and isinstance(positiva, (int, float)):
            df[objetivo] = (df[objetivo] == positiva).astype(int)
        else:
            df[objetivo] = (df[objetivo].astype(str).str.strip() == str(positiva)).astype(int)
    return df


def cargar_entrenamiento_y_evaluacion(cfg):
    """Devuelve (train, {nombre: df}) con los conjuntos de `datos.evaluacion`, validando columnas y objetivo."""
    d = cfg["datos"]
    objetivo = d["objetivo"]
    train = cargar_csv(d["train"], cfg)
    if objetivo not in train.columns:
        raise ValueError(f"A train le falta la columna objetivo '{objetivo}'")

    evaluaciones = {}
    for nombre, ruta in d["evaluacion"].items():
        df = cargar_csv(ruta, cfg)
        if objetivo not in df.columns:
            raise ValueError(f"El conjunto de evaluación '{nombre}' ({ruta}) no tiene la columna '{objetivo}'")
        faltan = sorted(set(train.columns) - set(df.columns))
        if faltan:
            raise ValueError(f"Columnas de train que no están en {nombre}: {faltan}")
        evaluaciones[nombre] = df[train.columns]

    for clave in ("conjunto_seleccion", "conjunto_confirmacion"):
        if d.get(clave) and d[clave] not in evaluaciones:
            raise ValueError(f"datos.{clave} = '{d[clave]}' no está en datos.evaluacion")
    return train, evaluaciones


def _resumen(nombre, ruta, cfg):
    df = cargar_csv(ruta, cfg, quitar_ids=False)
    d = cfg["datos"]
    fechas = df[d["columnas_id"][0]] if d.get("columnas_id") else None
    periodo = f" | cortes: {fechas.nunique()} ({fechas.iloc[0]} ... {fechas.iloc[-1]})" if fechas is not None else ""
    y = f" | % retiros: {df[d['objetivo']].mean():.2%}" if d["objetivo"] in df.columns else " | sin Y"
    print(f"{nombre:<10} {len(df):>7} filas | nulos: {int(df.isna().sum().sum())}{periodo}{y}")


def main():
    cfg = cargar_config()
    d = cfg["datos"]
    _resumen("train", d["train"], cfg)
    for nombre, ruta in d["evaluacion"].items():
        _resumen(nombre, ruta, cfg)
    _resumen("inferencia", cfg["inferencia"]["fuente"], cfg)

    train, _ = cargar_entrenamiento_y_evaluacion(cfg)
    variables = train.drop(columns=[d["objetivo"]])
    print("Numéricas:", list(variables.select_dtypes("number").columns))
    print("Categóricas:", list(variables.select_dtypes(exclude="number").columns))
    print("Validación OK: train y los conjuntos de evaluación tienen las mismas columnas y la Y")


if __name__ == "__main__":
    main()
