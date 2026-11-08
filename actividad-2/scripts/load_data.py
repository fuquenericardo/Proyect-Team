"""Carga de los CSV de entrenamiento y prueba, con validación.

Ejecución (desde la raíz del repositorio) para revisar los datos antes de entrenar:
    poetry run python scripts/load_data.py

Muestra filas, nulos, proporción de la clase positiva y qué columnas son
numéricas o categóricas (útil para llenar `custom` en configs/features.yaml).
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # raíz del repo, para `python scripts/load_data.py`

from scripts.config import cargar_config  # noqa: E402


def cargar_csv(ruta, objetivo=None, columnas_excluir=None, clase_positiva=1):
    """Lee un CSV, quita las columnas excluidas y deja el objetivo como 0/1.

    Si `objetivo` es None o no está en el CSV (datos nuevos sin etiqueta), solo devuelve las variables.
    """
    df = pd.read_csv(ruta)
    if df.empty:
        raise ValueError(f"El CSV {ruta} está vacío")

    excluir = [c for c in (columnas_excluir or []) if c in df.columns]
    df = df.drop(columns=excluir)

    if objetivo is not None and objetivo in df.columns:
        valores = set(df[objetivo].dropna().unique())
        if len(valores) != 2:
            raise ValueError(f"'{objetivo}' debe ser binaria y tiene los valores {sorted(map(str, valores))}")
        if pd.api.types.is_numeric_dtype(df[objetivo]) and isinstance(clase_positiva, (int, float)):
            df[objetivo] = (df[objetivo] == clase_positiva).astype(int)
        else:
            df[objetivo] = (df[objetivo].astype(str).str.strip() == str(clase_positiva)).astype(int)
    return df


def cargar_train_test(cfg):
    """Devuelve (train_df, test_df) validando que ambos tengan las mismas columnas y el objetivo."""
    d = cfg["datos"]
    train = cargar_csv(d["train"], d["objetivo"], d.get("columnas_excluir"), d.get("clase_positiva", 1))
    test = cargar_csv(d["test"], d["objetivo"], d.get("columnas_excluir"), d.get("clase_positiva", 1))

    for nombre, df in (("train", train), ("test", test)):
        if d["objetivo"] not in df.columns:
            raise ValueError(f"Al CSV de {nombre} le falta la columna objetivo '{d['objetivo']}'")

    faltan = sorted(set(train.columns) - set(test.columns))
    if faltan:
        raise ValueError(f"Columnas de train que no están en test: {faltan}")
    return train, test[train.columns]


def main():
    cfg = cargar_config()
    train, test = cargar_train_test(cfg)
    objetivo = cfg["datos"]["objetivo"]
    variables = train.drop(columns=[objetivo])

    for nombre, df in (("train", train), ("test", test)):
        print(
            f"{nombre}: {len(df)} filas | {df.shape[1] - 1} variables | "
            f"nulos: {int(df.isna().sum().sum())} | % clase positiva: {df[objetivo].mean():.2%}"
        )
    print("Numéricas:", list(variables.select_dtypes("number").columns))
    print("Categóricas:", list(variables.select_dtypes(exclude="number").columns))


if __name__ == "__main__":
    main()
