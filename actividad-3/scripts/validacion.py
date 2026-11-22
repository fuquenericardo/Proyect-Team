"""Validaciones de datos: esquema, rangos, categorías y nulos (Tarea 3).

Funciones puras: reciben un DataFrame y las reglas de `validacion` en configs/config.yaml y
devuelven una lista de problemas (vacía = todo bien). Se usan en los tests y se pueden llamar
desde cualquier flow antes de entrenar o puntuar.
"""


def validar_esquema(df, columnas, objetivo=None):
    """Comprueba que estén todas las columnas esperadas (y el objetivo, si se pide)."""
    esperadas = list(columnas) + ([objetivo] if objetivo else [])
    faltan = [c for c in esperadas if c not in df.columns]
    return [f"faltan columnas: {faltan}"] if faltan else []


def validar_rangos(df, rangos):
    """Comprueba que cada columna numérica esté dentro de [mínimo, máximo] (None = sin límite). Ignora vacíos."""
    problemas = []
    for columna, (minimo, maximo) in rangos.items():
        if columna not in df.columns:
            continue
        valores = df[columna].dropna()
        if minimo is not None and (valores < minimo).any():
            problemas.append(f"{columna}: {int((valores < minimo).sum())} valores menores que {minimo}")
        if maximo is not None and (valores > maximo).any():
            problemas.append(f"{columna}: {int((valores > maximo).sum())} valores mayores que {maximo}")
    return problemas


def validar_categorias(df, categorias):
    """Comprueba que las columnas categóricas solo tengan los valores permitidos. Ignora vacíos."""
    problemas = []
    for columna, permitidas in categorias.items():
        if columna in df.columns:
            raras = sorted(set(df[columna].dropna().unique()) - set(permitidas))
            if raras:
                problemas.append(f"{columna}: valores no permitidos {raras}")
    return problemas


def validar_nulos(df, nulos_maximos, columnas=None):
    """Comprueba la fracción de vacíos: cada columna como máximo su límite; las no listadas, 0."""
    problemas = []
    for columna in columnas or df.columns:
        if columna not in df.columns:
            continue
        fraccion = float(df[columna].isna().mean())
        limite = (nulos_maximos or {}).get(columna, 0.0)
        if fraccion > limite:
            problemas.append(f"{columna}: {fraccion:.2%} vacíos (máximo {limite:.2%})")
    return problemas


def validar_objetivo(df, objetivo):
    """El objetivo debe existir, no tener vacíos y ser 0/1."""
    if objetivo not in df.columns:
        return [f"falta la columna objetivo '{objetivo}'"]
    if df[objetivo].isna().any():
        return [f"'{objetivo}' tiene vacíos"]
    raros = sorted(set(df[objetivo].unique()) - {0, 1})
    return [f"'{objetivo}' tiene valores distintos de 0/1: {raros}"] if raros else []


def validar_datos(df, reglas, objetivo=None):
    """Todas las validaciones juntas. `reglas` = sección `validacion` del YAML."""
    problemas = validar_esquema(df, reglas["columnas"], objetivo)
    problemas += validar_rangos(df, reglas.get("rangos", {}))
    problemas += validar_categorias(df, reglas.get("categorias", {}))
    problemas += validar_nulos(df, reglas.get("nulos_maximos", {}), reglas["columnas"])
    if objetivo:
        problemas += validar_objetivo(df, objetivo)
    return problemas
