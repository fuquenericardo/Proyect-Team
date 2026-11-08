"""Lógica de negocio del modelo: variables, pipelines, métricas y regla de promoción.

No usa Prefect ni MLflow: se puede probar sin servicios levantados (ver tests/).
"""
import matplotlib

matplotlib.use("Agg")  # sin ventana: las figuras se guardan como archivo
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # noqa: E402
from xgboost import XGBClassifier  # noqa: E402

SELECCIONES = ("ALL", "NUMERIC", "CUSTOM")
TIPOS_MODELO = ("logistic_regression", "random_forest", "xgboost")


# ---------------------------------------------------------------------------
# Variables
# ---------------------------------------------------------------------------
def resolver_features(df, objetivo, seleccion, custom=None):
    """Devuelve (numericas, categoricas) según el alias de configs/features.yaml.

    Los tipos se detectan del DataFrame: número -> numérica; texto/categoría/bool -> categórica.
    """
    seleccion = str(seleccion).upper()
    if seleccion not in SELECCIONES:
        raise ValueError(f"seleccion '{seleccion}' no válida. Usa una de {SELECCIONES}")

    variables = df.drop(columns=[objetivo], errors="ignore")
    numericas = list(variables.select_dtypes("number").columns)
    categoricas = list(variables.select_dtypes(exclude="number").columns)

    if seleccion == "NUMERIC":
        categoricas = []
    elif seleccion == "CUSTOM":
        custom = list(custom or [])
        desconocidas = [c for c in custom if c not in variables.columns]
        if desconocidas:
            raise ValueError(f"Variables de 'custom' que no están en el CSV: {desconocidas}")
        numericas = [c for c in numericas if c in custom]
        categoricas = [c for c in categoricas if c in custom]

    if not numericas and not categoricas:
        raise ValueError(f"La selección '{seleccion}' no deja ninguna variable")
    return numericas, categoricas


# ---------------------------------------------------------------------------
# Modelos
# ---------------------------------------------------------------------------
def crear_estimador(tipo, parametros):
    """Crea el estimador de sklearn/xgboost a partir del tipo del YAML."""
    parametros = dict(parametros or {})
    if tipo == "logistic_regression":
        return LogisticRegression(**parametros)
    if tipo == "random_forest":
        return RandomForestClassifier(**parametros)
    if tipo == "xgboost":
        return XGBClassifier(**parametros)
    raise ValueError(f"Modelo '{tipo}' no soportado. Usa uno de {TIPOS_MODELO}")


def construir_modelo(tipo, parametros, numericas, categoricas):
    """Pipeline completo: imputación + escalado/one-hot + estimador.

    Guardar el preprocesado dentro del pipeline hace que la inferencia y la API
    reciban los datos crudos del cliente y apliquen exactamente la misma transformación.
    """
    transformadores = []
    if numericas:
        num = Pipeline([("imputar", SimpleImputer(strategy="median")), ("escalar", StandardScaler())])
        transformadores.append(("num", num, numericas))
    if categoricas:
        cat = Pipeline(
            [
                ("imputar", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )
        transformadores.append(("cat", cat, categoricas))

    preprocesado = ColumnTransformer(transformadores)
    return Pipeline([("preprocesado", preprocesado), ("modelo", crear_estimador(tipo, parametros))])


# ---------------------------------------------------------------------------
# Evaluación
# ---------------------------------------------------------------------------
def calcular_metricas(y_true, proba, umbral=0.5):
    """Métricas de clasificación binaria a partir de la probabilidad de la clase positiva."""
    pred = (proba >= umbral).astype(int)
    return {
        "auc": float(roc_auc_score(y_true, proba)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
    }


def figura_roc(y_true, probas_por_modelo):
    """Curva ROC de uno o varios modelos en la misma figura. probas_por_modelo = {nombre: proba}."""
    fig, ax = plt.subplots(figsize=(6, 5))
    for nombre, proba in probas_por_modelo.items():
        RocCurveDisplay.from_predictions(y_true, proba, name=nombre, ax=ax)
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=1)
    ax.set_title("Curva ROC (conjunto de test)")
    fig.tight_layout()
    return fig


def figura_matriz_confusion(y_true, proba, umbral=0.5, titulo="Matriz de confusión"):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_predictions(y_true, (proba >= umbral).astype(int), ax=ax, colorbar=False)
    ax.set_title(titulo)
    fig.tight_layout()
    return fig


def tabla_markdown(df, decimales=4):
    """Convierte un DataFrame en tabla Markdown (sin dependencias extra)."""
    cab = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "|".join(["---"] * len(df.columns)) + "|"
    filas = [
        "| " + " | ".join(f"{v:.{decimales}f}" if isinstance(v, float) else str(v) for v in fila) + " |"
        for fila in df.itertuples(index=False)
    ]
    return "\n".join([cab, sep, *filas])


# ---------------------------------------------------------------------------
# Regla de promoción a champion
# ---------------------------------------------------------------------------
def elegir_mejor(resultados, recall_minimo=None):
    """Mayor AUC entre los candidatos que cumplen el recall mínimo; si ninguno lo cumple, mayor AUC global.

    resultados = lista de dicts con la clave "metricas".
    """
    aptos = [r for r in resultados if recall_minimo is None or r["metricas"]["recall"] >= recall_minimo]
    return max(aptos or resultados, key=lambda r: r["metricas"]["auc"])


def decidir_promocion(metricas, auc_champion, reglas):
    """Decide si el modelo nuevo reemplaza al champion. Devuelve (promover: bool, motivo: str).

    reglas = sección `registro` de config.yaml (umbral_auc, mejora_minima, recall_minimo).
    auc_champion = AUC del champion vigente, o None si todavía no hay champion.
    """
    auc = metricas["auc"]
    umbral = reglas["umbral_auc"]
    if auc < umbral:
        return False, f"AUC {auc:.4f} por debajo del mínimo {umbral}"

    recall_min = reglas.get("recall_minimo")
    if recall_min is not None and metricas["recall"] < recall_min:
        return False, f"recall {metricas['recall']:.4f} por debajo del mínimo {recall_min}"

    if auc_champion is None:
        return True, f"no había champion y AUC {auc:.4f} supera el mínimo {umbral}"

    mejora = reglas.get("mejora_minima", 0.0) or 0.0
    if auc < auc_champion + mejora or auc <= auc_champion:
        return False, f"AUC {auc:.4f} no supera al champion ({auc_champion:.4f}) en al menos {mejora}"
    return True, f"AUC {auc:.4f} supera al champion ({auc_champion:.4f})"
