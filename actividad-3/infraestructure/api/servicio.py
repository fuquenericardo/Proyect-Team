"""Construcción de la API de FastAPI SIN MLflow: recibe el modelo ya cargado.

`main.py` carga el champion desde MLflow y llama a `crear_app`. Los tests llaman a `crear_app`
con un modelo falso (tests/test_api.py), así la API se prueba sin servicios levantados.
"""
import re
from dataclasses import dataclass, field

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, ConfigDict, Field, create_model


@dataclass
class InfoModelo:
    """Lo que la API necesita saber del modelo servido (sale de MLflow en main.py)."""

    nombre: str
    alias: str
    version: str
    run_id: str = ""
    tags: dict = field(default_factory=dict)
    metricas: dict = field(default_factory=dict)
    tipos: dict = field(default_factory=dict)  # {columna: int | float | str | bool}
    ejemplo: dict | None = None


class Prediccion(BaseModel):
    probabilidad_retiro: float
    prediccion: int
    modelo: str
    version_modelo: str


def esquema_entrada(columnas, tipos, ejemplo=None):
    """Modelo de Pydantic con un campo por columna del modelo (admite null: el pipeline imputa)."""
    campos = {}
    for col in columnas:
        nombre_py = re.sub(r"\W+", "_", col).strip("_") or "col"
        campos[nombre_py] = (tipos.get(col, float) | None, Field(alias=col))
    return create_model(
        "Asociado",
        __config__=ConfigDict(populate_by_name=True, extra="ignore", json_schema_extra={"example": ejemplo}),
        **campos,
    )


def crear_app(modelo, info, umbral=0.5, titulo="API de retiro de asociados"):
    """FastAPI con /health, /modelo, /predict y /predict/lote para `modelo` (cualquier objeto con
    `feature_names_in_` y `predict_proba`)."""
    columnas = list(modelo.feature_names_in_)
    Asociado = esquema_entrada(columnas, info.tipos, info.ejemplo)

    app = FastAPI(
        title=titulo,
        description=f"Predicción con `models:/{info.nombre}@{info.alias}` (versión {info.version}).",
        version="1.0.0",
    )

    def _predecir(registros):
        datos = pd.DataFrame(registros)[columnas]
        proba = modelo.predict_proba(datos)[:, 1]
        return [
            Prediccion(
                probabilidad_retiro=round(float(p), 4),
                prediccion=int(p >= umbral),
                modelo=info.nombre,
                version_modelo=str(info.version),
            )
            for p in proba
        ]

    @app.get("/", include_in_schema=False)
    def raiz():
        return RedirectResponse(url="/docs")

    @app.get("/health")
    def health():
        return {"estado": "ok", "modelo": info.nombre, "alias": info.alias, "version": str(info.version)}

    @app.get("/modelo")
    def info_modelo():
        """Versión, run de entrenamiento, tags, variables y métricas del champion servido."""
        return {
            "uri": f"models:/{info.nombre}@{info.alias}",
            "version": str(info.version),
            "run_id": info.run_id,
            "tags": info.tags,
            "variables": columnas,
            "metricas_test": {k: round(v, 4) for k, v in info.metricas.items()},
            "umbral": umbral,
        }

    @app.post("/predict", response_model=Prediccion)
    def predict(asociado: Asociado):
        """Predice el riesgo de retiro de un asociado. Recibe las columnas originales del CSV (ver ejemplo)."""
        try:
            return _predecir([asociado.model_dump(by_alias=True)])[0]
        except Exception as e:  # datos con formato inesperado
            raise HTTPException(status_code=422, detail=str(e)) from e

    @app.post("/predict/lote", response_model=list[Prediccion])
    def predict_lote(asociados: list[Asociado]):
        """Predice para varios asociados en una sola llamada."""
        if not asociados:
            raise HTTPException(status_code=422, detail="La lista de asociados está vacía")
        return _predecir([a.model_dump(by_alias=True) for a in asociados])

    return app
