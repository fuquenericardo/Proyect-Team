"""Actividad 3: pipeline de drift de EVENTO, es decir, del target Y = retiro (Tarea 2).

Pasos:  referencia (Y de train) → lote (copia que dejó la inferencia) → validar etiquetas
        → test chi-cuadrado de la tasa de retiro → alerta → registro

- El lote debe traer la Y real. En producción llega con retraso; si falta, el flow falla con un
  mensaje claro en vez de calcular algo sin sentido.
- ALERTA si el p-valor es menor que `monitoreo.alfa_evento` (la tasa de retiro cambió).

Ejecución (desde la carpeta actividad-3, después de la inferencia):
    poetry run python infraestructure/flows/drift_evento_flow.py                       # último lote
    poetry run python infraestructure/flows/drift_evento_flow.py lotes/<archivo>.csv   # un lote concreto
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import json  # noqa: E402

from prefect import flow, get_run_logger, task  # noqa: E402
from prefect.cache_policies import NO_CACHE  # noqa: E402

from domain.inferencia import nivel_desde_nombre  # noqa: E402
from domain.monitoreo import drift_de_target, mensaje_alerta, validar_lote  # noqa: E402
from scripts.config import cargar_config  # noqa: E402
from scripts.monitoreo_io import cargar_lote, cargar_referencia, registrar_en_mlflow, resolver_lote  # noqa: E402


@task(name="referencia-objetivo", cache_policy=NO_CACHE)
def referencia(cfg):
    ref = cargar_referencia(cfg)[cfg["datos"]["objetivo"]]
    print(f"Referencia: tasa de retiro en entrenamiento {ref.mean():.2%} ({len(ref)} filas)")
    return ref


@task(name="evaluar-evento", cache_policy=NO_CACHE)
def evaluar(ref, cfg, ruta):
    objetivo = cfg["datos"]["objetivo"]
    lote = cargar_lote(ruta, cfg)
    validar_lote(lote, objetivo)  # falla con mensaje claro si no hay etiquetas
    r = drift_de_target(ref, lote[objetivo], cfg["monitoreo"]["alfa_evento"])
    print(
        f"Tasa de retiro: entrenamiento {r['tasa_referencia']:.2%} | lote {r['tasa_lote']:.2%} "
        f"({r['diferencia']:+.2%}) | p-valor {r['p_valor']:.3g}"
    )
    carpeta = Path(cfg["monitoreo"]["carpeta_reportes"])
    carpeta.mkdir(parents=True, exist_ok=True)
    ruta_json = carpeta / f"drift_evento_{Path(ruta).stem}.json"
    ruta_json.write_text(json.dumps({"lote": str(ruta), **r}, indent=2), encoding="utf-8")
    print(f"Resumen: {ruta_json}")
    return r, str(ruta_json)


@task(name="alerta", cache_policy=NO_CACHE)
def alerta(r, alfa):
    texto = mensaje_alerta(
        "evento (tasa de retiro)", r["drift"],
        f"{r['tasa_lote']:.2%} frente a {r['tasa_referencia']:.2%}, p-valor {r['p_valor']:.3g} (alfa {alfa})",
    )
    if r["drift"]:
        get_run_logger().warning(texto)  # ALERTA: se ve en amarillo en los logs de Prefect
    else:
        print(texto)
    return r["drift"]


@flow(name="drift-evento", log_prints=True)
def drift_evento_flow(ruta_lote: str | None = None):
    cfg = cargar_config()
    ruta = resolver_lote(cfg, ruta_lote)
    r, ruta_json = evaluar(referencia(cfg), cfg, ruta)
    activa = alerta(r, cfg["monitoreo"]["alfa_evento"])
    registrar_en_mlflow(
        cfg,
        f"drift-evento-{ruta.stem}",
        {"tipo_drift": "evento", "lote": str(ruta), "nivel_drift": str(nivel_desde_nombre(ruta)), "alerta": str(activa)},
        {"tasa_referencia": r["tasa_referencia"], "tasa_lote": r["tasa_lote"], "p_valor": r["p_valor"],
         "alerta": int(activa)},
        [ruta_json],
    )
    return {"lote": str(ruta), **r, "alerta": activa}


if __name__ == "__main__":
    drift_evento_flow(sys.argv[1] if len(sys.argv) > 1 else None)
