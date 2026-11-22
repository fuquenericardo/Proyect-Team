# Datos de la actividad 2

Archivos esperados (separados por `;`):

| Archivo | Uso |
|---|---|
| `train.csv` | entrenamiento (sep-2023 a dic-2025, con Y) |
| `test1_con_y.csv` | evaluación: elige el mejor modelo (ene-jun 2026) |
| `test2_con_y.csv` | evaluación: confirma con retiros reales (jul-sep 2026) |
| `test1_sin_y.csv`, `test2_sin_y.csv` | lotes para inferencia (sin la Y) |

Son datos reales de asociados, por eso `data/*.csv` está en `.gitignore`.
Compártelos con el grupo por un canal privado (OneDrive, Drive) y cópialos aquí.
Si cambian los nombres, ajústalos en `configs/config.yaml` (secciones `datos` e `inferencia`).
