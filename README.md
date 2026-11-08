# Proyect-Team – MLOps

Proyecto del curso de MLOps (Maestría en Ciencia de Datos). Cada actividad es un proyecto de Poetry independiente dentro de este repositorio.

## Estructura

```
Proyect-Team/
├── actividad-1/   # Fundamentos: estructura, entorno con Poetry y primeros flows
├── actividad-2/   # Pipelines de entrenamiento e inferencia (Prefect + MLflow) y API (FastAPI)
├── services/      # MLflow y Prefect (Docker Compose o sin Docker), compartidos por todas las actividades
└── .gitignore
```

## Uso

1. Levanta los servicios desde `services/` (ver `services/README.md`).
2. Entra en la carpeta de la actividad y crea su entorno:

```powershell
cd actividad-2
poetry install
```

Los comandos de cada actividad se ejecutan desde su propia carpeta; ver el README de cada una.

## Integrantes

- Ricardo Fuquene
- Katherine Barreto
- Rafael Davila
