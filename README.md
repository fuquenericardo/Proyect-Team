# \# Proyect-Team – MLOps

# 

# Proyecto del curso de MLOps (Maestría en Ciencia de Datos).

# 

# \## Requisitos

# 

# \- Git

# \- Python 3.13 o 3.14

# \- Poetry

# 

# \## Estructura del proyecto

# 

# ```

# Proyect-Team/

# ├── configs/        # Archivos de configuración (YAML)

# ├── docs/           # Documentación

# ├── flows/          # Flujos de Prefect

# ├── notebooks/      # Notebooks de exploración

# ├── scripts/        # Scripts de Python

# ├── tests/          # Pruebas con pytest

# ├── pyproject.toml  # Dependencias del proyecto

# ├── poetry.lock     # Versiones exactas de las dependencias

# └── README.md

# ```

# 

# \## Pasos de ejecución

# 

# \### 1. Clonar el repositorio

# 

# ```bash

# git clone https://github.com/fuquenericardo/Proyect-Team.git

# cd Proyect-Team

# ```

# 

# \### 2. Instalar Poetry (si no lo tienes)

# 

# ```bash

# pip install poetry

# ```

# 

# \### 3. Instalar las dependencias

# 

# ```bash

# poetry install

# ```

# 

# Esto crea el entorno virtual e instala las versiones exactas que están en `poetry.lock`.

# 

# \### 4. Verificar la instalación

# 

# ```bash

# poetry --version

# poetry env info

# poetry show --top-level

# ```

# 

# \### 5. Ejecutar las pruebas

# 

# ```bash

# poetry run pytest

# ```

# 

# \## Dependencias

# 

# | Paquete      | Versión |

# |--------------|---------|

# | prefect      | 3.8.8   |

# | mlflow       | 3.17.0  |

# | scikit-learn | 1.9.1   |

# | pandas       | 3.0.6   |

# | numpy        | 2.5.4   |

# | pyyaml       | 6.0.3   |

# | pytest (dev) | 9.1.1   |

# 

# \## Integrantes

# 

# \- Ricardo Fuquene

# \- Katherine Barreto

# \- Rafael Davila

