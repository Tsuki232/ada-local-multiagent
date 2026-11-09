"""Genera activos de prueba reproducibles en data/samples/ para verificar los Proyectos 1 y 2 de ADA."""

from __future__ import annotations

from ada.ingest.demo_assets import (
    create_graphical_ode_image,
    create_sample_sqlite_database,
)

if __name__ == "__main__":
    db = create_sample_sqlite_database()
    img = create_graphical_ode_image()
    print(f"DB creada: {db}")
    print(f"Imagen EDO creada: {img}")
