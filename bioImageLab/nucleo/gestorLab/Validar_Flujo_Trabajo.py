from __future__ import annotations

from typing import Set, List

from ..controlador.Resultado_Either import Resultado, Ok, Err
from ..controlador.Controlador_BioImagen import ErrorBioImagen

from .Flujo_Trabajo import GrafoPipeline
from .Categoria_Operacion import CategoriaOperacion
from .Validaciones_Operaciones import (
    es_compatible,
    requiere_adaptador,
    obtener_adaptador,
    tipo_salida,
    tipo_entrada,
)


def validar_pipeline(grafo: GrafoPipeline) -> Resultado[bool, ErrorBioImagen]:
    """
    Valida completamente el pipeline antes de ejecutar.

    ✔ orden de categorías
    ✔ compatibilidad semántica
    ✔ dependencias
    ✔ DAG (sin ciclos)
    ✔ adaptadores posibles

    NO lanza excepción → devuelve Err acumulable
    """

    errores: List[str] = []
    warnings: List[str] = []

    # -------------------------------------------------
    # 1. Validar DAG (sin ciclos)
    # -------------------------------------------------
    try:
        orden = grafo.orden_topologico()
    except Exception as e:
        return Err(ErrorBioImagen(
            etapa="validacion_pipeline",
            mensaje=f"Grafo inválido (ciclo detectado): {e}"
        ))

    # -------------------------------------------------
    # 2. Validar cada arista (core)
    # -------------------------------------------------
    for arista in grafo.aristas:

        op = arista.operacion
        cat_actual = op.categoria

        # Nodo origen → obtener categoría previa si existe
        entrantes = grafo.entrantes(arista.origen)

        categorias_previas: Set[CategoriaOperacion] = {
            a.operacion.categoria for a in entrantes
        }

        # -------------------------------------------------
        # 2.1 Dependencias
        # -------------------------------------------------
        ok, errs = cat_actual.validar_dependencias(categorias_previas)
        if not ok:
            errores.extend([
                f"[{op.nombre}] " + e for e in errs
            ])

        # -------------------------------------------------
        # 2.2 Compatibilidad con operaciones previas
        # -------------------------------------------------
        for a_prev in entrantes:

            cat_prev = a_prev.operacion.categoria

            # Orden lógico
            if not cat_prev.puede_preceder_a(cat_actual):
                errores.append(
                    f"Orden inválido: {cat_prev.name} → {cat_actual.name}"
                )

            # Compatibilidad semántica
            if not es_compatible(cat_prev, cat_actual):

                if requiere_adaptador(cat_prev, cat_actual):
                    adapter = obtener_adaptador(cat_prev, cat_actual)

                    if adapter:
                        warnings.append(
                            f"Adaptador sugerido ({adapter}) "
                            f"entre {cat_prev.name} → {cat_actual.name}"
                        )
                    else:
                        errores.append(
                            f"Incompatibilidad sin adaptador: "
                            f"{cat_prev.name} → {cat_actual.name}"
                        )
                else:
                    errores.append(
                        f"Incompatibilidad: {cat_prev.name} → {cat_actual.name}"
                    )

    # -------------------------------------------------
    # 3. Resultado final
    # -------------------------------------------------
    if errores:
        return Err(ErrorBioImagen(
            etapa="validacion_pipeline",
            mensaje="Pipeline inválido",
            causa={"errores": errores, "warnings": warnings}
        ))

    # warnings no invalidan pipeline
    return Ok(True)