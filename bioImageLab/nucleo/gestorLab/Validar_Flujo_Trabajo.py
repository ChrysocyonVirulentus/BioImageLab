from __future__ import annotations

from typing import Set, List

from ..controlador.Resultado_Either import Resultado, Ok, Err
from ..controlador.Controlador_BioImagen import ErrorBioImagen

from .Flujo_Trabajo import GrafoPipeline
from .Categoria_Operacion import CategoriaOperacion

# CORREGIDO: nombre correcto del módulo
from .Validacion_Operacion import (
    es_compatible,
    requiere_adaptador,
    obtener_adaptador,
    tipo_salida,
    tipo_entrada,
)


def validar_pipeline(grafo: GrafoPipeline) -> Resultado[bool, ErrorBioImagen]:
    """
    Valida el pipeline completo antes de ejecutar.
      ✔ DAG sin ciclos
      ✔ Orden de categorías
      ✔ Compatibilidad semántica de tipos de dato
      ✔ Dependencias entre etapas
      ✔ Adaptadores disponibles cuando son necesarios
    """
    errores:  List[str] = []
    warnings: List[str] = []

    # ── 1. Validar DAG ──────────────────────────────────────────
    try:
        grafo.orden_topologico()
    except Exception as e:
        return Err(ErrorBioImagen(
            etapa="validacion_pipeline",
            mensaje=f"Grafo inválido (ciclo detectado): {e}",
            causa=e
        ))

    # ── 2. Validar cada arista ───────────────────────────────────
    for arista in grafo.aristas:
        op         = arista.operacion
        cat_actual = op.categoria

        entrantes          = grafo.entrantes(arista.origen)
        categorias_previas: Set[CategoriaOperacion] = {
            a.operacion.categoria for a in entrantes
        }

        # 2.1 Dependencias declaradas en CategoriaOperacion
        ok, errs = cat_actual.validar_dependencias(categorias_previas)
        if not ok:
            errores.extend(f"[{op.nombre}] {e}" for e in errs)

        # 2.2 Compatibilidad con cada operación previa
        for a_prev in entrantes:
            cat_prev = a_prev.operacion.categoria

            if not cat_prev.puede_preceder_a(cat_actual):
                errores.append(
                    f"Orden inválido: {cat_prev.name} → {cat_actual.name}"
                )

            if not es_compatible(cat_prev, cat_actual):
                if requiere_adaptador(cat_prev, cat_actual):
                    adapter = obtener_adaptador(cat_prev, cat_actual)
                    if adapter:
                        warnings.append(
                            f"Adaptador sugerido '{adapter}': "
                            f"{cat_prev.name} → {cat_actual.name}"
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

    # ── 3. Resultado ─────────────────────────────────────────────
    if warnings:
        for w in warnings:
            print(f"[WARN] {w}")

    if errores:
        # CORREGIDO: causa debe ser Exception|None, no dict
        detalle = "\n".join(errores)
        return Err(ErrorBioImagen(
            etapa="validacion_pipeline",
            mensaje=f"Pipeline inválido ({len(errores)} errores):\n{detalle}",
            causa=ValueError(detalle)
        ))

    return Ok(True)