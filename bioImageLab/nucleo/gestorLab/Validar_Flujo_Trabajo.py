# === gestorLab/Validar_Flujo_Trabajo.py ===

from __future__ import annotations

from typing import Set, List

from ..controlador.Resultado_Either import Resultado, Ok, Err, LogEvento, NivelLog
from ..controlador.Controlador_BioImagen import ErrorBioImagen

from .Flujo_Trabajo import GrafoPipeline
from .Categoria_Operacion import CategoriaOperacion

from .Validacion_Operacion import (
    es_compatible,
    requiere_adaptador,
    obtener_adaptador,
    tipo_salida,
    tipo_entrada,
)


def validar_pipeline(
    grafo: GrafoPipeline,
) -> Resultado[List[LogEvento], ErrorBioImagen]:
    """
    Valida el pipeline con dos niveles de severidad.

    Retorna:
      Ok(lista_de_warnings)  → pipeline válido, warnings acumulados en el valor
      Err(ErrorBioImagen)    → pipeline inválido, errores duros
    """
    errores_duros: List[str] = []
    warnings:      List[str] = []

    # ── 1. DAG sin ciclos ────────────────────────────────────
    try:
        grafo.orden_topologico()
    except Exception as e:
        return Err(ErrorBioImagen(
            etapa="validacion_pipeline",
            mensaje=f"Grafo inválido (ciclo detectado): {e}",
            causa=e,
        ))

    # ── 2. Validar cada arista ───────────────────────────────
    for arista in grafo.aristas:
        op         = arista.operacion
        cat_actual = op.categoria

        entrantes = grafo.entrantes(arista.origen)

        categorias_previas: Set[CategoriaOperacion] = {
            a.operacion.categoria for a in entrantes
        }

        # 2.1 Dependencias con dos niveles
        valido, errs, warns = cat_actual.validar_dependencias(categorias_previas)

        errores_duros.extend(f"[{op.nombre}] {e}" for e in errs)
        warnings.extend(f"[{op.nombre}] {w}" for w in warns)

        # 2.2 Compatibilidad semántica con operaciones previas
        for a_prev in entrantes:
            cat_prev = a_prev.operacion.categoria

            if not cat_prev.puede_preceder_a(cat_actual):
                errores_duros.append(
                    f"Conexión inválida: {cat_prev.name} → {cat_actual.name}"
                )

            elif not es_compatible(cat_prev, cat_actual):
                if requiere_adaptador(cat_prev, cat_actual):
                    adapter = obtener_adaptador(cat_prev, cat_actual)
                    if adapter:
                        warnings.append(
                            f"Adaptador necesario '{adapter}': "
                            f"{cat_prev.name} → {cat_actual.name}"
                        )
                    else:
                        errores_duros.append(
                            f"Incompatibilidad sin adaptador: "
                            f"{cat_prev.name} → {cat_actual.name}"
                        )

    # ── 3. Resultado ─────────────────────────────────────────
    if errores_duros:
        detalle = "\n".join(errores_duros)
        return Err(ErrorBioImagen(
            etapa="validacion_pipeline",
            mensaje=f"Pipeline inválido ({len(errores_duros)} errores):\n{detalle}",
            causa=ValueError(detalle),
        ))

    # FIX: usar NivelLog.WARN para warnings, no NivelLog.ERROR
    log_warnings = [
        LogEvento(
            etapa   = "validacion_pipeline",
            mensaje = w,
            nivel   = NivelLog.WARN,
        )
        for w in warnings
    ]

    return Ok(log_warnings)