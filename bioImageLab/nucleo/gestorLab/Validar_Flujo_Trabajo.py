# === gestorLab/Validar_Flujo_Trabajo.py ===
"""
Validación del pipeline con patrón Result puro.

PRINCIPIO: la validación NUNCA bloquea la ejecución.
Devuelve siempre Ok(DiagnosticoPipeline) con la lista completa de eventos
(errores duros, warnings y confirmaciones). El llamador decide si aborta.

GestorLab lee el diagnóstico y:
  - En modo normal  → ejecuta siempre, loguea todo
  - En modo estricto → puede abortar si hay errores duros (opt-in)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Set, List

from ..controlador.Resultado_Either import Resultado, Ok, LogEvento, NivelLog
from .Flujo_Trabajo import GrafoPipeline
from .Categoria_Operacion import CategoriaOperacion
from .Validacion_Operacion import (
    es_compatible,
    requiere_adaptador,
    obtener_adaptador,
)


# =========================================================
# DIAGNÓSTICO — reemplaza el Err duro
# =========================================================

@dataclass
class DiagnosticoPipeline:
    """
    Resultado de validar un pipeline.
    Siempre se produce — nunca rompe la ejecución.
    """
    eventos:      List[LogEvento] = field(default_factory=list)

    @property
    def errores(self) -> List[LogEvento]:
        return [e for e in self.eventos if e.nivel == NivelLog.ERROR]

    @property
    def warnings(self) -> List[LogEvento]:
        return [e for e in self.eventos if e.nivel == NivelLog.WARN]

    @property
    def infos(self) -> List[LogEvento]:
        return [e for e in self.eventos if e.nivel == NivelLog.INFO]

    @property
    def tiene_errores_duros(self) -> bool:
        return len(self.errores) > 0

    @property
    def es_valido(self) -> bool:
        return not self.tiene_errores_duros

    def resumen(self) -> str:
        estado = "✓ VÁLIDO" if self.es_valido else "✗ CON ERRORES"
        return (
            f"[Validación {estado}] "
            f"{len(self.errores)} errores  "
            f"{len(self.warnings)} warnings  "
            f"{len(self.infos)} infos"
        )

    def __repr__(self) -> str:
        return f"<DiagnosticoPipeline {self.resumen()}>"


# =========================================================
# VALIDADOR
# =========================================================

def validar_pipeline(
    grafo: GrafoPipeline,
) -> Resultado[DiagnosticoPipeline, None]:
    """
    Valida el pipeline y devuelve SIEMPRE Ok(DiagnosticoPipeline).

    Nunca retorna Err — el patrón Result se usa como log estructurado,
    no como mecanismo de aborto. El llamador inspecciona
    diagnostico.tiene_errores_duros si quiere decidir si continuar.

    Niveles de evento:
      INFO  → verificación pasada correctamente
      WARN  → práctica subóptima, no bloquea
      ERROR → violación de contrato del dominio (registrada, no lanzada)
    """
    diag = DiagnosticoPipeline()

    # ── 1. DAG sin ciclos ────────────────────────────────────
    try:
        orden = grafo.orden_topologico()
        diag.eventos.append(LogEvento(
            etapa   = "validacion_dag",
            mensaje = f"DAG válido — {len(orden)} nodos en orden topológico",
            nivel   = NivelLog.INFO,
        ))
    except Exception as e:
        diag.eventos.append(LogEvento(
            etapa   = "validacion_dag",
            mensaje = f"Ciclo detectado en el grafo: {e}",
            nivel   = NivelLog.ERROR,
            metadata= {"excepcion": str(e)},
        ))
        # Ciclo impide cualquier análisis posterior — retornar ya
        return Ok(diag)

    # ── 2. Nodo inicial y final ──────────────────────────────
    iniciales = grafo.nodos_iniciales()
    finales   = grafo.nodos_finales()

    if not iniciales:
        diag.eventos.append(LogEvento(
            etapa="validacion_estructura",
            mensaje="El grafo no tiene nodo inicial (nodo sin entrantes)",
            nivel=NivelLog.ERROR,
        ))
    else:
        diag.eventos.append(LogEvento(
            etapa="validacion_estructura",
            mensaje=f"Nodos iniciales: {[n.id for n in iniciales]}",
            nivel=NivelLog.INFO,
        ))

    if not finales:
        diag.eventos.append(LogEvento(
            etapa="validacion_estructura",
            mensaje="El grafo no tiene nodo final (nodo sin salientes)",
            nivel=NivelLog.ERROR,
        ))
    else:
        diag.eventos.append(LogEvento(
            etapa="validacion_estructura",
            mensaje=f"Nodos finales: {[n.id for n in finales]}",
            nivel=NivelLog.INFO,
        ))

    # ── 3. Validar cada arista ───────────────────────────────
    for arista in grafo.aristas:
        op         = arista.operacion
        cat_actual = op.categoria
        entrantes  = grafo.entrantes(arista.origen)

        categorias_previas: Set[CategoriaOperacion] = {
            a.operacion.categoria for a in entrantes
        }

        # 3.1 Dependencias con dos niveles
        valido, errs_dep, warns_dep = cat_actual.validar_dependencias(categorias_previas)

        for e in errs_dep:
            diag.eventos.append(LogEvento(
                etapa   = f"validacion_dependencias/{op.nombre}",
                mensaje = e,
                nivel   = NivelLog.ERROR,
                metadata= {"operacion": op.nombre, "categoria": cat_actual.name},
            ))

        for w in warns_dep:
            diag.eventos.append(LogEvento(
                etapa   = f"validacion_dependencias/{op.nombre}",
                mensaje = w,
                nivel   = NivelLog.WARN,
                metadata= {"operacion": op.nombre, "categoria": cat_actual.name},
            ))

        if valido and not warns_dep:
            diag.eventos.append(LogEvento(
                etapa   = f"validacion_dependencias/{op.nombre}",
                mensaje = f"Dependencias OK para '{op.nombre}' ({cat_actual.name})",
                nivel   = NivelLog.INFO,
            ))

        # 3.2 Compatibilidad semántica con operaciones previas
        for a_prev in entrantes:
            cat_prev = a_prev.operacion.categoria

            if not cat_prev.puede_preceder_a(cat_actual):
                diag.eventos.append(LogEvento(
                    etapa   = f"validacion_semantica/{op.nombre}",
                    mensaje = f"Conexión semánticamente inválida: {cat_prev.name} → {cat_actual.name}",
                    nivel   = NivelLog.ERROR,
                    metadata= {"desde": cat_prev.name, "hacia": cat_actual.name},
                ))

            elif not es_compatible(cat_prev, cat_actual):
                if requiere_adaptador(cat_prev, cat_actual):
                    adapter = obtener_adaptador(cat_prev, cat_actual)
                    if adapter:
                        diag.eventos.append(LogEvento(
                            etapa   = f"validacion_semantica/{op.nombre}",
                            mensaje = f"Adaptador necesario '{adapter}': {cat_prev.name} → {cat_actual.name}",
                            nivel   = NivelLog.WARN,
                            metadata= {"adaptador": adapter},
                        ))
                    else:
                        diag.eventos.append(LogEvento(
                            etapa   = f"validacion_semantica/{op.nombre}",
                            mensaje = f"Incompatibilidad sin adaptador conocido: {cat_prev.name} → {cat_actual.name}",
                            nivel   = NivelLog.ERROR,
                            metadata= {"desde": cat_prev.name, "hacia": cat_actual.name},
                        ))
            else:
                diag.eventos.append(LogEvento(
                    etapa   = f"validacion_semantica/{op.nombre}",
                    mensaje = f"Compatibilidad OK: {cat_prev.name} → {cat_actual.name}",
                    nivel   = NivelLog.INFO,
                ))

        # 3.3 Nodo merge — verificar que tiene merge_fn
        nodo_destino = grafo.nodos.get(arista.destino)
        if nodo_destino and nodo_destino.es_merge:
            if nodo_destino.merge_fn is None:
                diag.eventos.append(LogEvento(
                    etapa   = f"validacion_merge/{arista.destino}",
                    mensaje = f"Nodo merge '{arista.destino}' no tiene merge_fn asignada",
                    nivel   = NivelLog.ERROR,
                ))
            else:
                n_entrantes = len(grafo.entrantes(arista.destino))
                diag.eventos.append(LogEvento(
                    etapa   = f"validacion_merge/{arista.destino}",
                    mensaje = f"Nodo merge '{arista.destino}' OK — espera {n_entrantes} inputs",
                    nivel   = NivelLog.INFO,
                ))

    # ── 4. Resumen final ─────────────────────────────────────
    diag.eventos.append(LogEvento(
        etapa   = "validacion_resumen",
        mensaje = diag.resumen(),
        nivel   = NivelLog.INFO if diag.es_valido else NivelLog.WARN,
    ))

    return Ok(diag)
