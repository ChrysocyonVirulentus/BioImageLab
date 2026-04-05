# === gestorLab/Flujo_Trabajo.py ===

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, List, Any, Optional, Set

from ..controlador.Resultado_Either import Resultado, Ok, Err
from ..controlador.Controlador_BioImagen import ErrorBioImagen
from .Operacion import Operacion
from .Categoria_Operacion import TipoDato


# =========================================================
# NODO
# =========================================================

@dataclass
class NodoPipeline:
    id: str
    tipo_dato: TipoDato
    data: List[Any] = field(default_factory=list)

    def __repr__(self):
        return f"<Nodo {self.id} ({self.tipo_dato.name})>"


# =========================================================
# ARISTA (OPERACIÓN)
# =========================================================

@dataclass
class AristaOperacion:
    origen:    str
    destino:   str
    operacion: Operacion

    def __repr__(self):
        return f"{self.origen} --[{self.operacion.nombre}]--> {self.destino}"


# =========================================================
# GRAFO PIPELINE
# =========================================================

@dataclass
class GrafoPipeline:
    nodos:   Dict[str, NodoPipeline] = field(default_factory=dict)
    aristas: List[AristaOperacion]   = field(default_factory=list)

    def agregar_nodo(self, nodo: NodoPipeline):
        self.nodos[nodo.id] = nodo

    def agregar_arista(self, arista: AristaOperacion):
        if arista.origen not in self.nodos or arista.destino not in self.nodos:
            raise ValueError(
                f"Nodo origen '{arista.origen}' o destino '{arista.destino}' no existe"
            )
        self.aristas.append(arista)

    def salientes(self, nodo_id: str)  -> List[AristaOperacion]:
        return [a for a in self.aristas if a.origen  == nodo_id]

    def entrantes(self, nodo_id: str)  -> List[AristaOperacion]:
        return [a for a in self.aristas if a.destino == nodo_id]

    def nodos_iniciales(self) -> List[NodoPipeline]:
        return [n for n in self.nodos.values() if not self.entrantes(n.id)]

    def nodos_finales(self) -> List[NodoPipeline]:
        return [n for n in self.nodos.values() if not self.salientes(n.id)]

    def orden_topologico(self) -> List[str]:
        """Kahn's Algorithm."""
        in_degree = {n: 0 for n in self.nodos}
        for a in self.aristas:
            in_degree[a.destino] += 1

        cola  = [n for n, deg in in_degree.items() if deg == 0]
        orden = []

        while cola:
            actual = cola.pop(0)
            orden.append(actual)
            for arista in self.salientes(actual):
                in_degree[arista.destino] -= 1
                if in_degree[arista.destino] == 0:
                    cola.append(arista.destino)

        if len(orden) != len(self.nodos):
            raise ValueError("El grafo tiene ciclos — no es un DAG válido")

        return orden

    # -----------------------------
    # Orden topológico (DAG)
    # -----------------------------

    def orden_topologico(self) -> List[str]:
        """
        Kahn's Algorithm
        """
        in_degree = {n: 0 for n in self.nodos}

        for a in self.aristas:
            in_degree[a.destino] += 1

        cola = [n for n, deg in in_degree.items() if deg == 0]
        orden = []

        while cola:
            actual = cola.pop(0)
            orden.append(actual)

            for arista in self.salientes(actual):
                in_degree[arista.destino] -= 1
                if in_degree[arista.destino] == 0:
                    cola.append(arista.destino)

        if len(orden) != len(self.nodos):
            raise ValueError("El grafo tiene ciclos")

        return orden


# =========================================================
# FLUJO DE TRABAJO (PIPELINE)
# =========================================================

class FlujoTrabajo:
    """
    Pipeline basado en DAG.
    Nodo = dato en tránsito. Arista = operación que lo transforma.

    El error que viaja dentro de Resultado puede ser cualquier dataclass
    con campos 'etapa' y 'mensaje' — no solo ErrorBioImagen.
    """

    def __init__(self, grafo: GrafoPipeline):
        self.grafo  = grafo
        self.nombre = ""

    # =========================================================
    # EJECUCIÓN
    # =========================================================

    def ejecutar(self, input_inicial: Any) -> Resultado[Dict[str, Any], Any]:

        try:
            orden = self.grafo.orden_topologico()
        except Exception as e:
            return Err(ErrorBioImagen(
                etapa="pipeline", mensaje=f"Grafo inválido: {e}", causa=e
            ))

        iniciales = self.grafo.nodos_iniciales()
        if not iniciales:
            return Err(ErrorBioImagen(etapa="pipeline", mensaje="Sin nodo inicial"))

        for nodo in iniciales:
            nodo.data = [input_inicial]

        for nodo_id in orden:
            nodo = self.grafo.nodos[nodo_id]
            if not nodo.data:
                continue

            # Merge simple: tomar el último dato acumulado
            input_data = nodo.data[-1]

            for arista in self.grafo.salientes(nodo_id):
                resultado = arista.operacion.ejecutar(input_data)

                if resultado.es_err():
                    error = resultado.error
                    # replace funciona para cualquier @dataclass(frozen=True)
                    # siempre que tenga los campos 'etapa' y 'mensaje'
                    try:
                        error_enriquecido = replace(
                            error,
                            etapa   = f"pipeline -> {error.etapa}",
                            mensaje = (
                                f"[{nodo_id} → {arista.destino}] "
                                f"'{arista.operacion.nombre}' falló: {error.mensaje}"
                            )
                        )
                    except Exception:
                        # Fallback si el error no es dataclass o le faltan campos
                        error_enriquecido = error
                    return Err(error_enriquecido)

                self.grafo.nodos[arista.destino].data.append(resultado.unwrap())

        salida = {
            n.id: n.data[-1] if n.data else None
            for n in self.grafo.nodos_finales()
        }
        return Ok(salida)

    # =========================================================
    # SUBGRAFOS
    # =========================================================

    def subgrafo_desde(self, nodo_id: str) -> GrafoPipeline:
        visitados: Set[str]           = set()
        aristas_sub: List[AristaOperacion] = []

        def dfs(actual: str):
            if actual in visitados: return
            visitados.add(actual)
            for a in self.grafo.salientes(actual):
                aristas_sub.append(a)
                dfs(a.destino)

        dfs(nodo_id)
        return GrafoPipeline(
            nodos   = {nid: self.grafo.nodos[nid] for nid in visitados},
            aristas = aristas_sub
        )

    def subgrafo_hasta(self, nodo_id: str) -> GrafoPipeline:
        visitados: Set[str]           = set()
        aristas_sub: List[AristaOperacion] = []

        def dfs(actual: str):
            if actual in visitados: return
            visitados.add(actual)
            for a in self.grafo.entrantes(actual):
                aristas_sub.append(a)
                dfs(a.origen)

        dfs(nodo_id)
        return GrafoPipeline(
            nodos   = {nid: self.grafo.nodos[nid] for nid in visitados},
            aristas = aristas_sub
        )

    # =========================================================
    # VALIDACIÓN INTRÍNSECA
    # =========================================================

    def validar_pipeline(self):
        errores = []
        for arista in self.grafo.aristas:
            nodo_destino = self.grafo.nodos[arista.destino]
            op           = arista.operacion

            if op.tipo_dato_salida is None:
                continue

            if nodo_destino.tipo_dato != op.tipo_dato_salida:
                errores.append(
                    f"TipoDato inconsistente en {arista}: "
                    f"nodo={nodo_destino.tipo_dato.name} "
                    f"op={op.tipo_dato_salida.name}"
                )
        return len(errores) == 0, errores

    # =========================================================
    # UTILIDADES
    # =========================================================

    def reset_datos(self):
        """Limpia todos los nodos para re-ejecución."""
        for nodo in self.grafo.nodos.values():
            nodo.data = []   # CORREGIDO: era None — data es List[Any]

    def __repr__(self):
        return (
            f"<FlujoTrabajo '{self.nombre}' "
            f"nodos={len(self.grafo.nodos)} "
            f"aristas={len(self.grafo.aristas)}>"
        )