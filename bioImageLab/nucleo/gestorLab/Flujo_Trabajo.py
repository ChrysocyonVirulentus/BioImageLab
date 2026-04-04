# === gestorLab/Flujo_Trabajo.py ===

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Set, Tuple

from ..controlador.Resultado_Either import Resultado, Ok, Err
from ..controlador.Controlador_BioImagen import ErrorBioImagen
from .Operacion import Operacion
from .Categoria_Operacion import TipoDato


# =========================================================
# NODO
# =========================================================

@dataclass
class NodoPipeline:
    """
    Representa un estado de datos en el pipeline.
    """
    id: str
    tipo_dato: TipoDato
    data: Optional[Any] = None  # Se completa en ejecución

    def __repr__(self):
        return f"<Nodo {self.id} ({self.tipo_dato.name})>"


# =========================================================
# ARISTA (OPERACIÓN)
# =========================================================

@dataclass
class AristaOperacion:
    """
    Representa una transformación entre nodos.
    """
    origen: str
    destino: str
    operacion: Operacion

    def __repr__(self):
        return f"{self.origen} --[{self.operacion.nombre}]--> {self.destino}"


# =========================================================
# GRAFO PIPELINE
# =========================================================

@dataclass
class GrafoPipeline:
    nodos: Dict[str, NodoPipeline] = field(default_factory=dict)
    aristas: List[AristaOperacion] = field(default_factory=list)

    # -----------------------------
    # Helpers estructurales
    # -----------------------------

    def agregar_nodo(self, nodo: NodoPipeline):
        self.nodos[nodo.id] = nodo

    def agregar_arista(self, arista: AristaOperacion):
        if arista.origen not in self.nodos or arista.destino not in self.nodos:
            raise ValueError("Nodo origen/destino no existe")
        self.aristas.append(arista)

    def salientes(self, nodo_id: str) -> List[AristaOperacion]:
        return [a for a in self.aristas if a.origen == nodo_id]

    def entrantes(self, nodo_id: str) -> List[AristaOperacion]:
        return [a for a in self.aristas if a.destino == nodo_id]

    def nodos_iniciales(self) -> List[NodoPipeline]:
        return [n for n in self.nodos.values() if not self.entrantes(n.id)]

    def nodos_finales(self) -> List[NodoPipeline]:
        return [n for n in self.nodos.values() if not self.salientes(n.id)]

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

    - Nodo = dato
    - Arista = operación
    """

    def __init__(self, grafo: GrafoPipeline):
        self.grafo = grafo
        self.nombre = ""

    # =========================================================
    # EJECUCIÓN
    # =========================================================

    def ejecutar(self, input_inicial: Any) -> Resultado[Dict[str, Any], ErrorBioImagen]:
        """
        Ejecuta el pipeline completo.
        Retorna todos los nodos finales.
        """

        try:
            orden = self.grafo.orden_topologico()
        except Exception as e:
            return Err(ErrorBioImagen(
                etapa="pipeline",
                mensaje=f"Grafo inválido: {e}"
            ))

        # Inicializar nodo(s) raíz
        iniciales = self.grafo.nodos_iniciales()

        if not iniciales:
            return Err(ErrorBioImagen(
                etapa="pipeline",
                mensaje="No hay nodo inicial"
            ))

        for nodo in iniciales:
            nodo.data = input_inicial

        # Ejecutar
        for nodo_id in orden:
            nodo = self.grafo.nodos[nodo_id]

            if nodo.data is None:
                # puede ser nodo de merge → se llena después
                continue

            for arista in self.grafo.salientes(nodo_id):

                op = arista.operacion

                resultado = op.ejecutar(nodo.data)

                if resultado.es_err():
                    return resultado  # Writer Monad se encarga del log

                salida = resultado.unwrap()

                self.grafo.nodos[arista.destino].data = salida

        # Retornar nodos finales
        salida = {
            n.id: n.data
            for n in self.grafo.nodos_finales()
        }

        return Ok(salida)

    # =========================================================
    # SUBGRAFOS (RAMAS)
    # =========================================================

    def subgrafo_desde(self, nodo_id: str) -> GrafoPipeline:
        """
        Devuelve el subgrafo hacia adelante desde un nodo.
        """

        visitados: Set[str] = set()
        aristas_sub: List[AristaOperacion] = []

        def dfs(actual: str):
            if actual in visitados:
                return
            visitados.add(actual)

            for arista in self.grafo.salientes(actual):
                aristas_sub.append(arista)
                dfs(arista.destino)

        dfs(nodo_id)

        nodos_sub = {
            nid: self.grafo.nodos[nid]
            for nid in visitados
        }

        return GrafoPipeline(nodos=nodos_sub, aristas=aristas_sub)

    def subgrafo_hasta(self, nodo_id: str) -> GrafoPipeline:
        """
        Devuelve el subgrafo hacia atrás hasta un nodo.
        """

        visitados: Set[str] = set()
        aristas_sub: List[AristaOperacion] = []

        def dfs(actual: str):
            if actual in visitados:
                return
            visitados.add(actual)

            for arista in self.grafo.entrantes(actual):
                aristas_sub.append(arista)
                dfs(arista.origen)

        dfs(nodo_id)

        nodos_sub = {
            nid: self.grafo.nodos[nid]
            for nid in visitados
        }

        return GrafoPipeline(nodos=nodos_sub, aristas=aristas_sub)

    # =========================================================
    # UTILIDADES
    # =========================================================

    def reset_datos(self):
        """Limpia todos los nodos (para re-ejecutar)."""
        for nodo in self.grafo.nodos.values():
            nodo.data = None

    def __repr__(self):
        return f"<FlujoTrabajo nodos={len(self.grafo.nodos)} aristas={len(self.grafo.aristas)}>"