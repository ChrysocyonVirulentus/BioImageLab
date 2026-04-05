# === gestorLab/Flujo_Trabajo.py ===

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, List, Any, Optional, Set

from ..controlador.Resultado_Either import Resultado, Ok, Err, LogEvento, NivelLog
from ..controlador.Controlador_BioImagen import ErrorBioImagen
from .Operacion import Operacion
from .Categoria_Operacion import TipoDato


# =========================================================
# NODO
# =========================================================

@dataclass
class NodoPipeline:
    id:        str
    tipo_dato: TipoDato
    data:      List[Any] = field(default_factory=list)

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
    Pipeline DAG.

    ejecutar() retorna Ok((salida, logs)) donde:
        salida : Dict[nodo_id, dato_final]
        logs   : List[LogEvento] cosechados de todas las cajas intermedias
    """

    def __init__(self, grafo: GrafoPipeline):
        self.grafo  = grafo
        self.nombre = ""

    # =========================================================
    # EJECUCIÓN
    # =========================================================

    def ejecutar(
        self, input_inicial: Any
    ) -> Resultado[Tuple[Dict[str, Any], List[LogEvento]], Any]:
        """
        Retorna Ok((salida_dict, lista_logs)) o Err(error_enriquecido).
        Los logs de cada operación se cosechan de las cajas intermedias.
        """
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

        # Logs acumulados de todas las operaciones
        logs_acumulados: List[LogEvento] = []

        for nodo_id in orden:
            nodo = self.grafo.nodos[nodo_id]
            if not nodo.data:
                continue

            input_data = nodo.data[-1]

            for arista in self.grafo.salientes(nodo_id):
                resultado = arista.operacion.ejecutar(input_data)

                # Cosechar logs antes de abrir la caja
                log_intermedio = getattr(resultado, "_log", ())
                logs_acumulados.extend(
                    ev for ev in log_intermedio if isinstance(ev, LogEvento)
                )

                if resultado.es_err():
                    error = resultado.error
                    # Agregar log del error antes de propagar
                    logs_acumulados.append(LogEvento(
                        etapa   = f"pipeline -> {getattr(error, 'etapa', 'desconocido')}",
                        mensaje = (
                            f"[{nodo_id} → {arista.destino}] "
                            f"'{arista.operacion.nombre}' falló: "
                            f"{getattr(error, 'mensaje', str(error))}"
                        ),
                        nivel    = NivelLog.ERROR,
                        metadata = {"nodo_origen": nodo_id, "nodo_destino": arista.destino},
                    ))
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
                        error_enriquecido = error

                    # Retornar Err con los logs acumulados hasta el fallo
                    return Err(error_enriquecido, tuple(logs_acumulados))

                # Log de éxito por operación
                logs_acumulados.append(LogEvento(
                    etapa   = arista.operacion.categoria.name.lower(),
                    mensaje = f"'{arista.operacion.nombre}' OK",
                    nivel   = NivelLog.INFO,
                    metadata = {"nodo": nodo_id, "destino": arista.destino},
                ))

                self.grafo.nodos[arista.destino].data.append(resultado.unwrap())

        salida = {
            n.id: n.data[-1] if n.data else None
            for n in self.grafo.nodos_finales()
        }

        return Ok((salida, logs_acumulados))

    # =========================================================
    # SUBGRAFOS
    # =========================================================

    def subgrafo_desde(self, nodo_id: str) -> GrafoPipeline:
        visitados, aristas_sub = set(), []
        def dfs(actual):
            if actual in visitados: return
            visitados.add(actual)
            for a in self.grafo.salientes(actual):
                aristas_sub.append(a); dfs(a.destino)
        dfs(nodo_id)
        return GrafoPipeline(
            nodos={nid: self.grafo.nodos[nid] for nid in visitados},
            aristas=aristas_sub
        )

    def subgrafo_hasta(self, nodo_id: str) -> GrafoPipeline:
        visitados, aristas_sub = set(), []
        def dfs(actual):
            if actual in visitados: return
            visitados.add(actual)
            for a in self.grafo.entrantes(actual):
                aristas_sub.append(a); dfs(a.origen)
        dfs(nodo_id)
        return GrafoPipeline(
            nodos={nid: self.grafo.nodos[nid] for nid in visitados},
            aristas=aristas_sub
        )

    # =========================================================
    # VALIDACIÓN INTRÍNSECA
    # =========================================================

    def validar_pipeline(self):
        errores = []
        for arista in self.grafo.aristas:
            nodo_destino = self.grafo.nodos[arista.destino]
            op           = arista.operacion
            if op.tipo_dato_salida is None: continue
            if nodo_destino.tipo_dato != op.tipo_dato_salida:
                errores.append(
                    f"TipoDato inconsistente: nodo={nodo_destino.tipo_dato.name} "
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