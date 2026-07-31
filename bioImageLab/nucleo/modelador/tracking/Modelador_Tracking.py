"""
Tracking de objetos segmentados entre cortes Z.

Este módulo recibe los objetos segmentados obtenidos durante la
cuantificación y reconstruye las trayectorias de cada objeto a lo
largo de los distintos planos Z.

Actualmente implementa:

    - Vecino más cercano por centroide.

Características:
    - Un único tiempo t.
    - Tracking únicamente sobre Z.
    - Sin gate.
    - Sin resolución de conflictos.
    - Sin manejo de gaps.
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple, Set, TypedDict

from ...gestorLab.Registro_Metodos import registrar_en


# =============================================================================
# ESTRUCTURAS DE DATOS
# =============================================================================

Pixel = Tuple[int, int]


# ---------------- INPUT ----------------

class ObjetoSegmentado(TypedDict):
    label: int
    pixeles: Set[Pixel]
    centroide: Tuple[float, float]
    area: int


ObjetosPorCorte = Dict[Tuple[int, int], List[ObjetoSegmentado]]


# ---------------- OUTPUT ----------------

class DatosObjetoZ(TypedDict):
    pixeles: Set[Pixel]
    centroide: Tuple[float, float]
    area: int
    label_original: int


TrazaObjeto = Dict[int, DatosObjetoZ]
Trazas = Dict[int, TrazaObjeto]


# =============================================================================
# FUNCIONES AUXILIARES
# =============================================================================

def _distancia_euclidiana(
    p1: Tuple[float, float],
    p2: Tuple[float, float]
) -> float:
    """Calcula distancia euclidiana entre dos centroides."""
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


# =============================================================================
# CLASE BASE
# =============================================================================

class TrackingBase:
    """
    Clase base para algoritmos de tracking.

    Todos los algoritmos reciben ObjetosPorCorte y retornan Trazas.
    """

    nombre = "tracking_base"

    def rastrear(
        self,
        objetos_por_corte: ObjetosPorCorte
    ) -> Trazas:
        raise NotImplementedError

    def _validar_entrada(
        self,
        objetos_por_corte: ObjetosPorCorte
    ) -> None:

        if not isinstance(objetos_por_corte, dict):
            raise ValueError(
                "objetos_por_corte debe ser un diccionario."
            )


# =============================================================================
# TRACKING POR VECINO MÁS CERCANO
# =============================================================================

@registrar_en("modelado")
class VecinoMasCercanoTracking(TrackingBase):
    """
    Tracking mediante vecino más cercano usando centroides.

    Algoritmo:

        Para cada objeto del corte z:

            1. Buscar el objeto de z+1 cuyo centroide esté más cerca.
            2. Continuar la misma traza.
            3. Si un objeto aparece por primera vez,
               crear una nueva traza.

    Limitaciones:

        - Sin gate.
        - Sin resolución de conflictos.
        - Sin manejo de gaps.
    """

    nombre = "vecino_mas_cercano"

    def rastrear(
        self,
        objetos_por_corte: ObjetosPorCorte
    ) -> Trazas:

        self._validar_entrada(objetos_por_corte)

        trazas: Trazas = {}
        siguiente_id = 0

        if not objetos_por_corte:
            return trazas

        # Se toma únicamente el primer instante temporal.
        t = next(iter(objetos_por_corte))[1]

        zs = sorted(
            z
            for (z, t_clave) in objetos_por_corte.keys()
            if t_clave == t
        )

        # label local -> id global
        id_global_actual: Dict[int, int] = {}

        # ==========================================================
        # Inicialización
        # ==========================================================

        primer_z = zs[0]

        for obj in objetos_por_corte[(primer_z, t)]:

            id_global = siguiente_id
            siguiente_id += 1

            id_global_actual[obj["label"]] = id_global

            trazas[id_global] = {
                primer_z: {
                    "pixeles": obj["pixeles"],
                    "centroide": obj["centroide"],
                    "area": obj["area"],
                    "label_original": obj["label"],
                }
            }

        # ==========================================================
        # Tracking
        # ==========================================================

        for z, z_sig in zip(zs[:-1], zs[1:]):

            objetos_z = objetos_por_corte[(z, t)]
            objetos_sig = objetos_por_corte[(z_sig, t)]

            id_global_siguiente: Dict[int, int] = {}

            if objetos_sig:

                for obj_z in objetos_z:

                    id_global = id_global_actual.get(obj_z["label"])

                    if id_global is None:
                        continue

                    # Buscar el objeto del siguiente corte cuyo
                    # centroide esté a menor distancia.
                    mas_cercano = min(
                        objetos_sig,
                        key=lambda o: _distancia_euclidiana(
                            obj_z["centroide"],
                            o["centroide"],
                        ),
                    )

                    # Asignar ese objeto a la misma traza.
                    # Si dos objetos eligen el mismo, el último pisa
                    # la asignación anterior (limitación conocida).
                    id_global_siguiente[
                        mas_cercano["label"]
                    ] = id_global

            # ======================================================
            # Objetos nuevos
            # ======================================================

            for obj_sig in objetos_sig:

                if obj_sig["label"] not in id_global_siguiente:

                    id_global_siguiente[obj_sig["label"]] = siguiente_id
                    siguiente_id += 1

            # ======================================================
            # Guardar resultados
            # ======================================================

            for obj_sig in objetos_sig:

                id_global = id_global_siguiente[obj_sig["label"]]

                trazas.setdefault(id_global, {})[z_sig] = {
                    "pixeles": obj_sig["pixeles"],
                    "centroide": obj_sig["centroide"],
                    "area": obj_sig["area"],
                    "label_original": obj_sig["label"],
                }

            # El corte siguiente pasa a ser el actual.
            id_global_actual = id_global_siguiente

        return trazas


# =============================================================================
# PIPELINE COMPLETO
# =============================================================================

def pipeline_tracking_completo(
    objetos_por_corte: ObjetosPorCorte,
    metodo: str = "vecino_mas_cercano",
):
    """
    Ejecuta un algoritmo de tracking.

    Parameters
    ----------
    objetos_por_corte
        Diccionario (z,t) -> lista de objetos segmentados.

    metodo
        Algoritmo de tracking a utilizar.

    Returns
    -------
    dict con:

        - trazas
        - modelo
        - metodo
    """

    if metodo == "vecino_mas_cercano":
        tracker = VecinoMasCercanoTracking()

    else:
        raise ValueError(f"Método no reconocido: {metodo}")

    trazas = tracker.rastrear(objetos_por_corte)

    return {
        "trazas": trazas,
        "modelo": tracker,
        "metodo": metodo,
    }