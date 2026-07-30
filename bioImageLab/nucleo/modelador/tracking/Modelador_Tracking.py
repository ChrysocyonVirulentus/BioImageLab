from __future__ import annotations

import math
from typing import Dict, List, Tuple, Set, TypedDict


# ESTRUCTURAS DE DATOS (input / output)

Pixel = Tuple[int, int]

# INPUT 
class ObjetoSegmentado(TypedDict):
    label: int
    pixeles: Set[Pixel]
    centroide: Tuple[float, float]
    area: int

ObjetosPorCorte = Dict[Tuple[int, int], List[ObjetoSegmentado]]

# OUTPUT
class DatosObjetoZ(TypedDict):
    pixeles: Set[Pixel]
    centroide: Tuple[float, float]
    area: int
    label_original: int

TrazaObjeto = Dict[int, DatosObjetoZ]
Trazas = Dict[int, TrazaObjeto]


# ALGORITMO 1: VECINO MÁS CERCANO POR CENTROIDE (greedy, sin gate)

def _distancia_euclidiana(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def rastrear_vecino_mas_cercano(objetos_por_corte: ObjetosPorCorte) -> Trazas:
    """
    Algoritmo 1 (el más simple posible): para cada objeto del corte z,
    lo asocia con el objeto de z+1 cuyo centroide está a menor distancia
    euclidiana. Sin umbral de corte (gate) y sin resolver conflictos:
    si dos objetos de z eligen como "más cercano" al mismo objeto de
    z+1, la asignación que sobrevive es la del último objeto de z
    procesado — la anterior queda pisada y esa traza simplemente
    termina ahí, sin aviso.

    Asume un único t (se toma el que aparezca en las claves de
    objetos_por_corte; si hubiera más de uno, se ignoran los demás).
    Tracking exclusivamente a través de Z.

    Args:
        objetos_por_corte: dict (z, t) -> lista de objetos segmentados
            en ese corte.

    Returns:
        Trazas: dict id_objeto_global -> {z -> DatosObjetoZ}.

    Limitaciones (por diseño, es el algoritmo base):
        - No hay gate: incluso un objeto centroide-lejano se matchea
          si es el "menos lejano" disponible.
        - No hay resolución de conflictos: puede haber dos objetos de z
          asignados al mismo objeto de z+1; la traza que pierde el
          conflicto se corta silenciosamente.
        - No tolera gaps: si un objeto no aparece en z+1, la traza
          termina ahí aunque reaparezca en z+2.

    Complejidad:
        O(Z * N * M) — para cada par de cortes consecutivos, por cada
        objeto de z se recorren todos los de z+1 (N, M = objetos por
        corte).
    """
    trazas: Trazas = {}
    siguiente_id: int = 0

    if not objetos_por_corte:
        return trazas

    t = next(iter(objetos_por_corte))[1]
    zs = sorted(z for (z, t_clave) in objetos_por_corte.keys() if t_clave == t)

    # label local del corte actual -> id_global de su traza
    id_global_actual: Dict[int, int] = {}

    # Inicializar: todo objeto del primer z arranca traza nueva
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

    # Recorrer pares consecutivos (z, z_siguiente)
    for z, z_sig in zip(zs[:-1], zs[1:]):
        objetos_z = objetos_por_corte[(z, t)]
        objetos_sig = objetos_por_corte[(z_sig, t)]

        # label local de z_siguiente -> id_global asignado
        id_global_siguiente: Dict[int, int] = {}

        if objetos_sig:
            for obj_z in objetos_z:
                id_global = id_global_actual.get(obj_z["label"])
                if id_global is None:
                    continue  # robustez: no debería faltar

                mas_cercano = min(
                    objetos_sig,
                    key=lambda o: _distancia_euclidiana(
                        obj_z["centroide"], o["centroide"]
                    ),
                )
                # sin chequeo de conflicto: si ya había una asignación
                # para este label, se pisa acá
                id_global_siguiente[mas_cercano["label"]] = id_global

        # objetos de z_siguiente que nadie "eligió" -> nacen como traza nueva
        for obj_sig in objetos_sig:
            if obj_sig["label"] not in id_global_siguiente:
                id_global_siguiente[obj_sig["label"]] = siguiente_id
                siguiente_id += 1

        # volcar resultados de z_siguiente a Trazas
        for obj_sig in objetos_sig:
            id_global = id_global_siguiente[obj_sig["label"]]
            trazas.setdefault(id_global, {})[z_sig] = {
                "pixeles": obj_sig["pixeles"],
                "centroide": obj_sig["centroide"],
                "area": obj_sig["area"],
                "label_original": obj_sig["label"],
            }

        id_global_actual = id_global_siguiente

    return trazas