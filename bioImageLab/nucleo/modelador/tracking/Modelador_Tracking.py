from __future__ import annotations

import math
from typing import Dict, List, Tuple, Set, TypedDict, Optional
import numpy as np
from scipy.optimize import linear_sum_assignment


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

# ESPACIALES
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

    t = next(iter(objetos_por_corte))[1] # obtiene la primera t
    zs = sorted(z for (z, t_clave) in objetos_por_corte.keys() if t_clave == t) # ordena todas las zs del primer t

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


# ALGORITMO 2: ASIGNACIÓN GREEDY GLOBAL POR PARES (con gate)

def rastrear_asignacion_greedy_global(
    objetos_por_corte: ObjetosPorCorte,
    distancia_maxima: Optional[float] = None,
) -> Trazas:
    """
    Algoritmo 2: mejora el Algoritmo 1 resolviendo el problema del
    "pisado silencioso". En vez de que cada objeto de z elija su
    vecino más cercano de forma independiente (lo cual depende del
    orden de iteración cuando hay conflictos), acá se arma la lista
    de TODOS los pares posibles (obj_z, obj_siguiente) entre dos
    cortes consecutivos, se ordena por distancia ascendente, y se
    van confirmando matches de a uno: apenas un objeto (de cualquiera
    de los dos lados) es tomado, queda descartado para el resto de
    los pares. Es un matching bipartito greedy (aproximación barata
    al problema de asignación óptima, tipo Hungarian, pero sin
    garantía de optimalidad global -- sí determinístico y ya no
    depende del orden de iteración, solo de la distancia real).
 
    También agrega gate opcional: si `distancia_maxima` no es None,
    ningún par con distancia mayor a ese umbral se matchea, aunque
    fuera "el par disponible más cercano". Esto evita que un objeto
    lejano se asocie por descarte.
 
    Sigue sin tolerar gaps (si un objeto no aparece en z+1, la traza
    termina ahí aunque reaparezca en z+2) -- eso queda para un
    Algoritmo 3.
 
    Args:
        objetos_por_corte: dict (z, t) -> lista de objetos segmentados.
        distancia_maxima: gate opcional. Si se especifica, pares con
            distancia mayor a este valor nunca se matchean.
 
    Returns:
        Trazas: dict id_objeto_global -> {z -> DatosObjetoZ}.
 
    Complejidad:
        O(Z * N * M * log(N*M)) -- por el sort de pares en cada
        transición entre cortes consecutivos (N, M = objetos por corte).
    """
    trazas: Trazas = {}
    siguiente_id: int = 0
 
    if not objetos_por_corte:
        return trazas
 
    t = next(iter(objetos_por_corte))[1]
    zs = sorted(z for (z, t_clave) in objetos_por_corte.keys() if t_clave == t)
 
    id_global_actual: Dict[int, int] = {}
 
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
 
    for z, z_sig in zip(zs[:-1], zs[1:]):
        objetos_z = objetos_por_corte[(z, t)]
        objetos_sig = objetos_por_corte[(z_sig, t)]
 
        # armar todos los pares posibles (obj_z, obj_sig) con su distancia
        pares: List[Tuple[float, ObjetoSegmentado, ObjetoSegmentado]] = []
        for obj_z in objetos_z:
            if obj_z["label"] not in id_global_actual:
                continue  # robustez: no debería faltar
            for obj_sig in objetos_sig:
                d = _distancia_euclidiana(obj_z["centroide"], obj_sig["centroide"])
                if distancia_maxima is None or d <= distancia_maxima:
                    pares.append((d, obj_z, obj_sig))
 
        # ordenar por distancia ascendente -> los matches "más seguros" primero
        pares.sort(key=lambda p: p[0])
 
        tomados_z: Set[int] = set()
        tomados_sig: Set[int] = set()
        id_global_siguiente: Dict[int, int] = {}
 
        for d, obj_z, obj_sig in pares:
            if obj_z["label"] in tomados_z or obj_sig["label"] in tomados_sig:
                continue  # alguno de los dos ya fue asignado a otro match
            id_global_siguiente[obj_sig["label"]] = id_global_actual[obj_z["label"]]
            tomados_z.add(obj_z["label"])
            tomados_sig.add(obj_sig["label"])
 
        # objetos de z_siguiente sin match -> nacen como traza nueva
        for obj_sig in objetos_sig:
            if obj_sig["label"] not in id_global_siguiente:
                id_global_siguiente[obj_sig["label"]] = siguiente_id
                siguiente_id += 1
 
        # objetos de z que no encontraron match -> sus trazas mueren acá
        # (no requiere código extra: simplemente no se propagan)
 
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


# ALGORITMO 3: ASIGNACIÓN ÓPTIMA GLOBAL (HÚNGARO)

def rastrear_hungaro(
    objetos_por_corte: ObjetosPorCorte,
    distancia_maxima: Optional[float] = None,
) -> Trazas:
    """
    Algoritmo 3: en vez del matching greedy del Algoritmo 2 (que toma
    los pares mas cercanos primero, sin garantia de que la suma total
    sea la minima posible), usa el algoritmo hungaro para encontrar la
    asignacion que minimiza el COSTO TOTAL de la transicion entre dos
    cortes consecutivos.

    Por que importa: el greedy es "avaro" -- toma el mejor par
    disponible en cada paso, pero eso puede forzar matches pesimos mas
    adelante. Ejemplo: A-C cuesta 3, B-D cuesta 3, pero A-D cuesta 1 y
    B-C cuesta 10. El greedy toma A-D primero (el par mas barato de
    todos) y fuerza B-C (total 11). El optimo real es A-C + B-D
    (total 6). El hungaro siempre encuentra el optimo global.

    Matrices no cuadradas (distinta cantidad de objetos en z y
    z_siguiente) se manejan agregando el resultado de scipy: los
    objetos que no reciben match nacen o mueren, igual que en el
    Algoritmo 2.

    Sigue sin tolerar gaps -- eso queda para otro algoritmo.

    Args:
        objetos_por_corte: dict (z, t) -> lista de objetos segmentados.
        distancia_maxima: gate opcional. Pares con distancia mayor a
            este valor se penalizan con costo infinito, forzando a que
            el hungaro nunca los elija (equivalente a descartarlos).

    Returns:
        Trazas: dict id_objeto_global -> {z -> DatosObjetoZ}.

    Complejidad:
        O(Z * max(N, M)^3) -- el hungaro es O(n^3) por cada transicion
        entre cortes consecutivos.
    """
    trazas: Trazas = {}
    siguiente_id: int = 0

    if not objetos_por_corte:
        return trazas

    t = next(iter(objetos_por_corte))[1]
    zs = sorted(z for (z, t_clave) in objetos_por_corte.keys() if t_clave == t)

    id_global_actual: Dict[int, int] = {}

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

    for z, z_sig in zip(zs[:-1], zs[1:]):
        objetos_z = [o for o in objetos_por_corte[(z, t)] if o["label"] in id_global_actual]
        objetos_sig = objetos_por_corte[(z_sig, t)]

        id_global_siguiente: Dict[int, int] = {}

        if objetos_z and objetos_sig:
            n, m = len(objetos_z), len(objetos_sig)
            costo = np.zeros((n, m))
            for i, obj_z in enumerate(objetos_z):
                for j, obj_sig in enumerate(objetos_sig):
                    d = _distancia_euclidiana(obj_z["centroide"], obj_sig["centroide"])
                    if distancia_maxima is not None and d > distancia_maxima:
                        d = 1e9  # costo altisimo -> el hungaro lo evita si puede
                    costo[i, j] = d

            filas, columnas = linear_sum_assignment(costo)

            for i, j in zip(filas, columnas):
                if costo[i, j] >= 1e9:
                    continue  # gate: descartar aunque el solver lo haya propuesto
                id_global_siguiente[objetos_sig[j]["label"]] = id_global_actual[objetos_z[i]["label"]]

        for obj_sig in objetos_sig:
            if obj_sig["label"] not in id_global_siguiente:
                id_global_siguiente[obj_sig["label"]] = siguiente_id
                siguiente_id += 1

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
 