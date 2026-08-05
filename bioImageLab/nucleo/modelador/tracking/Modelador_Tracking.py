from dataclasses import dataclass
from typing import Dict, List, Set, Tuple, TypedDict
from collections import defaultdict

Pixel = Tuple[int, int]
BBox = Tuple[int, int, int, int]  # min_x, min_y, max_x, max_y

@dataclass
class ObjetoSegmentado:
    label: int
    pixeles: Set[Pixel]
    bbox: BBox

class ObjetoUnificado(TypedDict):
    label: int
    zs: Dict[int, Set[Pixel]]

# ---------------------------------------------------------------------------
# Helpers privados
# ---------------------------------------------------------------------------

def _bbox_se_tocan(b1: BBox, b2: BBox) -> bool:
    """Chequeo rápido de colisión en 2D antes de mirar píxeles."""
    return not (b1[2] < b2[0] or b2[2] < b1[0] or
                b1[3] < b2[1] or b2[3] < b1[1])

def _hay_overlap_xy(a: ObjetoSegmentado, b: ObjetoSegmentado) -> bool:
    """True si comparten al menos un píxel en la proyección XY."""
    if not _bbox_se_tocan(a.bbox, b.bbox):
        return False
    return not a.pixeles.isdisjoint(b.pixeles) # A = {1, 2, 3}, B = {3, 4, 5} → A.isdisjoint(B) es False

def _find(padre: List[int], x: int) -> int:
    """Path compression."""
    while padre[x] != x:
        padre[x] = padre[padre[x]]
        x = padre[x]
    return x

def _union(padre: List[int], rango: List[int], x: int, y: int) -> None:
    """Union by rank."""
    rx, ry = _find(padre, x), _find(padre, y)
    if rx == ry:
        return
    if rango[rx] < rango[ry]:
        rx, ry = ry, rx
    padre[ry] = rx
    if rango[rx] == rango[ry]:
        rango[rx] += 1

# ---------------------------------------------------------------------------
# Función pública pura
# ---------------------------------------------------------------------------

def unificar_objetos(
    objetos_por_z: Dict[int, List[ObjetoSegmentado]],
    conectividad_z: int = 1
) -> List[ObjetoUnificado]:
    """
    Agrupa objetos segmentados de distintos slices Z en componentes
    conectadas 3D (un único objeto real).

    Args:
        objetos_por_z: mapeo z -> lista de objetos segmentados en ese slice.
        conectividad_z: máxima distancia en Z para considerar conexión.
                        1 = solo slices adyacentes (z y z+1).

    Returns:
        Lista de objetos unificados con sus píxeles agrupados por z.
    """
    if not objetos_por_z:
        return []

    # Aplanar: asignar id numérico global a cada objeto
    # idx_a_obj[id] = (z, ObjetoSegmentado)
    idx_a_obj: List[Tuple[int, ObjetoSegmentado]] = []
    ids_por_z: Dict[int, List[int]] = defaultdict(list)

    for z in sorted(objetos_por_z.keys()):
        for obj in objetos_por_z[z]:
            ids_por_z[z].append(len(idx_a_obj))
            idx_a_obj.append((z, obj))

    n = len(idx_a_obj)
    if n == 0:
        return []

    # Union-Find sobre ids numéricos
    padre = list(range(n))
    rango = [0] * n

    zs = sorted(ids_por_z.keys())

    for i, z in enumerate(zs):
        for j in range(i + 1, len(zs)):
            z2 = zs[j]
            if z2 - z > conectividad_z:
                break
            for id_a in ids_por_z[z]:
                for id_b in ids_por_z[z2]:
                    _, obj_a = idx_a_obj[id_a]
                    _, obj_b = idx_a_obj[id_b]
                    if _hay_overlap_xy(obj_a, obj_b):
                        _union(padre, rango, id_a, id_b)

    # Agrupar ids por raíz
    comp: Dict[int, Dict[int, Set[Pixel]]] = defaultdict(lambda: defaultdict(set))
    for i in range(n):
        raiz = _find(padre, i)
        z, obj = idx_a_obj[i]
        comp[raiz][z].update(obj.pixeles)

    # Armar output inmutable (nuevos sets, no referencias a los inputs)
    return [
        ObjetoUnificado(
            label=label,
            zs={z: set(pixeles) for z, pixeles in sorted(zs_data.items())}
        )
        for label, (_, zs_data) in enumerate(sorted(comp.items()), start=1)
    ]