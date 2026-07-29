"""
Tracking de objetos (núcleos) a través de cortes Z.

Problema que resuelve:
    Cada corte Z se segmenta de forma independiente (Controlador_Segmentador,
    estrategia PorCorteZ). Esto significa que el label de una región en el
    corte z=5 NO tiene ninguna relación garantizada con el label de la misma
    región física en z=6: son etiquetas locales a cada corte (típicamente
    asignadas por connected components o watershed dentro de ese slice).

    Este módulo enlaza esas identidades locales en una identidad global
    consistente a través de z, asumiendo que el mismo núcleo real ocupa
    posiciones (centroide) y tamaños (área) similares entre cortes
    consecutivos.

Separación de responsabilidades:
    - NO segmenta (eso es Controlador_Segmentador)
    - NO cuantifica área/centroide desde imágenes (eso es Cuantificadores_Morfometricos)
    - SÍ: arma el DataFrame "largo" por corte a partir de una máscara ya
      etiquetada (construir_dataframe_cortes), y SÍ resuelve la
      correspondencia entre objetos de cortes consecutivos (rastreadores)

Contrato de datos — DataFrame de entrada (formato "largo", una fila por
objeto por corte):

    ┌────┬─────┬────┬───────┬───────┬─────────────┬─────────────┐
    │ t  │  z  │ c  │ label │ area  │ centroide_y │ centroide_x │
    ├────┼─────┼────┼───────┼───────┼─────────────┼─────────────┤
    │ 0  │  0  │ 0  │   1   │ 312.0 │    88.4      │   140.2     │
    │ 0  │  0  │ 0  │   2   │ 289.0 │    210.1     │   55.7      │
    │ 0  │  1  │ 0  │   1   │ 340.5 │    89.0      │   141.0     │
    └────┴─────┴────┴───────┴───────┴─────────────┴─────────────┘

    - t, z, c: índices del corte de origen (t y c fijos en el caso de uso
      actual: tracking solo a través de Z para un timepoint/canal dado)
    - label: id local del objeto asignado por el etiquetado de ese corte
      (NO es estable entre cortes — es justamente lo que este módulo resuelve)
    - area, centroide_y, centroide_x: salida de Cuantificadores_Morfometricos

Contrato de datos — DataFrame de salida:
    Igual al de entrada + una columna nueva `id_traza`: identificador entero
    global, consistente para el mismo objeto físico a través de todos los
    cortes z en los que aparece.

Algoritmos disponibles (ver docstrings de cada clase para el detalle):
    - RastreadorCentroide: matching solo por distancia euclidiana de centroides
    - RastreadorCentroideArea (recomendado por defecto): score combinado de
      distancia de centroides + diferencia relativa de área
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple
from scipy.optimize import linear_sum_assignment
from skimage import measure

from ...gestorLab.Registro_Metodos import registrar_en


# =============================================================================
# 1. CONSTRUCCIÓN DEL DATAFRAME DE ENTRADA
# =============================================================================

def construir_dataframe_cortes(
    volumen_etiquetado: np.ndarray,
    t: int = 0,
    c: int = 0,
) -> pd.DataFrame:
    """
    Arma el DataFrame "largo" (una fila por objeto por corte) a partir de
    un volumen ya segmentado y etiquetado.

    Args:
        volumen_etiquetado: array 3D (Z, Y, X) de enteros, donde cada región
            conexa tiene un label distinto (0 = fondo). Es lo que produce
            Controlador_Segmentador con una estrategia PorCorteZ + un método
            instancial (watershed, connected components, etc.) — extraído
            del tensor 5D como data.datos[t, :, c, :, :].
        t, c: índices de timepoint y canal de origen, para dejar registro
            en el DataFrame (no se usan para calcular nada acá).

    Returns:
        DataFrame con columnas: t, z, c, label, area, centroide_y, centroide_x

    Nota:
        Usa skimage.measure.regionprops directamente (en vez de encadenar
        las clases Area/Centroide de Cuantificadores_Morfometricos) para
        garantizar que area y centroide de una misma fila correspondan
        exactamente al mismo label — evita depender de que dos llamadas
        separadas devuelvan las regiones en el mismo orden.

    Complejidad:
        O(Z * N_pixeles) — una pasada de regionprops por corte.
    """
    if volumen_etiquetado.ndim != 3:
        raise ValueError(
            f"Se espera un volumen 3D (Z, Y, X), recibido shape "
            f"{volumen_etiquetado.shape}"
        )

    filas = []
    for z in range(volumen_etiquetado.shape[0]):
        corte = volumen_etiquetado[z]
        if not np.any(corte):
            continue  # corte vacío, sin objetos — no es un error
        for region in measure.regionprops(corte.astype(int)):
            cy, cx = region.centroid
            filas.append({
                "t": t,
                "z": z,
                "c": c,
                "label": region.label,
                "area": float(region.area),
                "centroide_y": float(cy),
                "centroide_x": float(cx),
            })

    columnas = ["t", "z", "c", "label", "area", "centroide_y", "centroide_x"]
    if not filas:
        return pd.DataFrame(columns=columnas)
    return pd.DataFrame(filas, columns=columnas)


# =============================================================================
# 2. RESULTADO DE MATCHING ENTRE DOS CORTES CONSECUTIVOS
# =============================================================================

@dataclass(frozen=True)
class MatchingCorte:
    """
    Resultado de resolver la correspondencia entre el corte z y z+1.

    pares: lista de (label_z, label_z_siguiente) que se consideran el mismo
        objeto físico.
    nuevos: labels del corte z+1 que no matchean con nada en z (aparecen
        por primera vez — el núcleo "empieza" en ese plano).
    perdidos: labels del corte z que no matchean con nada en z+1 (el
        núcleo "termina" en ese plano).
    """
    pares: List[Tuple[int, int]]
    nuevos: List[int]
    perdidos: List[int]


# =============================================================================
# 3. RASTREADORES (resuelven matching entre DOS cortes)
# =============================================================================

class RastreadorBase:
    """
    Clase base: arma la matriz de costos, resuelve la asignación óptima
    (Hungarian) y filtra pares cuyo costo supera un umbral (gate) — un
    costo alto significa "en realidad no es el mismo objeto", así que se
    trata como nacimiento + muerte en vez de forzar el match.

    Subclases solo necesitan implementar `_matriz_costos`.
    """
    nombre = "rastreador_base"

    def __init__(self, costo_maximo: float = 1.0):
        """
        Args:
            costo_maximo: costo por encima del cual un par (objeto_z,
                objeto_z+1) se descarta y se trata como nacimiento/muerte
                en vez de continuación de traza. Depende de la escala de
                costo de cada subclase (ver sus docstrings).
        """
        self.costo_maximo = costo_maximo

    def _matriz_costos(self, df_z: pd.DataFrame, df_z_siguiente: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError

    def matchear(self, df_z: pd.DataFrame, df_z_siguiente: pd.DataFrame) -> MatchingCorte:
        """
        Resuelve la correspondencia entre dos cortes consecutivos.

        Args:
            df_z, df_z_siguiente: sub-DataFrames (mismo formato que
                construir_dataframe_cortes) correspondientes a un único
                corte cada uno.

        Complejidad:
            O(N^3) por el Hungarian (N = objetos en el corte más chico),
            irrelevante en la práctica para conteos de núcleos por campo.
        """
        labels_z = df_z["label"].to_numpy()
        labels_sig = df_z_siguiente["label"].to_numpy()

        if len(labels_z) == 0:
            return MatchingCorte(pares=[], nuevos=list(labels_sig), perdidos=[])
        if len(labels_sig) == 0:
            return MatchingCorte(pares=[], nuevos=[], perdidos=list(labels_z))

        costos = self._matriz_costos(df_z, df_z_siguiente)
        filas_idx, cols_idx = linear_sum_assignment(costos)

        pares, matcheados_z, matcheados_sig = [], set(), set()
        for i, j in zip(filas_idx, cols_idx):
            if costos[i, j] <= self.costo_maximo:
                pares.append((int(labels_z[i]), int(labels_sig[j])))
                matcheados_z.add(int(labels_z[i]))
                matcheados_sig.add(int(labels_sig[j]))

        nuevos = [int(l) for l in labels_sig if int(l) not in matcheados_sig]
        perdidos = [int(l) for l in labels_z if int(l) not in matcheados_z]
        return MatchingCorte(pares=pares, nuevos=nuevos, perdidos=perdidos)


@registrar_en("modelado")
class RastreadorCentroide(RastreadorBase):
    """
    Matching entre cortes consecutivos usando solo distancia euclidiana
    entre centroides.

    Algoritmo:
        costo[i,j] = || centroide_i(z) - centroide_j(z+1) ||_2
        Asignación óptima global vía Hungarian (minimiza costo total, no
        greedy — evita que un match "bueno" temprano bloquee uno mejor
        para otro objeto).

    Ventajas:
        - Simple, rápido, sin parámetros de ponderación que tunear
        - Buen default cuando los núcleos están bien separados espacialmente
    Desventajas:
        - Ciego a la forma/tamaño: dos núcleos muy próximos pueden
          intercambiar identidad (swap) si sus centroides quedan más
          cerca del vecino ajeno que del propio
        - `costo_maximo` acá está en píxeles: hay que calibrarlo según el
          desplazamiento típico esperado entre z consecutivos
    """
    nombre = "rastreador_centroide"

    def _matriz_costos(self, df_z: pd.DataFrame, df_z_siguiente: pd.DataFrame) -> np.ndarray:
        p1 = df_z[["centroide_y", "centroide_x"]].to_numpy()
        p2 = df_z_siguiente[["centroide_y", "centroide_x"]].to_numpy()
        diff = p1[:, None, :] - p2[None, :, :]
        return np.sqrt((diff ** 2).sum(axis=-1))


@registrar_en("modelado")
class RastreadorCentroideArea(RastreadorBase):
    """
    Matching combinado: distancia de centroides + diferencia relativa de
    área, normalizadas y ponderadas. Recomendado por defecto para tracking
    de núcleos en Z.

    Algoritmo:
        d_norm[i,j]    = ||centroide_i(z) - centroide_j(z+1)|| / distancia_normalizacion
        area_norm[i,j] = |area_i(z) - area_j(z+1)| / max(area_i(z), area_j(z+1))
        costo[i,j]     = peso_distancia * d_norm[i,j] + peso_area * area_norm[i,j]

    Por qué el área ayuda acá:
        Un núcleo real es aprox. esférico/elipsoidal — su sección transversal
        crece y decrece gradualmente a medida que se atraviesa el volumen en
        z. Un salto brusco de área entre "candidatos" cercanos en el espacio
        es evidencia de que probablemente no es el mismo objeto (o que hubo
        un split/merge de segmentación), incluso si los centroides son
        parecidos.

    Ventajas:
        - Usa toda la información que ya calculan los cuantificadores
          morfométricos, sin necesitar las máscaras completas
        - Más robusto que centroide solo en zonas de alta densidad nuclear
    Desventajas:
        - Dos hiperparámetros más para calibrar (pesos + distancia de
          normalización), aunque los defaults son razonables como punto
          de partida
    """
    nombre = "rastreador_centroide_area"

    def __init__(
        self,
        costo_maximo: float = 1.0,
        peso_distancia: float = 0.7,
        peso_area: float = 0.3,
        distancia_normalizacion: float = 20.0,
    ):
        """
        Args:
            costo_maximo: umbral de costo combinado (adimensional, 0 a
                ~peso_distancia+peso_area en la práctica) por encima del
                cual se descarta el match.
            peso_distancia, peso_area: ponderación relativa de cada
                término. Default prioriza posición sobre tamaño.
            distancia_normalizacion: escala (en píxeles) que representa
                un desplazamiento "típico" del centroide entre cortes
                consecutivos — usarla para que d_norm quede en un rango
                comparable al término de área (que ya es relativo, 0-1).
                Ajustar según el z-step y el tamaño de núcleo esperado.
        """
        super().__init__(costo_maximo=costo_maximo)
        self.peso_distancia = peso_distancia
        self.peso_area = peso_area
        self.distancia_normalizacion = distancia_normalizacion

    def _matriz_costos(self, df_z: pd.DataFrame, df_z_siguiente: pd.DataFrame) -> np.ndarray:
        p1 = df_z[["centroide_y", "centroide_x"]].to_numpy()
        p2 = df_z_siguiente[["centroide_y", "centroide_x"]].to_numpy()
        diff = p1[:, None, :] - p2[None, :, :]
        distancia = np.sqrt((diff ** 2).sum(axis=-1))
        d_norm = distancia / self.distancia_normalizacion

        a1 = df_z["area"].to_numpy()[:, None]
        a2 = df_z_siguiente["area"].to_numpy()[None, :]
        area_norm = np.abs(a1 - a2) / np.maximum(a1, a2)

        return self.peso_distancia * d_norm + self.peso_area * area_norm


# =============================================================================
# 4. ARMADO DE TRAZAS A TRAVÉS DE TODOS LOS CORTES Z
# =============================================================================

def armar_trazas_z(
    df_cortes: pd.DataFrame,
    rastreador: Optional[RastreadorBase] = None,
) -> pd.DataFrame:
    """
    Encadena el matching corte-a-corte a lo largo de todo el eje Z y asigna
    un id_traza global consistente a cada objeto físico.

    Args:
        df_cortes: DataFrame como el que devuelve construir_dataframe_cortes
            (se espera un único t y c — filtrar antes si hay varios).
        rastreador: instancia de RastreadorBase a usar. Default:
            RastreadorCentroideArea() con sus parámetros por defecto.

    Returns:
        Copia de df_cortes + columna `id_traza` (int, consistente para el
        mismo objeto a través de z).

    Algoritmo:
        1. Recorre pares de cortes consecutivos (z, z+1) en orden.
        2. Para cada par, resuelve el matching con `rastreador.matchear`.
        3. Los "pares" propagan el id_traza del objeto en z al objeto en z+1.
        4. Los "nuevos" (sin match hacia atrás) reciben un id_traza fresco.
        5. Los "perdidos" simplemente no se propagan más (la traza termina
           en ese z) — no se tratan como error.

    Nota:
        No maneja gaps (un objeto que desaparece un plano y reaparece en el
        siguiente) — cada corte solo se compara contra el inmediato
        anterior. Si hace falta tolerar gaps, es una extensión natural:
        comparar también contra z-2 para los "perdidos" que no encontraron
        continuación.

    Complejidad:
        O(Z * N^3) en el peor caso (N = objetos por corte, dominado por el
        Hungarian de cada par consecutivo).
    """
    if rastreador is None:
        rastreador = RastreadorCentroideArea()

    if df_cortes.empty:
        return df_cortes.assign(id_traza=pd.Series(dtype=int))

    cortes_z = sorted(df_cortes["z"].unique())
    df_cortes = df_cortes.reset_index(drop=True)

    id_traza_por_fila: Dict[int, int] = {}
    siguiente_id_traza = 0

    # Inicializar: todo objeto del primer corte arranca una traza nueva
    df_z0 = df_cortes[df_cortes["z"] == cortes_z[0]]
    id_traza_actual: Dict[int, int] = {}  # label (del corte actual) -> id_traza
    for idx, fila in df_z0.iterrows():
        id_traza_por_fila[idx] = siguiente_id_traza
        id_traza_actual[int(fila["label"])] = siguiente_id_traza
        siguiente_id_traza += 1

    for z, z_siguiente in zip(cortes_z[:-1], cortes_z[1:]):
        df_z = df_cortes[df_cortes["z"] == z]
        df_sig = df_cortes[df_cortes["z"] == z_siguiente]

        matching = rastreador.matchear(df_z, df_sig)

        id_traza_siguiente: Dict[int, int] = {}
        for label_z, label_sig in matching.pares:
            id_traza_siguiente[label_sig] = id_traza_actual[label_z]

        for label_nuevo in matching.nuevos:
            id_traza_siguiente[label_nuevo] = siguiente_id_traza
            siguiente_id_traza += 1

        for _, fila in df_sig.iterrows():
            idx = fila.name
            id_traza_por_fila[idx] = id_traza_siguiente[int(fila["label"])]

        id_traza_actual = id_traza_siguiente

    resultado = df_cortes.copy()
    resultado["id_traza"] = resultado.index.map(id_traza_por_fila).astype(int)
    return resultado


# =============================================================================
# 5. PIPELINE COMPLETO (helper de alto nivel)
# =============================================================================

def pipeline_tracking_z(
    volumen_etiquetado: np.ndarray,
    t: int = 0,
    c: int = 0,
    rastreador: Optional[RastreadorBase] = None,
) -> pd.DataFrame:
    """
    Atajo: de volumen etiquetado (Z, Y, X) a DataFrame con id_traza, en un
    solo llamado.

    Args:
        volumen_etiquetado: ver construir_dataframe_cortes
        t, c: índices de origen a registrar en el DataFrame
        rastreador: ver armar_trazas_z

    Returns:
        DataFrame: t, z, c, label, area, centroide_y, centroide_x, id_traza
    """
    df_cortes = construir_dataframe_cortes(volumen_etiquetado, t=t, c=c)
    return armar_trazas_z(df_cortes, rastreador=rastreador)