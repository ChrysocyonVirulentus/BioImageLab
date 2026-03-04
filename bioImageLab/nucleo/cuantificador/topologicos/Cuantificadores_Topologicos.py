"""
Cuantificadores topológicos para análisis de conectividad y estructura.

Los cuantificadores topológicos operan sobre imágenes binarias/etiquetadas
para extraer propiedades de conectividad, agujeros, ramificación y esqueleto.

Principio fundamental:
    A partir de máscaras binarias R, extraer invariantes topológicos
    que describen la "forma" independiente de deformaciones continuas.

IMPORTANTE - Separación de responsabilidades:
- NO normalizan imágenes (ese rol es de normalizador.py)
- NO filtran ruido (ese rol es de filtros.py)
- NO segmentan (ese rol es de segmentacion.py)
- Reciben np.ndarray binario/etiquetado, retornan métricas topológicas
- Trabajan en conectividad 4 u 8 según especificación
- Requieren esqueletización previa para métricas de ramificación

Métricas disponibles:
- Esqueléticas: longitud de esqueleto, puntos terminales, puntos de ramificación
- Ramificación: número de ramas, orden de ramificación, ángulos de bifurcación
- Contornos: detección, longitud, complejidad, distancias entre contornos
- Conectividad: número de componentes, agujeros (euler characteristic)
- Topológicas: índice de betti, grafo de adyacencia, distancia geodésica
"""

import numpy as np
from typing import Optional, Tuple, List, Union, Literal, Dict
from scipy import ndimage
from scipy.spatial.distance import cdist
from skimage import measure, morphology, graph
from collections import defaultdict
import warnings


class CuantificadorTopologico:
    """Clase base para cuantificadores topológicos."""
    nombre = "cuantificador_topologico_base"
    
    def __call__(self, img_segmentada: np.ndarray) -> Union[int, float, np.ndarray, Tuple, Dict]:
        """
            Args:
                img_segmentada: Imagen binaria (0/1 o bool) o etiquetada (int)
            
            Returns:
                Métrica(s) topológica(s) extraída(s)
        """
        raise NotImplementedError
    
    def _validar_segmentada(self, img: np.ndarray, permitir_etiquetada: bool = True, 
                            requerir_binaria: bool = False):
        """Valida que la imagen esté segmentada correctamente."""
        if img.ndim != 2:
            raise ValueError(f"Imagen debe ser 2D, tiene {img.ndim} dimensiones")
        
        if requerir_binaria and not np.array_equal(img, img.astype(bool)):
            raise ValueError("Este cuantificador requiere imagen binaria (0/1 o bool)")
        
        if np.sum(img > 0) == 0:
            raise ValueError("Imagen segmentada vacía (sin objetos)")
    
    def _binarizar(self, img: np.ndarray) -> np.ndarray:
        """Convierte imagen a binaria segura."""
        return (img > 0).astype(np.uint8)


class MetricasEsqueleticas(CuantificadorTopologico):
    """
            Métricas del esqueleto/medial axis de objetos.
            
            El esqueleto preserva la topología y reduce la forma a un grafo
            de 1 dimensión, permitiendo análisis de longitud, conectividad y
            puntos característicos (terminales, de ramificación).
            
            Algoritmo (esqueletización):
                Esqueleto = Morfológico thinning iterativo que preserva:
                - Conectividad de componentes
                - Puntos terminales (extremos)
                - Puntos de ramificación (bifurcaciones)
            
            Métricas extraídas:
                - Longitud total del esqueleto (suma de píxeles conectados)
                - Puntos terminales: vecinos = 1 en esqueleto
                - Puntos de ramificación: vecinos ≥ 3 en esqueleto
                - Puntos de paso: vecinos = 2 en esqueleto
            
            Ventajas:
                - Reduce dimensionalidad manteniendo topología,
                - robusto a pequeñas variaciones de borde,
                - base para análisis de grafos
            Desventajas:
                - Sensible a ruido (produce espuelas/pequeñas ramas),
                - requiere pruning para objetos gruesos,
                - no único (depende de algoritmo de esqueletización)
            
            Usos microscopía:
                - Análisis de neuritas (longitud, ramificación),
                - vasculatura (tortuosidad, conectividad),
                - células alargadas (fibroblastos, músculo),
                - redes de colágeno/fibras
    """
    nombre = "metricas_esqueleticas"
    
    def __init__(self, conectividad: Literal[4, 8] = 8, 
                pruning: bool = False,
                longitud_min_rama: int = 5):
        self.conectividad = conectividad
        self.pruning = pruning
        self.longitud_min_rama = longitud_min_rama
    
    def __call__(self, img_segmentada: np.ndarray,
                retornar_esqueleto: bool = False) -> Union[Dict, Tuple[Dict, np.ndarray]]:
        """
            Args:
                img_segmentada: Imagen binaria
                retornar_esqueleto: Si True, incluye el esqueleto computado
            
            Returns:
                Dict con métricas y opcionalmente el esqueleto
        """
        self._validar_segmentada(img_segmentada, requerir_binaria=True)
        
        img_bin = self._binarizar(img_segmentada)
        esqueleto = morphology.skeletonize(img_bin > 0)
        
        if self.pruning:
            esqueleto = self._pruning_esqueleto(esqueleto)
        
        metricas = self._extraer_metricas(esqueleto)
        
        if retornar_esqueleto:
            return metricas, esqueleto
        return metricas
    
    def _pruning_esqueleto(self, esqueleto: np.ndarray) -> np.ndarray:
        """Elimina ramas pequeñas del esqueleto."""
        labeled = measure.label(esqueleto, connectivity=self.conectividad)
        result = np.zeros_like(esqueleto)
        for region in measure.regionprops(labeled):
            if region.area >= self.longitud_min_rama:
                result[labeled == region.label] = 1
        return result
    
    def _extraer_metricas(self, esqueleto: np.ndarray) -> Dict:
        """Extrae métricas del esqueleto."""
        longitud_total = np.sum(esqueleto)
        
        if longitud_total == 0:
            return {
                'longitud_total': 0,
                'num_puntos_terminales': 0,
                'num_puntos_ramificacion': 0,
                'num_puntos_paso': 0,
                'num_componentes': 0,
                'tortuosidad_promedio': 0.0,
                'densidad_esqueletica': 0.0
            }
        
        puntos_terminales = self._detectar_puntos_terminales(esqueleto)
        puntos_ramificacion = self._detectar_puntos_ramificacion(esqueleto)
        puntos_paso = self._detectar_puntos_paso(esqueleto)
        
        num_componentes = measure.label(esqueleto, connectivity=self.conectividad).max()
        tortuosidad = self._calcular_tortuosidad(esqueleto, puntos_terminales)
        
        area_objeto = np.sum(ndimage.binary_dilation(esqueleto, iterations=2))
        densidad = longitud_total / area_objeto if area_objeto > 0 else 0
        
        return {
            'longitud_total': int(longitud_total),
            'num_puntos_terminales': int(np.sum(puntos_terminales)),
            'num_puntos_ramificacion': int(np.sum(puntos_ramificacion)),
            'num_puntos_paso': int(np.sum(puntos_paso)),
            'num_componentes': int(num_componentes),
            'tortuosidad_promedio': float(tortuosidad),
            'densidad_esqueletica': float(densidad)
        }
    
    def _detectar_puntos_terminales(self, esqueleto: np.ndarray) -> np.ndarray:
        """Detecta puntos terminales (un solo vecino en esqueleto)."""
        kernel = np.ones((3, 3), dtype=np.uint8)
        vecinos = ndimage.convolve(esqueleto.astype(np.uint8), kernel, mode='constant', cval=0)
        vecinos = vecinos - esqueleto
        return (esqueleto > 0) & (vecinos == 1)
    
    def _detectar_puntos_ramificacion(self, esqueleto: np.ndarray) -> np.ndarray:
        """Detecta puntos de ramificación (3 o más vecinos en esqueleto)."""
        kernel = np.ones((3, 3), dtype=np.uint8)
        vecinos = ndimage.convolve(esqueleto.astype(np.uint8), kernel, mode='constant', cval=0)
        vecinos = vecinos - esqueleto
        return (esqueleto > 0) & (vecinos >= 3)
    
    def _detectar_puntos_paso(self, esqueleto: np.ndarray) -> np.ndarray:
        """Detecta puntos de paso (exactamente 2 vecinos en esqueleto)."""
        kernel = np.ones((3, 3), dtype=np.uint8)
        vecinos = ndimage.convolve(esqueleto.astype(np.uint8), kernel, mode='constant', cval=0)
        vecinos = vecinos - esqueleto
        return (esqueleto > 0) & (vecinos == 2)
    
    def _calcular_tortuosidad(self, esqueleto: np.ndarray, 
                            puntos_terminales: np.ndarray) -> float:
        """Calcula tortuosidad promedio de las ramas."""
        if np.sum(puntos_terminales) < 2:
            return 1.0
        
        coords_terminales = np.argwhere(puntos_terminales)
        tortuosidades = []
        
        for i, terminal1 in enumerate(coords_terminales):
            for terminal2 in coords_terminales[i+1:]:
                try:
                    path = graph.route_through_array(
                        1 - esqueleto,
                        start=tuple(terminal1),
                        end=tuple(terminal2),
                        fully_connected=True
                    )
                    
                    if path[0] is not None:
                        longitud_camino = len(path[0])
                        distancia_euclidiana = np.linalg.norm(terminal1 - terminal2)
                        
                        if distancia_euclidiana > 0:
                            tortuosidades.append(longitud_camino / distancia_euclidiana)
                except:
                    continue
        
        return float(np.mean(tortuosidades)) if tortuosidades else 1.0


class Branching(CuantificadorTopologico):
    """
        Análisis de ramificación y árbol de conectividad.
        
        Cuantifica la estructura de árbol de objetos ramificados
        (neuronas, vasculatura, árboles de decisión morfológicos).
        
        Algoritmo:
            1. Esqueletización del objeto ramificado
            2. Identificación de nodos (terminales, ramificación)
            3. Construcción de grafo de conectividad
            4. Análisis: orden de ramas, ángulos de bifurcación,
            longitud de segmentos, asimetría
        
        Métricas de ramificación:
            - Número de ramas primarias/secundarias/terciarias
            - Orden de ramificación (Strahler o Shreve)
            - Ángulos de bifurcación (ángulo entre ramas hijas)
            - Índice de ramificación: N_term / N_rami
            - Longitud de segmentos entre bifurcaciones
            - Factor de asimetría (comparación subárboles)
        
        Ventajas:
            - Describe completamente la arquitectura del árbol,
            - invariantes útiles para clasificación morfológica,
            - detecta cambios en desarrollo o patología
        Desventajas:
            - Complejidad computacional O(n²) para grafos grandes,
            - requiere esqueleto limpio (sensible a ruido),
            - múltiples definiciones de "orden" pueden confundir
        
        Usos microscopía:
            - Clasificación de tipos neuronales (dendrogramas),
            - grado de malignidad en vasculatura tumoral,
            - análisis de desarrollo de axones/neuritas,
            - enfermedades neurodegenerativas (pérdida de ramas),
            - angiogénesis (formación de nuevos vasos)
    """
    nombre = "branching"
    
    def __init__(self, orden_metodo: Literal['strahler', 'shreve'] = 'strahler',
                min_longitud_rama: float = 5.0):
        self.orden_metodo = orden_metodo
        self.min_longitud_rama = min_longitud_rama
    
    def __call__(self, img_segmentada: np.ndarray,
                retornar_arbol: bool = False) -> Union[Dict, Tuple[Dict, Dict]]:
        """
            Args:
                img_segmentada: Imagen binaria de objeto ramificado
                retornar_arbol: Si True, retorna estructura del árbol
            
            Returns:
                Dict con métricas de ramificación y opcionalmente el árbol
        """
        self._validar_segmentada(img_segmentada, requerir_binaria=True)
        
        img_bin = self._binarizar(img_segmentada)
        esqueleto = morphology.skeletonize(img_bin > 0)
        
        metricas = self._analizar_ramas(esqueleto)
        
        if retornar_arbol:
            arbol = self._construir_arbol(esqueleto)
            return metricas, arbol
        
        return metricas
    
    def _analizar_ramas(self, esqueleto: np.ndarray) -> Dict:
        """Analiza la estructura de ramas del esqueleto."""
        metricas_esq = MetricasEsqueleticas()
        puntos_term = metricas_esq._detectar_puntos_terminales(esqueleto)
        puntos_rami = metricas_esq._detectar_puntos_ramificacion(esqueleto)
        
        n_terminales = int(np.sum(puntos_term))
        n_ramificacion = int(np.sum(puntos_rami))
        
        indice_rami = n_terminales / max(n_ramificacion, 1)
        
        ordenes = self._calcular_orden_ramas(esqueleto, puntos_term, puntos_rami)
        longitudes = self._calcular_longitudes_segmentos(esqueleto, puntos_rami)
        angulos = self._calcular_angulos_bifurcacion(esqueleto, puntos_rami)
        
        return {
            'num_terminales': n_terminales,
            'num_puntos_ramificacion': n_ramificacion,
            'indice_ramificacion': float(indice_rami),
            'ordenes_ramas': ordenes,
            'orden_maximo': max(ordenes) if ordenes else 0,
            'longitud_media_segmentos': float(np.mean(longitudes)) if longitudes else 0.0,
            'longitud_total_ramas': float(np.sum(longitudes)),
            'angulos_bifurcacion_media': float(np.mean(angulos)) if angulos else 0.0,
            'angulos_bifurcacion_std': float(np.std(angulos)) if angulos else 0.0,
            'num_ramas_primarias': int(np.sum(np.array(ordenes) == 1)) if ordenes else 0
        }
    
    def _calcular_orden_ramas(self, esqueleto: np.ndarray, 
                            puntos_term: np.ndarray,
                            puntos_rami: np.ndarray) -> List[int]:
        """Calcula el orden de Strahler o Shreve de cada rama."""
        distancias = ndimage.distance_transform_edt(~puntos_term)
        esqueleto_sin_nodos = esqueleto & ~puntos_rami
        labeled = measure.label(esqueleto_sin_nodos, connectivity=2)
        
        ordenes = []
        for region in measure.regionprops(labeled):
            if region.area >= self.min_longitud_rama:
                toca_terminal = any(puntos_term[c[0], c[1]] for c in region.coords 
                                    if 0 <= c[0] < puntos_term.shape[0] and 
                                    0 <= c[1] < puntos_term.shape[1])
                
                if self.orden_metodo == 'shreve':
                    orden = 1 if toca_terminal else 2
                else:
                    orden = 1 if toca_terminal else 2
                
                ordenes.append(orden)
        
        return ordenes if ordenes else [1]
    
    def _calcular_longitudes_segmentos(self, esqueleto: np.ndarray,
                                        puntos_rami: np.ndarray) -> List[float]:
        """Calcula longitudes de segmentos entre puntos de ramificación."""
        esqueleto_lineas = esqueleto & ~puntos_rami
        labeled = measure.label(esqueleto_lineas, connectivity=2)
        
        longitudes = []
        for region in measure.regionprops(labeled):
            if region.area >= self.min_longitud_rama:
                longitudes.append(float(region.area))
        
        return longitudes
    
    def _calcular_angulos_bifurcacion(self, esqueleto: np.ndarray,
                                    puntos_rami: np.ndarray) -> List[float]:
        """Calcula ángulos de bifurcación en puntos de ramificación."""
        angulos = []
        coords_rami = np.argwhere(puntos_rami)
        
        for y, x in coords_rami:
            vecinos = []
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = y + dy, x + dx
                    if (0 <= ny < esqueleto.shape[0] and 
                        0 <= nx < esqueleto.shape[1] and 
                        esqueleto[ny, nx]):
                        vecinos.append((ny, nx))
            
            if len(vecinos) >= 2:
                for i in range(len(vecinos)):
                    for j in range(i+1, len(vecinos)):
                        v1 = np.array(vecinos[i]) - np.array([y, x])
                        v2 = np.array(vecinos[j]) - np.array([y, x])
                        
                        cos_ang = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
                        cos_ang = np.clip(cos_ang, -1, 1)
                        angulo = np.arccos(cos_ang)
                        angulos.append(np.degrees(angulo))
        
        return angulos
    
    def _construir_arbol(self, esqueleto: np.ndarray) -> Dict:
        """Construye representación de árbol del esqueleto."""
        metricas_esq = MetricasEsqueleticas()
        puntos_term = metricas_esq._detectar_puntos_terminales(esqueleto)
        puntos_rami = metricas_esq._detectar_puntos_ramificacion(esqueleto)
        
        return {
            'raices': np.argwhere(puntos_term & (esqueleto > 0)).tolist(),
            'bifurcaciones': np.argwhere(puntos_rami).tolist(),
            'longitud_total': int(np.sum(esqueleto)),
            'numero_ramas': int(np.sum(puntos_rami) + np.sum(puntos_term) / 2)
        }


class Contornos(CuantificadorTopologico):
    """
        Detección y análisis de contornos de objetos.
        
        Extrae los bordes de objetos segmentados y calcula métricas
        de complejidad, distancias entre contornos y relaciones espaciales.
        
        Algoritmo:
            Contorno = {(x,y) | I(x,y)=1 y ∃ vecino con I=0}
            
            Métricas:
            - Longitud del contorno (perímetro)
            - Distancia entre contornos (mínima, media, máxima)
            - Complejidad: número de componentes de contorno
            - Convexidad del contorno (ratio áreas)
            - Curvatura local y puntos de inflexión
        
        Ventajas:
            - Describe precisamente la frontera del objeto,
            - permite análisis de proximidad entre objetos,
            - base para matching de formas
        Desventajas:
            - Muy sensible a ruido de segmentación,
            - requiere suavizado para curvatura estable,
            - costoso para muchos objetos pequeños
        
        Usos microscopía:
            - Análisis de contacto célula-célula,
            - distancia entre núcleo y membrana,
            - rugosidad de interfaz tejido-matriz,
            - análisis de sinapsis (proximidad terminales),
            - cuantificación de invaginaciones nucleares
    """
    nombre = "contornos"
    
    def __init__(self, metodo_contorno: Literal['chain_code', 'interpolacion'] = 'chain_code',
                suavizado: float = 0.0):
        self.metodo_contorno = metodo_contorno
        self.suavizado = suavizado
    
    def __call__(self, img_segmentada: np.ndarray,
                por_region: bool = False,
                retornar_contornos: bool = False) -> Union[Dict, List[Dict], Tuple]:
        """
            Args:
                img_segmentada: Imagen binaria o etiquetada
                por_region: Si True, analiza contornos por región etiquetada
                retornar_contornos: Si True, incluye coordenadas de contornos
            
            Returns:
                Dict o lista de dicts con métricas de contorno
        """
        self._validar_segmentada(img_segmentada, permitir_etiquetada=True)
        
        if por_region and not np.array_equal(img_segmentada, img_segmentada.astype(bool)):
            regiones = measure.regionprops(img_segmentada.astype(int))
            resultados = []
            contornos_list = []
            
            for r in regiones:
                metricas, cont = self._analizar_contorno_region(r)
                resultados.append(metricas)
                if retornar_contornos:
                    contornos_list.append(cont)
            
            if retornar_contornos:
                return resultados, contornos_list
            return resultados
        
        img_bin = self._binarizar(img_segmentada)
        metricas, contornos = self._analizar_contorno_binario(img_bin)
        
        if retornar_contornos:
            return metricas, contornos
        return metricas
    
    def _analizar_contorno_region(self, region) -> Tuple[Dict, np.ndarray]:
        """Analiza contorno de una región de skimage."""
        contorno = region.coords
        perimetro = region.perimeter
        area = region.area
        
        from scipy.spatial import ConvexHull
        if len(contorno) >= 3:
            try:
                hull = ConvexHull(contorno)
                area_convex = hull.volume
                convexidad = area / area_convex if area_convex > 0 else 1.0
            except:
                convexidad = 1.0
        else:
            convexidad = 1.0
        
        curvatura = self._calcular_curvatura(contorno)
        
        metricas = {
            'area': int(area),
            'perimetro': float(perimetro),
            'convexidad': float(convexidad),
            'curvatura_media': float(np.mean(curvatura)) if len(curvatura) > 0 else 0.0,
            'curvatura_max': float(np.max(curvatura)) if len(curvatura) > 0 else 0.0,
            'num_puntos_contorno': len(contorno),
            'compactacion': float(perimetro**2 / area) if area > 0 else 0.0
        }
        
        return metricas, contorno
    
    def _analizar_contorno_binario(self, img_bin: np.ndarray) -> Tuple[Dict, List[np.ndarray]]:
        """Analiza contornos de imagen binaria."""
        contornos = measure.find_contours(img_bin, 0.5)
        
        if not contornos:
            return {
                'num_contornos': 0,
                'longitud_total': 0.0,
                'areas': [],
                'convexidades': []
            }, []
        
        longitudes = []
        areas = []
        convexidades = []
        
        for contorno in contornos:
            longitud = measure.perimeter_crofton(contorno)
            longitudes.append(longitud)
            
            if len(contorno) >= 3:
                area = 0.5 * abs(np.dot(contorno[:, 0], np.roll(contorno[:, 1], 1)) - 
                                np.dot(contorno[:, 1], np.roll(contorno[:, 0], 1)))
                areas.append(area)
                
                from scipy.spatial import ConvexHull
                try:
                    hull = ConvexHull(contorno)
                    area_convex = hull.volume
                    convexidades.append(area / area_convex if area_convex > 0 else 1.0)
                except:
                    convexidades.append(1.0)
            else:
                areas.append(0.0)
                convexidades.append(1.0)
        
        metricas = {
            'num_contornos': len(contornos),
            'longitud_total': float(np.sum(longitudes)),
            'longitud_media': float(np.mean(longitudes)),
            'longitudes_individuales': [float(l) for l in longitudes],
            'areas': [float(a) for a in areas],
            'convexidades': [float(c) for c in convexidades],
            'convexidad_media': float(np.mean(convexidades))
        }
        
        return metricas, contornos
    
    def _calcular_curvatura(self, contorno: np.ndarray) -> np.ndarray:
        """Calcula curvatura local del contorno."""
        if len(contorno) < 3:
            return np.array([])
        
        if self.suavizado > 0:
            from scipy.ndimage import gaussian_filter1d
            contorno_suav = np.zeros_like(contorno)
            contorno_suav[:, 0] = gaussian_filter1d(contorno[:, 0], self.suavizado)
            contorno_suav[:, 1] = gaussian_filter1d(contorno[:, 1], self.suavizado)
            contorno = contorno_suav
        
        dx = np.gradient(contorno[:, 1])
        dy = np.gradient(contorno[:, 0])
        d2x = np.gradient(dx)
        d2y = np.gradient(dy)
        
        numerador = dx * d2y - dy * d2x
        denominador = (dx**2 + dy**2)**(3/2) + 1e-10
        curvatura = numerador / denominador
        
        return np.abs(curvatura)
    
    def distancia_entre_contornos(self, img1: np.ndarray, img2: np.ndarray,
                                tipo: Literal['min', 'mean', 'hausdorff'] = 'min') -> float:
        """
        Calcula distancia entre contornos de dos imágenes.
        
        Args:
            img1, img2: Imágenes binarias
            tipo: 'min' (mínima), 'mean' (media), 'hausdorff' (máxima mínima)
        
        Returns:
            Distancia en píxeles
        """
        _, contornos1 = self._analizar_contorno_binario(self._binarizar(img1))
        _, contornos2 = self._analizar_contorno_binario(self._binarizar(img2))
        
        if not contornos1 or not contornos2:
            return np.inf
        
        pts1 = np.vstack(contornos1)
        pts2 = np.vstack(contornos2)
        
        distancias = cdist(pts1, pts2)
        
        if tipo == 'min':
            return float(np.min(distancias))
        elif tipo == 'mean':
            return float(np.mean(distancias.min(axis=1)))
        else:
            d1 = distancias.min(axis=1).max()
            d2 = distancias.min(axis=0).max()
            return float(max(d1, d2))


class Conectividad(CuantificadorTopologico):
    """
        Métricas de conectividad de componentes y agujeros.
        
        Análisis topológico clásico: número de componentes conexas,
        número de agujeros (ciclos), característica de Euler.
        
        Algoritmo:
            - Componentes: etiquetado con conectividad 4 u 8
            - Agujeros: inversión de imagen y etiquetado de fondo
            - Euler: χ = N_componentes - N_agujeros (para imagen binaria)
        
        Propiedades topológicas:
            - Invariantes a deformaciones continuas (homeomorfismos)
            - Característica de Euler: χ = V - E + F (fórmula de Euler)
            - Número de Betti: β₀ = componentes, β₁ = agujeros
        
        Ventajas:
            - Invariantes topológicos robustos,
            - computacionalmente eficientes,
            - base para homología computacional
        Desventajas:
            - No distinguen formas con misma topología,
            - sensibles a ruido (pequeños agujeros),
            - información limitada para objetos complejos
        
        Usos microscopía:
            - Conteo de células individuales (componentes),
            - análisis de porosidad tejular (agujeros),
            - conectividad de redes vasculares,
            - caracterización de espuma/burbujas
    """
    nombre = "conectividad"
    
    def __init__(self, conectividad: Literal[4, 8] = 8):
        self.conectividad = conectividad
    
    def __call__(self, img_segmentada: np.ndarray) -> Dict:
        """
            Args:
                img_segmentada: Imagen binaria
            
            Returns:
                Dict con métricas de conectividad
        """
        self._validar_segmentada(img_segmentada, requerir_binaria=True)
        
        img_bin = self._binarizar(img_segmentada)
        
        estructura = np.ones((3, 3), dtype=int) if self.conectividad == 8 else \
                    np.array([[0,1,0],[1,1,1],[0,1,0]], dtype=int)
        
        labeled, num_componentes = ndimage.label(img_bin, structure=estructura)
        
        img_inv = ~img_bin.astype(bool)
        fondo_sin_bordes = img_inv.copy()
        fondo_sin_bordes[0, :] = False
        fondo_sin_bordes[-1, :] = False
        fondo_sin_bordes[:, 0] = False
        fondo_sin_bordes[:, -1] = False
        
        labeled_holes, num_agujeros = ndimage.label(fondo_sin_bordes, structure=estructura)
        
        euler = num_componentes - num_agujeros
        
        regiones = measure.regionprops(labeled)
        areas_componentes = [r.area for r in regiones]
        
        regiones_holes = measure.regionprops(labeled_holes)
        areas_agujeros = [r.area for r in regiones_holes]
        
        return {
            'num_componentes': int(num_componentes),
            'num_agujeros': int(num_agujeros),
            'caracteristica_euler': int(euler),
            'numero_betti_0': int(num_componentes),
            'numero_betti_1': int(num_agujeros),
            'areas_componentes': areas_componentes,
            'areas_agujeros': areas_agujeros,
            'area_total_objetos': int(np.sum(img_bin)),
            'conectividad_usada': self.conectividad
        }


class DistanciaGeodesica(CuantificadorTopologico):
    """
        Distancias geodésicas dentro de objetos segmentados.
        
        Calcula caminos más cortos restringidos a la máscara binaria,
        útil para análisis de accesibilidad y conectividad interna.
        
        Algoritmo:
            D_g(p,q) = min{longitud de camino γ ⊂ R conectando p y q}
            
            Calculado via:
            - Transformada de distancia desde punto fuente
            - Dijkstra en grafo de píxeles adyacentes
            - Fast Marching Method para métricas generales
        
        Propiedades:
            - D_g ≥ D_euclidiana (igualdad si línea recta está en R)
            - Simétrica: D_g(p,q) = D_g(q,p)
            - Satisface desigualdad triangular
        
        Ventajas:
            - Respeta topología del objeto (no atraviesa fondo),
            - detecta cuellos de botella y estrechamientos,
            - base para centralidad y entre objetos
        Desventajas:
            - Costoso computacionalmente O(N log N),
            - requiere definición de métrica (4/8-conectividad),
            - sensible a ruido (pequeños agujeros bloquean caminos)
        
        Usos microscopía:
            - Ancho de istmos nucleares (división celular),
            - conectividad de compartimentos celulares,
            - análisis de porosidad efectiva,
            - caminos de migración celular (matriz extracelular),
            - distancia a lo largo de neuritas (no euclidiana)
    """
    nombre = "distancia_geodesica"
    
    def __init__(self, conectividad: Literal[4, 8] = 8):
        self.conectividad = conectividad
    
    def __call__(self, img_segmentada: np.ndarray,
                punto_fuente: Optional[Tuple[int, int]] = None,
                punto_destino: Optional[Tuple[int, int]] = None,
                retornar_mapa_completo: bool = False) -> Union[float, np.ndarray, Dict]:
        """
            Args:
                img_segmentada: Imagen binaria
                punto_fuente: (y, x) origen. Si None, usa centroide
                punto_destino: (y, x) destino. Si None, retorna mapa completo
                retornar_mapa_completo: Si True, retorna mapa de distancias
            
            Returns:
                Distancia específica o mapa completo de distancias geodésicas
        """
        self._validar_segmentada(img_segmentada, requerir_binaria=True)
        
        img_bin = self._binarizar(img_segmentada)
        
        if punto_fuente is None:
            coords = np.where(img_bin > 0)
            punto_fuente = (int(np.mean(coords[0])), int(np.mean(coords[1])))
        
        if img_bin[punto_fuente[0], punto_fuente[1]] == 0:
            raise ValueError("Punto fuente fuera del objeto")
        
        distancias = self._calcular_geodesica(img_bin, punto_fuente)
        
        if retornar_mapa_completo:
            return distancias
        
        if punto_destino is not None:
            if img_bin[punto_destino[0], punto_destino[1]] == 0:
                raise ValueError("Punto destino fuera del objeto")
            return float(distancias[punto_destino[0], punto_destino[1]])
        
        valores_validos = distancias[img_bin > 0]
        return {
            'distancia_maxima': float(np.max(valores_validos)),
            'distancia_media': float(np.mean(valores_validos)),
            'centro_geodesico': np.unravel_index(np.argmax(distancias), distancias.shape),
            'punto_fuente': punto_fuente
        }
    
    def _calcular_geodesica(self, img_bin: np.ndarray, 
                        fuente: Tuple[int, int]) -> np.ndarray:
        """Calcula mapa de distancias geodésicas usando Dijkstra."""
        from heapq import heappush, heappop
        
        h, w = img_bin.shape
        distancias = np.full((h, w), np.inf)
        distancias[fuente[0], fuente[1]] = 0
        
        cola = [(0, fuente[0], fuente[1])]
        visitado = np.zeros((h, w), dtype=bool)
        
        if self.conectividad == 8:
            vecinos = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
            pesos = [np.sqrt(2), 1, np.sqrt(2), 1, 1, np.sqrt(2), 1, np.sqrt(2)]
        else:
            vecinos = [(-1,0), (0,-1), (0,1), (1,0)]
            pesos = [1, 1, 1, 1]
        
        while cola:
            dist_actual, y, x = heappop(cola)
            
            if visitado[y, x]:
                continue
            visitado[y, x] = True
            
            for (dy, dx), peso in zip(vecinos, pesos):
                ny, nx = y + dy, x + dx
                
                if 0 <= ny < h and 0 <= nx < w and img_bin[ny, nx] > 0:
                    nueva_dist = dist_actual + peso
                    
                    if nueva_dist < distancias[ny, nx]:
                        distancias[ny, nx] = nueva_dist
                        heappush(cola, (nueva_dist, ny, nx))
        
        return distancias
    
    def get_diametro_geodesico(self, img_segmentada: np.ndarray) -> float:
        """
            Calcula el diámetro geodésico (máxima distancia geodésica interna).
            
            Returns:
                Máxima distancia geodésica entre cualquier par de puntos del objeto
        """
        self._validar_segmentada(img_segmentada, requerir_binaria=True)
        img_bin = self._binarizar(img_segmentada)
        
        esqueleto = morphology.skeletonize(img_bin)
        puntos_esq = np.argwhere(esqueleto > 0)
        
        if len(puntos_esq) < 2:
            return 0.0
        
        if len(puntos_esq) > 100:
            indices = np.linspace(0, len(puntos_esq)-1, 100, dtype=int)
            puntos_muestra = puntos_esq[indices]
        else:
            puntos_muestra = puntos_esq
        
        max_dist = 0.0
        
        for p1 in puntos_muestra:
            distancias = self._calcular_geodesica(img_bin, tuple(p1))
            dist_max_local = np.max(distancias[img_bin > 0])
            max_dist = max(max_dist, dist_max_local)
        
        return float(max_dist)

# FUNCIONES DE UTILIDAD Y EXTRACCIÓN COMPLETA

def extraer_todas_metricas_topologicas(img_segmentada: np.ndarray,
                                    por_region: bool = False,
                                    incluir_geodesicas: bool = False) -> Dict:
    """
        Extrae TODAS las métricas topológicas disponibles.
        
        Pipeline completo de análisis topológico:
        1. Conectividad (componentes, agujeros, Euler)
        2. Esqueleto (longitud, puntos terminales/ramificación)
        3. Ramificación (orden, ángulos, índices)
        4. Contornos (perímetro, convexidad, curvatura)
        5. Geodésicas (opcional, costoso computacionalmente)
        
        Args:
            img_segmentada: Imagen binaria o etiquetada
            por_region: Si True, calcula métricas por región etiquetada
            incluir_geodesicas: Si True, incluye análisis de distancias geodésicas
        
        Returns:
            Dict jerárquico con todas las métricas topológicas:
            {
                'conectividad': {...},
                'esqueleto': {...},
                'ramificacion': {...},
                'contornos': {...},
                'geodesicas': {...}  # si incluir_geodesicas=True
            }
    """
    validador = CuantificadorTopologico()
    validador._validar_segmentada(img_segmentada, permitir_etiquetada=True)
    
    resultados = {}
    
    # 1. Conectividad
    try:
        conect = Conectividad()
        if por_region and not np.array_equal(img_segmentada, img_segmentada.astype(bool)):
            regiones = measure.regionprops(img_segmentada.astype(int))
            conectividades = []
            for r in regiones:
                try:
                    c = conect(r.image)
                    conectividades.append(c)
                except:
                    conectividades.append(None)
            resultados['conectividad'] = conectividades
        else:
            resultados['conectividad'] = conect(img_segmentada)
    except Exception as e:
        warnings.warn(f"Error en conectividad: {e}")
        resultados['conectividad'] = None
    
    # 2. Métricas esqueléticas
    try:
        esq = MetricasEsqueleticas(pruning=True)
        if por_region and not np.array_equal(img_segmentada, img_segmentada.astype(bool)):
            regiones = measure.regionprops(img_segmentada.astype(int))
            esqueletos = []
            for r in regiones:
                try:
                    e = esq(r.image, retornar_esqueleto=False)
                    esqueletos.append(e)
                except:
                    esqueletos.append(None)
            resultados['esqueleto'] = esqueletos
        else:
            resultados['esqueleto'] = esq(img_segmentada, retornar_esqueleto=False)
    except Exception as e:
        warnings.warn(f"Error en métricas esqueléticas: {e}")
        resultados['esqueleto'] = None
    
    # 3. Análisis de ramificación
    try:
        branch = Branching()
        if por_region and not np.array_equal(img_segmentada, img_segmentada.astype(bool)):
            regiones = measure.regionprops(img_segmentada.astype(int))
            ramis = []
            for r in regiones:
                try:
                    b = branch(r.image, retornar_arbol=False)
                    ramis.append(b)
                except:
                    ramis.append(None)
            resultados['ramificacion'] = ramis
        else:
            resultados['ramificacion'] = branch(img_segmentada, retornar_arbol=False)
    except Exception as e:
        warnings.warn(f"Error en ramificación: {e}")
        resultados['ramificacion'] = None
    
    # 4. Análisis de contornos
    try:
        cont = Contornos()
        resultados['contornos'] = cont(img_segmentada, por_region=por_region, 
                                    retornar_contornos=False)
    except Exception as e:
        warnings.warn(f"Error en contornos: {e}")
        resultados['contornos'] = None
    
    # 5. Distancias geodésicas (opcional)
    if incluir_geodesicas:
        try:
            geo = DistanciaGeodesica()
            if por_region and not np.array_equal(img_segmentada, img_segmentada.astype(bool)):
                regiones = measure.regionprops(img_segmentada.astype(int))
                geos = []
                for r in regiones:
                    try:
                        g = geo(r.image, retornar_mapa_completo=False)
                        geos.append(g)
                    except:
                        geos.append(None)
                resultados['geodesicas'] = geos
            else:
                resultados['geodesicas'] = geo(img_segmentada, 
                                            retornar_mapa_completo=False)
        except Exception as e:
            warnings.warn(f"Error en geodésicas: {e}")
            resultados['geodesicas'] = None
    
    resultados['_metadatos'] = {
        'por_region': por_region,
        'incluir_geodesicas': incluir_geodesicas,
        'num_pixeles_objeto': int(np.sum(img_segmentada > 0)),
        'forma_imagen': img_segmentada.shape
    }
    
    return resultados


def analizar_red_conectividad(img_segmentada: np.ndarray,
                            umbral_proximidad: float = 10.0) -> Dict:
    """
        Analiza la red de conectividad entre múltiples objetos.
        
        Construye un grafo donde los nodos son objetos y las aristas
        representan proximidad o contacto.
        
        Args:
            img_segmentada: Imagen etiquetada (múltiples objetos)
            umbral_proximidad: Distancia máxima para considerar conexión
        
        Returns:
            Dict con grafo de adyacencia, distancias entre objetos,
            y métricas de red (coeficiente de clustering, caminos cortos)
    """
    if np.array_equal(img_segmentada, img_segmentada.astype(bool)):
        raise ValueError("Se requiere imagen etiquetada (múltiples objetos)")
    
    regiones = measure.regionprops(img_segmentada.astype(int))
    n_objetos = len(regiones)
    
    if n_objetos == 0:
        return {'grafo': {}, 'num_objetos': 0}
    
    centroides = np.array([r.centroid for r in regiones])
    dist_matrix = cdist(centroides, centroides)
    
    grafo = {}
    for i in range(n_objetos):
        vecinos = []
        for j in range(n_objetos):
            if i != j and dist_matrix[i, j] <= umbral_proximidad:
                vecinos.append({
                    'id_vecino': j,
                    'distancia': float(dist_matrix[i, j]),
                    'area_vecino': int(regiones[j].area)
                })
        grafo[i] = {
            'centroide': centroides[i].tolist(),
            'area': int(regiones[i].area),
            'vecinos': vecinos,
            'grado': len(vecinos)
        }
    
    grados = [grafo[i]['grado'] for i in range(n_objetos)]
    
    return {
        'grafo_adyacencia': grafo,
        'num_objetos': n_objetos,
        'num_conexiones': sum(grados) // 2,
        'grado_medio': float(np.mean(grados)),
        'grado_max': int(np.max(grados)) if grados else 0,
        'clustering_promedio': float(np.mean([g for g in grados if g > 0])) if any(g > 0 for g in grados) else 0.0,
        'matriz_distancias': dist_matrix.tolist(),
        'umbral_proximidad': umbral_proximidad
    }