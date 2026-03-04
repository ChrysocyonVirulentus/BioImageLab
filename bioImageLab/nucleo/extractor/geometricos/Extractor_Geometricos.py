"""
Extractores de características geométricas de objetos segmentados.

Los extractores geométricos calculan propiedades de forma, tamaño y
posición de objetos previamente segmentados (binarios o etiquetados).
Son independientes del método de segmentación utilizado, operando
únicamente sobre la máscara resultante.

Principio fundamental:
Para cada objeto o región segmentada R, extraer descriptores geométricos
invariantes y métricos que caractericen su morfología espacial.

IMPORTANTE - Separación de responsabilidades:
- Estos métodos NO segmentan (ese rol es de Segmentadores_*.py)
- Asumen que la imagen de entrada ya está segmentada (binaria o etiquetada)
- Trabajan con coordenadas de píxeles (unidades espaciales)
- No aplican calibración física (micrómetros) - eso es rol de calibrador.py
- Para análisis estadísticos de poblaciones, usar Analizador_Poblacional.py

Tipos de descriptores geométricos:
- Posicionales: Ubicación espacial (centroide, caja envolvente)
- Métricos: Tamaño (área, perímetro, volumen proxy)
- Forma: Proporciones (circularidad, elongación, convexidad)
- Estructurales: Relaciones espaciales (vecindad, distancias)

Métodos disponibles:
- Centroides: Centro de masa y variantes ponderadas
- CajaFrontera: Bounding box y propiedades derivadas
- Area: Medidas de tamaño 2D y densidad
- Diametro: Feret diameters y distancias extremas
- Perimetro: Longitud de contorno y suavidad
- Forma: Descriptores de circularidad, elongación, solidez
- Orientacion: Eje principal, ángulo de rotación
- Convexidad: Hull convexo y defectos de convexidad
- Compactacion: Relaciones área-perímetro y densidad
- DistanciasInternas: Transformada de distancia y estadísticos
"""

import numpy as np
import cv2
from typing import Optional, Tuple, List, Dict, Union, Literal
from scipy import ndimage
from skimage import measure, morphology
from skimage.measure import regionprops, find_contours
from skimage.morphology import convex_hull_image
import warnings


class ExtractorGeometrico:
    """
        Clase base para extractores de características geométricas.
        
        Los extractores geométricos operan sobre imágenes segmentadas
        (binarias o etiquetadas) para calcular propiedades de forma.
        
        Conceptos clave:
            - Objeto/Región: Conjunto conectado de píxeles con misma etiqueta
            - Centroide: Centro de masa geométrico
            - Bounding box: Rectángulo mínimo alineado a ejes que contiene el objeto
            - Feret diameter: Distancia máxima entre puntos del contorno
            - Convex hull: Envoltura convexa mínima del objeto
    """
    nombre = "extractor_geometrico_base"
    
    def __call__(self, 
                mascara: np.ndarray,
                etiquetas: Optional[np.ndarray] = None) -> Union[np.ndarray, Dict, List]:
        """
            Extrae características geométricas.
            
            Args:
                mascara: Imagen binaria (True/1 = objeto) o etiquetada (int > 0 = objetos)
                etiquetas: Si mascara es binaria, etiquetas pre-calculadas opcionales
                
            Returns:
                Resultado según extractor específico (array, diccionario o lista)
        """
        raise NotImplementedError("Subclases deben implementar __call__")
    
    def _validar_mascara(self, mascara: np.ndarray):
        """Valida que la máscara sea 2D y contenga objetos."""
        if mascara.ndim != 2:
            raise ValueError(f"Máscara debe ser 2D, tiene {mascara.ndim} dimensiones")
        if mascara.max() == 0:
            raise ValueError("Máscara vacía (sin objetos)")
    
    def _obtener_etiquetas(self, 
                            mascara: np.ndarray, 
                            etiquetas: Optional[np.ndarray] = None,
                            conectividad: int = 2) -> Tuple[np.ndarray, int]:
        """
            Obtiene imagen etiquetada y número de objetos.
            
            Args:
                mascara: Binaria o etiquetada
                etiquetas: Pre-calculadas (opcional)
                conectividad: 1 (4-vecinos) o 2 (8-vecinos)
                
            Returns:
                (etiquetas_array, n_objetos)
        """
        if etiquetas is not None:
            return etiquetas, etiquetas.max()
        
        if mascara.dtype == bool or mascara.max() == 1:
            # Binaria: etiquetar
            struct = np.ones((3, 3)) if conectividad == 2 else np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
            etiquetada, n = ndimage.label(mascara > 0, structure=struct)
            return etiquetada, n
        else:
            # Ya etiquetada
            return mascara.astype(int), mascara.max()
    
    def _pixeles_a_coordenadas(self, 
                                y: np.ndarray, 
                                x: np.ndarray) -> np.ndarray:
        """Convierte índices de píxeles a array de coordenadas (N, 2)."""
        return np.column_stack([y, x])


class Centroides(ExtractorGeometrico):
    """
        Cálculo de centroides geométricos y variantes ponderadas.
        
        El centroide (centro de masa) es el punto promedio de todos los
        píxeles del objeto. Variantes incluyen ponderación por intensidad
        original o por distancia al borde.
        
        Algoritmo (centroide geométrico):
            C_y = (1/N) Σ y_i,  C_x = (1/N) Σ x_i
            
            donde (y_i, x_i) son coordenadas de píxeles del objeto
        
        Variantes:
            - Geométrico: Peso uniforme para todos los píxeles
            - Ponderado: Peso por intensidad de imagen original
            - De distancia: Peso por distancia al borde (centro "profundo")
            - De intensidad inversa: Centróide de "hueso" (medial axis)
        
        Propiedades:
            - Invariante a traslación
            - No invariante a escala (cambia con tamaño) ni rotación (cambia posición)
            - Único para cada objeto conexo
        
        Ventajas:
            - Descriptor posicional robusto (promedio de ruido)
            - Base para tracking de objetos
            - Cálculo computacionalmente eficiente
            - Bien definido matemáticamente
        
        Desventajas:
            - Puede caer fuera del objeto para formas cóncavas
            - No representa forma, solo ubicación
            - Sensible a artefactos de segmentación (píxeles aislados)
            - Para objetos alargados, no coincide con punto "central" perceptual
        
        Usos típicos en microscopía:
            - Tracking de células en time-lapse (centroides como posiciones)
            - Análisis de distribución espacial (patrones de clustering)
            - Registro de imágenes (puntos de referencia)
            - Medición de desplazamiento celular (migración)
            - Análisis de división celular (separación de centroides)
    """
    nombre = "centroides"
    
    def __init__(self,
                tipo: Literal['geometrico', 'ponderado', 'distancia', 'intensidad_inversa'] = 'geometrico',
                img_original: Optional[np.ndarray] = None):
        """
            Args:
                tipo: Tipo de centroide a calcular
                        'geometrico': Peso uniforme (default)
                        'ponderado': Peso por intensidad de img_original
                        'distancia': Peso por distancia al borde (más central)
                        'intensidad_inversa': Centroide de eje medial
                
                img_original: Imagen de intensidad para variantes ponderadas        
        """
        self.tipo = tipo
        self.img_original = img_original
        
        if tipo in ['ponderado', 'intensidad_inversa'] and img_original is None:
            raise ValueError(f"Tipo '{tipo}' requiere img_original")
    
    def __call__(self,
                mascara: np.ndarray,
                etiquetas: Optional[np.ndarray] = None) -> Union[Tuple[float, float], List[Tuple[float, float]], np.ndarray]:
        """
            Calcula centroides de objetos en máscara.
            
            Args:
                mascara: Binaria o etiquetada
                etiquetas: Pre-calculadas (opcional)
                
            Returns:
                Si un objeto: (cy, cx)
                Si múltiples objetos: [(cy1, cx1), (cy2, cx2), ...] o array (n, 2)
        """
        self._validar_mascara(mascara)
        etiquetada, n_objetos = self._obtener_etiquetas(mascara, etiquetas)
        
        if n_objetos == 0:
            return [] if self.tipo != 'geometrico' else (np.nan, np.nan)
        
        centroides = []
        
        for i in range(1, n_objetos + 1):
            mask_obj = (etiquetada == i)
            cy, cx = self._calcular_centroide_objeto(mask_obj)
            centroides.append((cy, cx))
        
        if n_objetos == 1:
            return centroides[0]
        
        return np.array(centroides)  # (n_objetos, 2)
    
    def _calcular_centroide_objeto(self, mascara_obj: np.ndarray) -> Tuple[float, float]:
        """Calcula centroide para un objeto individual."""
        y, x = np.where(mascara_obj)
        
        if len(y) == 0:
            return (np.nan, np.nan)
        
        if self.tipo == 'geometrico':
            pesos = np.ones_like(y, dtype=float)
        
        elif self.tipo == 'ponderado':
            pesos = self.img_original[mascara_obj].astype(float)
        
        elif self.tipo == 'distancia':
            from scipy.ndimage import distance_transform_edt
            dist = distance_transform_edt(mascara_obj)
            pesos = dist[mascara_obj]
        
        else:  # intensidad_inversa
            # Centroide de eje medial (máximos de distancia)
            from scipy.ndimage import distance_transform_edt
            dist = distance_transform_edt(mascara_obj)
            pesos = (dist[mascara_obj] > 0.9 * dist.max()).astype(float)
        
        # Normalizar pesos
        suma_pesos = pesos.sum()
        if suma_pesos == 0:
            return (y.mean(), x.mean())
        
        cy = np.average(y, weights=pesos)
        cx = np.average(x, weights=pesos)
        
        return (cy, cx)
    
    def get_centroide_global(self, 
                                mascara: np.ndarray,
                        qetiquetas: Optional[np.ndarray] = None) -> Tuple[float, float]:
        """
            Calcula centroide de todos los objetos combinados.
            
            Returns:
                (cy, cx) centroide global
        """
        self._validar_mascara(mascara)
        
        if self.tipo != 'geometrico':
            # Para ponderados, calcular sobre máscara combinada
            binaria = mascara > 0 if mascara.dtype != bool else mascara
            return self._calcular_centroide_objeto(binaria)
        
        # Para geométrico, promedio ponderado por área de centroides individuales
        centroides = self(mascara, etiquetas)
        
        if isinstance(centroides, tuple):
            return centroides
        
        # Ponderar por áreas
        etiquetada, n = self._obtener_etiquetas(mascara, etiquetas)
        areas = np.array([np.sum(etiquetada == i) for i in range(1, n + 1)])
        
        cy = np.average(centroides[:, 0], weights=areas)
        cx = np.average(centroides[:, 1], weights=areas)
        
        return (cy, cx)


class CajaFrontera(ExtractorGeometrico):
    """
        Bounding box (caja envolvente) y propiedades derivadas.
        
        Rectángulo mínimo alineado a los ejes que contiene completamente
        el objeto. Variantes incluyen caja mínima rotada y caja convexa.
        
        Algoritmo (axis-aligned):
            y_min = min(y_i), y_max = max(y_i)
            x_min = min(x_i), x_max = max(x_i)
            Ancho = x_max - x_min + 1, Alto = y_max - y_min + 1
        
        Variantes:
            - AABB: Axis-Aligned Bounding Box (ejes cartesianos)
            - OBB: Oriented Bounding Box (eje principal del objeto)
            - MBR: Minimum Bounding Rectangle (cualquier orientación)
        
        Propiedades derivadas:
            - Relación de aspecto: ancho / alto
            - Extensión: (ancho × alto) / área_objeto (ocupación)
            - Densidad: área_objeto / área_caja
        
        Ventajas:
            - Descriptor simple de extensión espacial
            - Base para normalización de objetos (crop, rotación)
            - Cálculo extremadamente rápido (min/max)
            - Útil para indexación espacial (R-trees, quadtrees)
        
        Desventajas:
            - AABB: Pobre para objetos rotados (mucho espacio vacío)
            - No describe forma, solo extensión
            - Sensible a proyecciones (objetos 3D alargados en 2D)
            - Para objetos cóncavos, incluye mucho espacio no-objeto
        
        Usos típicos en microscopía:
            - Crop de objetos individuales para análisis posterior
            - Normalización de tamaño de células para clasificación
            - Detección de objetos alargados (relación aspecto alta)
            - Indexación rápida para colocalización espacial
            - Medición de elongación celular (respuesta a estímulos mecánicos)
            - Análisis de orientación de células (ángulo de OBB)
    """
    nombre = "caja_frontera"
    
    def __init__(self,
                tipo: Literal['aabb', 'obb', 'convex'] = 'aabb',
                return_propiedades: bool = True):
        """
            Args:
                tipo: Tipo de caja envolvente
                    'aabb': Axis-Aligned (ejes cartesianos, default)
                    'obb': Oriented Bounding Box (eje principal del objeto)
                    'convex': Convex Hull (envoltura convexa, no rectangular)
                
                return_propiedades: Si True, devuelve propiedades derivadas
        """
        self.tipo = tipo
        self.return_propiedades = return_propiedades
    
    def __call__(self,
                mascara: np.ndarray,
                etiquetas: Optional[np.ndarray] = None) -> Union[Dict, List[Dict]]:
        """
        Calcula cajas envolventes de objetos.
        
        Args:
            mascara: Binaria o etiquetada
            etiquetas: Pre-calculadas (opcional)
            
        Returns:
            Si un objeto: dict con caja y propiedades
            Si múltiples: lista de dicts
        """
        self._validar_mascara(mascara)
        etiquetada, n_objetos = self._obtener_etiquetas(mascara, etiquetas)
        
        cajas = []
        
        for i in range(1, n_objetos + 1):
            mask_obj = (etiquetada == i)
            caja = self._calcular_caja(mask_obj)
            cajas.append(caja)
        
        if n_objetos == 1:
            return cajas[0]
        
        return cajas
    
    def _calcular_caja(self, mascara_obj: np.ndarray) -> Dict:
        """Calcula caja para un objeto."""
        y, x = np.where(mascara_obj)
        
        if len(y) == 0:
            return self._caja_vacia()
        
        if self.tipo == 'aabb':
            return self._calcular_aabb(y, x, mascara_obj)
        
        elif self.tipo == 'obb':
            return self._calcular_obb(y, x, mascara_obj)
        
        else:  # convex
            return self._calcular_convex_hull(y, x, mascara_obj)
    
    def _calcular_aabb(self, y: np.ndarray, x: np.ndarray, mask: np.ndarray) -> Dict:
        """Axis-Aligned Bounding Box."""
        y_min, y_max = y.min(), y.max()
        x_min, x_max = x.min(), x.max()
        
        caja = {
            'tipo': 'aabb',
            'y_min': int(y_min),
            'y_max': int(y_max),
            'x_min': int(x_min),
            'x_max': int(x_max),
            'ancho': int(x_max - x_min + 1),
            'alto': int(y_max - y_min + 1),
            'centro_y': (y_min + y_max) / 2,
            'centro_x': (x_min + x_max) / 2
        }
        
        if self.return_propiedades:
            area_obj = len(y)
            area_caja = caja['ancho'] * caja['alto']
            caja.update({
                'relacion_aspecto': caja['ancho'] / max(caja['alto'], 1),
                'extension': area_caja / max(area_obj, 1),
                'densidad': area_obj / max(area_caja, 1),
                'area_objeto': area_obj
            })
        
        return caja
    
    def _calcular_obb(self, y: np.ndarray, x: np.ndarray, mask: np.ndarray) -> Dict:
        """Oriented Bounding Box (usando PCA)."""
        coords = np.column_stack([y, x])
        
        # PCA para encontrar ejes principales
        mean = coords.mean(axis=0)
        centered = coords - mean
        
        if len(coords) < 2:
            return self._calcular_aabb(y, x, mask)
        
        cov = np.cov(centered.T)
        eigenvals, eigenvecs = np.linalg.eigh(cov)
        
        # Eje principal (mayor eigenvalor)
        eje_principal = eigenvecs[:, 1]  # Mayor eigenvalor en índice 1
        
        # Proyectar en ejes principales
        proyecciones = centered @ eigenvecs
        
        # Bounds en espacio rotado
        y_min, y_max = proyecciones[:, 1].min(), proyecciones[:, 1].max()
        x_min, x_max = proyecciones[:, 0].min(), proyecciones[:, 0].max()
        
        # Centro en espacio original
        centro_rotado = np.array([(x_min + x_max) / 2, (y_min + y_max) / 2])
        centro = mean + centro_rotado @ eigenvecs.T
        
        # Esquinas de la caja
        esquinas_rot = np.array([
            [x_min, y_min], [x_max, y_min],
            [x_max, y_max], [x_min, y_max]
        ])
        esquinas = mean + esquinas_rot @ eigenvecs.T
        
        caja = {
            'tipo': 'obb',
            'centro_y': centro[0],
            'centro_x': centro[1],
            'eje_principal_y': eje_principal[0],
            'eje_principal_x': eje_principal[1],
            'angulo': np.arctan2(eje_principal[0], eje_principal[1]),  # radianes
            'ancho': x_max - x_min,
            'alto': y_max - y_min,
            'esquinas': esquinas  # (4, 2) array
        }
        
        if self.return_propiedades:
            area_obj = len(y)
            area_caja = caja['ancho'] * caja['alto']
            caja.update({
                'relacion_aspecto': caja['ancho'] / max(caja['alto'], 1),
                'extension': area_caja / max(area_obj, 1),
                'densidad': area_obj / max(area_caja, 1),
                'area_objeto': area_obj
            })
        
        return caja
    
    def _calcular_convex_hull(self, y: np.ndarray, x: np.ndarray, mask: np.ndarray) -> Dict:
        """Convex Hull usando skimage."""
        from skimage.morphology import convex_hull_image
        
        hull = convex_hull_image(mask)
        area_hull = hull.sum()
        area_obj = mask.sum()
        
        # Bounding box del hull
        y_h, x_h = np.where(hull)
        
        caja = {
            'tipo': 'convex',
            'y_min': int(y_h.min()),
            'y_max': int(y_h.max()),
            'x_min': int(x_h.min()),
            'x_max': int(x_h.max()),
            'area_hull': int(area_hull),
            'area_objeto': int(area_obj),
            'solidez': area_obj / max(area_hull, 1),
            'convexidad': area_obj / max(area_hull, 1)
        }
        
        # Defectos de convexidad
        from skimage.measure import find_contours
        contour_obj = find_contours(mask, 0.5)[0] if find_contours(mask, 0.5) else np.array([])
        contour_hull = find_contours(hull, 0.5)[0] if find_contours(hull, 0.5) else np.array([])
        
        caja['perimetro_objeto'] = len(contour_obj)
        caja['perimetro_hull'] = len(contour_hull)
        
        return caja
    
    def _caja_vacia(self) -> Dict:
        """Retorna caja vacía para objeto no válido."""
        return {
            'tipo': self.tipo,
            'y_min': 0, 'y_max': 0, 'x_min': 0, 'x_max': 0,
            'ancho': 0, 'alto': 0, 'vacia': True
        }


class Area(ExtractorGeometrico):
    """
        Medidas de área 2D y densidad espacial.
        
        Calcula área en píxeles, área convexa, área de bounding box,
        y métricas derivadas de ocupación espacial.
        
        Algoritmo:
            Área = Σ 1 para todos los píxeles del objeto
        
        Métricas derivadas:
            - Densidad de ocupación: área / área_caja
            - Porosidad: 1 - (área_objeto / área_total_región)
            - Compactación: área_convexa / área_objeto
        
        Ventajas:
            - Descriptor fundamental de tamaño
            - Invariante a rotación y traslación
            - Robusto a ruido (promedio espacial)
            - Base para normalización de otras medidas
        
        Desventajas:
            - Dependiente de resolución (no físico sin calibración)
            - Para objetos 3D proyectados, área 2D ≠ sección real
            - No distingue formas con misma área
            - Sensible a umbralización de segmentación
        
        Usos típicos en microscopía:
            - Medición de tamaño celular (respuesta a tratamientos)
            - Cuantificación de área de colonias bacterianas
            - Análisis de spreading celular (adhesión)
            - Medición de área de lesiones o regiones de interés
            - Normalización de intensidades (por área)
            - Estimación de número de células (área total / área promedio)
    """
    nombre = "area"
    
    def __init__(self,
                incluir_hole_area: bool = False,
                calcular_densidad: bool = True):
        """
            Args:
                incluir_hole_area: Si True, resta área de agujeros internos
                calcular_densidad: Si True, calcula métricas de ocupación
        """
        self.incluir_hole_area = incluir_hole_area
        self.calcular_densidad = calcular_densidad
    
    def __call__(self,
                mascara: np.ndarray,
                etiquetas: Optional[np.ndarray] = None,
                region_total: Optional[Tuple[int, int, int, int]] = None) -> Union[Dict, List[Dict]]:
        """
            Calcula áreas de objetos.
            
            Args:
                mascara: Binaria o etiquetada
                etiquetas: Pre-calculadas (opcional)
                region_total: (y_min, y_max, x_min, x_max) para calcular densidad regional
                
            Returns:
                Dict o lista de dicts con medidas de área
        """
        self._validar_mascara(mascara)
        etiquetada, n_objetos = self._obtener_etiquetas(mascara, etiquetas)
        
        areas = []
        
        for i in range(1, n_objetos + 1):
            mask_obj = (etiquetada == i)
            area_dict = self._calcular_area_objeto(mask_obj, i)
            areas.append(area_dict)
        
        # Densidad regional si se especifica
        if region_total is not None and self.calcular_densidad:
            y_min, y_max, x_min, x_max = region_total
            area_total_reg = (y_max - y_min) * (x_max - x_min)
            area_ocupada = sum(a['area'] for a in areas)
            
            for a in areas:
                a['densidad_regional'] = area_ocupada / area_total_reg
                a['fraccion_ocupada'] = a['area'] / area_total_reg
        
        if n_objetos == 1:
            return areas[0]
        
        return areas
    
    def _calcular_area_objeto(self, mascara_obj: np.ndarray, id_obj: int) -> Dict:
        """Calcula área para un objeto."""
        area_pixels = int(mascara_obj.sum())
        
        resultado = {
            'id_objeto': id_obj,
            'area_pixels': area_pixels,
            'area': area_pixels  # alias
        }
        
        # Agujeros (holes)
        if self.incluir_hole_area:
            from scipy.ndimage import binary_fill_holes
            rellena = binary_fill_holes(mascara_obj)
            area_huecos = int(rellena.sum() - area_pixels)
            resultado['area_huecos'] = area_huecos
            resultado['area_neta'] = area_pixels - area_huecos
        
        # Bounding box para densidad
        y, x = np.where(mascara_obj)
        if len(y) > 0:
            area_caja = (y.max() - y.min() + 1) * (x.max() - x.min() + 1)
            resultado['area_caja'] = area_caja
            
            if self.calcular_densidad:
                resultado['densidad_caja'] = area_pixels / max(area_caja, 1)
                resultado['extension'] = area_caja / max(area_pixels, 1)
        
        # Convex hull
        try:
            from skimage.morphology import convex_hull_image
            hull = convex_hull_image(mascara_obj)
            area_hull = int(hull.sum())
            resultado['area_convexa'] = area_hull
            resultado['solidez'] = area_pixels / max(area_hull, 1)
        except:
            pass
        
        # Equivalent diameter (círculo con misma área)
        resultado['diametro_equivalente'] = 2 * np.sqrt(area_pixels / np.pi)
        
        return resultado
    
    def get_area_total(self,
                        mascara: np.ndarray,
                        etiquetas: Optional[np.ndarray] = None) -> int:
        """Retorna área total de todos los objetos combinados."""
        self._validar_mascara(mascara)
        binaria = mascara > 0 if mascara.dtype != bool else mascara
        return int(binaria.sum())


class Diametro(ExtractorGeometrico):
    """
        Diámetros de Feret y distancias extremas del objeto.
        
        El diámetro de Feret es la distancia máxima entre dos puntos
        paralelos tangentes al objeto en una dirección dada. Incluye
        diámetro máximo, mínimo, y promedio en todas las direcciones.
        
        Algoritmo (Feret diameter):
            Para ángulo θ:
                Proyectar todos los puntos del objeto en dirección θ:
                p_i = x_i·cosθ + y_i·sinθ
                Feret(θ) = max(p_i) - min(p_i)
        
        Diámetros característicos:
            - Feret máximo: distancia entre puntos más alejados (caliper máximo)
            - Feret mínimo: "grosor" mínimo del objeto
            - Feret medio: promedio en todas las direcciones
            - Martin's diameter: distancia entre puntos de división de área
        
        Ventajas:
            - Describe tamaño en todas las direcciones (anisotropía)
            - Invariante a rotación (el máximo es invariante)
            - Relación Feret_max/Feret_min describe elongación
            - Base para estimación de tamaño de partículas
        
        Desventajas:
            - Costoso calcular en muchas direcciones
            - Para formas complejas, el máximo puede no representar "tamaño típico"
            - Sensible a proyecciones (objetos 3D alargados)
            - Requiere muestreo angular suficiente
        
        Usos típicos en microscopía:
            - Medición de elongación celular (fibroblastos, músculo)
            - Análisis de forma de partículas (circulares vs alargadas)
            - Estimación de tamaño de células en suspensión (Feret medio)
            - Análisis de deformación mecánica (cambio en Feret_max/Feret_min)
            - Clasificación de morfología celular (redonda, alargada, irregular)
    """
    nombre = "diametro"
    
    def __init__(self,
                n_angulos: int = 180,
                calcular_feret: bool = True,
                calcular_martin: bool = False):
        """
        Args:
            n_angulos: Número de direcciones para muestrear Feret (default 180 = 1°)
            calcular_feret: Si True, calcula diámetros de Feret
            calcular_martin: Si True, calcula diámetro de Martin (división área)
        """
        self.n_angulos = n_angulos
        self.calcular_feret = calcular_feret
        self.calcular_martin = calcular_martin
    
    def __call__(self,
                 mascara: np.ndarray,
                 etiquetas: Optional[np.ndarray] = None) -> Union[Dict, List[Dict]]:
        """
        Calcula diámetros de objetos.
        
        Args:
            mascara: Binaria o etiquetada
            etiquetas: Pre-calculadas (opcional)
            
        Returns:
            Dict o lista de dicts con diámetros
        """
        self._validar_mascara(mascara)
        etiquetada, n_objetos = self._obtener_etiquetas(mascara, etiquetas)
        
        diametros = []
        
        for i in range(1, n_objetos + 1):
            mask_obj = (etiquetada == i)
            diam_dict = self._calcular_diametros_objeto(mask_obj, i)
            diametros.append(diam_dict)
        
        if n_objetos == 1:
            return diametros[0]
        
        return diametros
    
    def _calcular_diametros_objeto(self, mascara_obj: np.ndarray, id_obj: int) -> Dict:
        """Calcula diámetros para un objeto."""
        y, x = np.where(mascara_obj)
        
        if len(y) == 0:
            return {'id_objeto': id_obj, 'vacio': True}
        
        resultado = {'id_objeto': id_obj}
        
        # Feret diameters
        if self.calcular_feret:
            feret_dict = self._calcular_feret(y, x)
            resultado.update(feret_dict)
        
        # Martin's diameter
        if self.calcular_martin:
            resultado['martin_diameter'] = self._calcular_martin(mascara_obj)
        
        # Distancia máxima Euclidiana (diámetro geodésico)
        coords = np.column_stack([y, x])
        from scipy.spatial.distance import pdist
        if len(coords) > 1:
            dist_max = pdist(coords, metric='euclidean').max()
            resultado['diametro_maximo'] = dist_max
        
        # Diámetro equivalente (de área)
        area = len(y)
        resultado['diametro_equivalente'] = 2 * np.sqrt(area / np.pi)
        
        return resultado
    
    def _calcular_feret(self, y: np.ndarray, x: np.ndarray) -> Dict:
        """Calcula diámetros de Feret en múltiples direcciones."""
        coords = np.column_stack([y, x])
        
        # Centrar para simplificar
        centro = coords.mean(axis=0)
        centered = coords - centro
        
        angulos = np.linspace(0, np.pi, self.n_angulos, endpoint=False)
        ferets = []
        
        for theta in angulos:
            # Proyección en dirección theta
            u = np.cos(theta)
            v = np.sin(theta)
            proyecciones = centered[:, 0] * u + centered[:, 1] * v
            
            feret = proyecciones.max() - proyecciones.min()
            ferets.append(feret)
        
        ferets = np.array(ferets)
        
        return {
            'feret_maximo': ferets.max(),
            'feret_minimo': ferets.min(),
            'feret_promedio': ferets.mean(),
            'feret_mediano': np.median(ferets),
            'feret_desviacion': ferets.std(),
            'feret_all': ferets,  # Array completo para análisis direccional
            'angulo_feret_max': angulos[np.argmax(ferets)],
            'angulo_feret_min': angulos[np.argmin(ferets)],
            'elongacion_feret': ferets.max() / max(ferets.min(), 1e-10)
        }
    
    def _calcular_martin(self, mascara_obj: np.ndarray) -> float:
        """Calcula diámetro de Martin (división de área en dos)."""
        # Simplificación: usar distancia en dirección del momento principal
        y, x = np.where(mascara_obj)
        if len(y) == 0:
            return 0.0
        
        # Dirección de mayor variación
        coords = np.column_stack([y, x])
        cov = np.cov(coords.T)
        eigenvals, eigenvecs = np.linalg.eigh(cov)
        direccion = eigenvecs[:, 1]  # Mayor eigenvalor
        
        # Proyectar y encontrar punto medio del área
        proyecciones = coords @ direccion
        proyecciones_ordenadas = np.sort(proyecciones)
        
        # Índice de la mediana (divide área en dos)
        n = len(proyecciones_ordenadas)
        if n % 2 == 0:
            mediana = (proyecciones_ordenadas[n//2 - 1] + proyecciones_ordenadas[n//2]) / 2
        else:
            mediana = proyecciones_ordenadas[n//2]
        
        # Distancia entre puntos de corte
        p_min = proyecciones_ordenadas[0]
        p_max = proyecciones_ordenadas[-1]
        
        return p_max - p_min


class Perimetro(ExtractorGeometrico):
    """
    Longitud de perímetro y descriptores de contorno.
    
    Calcula perímetro mediante contorno de píxeles o aproximación
    poligonal, con correcciones para digitalización.
    
    Algoritmos:
        - Píxeles: contar transiciones objeto-fondo (4-conectado)
        - Cadena: código de Freeman, longitud de cadena
        - Poligonal: aproximar contorno con polígono, sumar lados
    
    Correcciones:
        - Factor √2 para diagonales (8-conectado)
        - Corrección de corner counting (Freeman)
        - Aproximación spline para suavidad
    
    Descriptores de contorno:
        - Rugosidad: perímetro / perímetro_convexo
        - Circularidad: 4π·área / perímetro² (1 = círculo perfecto)
        - Indentación: desviaciones del contorno convexo
    
    Ventajas:
        - Base para descriptores de forma (circularidad)
        - Sensibilidad a irregularidades de borde
        - Invariante a rotación y traslación
        - Robusto a escala (relación área-perímetro)
    
    Desventajas:
        - Muy sensible a ruido de segmentación (perímetro diverge con resolución)
        - Para fractales, perímetro depende de escala de medición
        - Digitalización introduce error sistemático
        - No único: diferentes contornos pueden tener mismo perímetro
    
    Usos típicos en microscopía:
        - Cuantificación de rugosidad de membrana (respuesta a tratamientos)
        - Medición de circularidad de núcleos (indicador de salud celular)
        - Análisis de spreading celular (perímetro vs área)
        - Detección de blebbing o protrusiones (aumento súbito de perímetro)
        - Clasificación de morfología (circular vs irregular)
    """
    nombre = "perimetro"
    
    def __init__(self,
                 metodo: Literal['pixel', 'cadena', 'poligonal'] = 'pixel',
                 correccion_diagonal: bool = True,
                 suavizar: bool = False,
                 sigma_suavizado: float = 1.0):
        """
        Args:
            metodo: Método de cálculo
                    'pixel': Conteo de píxeles de borde
                    'cadena': Código de Freeman (contorno 8-conectado)
                    'poligonal': Aproximación poligonal (Ramer-Douglas-Peucker)
            
            correccion_diagonal: Aplicar factor √2 a movimientos diagonales
            suavizar: Suavizar contorno antes de medir (reduce ruido)
            sigma_suavizado: Desviación para suavizado gaussiano
        """
        self.metodo = metodo
        self.correccion_diagonal = correccion_diagonal
        self.suavizar = suavizar
        self.sigma_suavizado = sigma_suavizado
    
    def __call__(self,
                 mascara: np.ndarray,
                 etiquetas: Optional[np.ndarray] = None) -> Union[Dict, List[Dict]]:
        """
        Calcula perímetros de objetos.
        
        Args:
            mascara: Binaria o etiquetada
            etiquetas: Pre-calculadas (opcional)
            
        Returns:
            Dict o lista de dicts con perímetros y descriptores
        """
        self._validar_mascara(mascara)
        etiquetada, n_objetos = self._obtener_etiquetas(mascara, etiquetas)
        
        perimetros = []
        
        for i in range(1, n_objetos + 1):
            mask_obj = (etiquetada == i)
            perim_dict = self._calcular_perimetro_objeto(mask_obj, i)
            perimetros.append(perim_dict)
        
        if n_objetos == 1:
            return perimetros[0]
        
        return perimetros
    
    def _calcular_perimetro_objeto(self, mascara_obj: np.ndarray, id_obj: int) -> Dict:
        """Calcula perímetro para un objeto."""
        resultado = {'id_objeto': id_obj}
        
        # Suavizar si se solicita
        if self.suavizar:
            from skimage.filters import gaussian
            mascara_obj = gaussian(mascara_obj.astype(float), self.sigma_suavizado) > 0.5
        
        if self.metodo == 'pixel':
            perim, contorno = self._perimetro_pixel(mascara_obj)
        elif self.metodo == 'cadena':
            perim, contorno = self._perimetro_cadena(mascara_obj)
        else:  # poligonal
            perim, contorno = self._perimetro_poligonal(mascara_obj)
        
        resultado['perimetro'] = perim
        resultado['n_puntos_contorno'] = len(contorno) if contorno is not None else 0
        
        # Descriptores de forma basados en perímetro
        area = mascara_obj.sum()
        if area > 0:
            # Circularidad (1.0 = círculo perfecto)
            circularidad = 4 * np.pi * area / (perim ** 2) if perim > 0 else 0
            resultado['circularidad'] = circularidad
            
            # Esfericidad (3D proxy)
            resultado['esfericidad'] = circularidad  # Mismo cálculo para 2D
            
            # Relación área-perímetro (compactación)
            resultado['compactacion'] = (perim ** 2) / area if area > 0 else 0
        
        # Perímetro convexo para rugosidad
        try:
            from skimage.morphology import convex_hull_image
            hull = convex_hull_image(mascara_obj)
            perim_hull, _ = self._perimetro_pixel(hull)
            resultado['perimetro_convexo'] = perim_hull
            resultado['rugosidad'] = perim / max(perim_hull, 1e-10)
        except:
            pass
        
        return resultado
    
    def _perimetro_pixel(self, mascara: np.ndarray) -> Tuple[float, np.ndarray]:
        """Perímetro por conteo de píxeles de borde."""
        # Encontrar píxeles de borde: vecinos en fondo
        from scipy.ndimage import binary_erosion
        
        interior = binary_erosion(mascara)
        borde = mascara & ~interior
        
        y, x = np.where(borde)
        
        if len(y) == 0:
            return 0.0, np.array([])
        
        # Contar vecinos en fondo para cada píxel de borde
        vecinos_fondo = 0
        vecinos_diagonal = 0
        
        for dy, dx in [(-1,0), (1,0), (0,-1), (0,1)]:  # 4-conectado
            vecinos = mascara[y+dy, x+dx] if 0 <= (y+dy).min() and (y+dy).max() < mascara.shape[0] and 0 <= (x+dx).min() and (x+dx).max() < mascara.shape[1] else np.zeros_like(y, dtype=bool)
            vecinos_fondo += np.sum(~vecinos)
        
        if self.correccion_diagonal:
            # Aproximación: asumir mitad de movimientos diagonales
            perim = len(y) * (2 - np.sqrt(2)/2)  # Factor de corrección promedio
        else:
            perim = len(y)
        
        return perim, np.column_stack([y, x])
    
    def _perimetro_cadena(self, mascara: np.ndarray) -> Tuple[float, np.ndarray]:
        """Perímetro por código de Freeman."""
        from skimage.measure import find_contours
        
        contornos = find_contours(mascara, 0.5)
        
        if not contornos:
            return 0.0, np.array([])
        
        # Tomar el contorno más largo (exterior)
        contorno = max(contornos, key=len)
        
        # Calcular longitud considerando diagonales
        diffs = np.diff(contorno, axis=0)
        distancias = np.sqrt((diffs ** 2).sum(axis=1))
        perim = distancias.sum()
        
        return perim, contorno
    
    def _perimetro_poligonal(self, mascara: np.ndarray, epsilon: float = 2.0) -> Tuple[float, np.ndarray]:
        """Perímetro por aproximación poligonal."""
        from skimage.measure import find_contours
        from cv2 import approxPolyDP, arcLength
        
        contornos = find_contours(mascara, 0.5)
        
        if not contornos:
            return 0.0, np.array([])
        
        contorno = max(contornos, key=len)
        
        # Aproximación poligonal (Ramer-Douglas-Peucker)
        contorno_cv = contorno.astype(np.float32).reshape(-1, 1, 2)
        perim_cv = arcLength(contorno_cv, True)
        
        # Aproximar
        aprox = approxPolyDP(contorno_cv, epsilon, True)
        perim_aprox = arcLength(aprox, True)
        
        return perim_aprox, aprox.reshape(-1, 2)


class Forma(ExtractorGeometrico):
    """
    Descriptores de forma compuestos (circularidad, elongación, convexidad).
    
    Calcula métricas adimensionales que describen la forma del objeto
    independientemente de tamaño, posición y orientación.
    
    Descriptores:
        - Circularidad: 4π·Área/Perímetro² (1 = círculo, <1 = otras formas)
        - Elongación: Eje mayor / eje menor (1 = circular, >>1 = alargado)
        - Solidez: Área / Área_convexa (1 = convexo, <1 = cóncavo)
        - Convexidad: Perímetro_convexo / Perímetro
        - Rectangularidad: Área / Área_rectángulo_mínimo
    
    Momentos de Hu:
        - 7 invariantes a traslación, escala y rotación
        - Usados para reconocimiento de patrones
    
    Ventajas:
        - Invariantes a transformaciones geométricas básicas
        - Descripción compacta de forma compleja
        - Base para clasificación automática
        - Normalizados (rango típico [0,1] o comparables)
    
    Desventajas:
        - Pérdida de información (muchos objetos pueden tener mismos descriptores)
        - Sensibles a calidad de segmentación
        - No capturan relaciones espaciales entre objetos
        - Para formas muy complejas, descriptores simples son insuficientes
    
    Usos típicos en microscopía:
        - Clasificación de morfología celular (redonda, alargada, estrellada)
        - Detección de apoptosis (circularidad aumenta al hacerse redonda)
        - Análisis de diferenciación (cambios en elongación)
        - Clasificación de tipos celulares (momentos de Hu)
        - Control de calidad de segmentación (formas imposibles)
    """
    nombre = "forma"
    
    def __init__(self,
                 calcular_momentos_hu: bool = False,
                 calcular_fourier: bool = False):
        """
        Args:
            calcular_momentos_hu: Si True, calcula 7 invariantes de Hu
            calcular_fourier: Si True, calcula descriptores de Fourier del contorno
        """
        self.calcular_momentos_hu = calcular_momentos_hu
        self.calcular_fourier = calcular_fourier
    
    def __call__(self,
                 mascara: np.ndarray,
                 etiquetas: Optional[np.ndarray] = None) -> Union[Dict, List[Dict]]:
        """
        Calcula descriptores de forma.
        
        Args:
            mascara: Binaria o etiquetada
            etiquetas: Pre-calculadas (opcional)
            
        Returns:
            Dict o lista de dicts con descriptores de forma
        """
        self._validar_mascara(mascara)
        etiquetada, n_objetos = self._obtener_etiquetas(mascara, etiquetas)
        
        formas = []
        
        for i in range(1, n_objetos + 1):
            mask_obj = (etiquetada == i)
            forma_dict = self._calcular_forma_objeto(mask_obj, i)
            formas.append(forma_dict)
        
        if n_objetos == 1:
            return formas[0]
        
        return formas
    
    def _calcular_forma_objeto(self, mascara_obj: np.ndarray, id_obj: int) -> Dict:
        """Calcula descriptores de forma para un objeto."""
        resultado = {'id_objeto': id_obj}
        
        # Usar regionprops para descriptores estándar
        props = regionprops(mascara_obj.astype(int))
        
        if len(props) == 0:
            return resultado
        
        prop = props[0]
        
        # Descriptores básicos de skimage
        resultado['circularidad'] = 4 * np.pi * prop.area / (prop.perimeter ** 2) if prop.perimeter > 0 else 0
        resultado['elongacion'] = prop.major_axis_length / max(prop.minor_axis_length, 1e-10)
        resultado['solidez'] = prop.solidity
        resultado['excentricidad'] = prop.eccentricity
        resultado['orientacion'] = prop.orientation  # radianes
        
        # Convexidad
        if hasattr(prop, 'convex_area'):
            resultado['convexidad_area'] = prop.area / prop.convex_area
        
        # Momentos de Hu si se solicita
        if self.calcular_momentos_hu:
            momentos = self._calcular_momentos_hu(mascara_obj)
            resultado['momentos_hu'] = momentos
        
        # Descriptores de Fourier si se solicita
        if self.calcular_fourier:
            fourier_desc = self._calcular_descriptores_fourier(mascara_obj)
            resultado['descriptores_fourier'] = fourier_desc
        
        return resultado
    
    def _calcular_momentos_hu(self, mascara: np.ndarray) -> np.ndarray:
        """Calcula 7 invariantes de momentos de Hu."""
        # Momentos raw
        moments = cv2.moments(mascara.astype(np.uint8))
        
        # Momentos de Hu
        hu = cv2.HuMoments(moments).flatten()
        
        # Log transform para manejar rango amplio
        hu_log = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)
        
        return hu_log
    
    def _calcular_descriptores_fourier(self, mascara: np.ndarray, n_descriptores: int = 10) -> np.ndarray:
        """Calcula descriptores de Fourier del contorno."""
        from skimage.measure import find_contours
        
        contornos = find_contours(mascara, 0.5)
        if not contornos:
            return np.zeros(n_descriptores)
        
        contorno = max(contornos, key=len)
        
        # Centrar
        contorno = contorno - contorno.mean(axis=0)
        
        # Representación compleja
        z = contorno[:, 0] + 1j * contorno[:, 1]
        
        # FFT
        fft = np.fft.fft(z)
        
        # Normalizar (invariante a escala, rotación, traslación)
        # Usar magnitudes de coeficientes (invariante a rotación y fase inicial)
        magnitudes = np.abs(fft)
        
        # Normalizar por DC (primer coeficiente)
        if magnitudes[0] > 0:
            magnitudes = magnitudes / magnitudes[0]
        
        # Retornar primeros n descriptores (sin DC que es 1 después de normalizar)
        return magnitudes[1:n_descriptores+1]


class Orientacion(ExtractorGeometrico):
    """
    Eje principal y ángulo de orientación del objeto.
    
    Calcula la dirección predominante del objeto mediante análisis de
    componentes principales (PCA) o momentos de inercia.
    
    Algoritmo (PCA):
        1. Centrar coordenadas de píxeles del objeto
        2. Calcular matriz de covarianza
        3. Eigenvalores/eigenvectores de covarianza
        4. Eje principal = eigenvector de mayor eigenvalor
    
    Propiedades:
        - Orientación: ángulo del eje principal respecto a horizontal
        - Ejes: longitudes de ejes mayor y menor (análogo a elipse)
        - Momento de inercia: distribución espacial de masa
    
    Ventajas:
        - Descriptor de dirección para objetos alargados
        - Base para normalización de rotación
        - Relación eje_mayor/eje_menor describe elongación
        - Invariante a traslación (centrado)
    
    Desventajas:
        - No definido para objetos circulares (eigenvalores similares)
        - Ambigüedad de 180° (dirección vs orientación)
        - Sensible a artefactos de segmentación (colas, protrusiones)
        - Para objetos ramificados, eje principal puede no ser intuitivo
    
    Usos típicos en microscopía:
        - Alineación de células en tejidos (orden orientacional)
        - Análisis de respuesta a campos direccionales (eléctrico, químico)
        - Normalización de orientación para análisis de forma
        - Medición de torque en células rotativas
        - Análisis de textura orientada (fibras, músculo)
    """
    nombre = "orientacion"
    
    def __init__(self,
                 return_elipse: bool = True,
                 normalizar: bool = False):
        """
        Args:
            return_elipse: Si True, devuelve parámetros de elipse equivalente
            normalizar: Si True, normaliza ejes por área (invariante escala)
        """
        self.return_elipse = return_elipse
        self.normalizar = normalizar
    
    def __call__(self,
                 mascara: np.ndarray,
                 etiquetas: Optional[np.ndarray] = None) -> Union[Dict, List[Dict]]:
        """
        Calcula orientación de objetos.
        
        Args:
            mascara: Binaria o etiquetada
            etiquetas: Pre-calculadas (opcional)
            
        Returns:
            Dict o lista de dicts con orientación y ejes
        """
        self._validar_mascara(mascara)
        etiquetada, n_objetos = self._obtener_etiquetas(mascara, etiquetas)
        
        orientaciones = []
        
        for i in range(1, n_objetos + 1):
            mask_obj = (etiquetada == i)
            ori_dict = self._calcular_orientacion_objeto(mask_obj, i)
            orientaciones.append(ori_dict)
        
        if n_objetos == 1:
            return orientaciones[0]
        
        return orientaciones
    
    def _calcular_orientacion_objeto(self, mascara_obj: np.ndarray, id_obj: int) -> Dict:
        """Calcula orientación para un objeto."""
        y, x = np.where(mascara_obj)
        
        if len(y) < 2:
            return {'id_objeto': id_obj, 'orientacion': np.nan, 'definido': False}
        
        coords = np.column_stack([y, x])
        
        # Centrar
        centro = coords.mean(axis=0)
        centered = coords - centro
        
        # PCA
        if len(centered) > 1:
            cov = np.cov(centered.T)
            eigenvals, eigenvecs = np.linalg.eigh(cov)
            
            # Ordenar de mayor a menor
            idx = eigenvals.argsort()[::-1]
            eigenvals = eigenvals[idx]
            eigenvecs = eigenvecs[:, idx]
        else:
            return {'id_objeto': id_obj, 'orientacion': 0, 'definido': False}
        
        # Eje principal
        eje_principal = eigenvecs[:, 0]
        angulo = np.arctan2(eje_principal[0], eje_principal[1])  # Respecto a eje x
        
        resultado = {
            'id_objeto': id_obj,
            'orientacion_rad': angulo,
            'orientacion_deg': np.degrees(angulo),
            'centro_y': centro[0],
            'centro_x': centro[1],
            'eje_mayor': 2 * np.sqrt(eigenvals[0]),  # Longitud ~2σ
            'eje_menor': 2 * np.sqrt(eigenvals[1]) if len(eigenvals) > 1 else 0,
            'eigenvalor_mayor': eigenvals[0],
            'eigenvalor_menor': eigenvals[1] if len(eigenvals) > 1 else 0,
            'definido': eigenvals[0] > eigenvals[1] * 1.1  # Ejes suficientemente distintos
        }
        
        # Elipse equivalente
        if self.return_elipse:
            resultado['elipse_ancho'] = resultado['eje_mayor']
            resultado['elipse_alto'] = resultado['eje_menor']
            resultado['elipse_angulo'] = resultado['orientacion_deg']
        
        # Normalización
        if self.normalizar:
            area = len(y)
            factor = np.sqrt(area)  # √area tiene unidades de longitud
            resultado['eje_mayor_norm'] = resultado['eje_mayor'] / factor
            resultado['eje_menor_norm'] = resultado['eje_menor'] / factor
        
        return resultado


class DistanciasInternas(ExtractorGeometrico):
    """
    Estadísticos de transformada de distancia interna del objeto.
    
    Calcula distribución de distancias desde cada píxel del objeto
    hasta el borde más cercano, proporcionando información de grosor
    y estructura interna.
    
    Métricas:
        - Distancia máxima: radio máximo (centro "profundo")
        - Distancia media: grosor típico
        - Distancia mediana: robusta a outliers
        - Desviación: variabilidad de grosor
        - Moda: grosor más frecuente
    
    Ventajas:
        - Describe grosor interno (no solo contorno)
        - Robustos a forma irregular
        - Base para estimación de volumen 3D desde 2D
        - Detecta heterogeneidad estructural (variación de grosor)
    
    Desventajas:
        - Para objetos planos (láminas), distancias son pequeñas y ruidosas
        - No distingue entre ramas delgadas y cuerpo grueso
        - Sensible a agujeros internos (distancia atraviesa agujeros)
    
    Usos típicos en microscopía:
        - Estimación de diámetro de vasos sanguíneos
        - Análisis de grosor de neuritas (atrafia vs hipertrofia)
        - Medición de diámetro celular (células redondas)
        - Detección de estructuras huecas vs sólidas
        - Análisis de porosidad (distribución de distancias bimodales)
    """
    nombre = "distancias_internas"
    
    def __init__(self,
                 metrica: Literal['euclidean', 'taxicab'] = 'euclidean',
                 calcular_histograma: bool = False,
                 n_bins: int = 20):
        """
        Args:
            metrica: Tipo de distancia ('euclidean' o 'taxicab')
            calcular_histograma: Si True, devuelve distribución de distancias
            n_bins: Número de bins para histograma
        """
        self.metrica = metrica
        self.calcular_histograma = calcular_histograma
        self.n_bins = n_bins
    
    def __call__(self,
                 mascara: np.ndarray,
                 etiquetas: Optional[np.ndarray] = None) -> Union[Dict, List[Dict]]:
        """
        Calcula estadísticos de distancia interna.
        
        Args:
            mascara: Binaria o etiquetada
            etiquetas: Pre-calculadas (opcional)
            
        Returns:
            Dict o lista de dicts con estadísticos de distancia
        """
        self._validar_mascara(mascara)
        etiquetada, n_objetos = self._obtener_etiquetas(mascara, etiquetas)
        
        distancias = []
        
        for i in range(1, n_objetos + 1):
            mask_obj = (etiquetada == i)
            dist_dict = self._calcular_distancias_objeto(mask_obj, i)
            distancias.append(dist_dict)
        
        if n_objetos == 1:
            return distancias[0]
        
        return distancias
    
    def _calcular_distancias_objeto(self, mascara_obj: np.ndarray, id_obj: int) -> Dict:
        """Calcula estadísticos de distancia para un objeto."""
        from scipy.ndimage import distance_transform_edt
        
        # Transformada de distancia
        if self.metrica == 'euclidean':
            dist = distance_transform_edt(mascara_obj)
        else:
            dist = ndimage.distance_transform_cdt(mascara_obj, metric=self.metrica)
        
        # Solo píxeles del objeto
        distancias_obj = dist[mascara_obj]
        
        if len(distancias_obj) == 0:
            return {'id_objeto': id_obj, 'vacio': True}
        
        resultado = {
            'id_objeto': id_obj,
            'distancia_maxima': float(distancias_obj.max()),
            'distancia_media': float(distancias_obj.mean()),
            'distancia_mediana': float(np.median(distancias_obj)),
            'distancia_std': float(distancias_obj.std()),
            'distancia_minima': float(distancias_obj.min()),
            'radio_equivalente': float(distancias_obj.max()),  # Alias común
            'diametro_equivalente': float(2 * distancias_obj.max()),
            'grosor_medio': float(2 * distancias_obj.mean()),  # Diámetro promedio
        }
        
        # Moda (valor más frecuente)
        from scipy.stats import mode
        moda = mode(distancias_obj, keepdims=True)[0][0]
        resultado['distancia_moda'] = float(moda)
        
        # Histograma si se solicita
        if self.calcular_histograma:
            hist, bins = np.histogram(distancias_obj, bins=self.n_bins)
            resultado['histograma'] = hist
            resultado['bins'] = bins
        
        # Centroide ponderado por distancia (punto más "central")
        y, x = np.where(mascara_obj)
        if len(y) > 0:
            cy = np.average(y, weights=dist[mascara_obj])
            cx = np.average(x, weights=dist[mascara_obj])
            resultado['centroide_profundo_y'] = cy
            resultado['centroide_profundo_x'] = cx
        
        return resultado