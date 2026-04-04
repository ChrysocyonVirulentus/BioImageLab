"""
Cuantificadores morfométricos para análisis de formas y geometría.

Los cuantificadores morfométricos operan sobre imágenes binarias/etiquetadas
para extraer propiedades geométricas de regiones segmentadas.

Principio fundamental:
    A partir de máscaras binarias R = {(x,y) | f(x,y) ∈ objeto},
    extraer métricas invariantes o características discriminantes.

IMPORTANTE - Separación de responsabilidades:
- NO normalizan imágenes (ese rol es de normalizador.py)
- NO filtran ruido (ese rol es de filtros.py)
- NO segmentan (ese rol es de segmentacion.py)
- Reciben np.ndarray binario/etiquetado, retornan métricas escalares o arrays
- Trabajan en float64 para precisión, retornan unidades en píxeles o normalizadas

Métricas disponibles:
- Geométricas básicas: área, perímetro, centroide, caja frontera
- Forma: circularidad, excentricidad, compacidad, convexidad
- Escala: diámetro equivalente, longitud mayor/menor eje
- Orientación: ángulo principal (momentos de inercia)
- Intensidad de forma: momentos de Hu, descriptores de Fourier
"""

import numpy as np
from typing import Optional, Tuple, List, Union, Literal
from scipy import ndimage
from scipy.spatial.distance import cdist
from skimage import measure
import warnings
from ...gestorLab.Registro_Metodos import registrar_en

class CuantificadorMorfometrico:
    """Clase base para cuantificadores morfométricos."""
    nombre = "cuantificador_morfometrico_base"
    
    def __call__(self, img_segmentada: np.ndarray) -> Union[float, np.ndarray, Tuple]:
        """
            Args:
                img_segmentada: Imagen binaria (0/1 o bool) o etiquetada (int)
            
            Returns:
                Métrica(s) morfométrica(s) extraída(s)
        """
        raise NotImplementedError
    
    def _validar_segmentada(self, img: np.ndarray, permitir_etiquetada: bool = True):
        """Valida que la imagen esté segmentada correctamente."""
        if img.ndim != 2:
            raise ValueError(f"Imagen debe ser 2D, tiene {img.ndim} dimensiones")
        
        if not permitir_etiquetada and not np.array_equal(img, img.astype(bool)):
            raise ValueError("Este cuantificador requiere imagen binaria (no etiquetada)")
        
        if np.sum(img > 0) == 0:
            raise ValueError("Imagen segmentada vacía (sin objetos)")
    
    def _get_region_props(self, img: np.ndarray) -> List:
        """Obtiene propiedades de regiones etiquetadas."""
        return measure.regionprops(img.astype(int))

@registrar_en("cuantificacion")
class Area(CuantificadorMorfometrico):
    """
            Área de regiones segmentadas en píxeles.
            
            Métrica fundamental para cuantificación de tamaño celular,
            cobertura tisular, o crecimiento de colonias.
            
            Algoritmo:
                A = Σ_x Σ_y I(x,y) donde I es máscara binaria
                Para etiquetadas: A_i = count(R_i) para cada región i
            
            Propiedades:
                - Invariante a traslación y rotación
                - Sensible a escala (calibración requerida)
                - Aditiva: A_total = Σ A_i
            
            Ventajas:
                - Intuitiva, robusta al ruido pequeño,
                - base para índices de proliferación
            Desventajas:
                - No distingue formas, sensible a sobre-segmentación,
                - unidades dependen de magnificación
            
            Usos microscopía:
                - Tamaño celular (hipertrofia, atrofia),
                - área de tejido viable vs necrosis,
                - cuantificación de colonias bacterianas,
                - análisis de wound healing assays
    """
    nombre = "area"
    
    def __call__(self, img_segmentada: np.ndarray, 
                por_region: bool = False) -> Union[int, np.ndarray]:
        """
            Args:
                img_segmentada: Imagen binaria o etiquetada
                por_region: Si True, retorna área por región etiquetada
            
            Returns:
                Área total (int) o array de áreas por región
        """
        self._validar_segmentada(img_segmentada, permitir_etiquetada=True)
        
        if por_region and not np.array_equal(img_segmentada, img_segmentada.astype(bool)):
            regiones = self._get_region_props(img_segmentada)
            return np.array([r.area for r in regiones])
        
        return int(np.sum(img_segmentada > 0))

@registrar_en("cuantificacion")
class Perimetro(CuantificadorMorfometrico):
    """
            Perímetro de regiones segmentadas.
            
            Longitud del contorno de objetos, útil para cuantificar
            rugosidad de membranas o interacción célula-matriz.
            
            Algoritmo:
                Método de Crofton (mejor que conteo de píxeles de borde):
                P = Σ (π/4) · (n_diag + n_ortho · √2) 
                donde n son píxeles de contorno en direcciones diagonales/ortogonales
            
            Propiedades:
                - Sensible a resolución y suavizado de bordes
                - No aditivo: P_total ≠ Σ P_i (bordes compartidos)
                - Relacionado con área vía isoperimetric inequality
            
            Ventajas:
                - Detecta cambios en morfología de membrana,
                - base para circularidad
            Desventajas:
                - Muy sensible a ruido de segmentación (escalado ~1/resolución),
                - subestima en bordes diagonales si no se corrige
            
            Usos microscopía:
                - Rugosidad de membrana celular (metástasis),
                - interacción célula-sustrato (spreading),
                - análisis de neuritas (longitud procesos),
                - cuantificación de invaginaciones nucleares
    """
    nombre = "perimetro"
    
    def __call__(self, img_segmentada: np.ndarray,
                por_region: bool = False,
                metodo: Literal['crofton', 'pixel'] = 'crofton') -> Union[float, np.ndarray]:
        """
            Args:
                img_segmentada: Imagen binaria o etiquetada
                por_region: Si True, retorna perímetro por región
                metodo: 'crofton' (preciso) o 'pixel' (simple)
            
            Returns:
                Perímetro total o array de perímetros por región
        """
        self._validar_segmentada(img_segmentada, permitir_etiquetada=True)
        
        if por_region:
            regiones = self._get_region_props(img_segmentada)
            return np.array([r.perimeter for r in regiones])
        
        if metodo == 'crofton':
            contornos = measure.find_contours(img_segmentada > 0, 0.5)
            perimetro = sum(measure.perimeter_crofton(c) for c in contornos)
        else:
            borde = ndimage.binary_dilation(img_segmentada > 0) ^ (img_segmentada > 0)
            perimetro = np.sum(borde)
        
        return float(perimetro)

@registrar_en("cuantificacion")
class Centroide(CuantificadorMorfometrico):
    """
            Centro geométrico (centro de masa) de regiones.
            
            Punto de equilibrio de la máscara binaria, útil para
            tracking de células y análisis de distribución espacial.
            
            Algoritmo:
                c_x = (1/A) Σ_x Σ_y x · I(x,y)
                c_y = (1/A) Σ_x Σ_y y · I(x,y)
            
            Propiedades:
                - Invariante a traslación, rotación, escala
                - Único para regiones convexas
                - Puede caer fuera de región no-convexa
            
            Ventajas:
                - Robusto, único por objeto,
                - base para análisis de vecindad
            Desventajas:
                - Puede no representar "centro biológico",
                - sensible a artefactos de segmentación (outliers)
            
            Usos microscopía:
                - Tracking de células en time-lapse,
                - análisis de patrones espaciales (CSR),
                - nucleación de clusters,
                - alineación de estructuras
    """
    nombre = "centroide"
    
    def __call__(self, img_segmentada: np.ndarray,
                por_region: bool = True) -> Union[Tuple[float, float], np.ndarray]:
        """
            Args:
                img_segmentada: Imagen binaria o etiquetada
                por_region: Si True, retorna centroides por región
            
            Returns:
                (cy, cx) único o array shape (N, 2) de centroides
        """
        self._validar_segmentada(img_segmentada, permitir_etiquetada=True)
        
        if por_region:
            regiones = self._get_region_props(img_segmentada)
            return np.array([r.centroid for r in regiones])
        
        coords = np.where(img_segmentada > 0)
        cy = np.mean(coords[0])
        cx = np.mean(coords[1])
        return (float(cy), float(cx))

@registrar_en("cuantificacion")
class CajaFrontera(CuantificadorMorfometrico):
    """
            Bounding box (caja envolvente) de regiones.
            
            Rectángulo mínimo alineado a ejes que contiene la región.
            Útil para recortes ROI y análisis de elongación.
            
            Algoritmo:
                bbox = (min_row, min_col, max_row, max_col)
                ancho = max_col - min_col
                alto = max_row - min_row
            
            Propiedades:
                - No invariante a rotación (usar min_area_rect para eso)
                - Computacionalmente eficiente
                - Útil para índices de elongación
            
            Ventajas:
                - Rápido, útil para ROI,
                - base para aspect ratio
            Desventajas:
                - No rota con objeto (sobreestima objetos rotados),
                - no información de forma interna
            
            Usos microscopía:
                - Recorte de células individuales,
                - análisis de elongación (shear stress),
                - filtrado por tamaño mínimo/máximo,
                - normalización de posición
    """
    nombre = "caja_frontera"
    
    def __call__(self, img_segmentada: np.ndarray,
                por_region: bool = True) -> Union[Tuple, List[Tuple]]:
        """
            Args:
                img_segmentada: Imagen binaria o etiquetada
                por_region: Si True, retorna bbox por región
            
            Returns:
                (min_r, min_c, max_r, max_c) o lista de tuplas
        """
        self._validar_segmentada(img_segmentada, permitir_etiquetada=True)
        
        if por_region:
            regiones = self._get_region_props(img_segmentada)
            return [r.bbox for r in regiones]
        
        coords = np.where(img_segmentada > 0)
        return (int(np.min(coords[0])), int(np.min(coords[1])),
                int(np.max(coords[0])), int(np.max(coords[1])))
    
    def get_dimensiones(self, bbox: Tuple) -> Tuple[int, int]:
        """Extrae alto y ancho de un bbox."""
        min_r, min_c, max_r, max_c = bbox
        return (max_r - min_r, max_c - min_c)

@registrar_en("cuantificacion")
class DiametroEquivalente(CuantificadorMorfometrico):
    """
            Diámetro de círculo con igual área que la región.
            
            Métrica de tamaño invariante a forma, útil para
            comparaciones entre objetos de morfología diferente.
            
            Algoritmo:
                D_eq = 2 · √(A/π)
            
            Propiedades:
                - Invariante a forma (mismo área = mismo D_eq)
                - Escalado lineal con tamaño característico
                - Relacionado con volumen en 3D (asumiendo esfera)
            
            Ventajas:
                - Normaliza comparaciones entre formas,
                - intuitivo (tamaño "equivalente")
            Desventajas:
                - Pierde información de forma,
                - asume convexidad implícita
            
            Usos microscopía:
                - Tamaño de células irregulares,
                - cuantificación de agregados plaquetarios,
                - comparación entre tipos celulares,
                - estimación de volumen celular
    """
    nombre = "diametro_equivalente"
    
    def __call__(self, img_segmentada: np.ndarray,
                por_region: bool = False) -> Union[float, np.ndarray]:
        """
            Args:
                img_segmentada: Imagen binaria o etiquetada
                por_region: Si True, retorna diámetro por región
            
            Returns:
                Diámetro equivalente en píxeles
        """
        self._validar_segmentada(img_segmentada, permitir_etiquetada=True)
        
        if por_region:
            regiones = self._get_region_props(img_segmentada)
            return np.array([r.equivalent_diameter for r in regiones])
        
        area = np.sum(img_segmentada > 0)
        return 2 * np.sqrt(area / np.pi)

@registrar_en("cuantificacion")
class Excentricidad(CuantificadorMorfometrico):
    """
            Excentricidad de la elipse equivalente.
            
            Métrica de elongación basada en momentos de inercia.
            0 = círculo, →1 = segmento de línea.
            
            Algoritmo (momentos centralizados):
                μ_pq = Σ_x Σ_y (x-x̄)^p (y-ȳ)^q I(x,y)
                
                λ₁, λ₂ = eigenvalores de matriz de covarianza:
                [μ₂₀  μ₁₁]
                [μ₁₁  μ₀₂]
                
                e = √(1 - (λ₂/λ₁)) donde λ₁ ≥ λ₂
            
            Propiedades:
                - Invariante a traslación, rotación, escala
                - 0 ≤ e < 1 (elipse degenerada en límite)
                - Relacionado con ratio de ejes
            
            Ventajas:
                - Robusto, bien definido matemáticamente,
                - sensible a elongación
            Desventajas:
                - Requiere momento de segundo orden (sensible a ruido),
                - no distingue formas con misma elipse equivalente
            
            Usos microscopía:
                - Polarización celular (migración),
                - alineación de fibras,
                - distinción morfología epitelial vs mesenquimal,
                - análisis de deformación mecánica
    """
    nombre = "excentricidad"
    
    def __call__(self, img_segmentada: np.ndarray,
                por_region: bool = False) -> Union[float, np.ndarray]:
        """
            Args:
                img_segmentada: Imagen binaria o etiquetada
                por_region: Si True, retorna excentricidad por región
            
            Returns:
                Excentricidad en [0, 1)
        """
        self._validar_segmentada(img_segmentada, permitir_etiquetada=True)
        
        if por_region:
            regiones = self._get_region_props(img_segmentada)
            return np.array([r.eccentricity for r in regiones])
        
        coords = np.where(img_segmentada > 0)
        if len(coords[0]) == 0:
            return 0.0
        
        y, x = coords[0], coords[1]
        cy, cx = np.mean(y), np.mean(x)
        
        mu20 = np.sum((x - cx)**2)
        mu02 = np.sum((y - cy)**2)
        mu11 = np.sum((x - cx) * (y - cy))
        
        matriz = np.array([[mu20, mu11], [mu11, mu02]])
        eigenvals = np.linalg.eigvalsh(matriz)
        eigenvals = np.sort(eigenvals)[::-1]
        
        if eigenvals[0] == 0:
            return 0.0
        
        return np.sqrt(1 - (eigenvals[1] / eigenvals[0]))

@registrar_en("cuantificacion")
class Circularidad(CuantificadorMorfometrico):
    """
            Circularidad (o forma) de regiones.
            
            Mide qué tan circular es un objeto. Máximo (1) para círculo perfecto.
            También conocida como "roundness" o "compactness" en algunos contextos.
            
            Algoritmo (índice de circularidad de Haralick):
                C = 4π · A / P²
            
            Propiedades:
                - C = 1 para círculo perfecto
                - C < 1 para otras formas (cuadrado: π/4 ≈ 0.785)
                - Invariante a traslación, rotación, escala
                - Sensible a rugosidad de borde (P aumenta)
            
            Ventajas:
                - Intuitiva, fácil de interpretar,
                - buena discriminación círculo vs elipse
            Desventajas:
                - Muy sensible a ruido de perímetro,
                - no distingue formas convexa vs cóncava
            
            Usos microscopía:
                - Redondez nuclear (cáncer: nucleos irregulares),
                - cuantificación de apoptóticos (redondez),
                - distinción morfología celular,
                - filtrado de artefactos de segmentación
    """
    nombre = "circularidad"
    
    def __call__(self, img_segmentada: np.ndarray,
                por_region: bool = False,
                suavizar: bool = True) -> Union[float, np.ndarray]:
        """
            Args:
                img_segmentada: Imagen binaria o etiquetada
                por_region: Si True, retorna circularidad por región
                suavizar: Si True, aplica suavizado ligero antes de perímetro
            
            Returns:
                Circularidad en (0, 1]
        """
        self._validar_segmentada(img_segmentada, permitir_etiquetada=True)
        
        if por_region:
            regiones = self._get_region_props(img_segmentada)
            circularidades = []
            for r in regiones:
                area = r.area
                perimetro = r.perimeter
                if perimetro == 0:
                    circularidades.append(0.0)
                else:
                    circularidades.append(4 * np.pi * area / (perimetro ** 2))
            return np.array(circularidades)
        
        img_bin = (img_segmentada > 0).astype(np.uint8)
        
        if suavizar:
            img_bin = ndimage.gaussian_filter(img_bin.astype(float), sigma=0.5) > 0.5
        
        area = np.sum(img_bin)
        perimetro = Perimetro()(img_bin, metodo='crofton')
        
        if perimetro == 0:
            return 0.0
        
        return 4 * np.pi * area / (perimetro ** 2)

@registrar_en("cuantificacion")
class Compactacion(CuantificadorMorfometrico):
    """
            Compacidad: ratio entre perímetro y área.
            
            Métrica alternativa a circularidad, más sensible a
            irregularidades de borde. Mínima para círculo.
            
            Algoritmo:
                K = P² / A  (inverso de circularidad escalado)
                o normalizada: K_norm = P / (2√(πA))
            
            Propiedades:
                - K ≥ 4π (igualdad para círculo)
                - Crece con rugosidad y elongación
                - Invariante a escala si se normaliza
            
            Ventajas:
                - Más sensible a rugosidad que circularidad,
                - usada en análisis de fractales
            Desventajas:
                - Unidades dependen de normalización,
                - menos intuitiva que circularidad
            
            Usos microscopía:
                - Rugosidad de membrana (metástasis),
                - complejidad de bordes nucleares,
                - análisis de dendritas (ramificación),
                - textura de contornos celulares
    """
    nombre = "compactacion"
    
    def __call__(self, img_segmentada: np.ndarray,
                por_region: bool = False,
                normalizada: bool = True) -> Union[float, np.ndarray]:
        """
            Args:
                img_segmentada: Imagen binaria o etiquetada
                por_region: Si True, retorna compacidad por región
                normalizada: Si True, retorna P/(2√(πA)), si no P²/A
            
            Returns:
                Compacidad (≥ 1 si normalizada, ≥ 4π si no)
        """
        self._validar_segmentada(img_segmentada, permitir_etiquetada=True)
        
        if por_region:
            regiones = self._get_region_props(img_segmentada)
            compactaciones = []
            for r in regiones:
                area = r.area
                perimetro = r.perimeter
                if area == 0:
                    compactaciones.append(np.inf)
                elif normalizada:
                    compactaciones.append(perimetro / (2 * np.sqrt(np.pi * area)))
                else:
                    compactaciones.append((perimetro ** 2) / area)
            return np.array(compactaciones)
        
        area = Area()(img_segmentada)
        perimetro = Perimetro()(img_segmentada, metodo='crofton')
        
        if area == 0:
            return np.inf
        
        if normalizada:
            return perimetro / (2 * np.sqrt(np.pi * area))
        return (perimetro ** 2) / area

@registrar_en("cuantificacion")
class Orientacion(CuantificadorMorfometrico):
    """
            Orientación principal de regiones (ángulo del eje mayor).
            
            Ángulo del eje de máxima inercia respecto al eje horizontal.
            Útil para analisar alineación celular y polaridad tisular.
            
            Algoritmo (momentos de segundo orden):
                θ = 0.5 · atan2(2μ₁₁, μ₂₀ - μ₀₂)
            
            donde:
                μ_pq = momentos centralizados
                θ ∈ [-π/2, π/2] (ortogonalidad de ejes)
            
            Propiedades:
                - Invariante a traslación y escala
                - Ambigüedad de 180° (ejes no orientados)
                - Sensible a forma alargada
            
            Ventajas:
                - Robusto para formas elongadas,
                - base para análisis de alineación
            Desventajas:
                - Indefinido para círculos perfectos (isotrópicos),
                - sensible a ruido en formas compactas
            
            Usos microscopía:
                - Alineación de fibroblastos (wound healing),
                - polaridad de células epiteliales,
                - orientación de fibras de colágeno,
                - análisis de campo direccional
    """
    nombre = "orientacion"
    
    def __call__(self, img_segmentada: np.ndarray,
                por_region: bool = False,
                modo: Literal['radians', 'degrees'] = 'degrees') -> Union[float, np.ndarray]:
        """
            Args:
                img_segmentada: Imagen binaria o etiquetada
                por_region: Si True, retorna orientación por región
                modo: 'radians' o 'degrees'
            
            Returns:
                Ángulo(s) en grados [-90, 90] o radianes [-π/2, π/2]
        """
        self._validar_segmentada(img_segmentada, permitir_etiquetada=True)
        
        if por_region:
            regiones = self._get_region_props(img_segmentada)
            angulos = []
            for r in regiones:
                angulo = r.orientation
                if modo == 'degrees':
                    angulo = np.degrees(angulo)
                angulos.append(angulo)
            return np.array(angulos)
        
        coords = np.where(img_segmentada > 0)
        if len(coords[0]) == 0:
            return 0.0
        
        y, x = coords[0], coords[1]
        cy, cx = np.mean(y), np.mean(x)
        
        mu20 = np.sum((x - cx)**2)
        mu02 = np.sum((y - cy)**2)
        mu11 = np.sum((x - cx) * (y - cy))
        
        if mu20 == mu02 and mu11 == 0:
            return 0.0
        
        angulo = 0.5 * np.arctan2(2 * mu11, mu20 - mu02)
        
        if modo == 'degrees':
            angulo = np.degrees(angulo)
        
        return float(angulo)
    
    def get_ejes_principales(self, img_segmentada: np.ndarray,
                            por_region: bool = False) -> Union[Tuple, List[Tuple]]:
        """
            Retorna longitudes de ejes mayor y menor (elipse equivalente).
            
            Returns:
                (eje_mayor, eje_menor) en píxeles
        """
        self._validar_segmentada(img_segmentada, permitir_etiquetada=True)
        
        if por_region:
            regiones = self._get_region_props(img_segmentada)
            return [(r.major_axis_length, r.minor_axis_length) for r in regiones]
        
        coords = np.where(img_segmentada > 0)
        if len(coords[0]) == 0:
            return (0.0, 0.0)
        
        y, x = coords[0], coords[1]
        cy, cx = np.mean(y), np.mean(x)
        
        mu20 = np.sum((x - cx)**2)
        mu02 = np.sum((y - cy)**2)
        mu11 = np.sum((x - cx) * (y - cy))
        
        matriz = np.array([[mu20, mu11], [mu11, mu02]])
        eigenvals = np.linalg.eigvalsh(matriz)
        eigenvals = np.sort(eigenvals)[::-1]
        
        eje_mayor = 4 * np.sqrt(eigenvals[0] / len(y))
        eje_menor = 4 * np.sqrt(eigenvals[1] / len(y)) if eigenvals[1] > 0 else 0
        
        return (float(eje_mayor), float(eje_menor))

@registrar_en("cuantificacion")
class Convexidad(CuantificadorMorfometrico):
    """
            Convexidad: ratio entre área y área del convex hull.
            
            Mide qué tan "llena" está una forma. Útil para detectar
            invaginaciones y procesos de extensión celular.
            
            Algoritmo:
                Conv = A_objeto / A_convex_hull
            
            Propiedades:
                - 0 < Conv ≤ 1 (igualdad para objetos convexos)
                - Invariante a traslación, rotación, escala
                - Sensible a indentaciones profundas
            
            Ventajas:
                - Detecta cóncavidades (nuclear indentations),
                - robusto a pequeñas irregularidades
            Desventajas:
                - Computacionalmente costoso (convex hull),
                - no distingue número de invaginaciones
            
            Usos microscopía:
                - Indentaciones nucleares (cáncer),
                - extensión de pseudópodos (fagocitosis),
                - análisis de clustering celular,
                - forma de glándulas/ductos
    """
    nombre = "convexidad"
    
    def __call__(self, img_segmentada: np.ndarray,
                por_region: bool = False) -> Union[float, np.ndarray]:
        """
            Args:
                img_segmentada: Imagen binaria o etiquetada
                por_region: Si True, retorna convexidad por región
            
            Returns:
                Convexidad en (0, 1]
        """
        self._validar_segmentada(img_segmentada, permitir_etiquetada=True)
        
        if por_region:
            regiones = self._get_region_props(img_segmentada)
            return np.array([r.solidity for r in regiones])
        
        img_bin = (img_segmentada > 0).astype(np.uint8)
        contornos = measure.find_contours(img_bin, 0.5)
        
        if not contornos:
            return 0.0
        
        contorno = max(contornos, key=len)
        
        from scipy.spatial import ConvexHull
        if len(contorno) < 3:
            return 1.0
        
        try:
            hull = ConvexHull(contorno)
            area_convex = hull.volume
        except:
            return 1.0
        
        area_objeto = np.sum(img_bin)
        return area_objeto / area_convex if area_convex > 0 else 0.0

@registrar_en("cuantificacion")
class Concavidad(CuantificadorMorfometrico):
    """
            Concavidad: medida de profundidad de indentaciones.
            
            Cuantifica qué tan "profundas" son las invaginaciones
            respecto al convex hull.
            
            Algoritmo:
                Conc = 1 - (A_objeto / A_convex_hull) = 1 - Convexidad
                o profundidad máxima: max(distancia de borde a convex hull)
            
            Propiedades:
                - 0 ≤ Conc < 1 (0 para convexo)
                - Complementaria a convexidad
                - Sensible a indentaciones nucleares profundas
            
            Ventajas:
                - Directamente relacionada con fenotipos de invaginación,
                - sensible a cambios morfológicos sutiles
            Desventajas:
                - Misma complejidad computacional que convexidad,
                - puede ser ruidosa para bordes irregulares
            
            Usos microscopía:
                - Grado de invaginación nuclear (cáncer),
                - formación de blebs (membrana),
                - análisis de morfología neuronal (espinas),
                - cuantificación de "cupping" en estructuras
    """
    nombre = "concavidad"
    
    def __call__(self, img_segmentada: np.ndarray,
                por_region: bool = False,
                modo: Literal['area', 'profundidad'] = 'area') -> Union[float, np.ndarray]:
        """
            Args:
                img_segmentada: Imagen binaria o etiquetada
                por_region: Si True, retorna concavidad por región
                modo: 'area' (1 - convexidad) o 'profundidad' (máxima distancia)
            
            Returns:
                Concavidad [0, 1) o profundidad máxima en píxeles
        """
        self._validar_segmentada(img_segmentada, permitir_etiquetada=True)
        
        if modo == 'area':
            convexidad = Convexidad()(img_segmentada, por_region=por_region)
            if por_region:
                return 1 - convexidad
            return 1 - convexidad
        
        if por_region:
            regiones = self._get_region_props(img_segmentada)
            return np.array([self._calcular_profundidad_max(r.image) for r in regiones])
        
        return self._calcular_profundidad_max(img_segmentada > 0)
    
    def _calcular_profundidad_max(self, img_bin: np.ndarray) -> float:
        """Calcula profundidad máxima de invaginaciones."""
        from scipy.ndimage import distance_transform_edt
        
        dist_interna = distance_transform_edt(img_bin)
        contornos = measure.find_contours(img_bin, 0.5)
        
        if not contornos:
            return 0.0
        
        contorno = max(contornos, key=len)
        
        from scipy.spatial import ConvexHull
        if len(contorno) < 3:
            return 0.0
        
        try:
            hull = ConvexHull(contorno)
            from skimage.draw import polygon
            hull_puntos = contorno[hull.vertices]
            rr, cc = polygon(hull_puntos[:, 0], hull_puntos[:, 1], shape=img_bin.shape)
            mascara_hull = np.zeros_like(img_bin, dtype=bool)
            mascara_hull[rr.astype(int), cc.astype(int)] = True
            
            diferencia = mascara_hull & ~img_bin
            return float(np.max(dist_interna[diferencia])) if np.any(diferencia) else 0.0
        except:
            return 0.0

@registrar_en("cuantificacion")
class Forma(CuantificadorMorfometrico):
    """
        Descriptor de forma compuesto (HU moments).
        
        Momentos invariantes de Hu: 7 descriptores invariantes a
        traslación, escala, rotación y reflexión.
        
        Algoritmo (momentos normalizados centrales):
            η_pq = μ_pq / μ₀₀^((p+q)/2 + 1)
            
            I₁ = η₂₀ + η₀₂
            I₂ = (η₂₀ - η₀₂)² + 4η₁₁²
            ... (hasta I₇)
        
        Propiedades:
            - Invariantes a transformaciones rígidas + escala
            - I₁, I₂ son invariantes a reflexión
            - I₇ cambia signo con reflexión (handedness)
            - Sensible a ruido para momentos altos
        
        Ventajas:
            - Descripción completa de forma,
            - invariantes ideales para clasificación
        Desventajas:
            - No son independientes (correlacionados),
            - momentos altos muy sensibles a ruido,
            - no son únicos (diferentes formas, mismos momentos)
        
        Usos microscopía:
            - Clasificación de tipos celulares,
            - matching de células en time-lapse,
            - análisis de morfología nuclear,
            - features para ML (SVM, Random Forest)
    """
    nombre = "forma_hu_moments"
    
    def __call__(self, img_segmentada: np.ndarray,
                por_region: bool = False,
                log_transform: bool = True) -> Union[np.ndarray, List[np.ndarray]]:
        """
            Args:
                img_segmentada: Imagen binaria o etiquetada
                por_region: Si True, retorna momentos por región
                log_transform: Si True, aplica -sign(I)*log10(|I|) para estabilidad
            
            Returns:
                Array de 7 momentos de Hu
        """
        self._validar_segmentada(img_segmentada, permitir_etiquetada=True)
        
        if por_region:
            regiones = self._get_region_props(img_segmentada)
            momentos_list = []
            for r in regiones:
                momentos = self._calcular_hu_moments(r.image)
                if log_transform:
                    momentos = self._log_transform(momentos)
                momentos_list.append(momentos)
            return momentos_list
        
        momentos = self._calcular_hu_moments(img_segmentada > 0)
        if log_transform:
            momentos = self._log_transform(momentos)
        return momentos
    
    def _calcular_hu_moments(self, img_bin: np.ndarray) -> np.ndarray:
        """Calcula los 7 momentos invariantes de Hu."""
        try:
            import cv2
            moments = cv2.moments(img_bin.astype(np.uint8))
        except ImportError:
            moments = self._calcular_momentos_manuales(img_bin)
        
        if moments['m00'] == 0:
            return np.zeros(7)
        
        mu20, mu02, mu11 = moments['mu20'], moments['mu02'], moments['mu11']
        mu30 = moments['mu30']
        mu12, mu21, mu03 = moments['mu12'], moments['mu21'], moments['mu03']
        
        m00 = moments['m00']
        eta20 = mu20 / (m00 ** 2)
        eta02 = mu02 / (m00 ** 2)
        eta11 = mu11 / (m00 ** 2)
        eta30 = mu30 / (m00 ** 2.5)
        eta12 = mu12 / (m00 ** 2.5)
        eta21 = mu21 / (m00 ** 2.5)
        eta03 = mu03 / (m00 ** 2.5)
        
        I1 = eta20 + eta02
        I2 = (eta20 - eta02) ** 2 + 4 * eta11 ** 2
        I3 = (eta30 - 3 * eta12) ** 2 + (3 * eta21 - eta03) ** 2
        I4 = (eta30 + eta12) ** 2 + (eta21 + eta03) ** 2
        I5 = (eta30 - 3 * eta12) * (eta30 + eta12) * ((eta30 + eta12) ** 2 - 3 * (eta21 + eta03) ** 2) + \
             (3 * eta21 - eta03) * (eta21 + eta03) * (3 * (eta30 + eta12) ** 2 - (eta21 + eta03) ** 2)
        I6 = (eta20 - eta02) * ((eta30 + eta12) ** 2 - (eta21 + eta03) ** 2) + \
             4 * eta11 * (eta30 + eta12) * (eta21 + eta03)
        I7 = (3 * eta21 - eta03) * (eta30 + eta12) * ((eta30 + eta12) ** 2 - 3 * (eta21 + eta03) ** 2) - \
             (eta30 - 3 * eta12) * (eta21 + eta03) * (3 * (eta30 + eta12) ** 2 - (eta21 + eta03) ** 2)
        
        return np.array([I1, I2, I3, I4, I5, I6, I7])
    
    def _calcular_momentos_manuales(self, img_bin: np.ndarray) -> dict:
        """Calcula momentos manualmente (fallback sin cv2)."""
        coords = np.where(img_bin > 0)
        if len(coords[0]) == 0:
            return {'m00': 0}
        
        y, x = coords[0], coords[1]
        m00 = len(y)
        cx = np.mean(x)
        cy = np.mean(y)
        
        return {
            'm00': m00,
            'mu20': np.sum((x - cx)**2),
            'mu02': np.sum((y - cy)**2),
            'mu11': np.sum((x - cx) * (y - cy)),
            'mu30': np.sum((x - cx)**3),
            'mu03': np.sum((y - cy)**3),
            'mu12': np.sum((x - cx)**2 * (y - cy)),
            'mu21': np.sum((x - cx) * (y - cy)**2)
        }
    
    def _log_transform(self, momentos: np.ndarray) -> np.ndarray:
        """Aplica transformación logarítmica para estabilidad numérica."""
        resultado = np.zeros_like(momentos)
        for i, m in enumerate(momentos):
            if abs(m) < 1e-10:
                resultado[i] = 0
            else:
                resultado[i] = -np.sign(m) * np.log10(abs(m))
        return resultado


# FUNCIONES DE UTILIDAD Y PIPELINE


def extraer_todas_metricas(img_segmentada: np.ndarray, 
                            por_region: bool = True) -> dict:
    """
        Extrae todas las métricas morfométricas disponibles.
        
        Args:
            img_segmentada: Imagen binaria o etiquetada
            por_region: Si True, calcula por cada región etiquetada
        
        Returns:
            Diccionario con todas las métricas morfométricas:
            {
                'area': [...],
                'perimetro': [...],
                'centroide': [...],
                'caja_frontera': [...],
                'diametro_equivalente': [...],
                'excentricidad': [...],
                'circularidad': [...],
                'compactacion': [...],
                'orientacion': [...],
                'convexidad': [...],
                'concavidad': [...],
                'forma_hu_moments': [...]
            }
    """
    metricas = {}
    
    cuantificadores = {
        'area': Area(),
        'perimetro': Perimetro(),
        'centroide': Centroide(),
        'caja_frontera': CajaFrontera(),
        'diametro_equivalente': DiametroEquivalente(),
        'excentricidad': Excentricidad(),
        'circularidad': Circularidad(),
        'compactacion': Compactacion(),
        'orientacion': Orientacion(),
        'convexidad': Convexidad(),
        'concavidad': Concavidad(),
        'forma_hu_moments': Forma(),
    }
    
    for nombre, cuantificador in cuantificadores.items():
        try:
            if nombre == 'orientacion':
                metricas[nombre] = cuantificador(img_segmentada, por_region=por_region, modo='degrees')
            elif nombre == 'compactacion':
                metricas[nombre] = cuantificador(img_segmentada, por_region=por_region, normalizada=True)
            elif nombre == 'concavidad':
                metricas[nombre] = cuantificador(img_segmentada, por_region=por_region, modo='area')
            elif nombre == 'forma_hu_moments':
                metricas[nombre] = cuantificador(img_segmentada, por_region=por_region, log_transform=True)
            else:
                metricas[nombre] = cuantificador(img_segmentada, por_region=por_region)
        except Exception as e:
            warnings.warn(f"Error calculando {nombre}: {e}")
            metricas[nombre] = None
    
    metricas['_metadatos'] = {
        'por_region': por_region,
        'num_pixeles_objeto': int(np.sum(img_segmentada > 0)),
        'forma_imagen': img_segmentada.shape,
        'tipo_analisis': 'morfometrico'
    }
    
    return metricas


def comparar_formas(img1: np.ndarray, img2: np.ndarray,
                    metricas: List[str] = None) -> dict:
    """
        Compara dos formas mediante sus descriptores.
        
        Args:
            img1, img2: Imágenes binarias de objetos a comparar
            metricas: Lista de métricas a comparar (None = todas)
        
        Returns:
            Diccionario con diferencias normalizadas
    """
    if metricas is None:
        metricas = ['area', 'perimetro', 'circularidad', 'excentricidad', 
                    'convexidad', 'compactacion']
    
    resultados = {}
    
    for metrica in metricas:
        try:
            clase = globals()[metrica.capitalize()]
            instancia = clase()
            
            val1 = float(instancia(img1, por_region=False))
            val2 = float(instancia(img2, por_region=False))
            
            if val1 == 0 and val2 == 0:
                diff = 0.0
            elif val1 == 0:
                diff = 1.0
            else:
                diff = abs(val1 - val2) / max(abs(val1), abs(val2))
            
            resultados[metrica] = {
                'valor_1': val1,
                'valor_2': val2,
                'diferencia_normalizada': diff
            }
        except Exception as e:
            resultados[metrica] = {'error': str(e)}
    
    return resultados