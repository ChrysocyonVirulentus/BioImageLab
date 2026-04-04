"""
Transformaciones proyectivas e integrales para análisis de proyecciones.

Incluye transformaciones que integran información a lo largo de líneas,
superficies o volúmenes: Radon (tomografía), Hough (detección de formas),
distancias transformadas, y otras integrales geométricas.

Principio fundamental:
Proyectar información de alta dimensión a espacios de menor dimensión
donde ciertas propiedades se vuelven explícitas y medibles.

IMPORTANTE - Separación de responsabilidades:
- NO normalizan imágenes (ese rol es de normalizador.py)
- Algunas son teóricamente invertibles (Radon) pero requieren datos completos
- Resolución angular y espacial críticas para calidad
- Artefactos de muestreo comunes en reconstrucciones

Métodos disponibles:
- Radon: Transformada de proyecciones (tomografía computarizada)
- Hough: Transformada para detección de formas paramétricas (líneas, círculos)
- TransformadaDistanciaGeodesica: Distancia a lo largo de superficie (no Euclidiana) # NO IMPLEMENTADO
- TransformadaHilbert: Transformada integral de Hilbert (análisis de fase)
- Abel: Transformada para simetría cilíndrica (proyecciones axiales)
- IntegralDeLinea : Contorno de un ojeto segun gradiente. 
"""

import numpy as np
import cv2
from typing import Optional, Tuple, Literal, Union, List
from scipy import ndimage
from skimage import transform
from skimage.transform import radon, iradon, hough_line, hough_circle
import warnings
from ...gestorLab.Registro_Metodos import registrar_en


class TransformadorProyectivo:
    """Clase base para transformaciones proyectivas e integrales."""
    nombre = "transformador_proyectivo_base"
    
    def __call__(self, img: np.ndarray) -> np.ndarray:
        raise NotImplementedError
    
    def _validar_imagen(self, img: np.ndarray):
        if img.ndim != 2:
            raise ValueError(f"Imagen debe ser 2D, tiene {img.ndim} dimensiones")

@registrar_en("transformacion")
class Radon(TransformadorProyectivo):
    """
        Transformada de Radon (proyecciones a lo largo de líneas).
        
        Proyecta imagen sobre líneas en diferentes ángulos, generando sinograma.
        Base de tomografía computarizada (CT) y detección de estructuras lineales.
        
        Algoritmo:
            R(ρ, θ) = ∫∫ f(x,y) · δ(x·cosθ + y·sinθ - ρ) dx dy
        
        Inversa (retroproyección filtrada):
            f(x,y) ≈ ∫ R(ρ, θ) * h(ρ) dθ, con h = filtro de rampa
        
        Ventajas:
            - Detección de líneas débiles (integración mejora SNR),
            - base de tomografía 3D, análisis de anisotropía
        Desventajas:
            - Pérdida de información local, artefactos de muestreo angular,
            - reconstrucción exacta requiere 180° completos
        
        Usos microscopía:
            - μCT (tomografía de rayos X), detección de fibras alineadas,
            - corrección de movimiento, análisis de cristalización,
            - phase retrieval en contraste de fase
    """
    nombre = "radon"
    
    def __init__(self, theta: Optional[np.ndarray] = None, circle: bool = True):
        self.theta = theta if theta is not None else np.linspace(0., 180., 180, endpoint=False)
        self.circle = circle
    
    def __call__(self, img: np.ndarray) -> np.ndarray:
        self._validar_imagen(img)
        return radon(img.astype(float), theta=self.theta, circle=self.circle)
    
    def inversa(self, sinograma: np.ndarray,
                filtro: Literal['ramp', 'shepp-logan', 'cosine', 'hamming', 'hann'] = 'ramp',
                interpolacion: Literal['linear', 'nearest'] = 'linear') -> np.ndarray:
        """Reconstrucción por retroproyección filtrada."""
        return iradon(sinograma, theta=self.theta, filter_name=filtro,
                    interpolation=interpolacion, circle=self.circle)
    
    def detectar_lineas(self, sinograma: np.ndarray, umbral: Optional[float] = None) -> List[Tuple[float, float]]:
        """Detecta líneas dominantes desde sinograma."""
        from skimage.feature import peak_local_max
        
        if umbral is None:
            umbral = np.percentile(sinograma, 95)
        
        picos = peak_local_max(sinograma, min_distance=5, threshold_abs=umbral)
        
        lineas = []
        for rho_idx, theta_idx in picos:
            theta_deg = self.theta[theta_idx] if theta_idx < len(self.theta) else 0
            n_rhos = sinograma.shape[0]
            rho = (rho_idx - n_rhos / 2)
            lineas.append((theta_deg, rho))
        
        return lineas

@registrar_en("transformacion")
class Hough(TransformadorProyectivo):
    """
        Transformada de Hough para detección de formas paramétricas.
        
        Mapea píxeles de borde a espacio de parámetros donde formas
        colineares se intersectan (votación acumulativa).
        
        Algoritmo (líneas):
            Cada punto (x,y) vota por todas las líneas que pasan por él:
            ρ = x·cosθ + y·sinθ, para θ ∈ [0, π]
            Máximos en espacio (ρ, θ) = líneas dominantes
        
        Algoritmo (círculos):
            Cada punto vota por centros (a,b) posibles a radio r:
            (a-x)² + (b-y)² = r²
        
        Ventajas:
            - Robusto a oclusión y ruido (votación múltiple),
            - detecta formas parciales, paralelizable
        Desventajas:
            - Memoria intensiva (espacio de parámetros discretizado),
            - precisión limitada por binning, lento para muchas formas
        
        Usos microscopía:
            - Detección de líneas de división celular, cristales rectangulares,
            - conteo de células redondas (Hough circular), detección de vesículas,
            - análisis de organización tisular (parallelismo)
    """
    nombre = "hough"
    
    def __init__(self, forma: Literal['linea', 'circulo'] = 'linea'):
        self.forma = forma
    
    def __call__(self, img: np.ndarray, 
                rango_radios: Optional[Tuple[int, int]] = None) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        self._validar_imagen(img)
        
        # Detectar bordes primero
        from skimage.filters import canny
        bordes = canny(img.astype(float) / img.max() if img.max() > 0 else img.astype(float))
        
        if self.forma == 'linea':
            return self._hough_lineas(bordes)
        else:
            if rango_radios is None:
                # Estimar rango típico
                min_r = max(3, min(img.shape) // 50)
                max_r = min(img.shape) // 5
                rango_radios = (min_r, max_r)
            return self._hough_circulos(bordes, rango_radios)
    
    def _hough_lineas(self, bordes: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Transformada de Hough para líneas."""
        h, theta, d = hough_line(bordes)
        return h, theta, d
    
    def _hough_circulos(self, bordes: np.ndarray, rango: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Transformada de Hough para círculos."""
        hough_radii = np.arange(rango[0], rango[1], 2)
        hough_res = hough_circle(bordes, hough_radii)
        
        return hough_res, hough_radii
    
    def get_lineas_principales(self, hough_space: np.ndarray, theta: np.ndarray, d: np.ndarray,
                                n_lineas: int = 5) -> List[Tuple[float, float]]:
        """Extrae las n líneas más votadas."""
        from skimage.feature import peak_local_max
        
        picos = peak_local_max(hough_space, min_distance=20, num_peaks=n_lineas)
        
        lineas = []
        for y, x in picos:
            lineas.append((np.rad2deg(theta[x]), d[y]))
        
        return lineas
    
    def get_circulos(self, hough_res: np.ndarray, hough_radii: np.ndarray,
                    umbral: float = 0.5) -> List[Tuple[int, int, int]]:
        """Extrae círculos detectados."""
        from skimage.feature import peak_local_max
        
        circulos = []
        for i, radius in enumerate(hough_radii):
            # Picos para este radio
            peaks = peak_local_max(hough_res[i], min_distance=30, threshold_rel=umbral)
            for y, x in peaks:
                circulos.append((x, y, radius))
        
        return circulos

@registrar_en("transformacion")
class DistanciaGeodesica(TransformadorProyectivo):
    """
        Transformada de distancia geodésica (a lo largo de superficie).
        
        Calcula distancia más corta entre puntos restringida a permanecer
        dentro de una región o sobre una superficie (no línea recta).
        
        Algoritmo (fast marching):
            Propagar frente de onda desde semilla con velocidad dependiente
            de imagen, encontrar tiempo de llegada = distancia geodésica.
        
        Ventajas:
            - Respeta topología de la imagen (no atraviesa barreras),
            - distancia semántica (sigue estructuras), robusto a ruido
        Desventajas:
            - Costoso computacionalmente, requiere definir métrica de velocidad,
            - sensible a conectividad de la región
        
        Usos microscopía:
            - Seguimiento de neuritas (distancia a lo largo de axones),
            - análisis de red vascular (camino más corto entre puntos),
            - segmentación geodésica (snakes), interpolación de cortes histológicos
    """
    nombre = "distancia_geodesica"
    
    def __init__(self, modo: Literal['fast_marching', 'dijkstra'] = 'fast_marching'):
        self.modo = modo
    
    def __call__(self, 
                img: np.ndarray,
                semillas: np.ndarray,
                mascara: Optional[np.ndarray] = None,
                peso: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Calcula distancia geodésica desde semillas.
        
        Args:
            img: Imagen base (define métrica si peso=None)
            semillas: Máscara binaria de puntos de inicio
            mascara: Región permitida (opcional)
            peso: Mapa de velocidad/costo (opcional, usa gradiente de img si None)
        """
        self._validar_imagen(img)
        
        try:
            from skimage.segmentation import morphological_geodesic_active_contour
            from skimage.filters import sobel
        except ImportError:
            pass
        
        # Calcular peso (métrica) si no se proporciona
        if peso is None:
            # Usar gradiente como costo (alto gradiente = baja velocidad)
            peso = 1.0 / (1.0 + sobel(img.astype(float)) ** 2)
        
        # Aplicar máscara
        if mascara is not None:
            peso = peso * mascara
        
        # Fast marching (simplificado: usar scipy.ndimage.distance_transform_edt con pesos)
        # Implementación completa requiere skimage.future.graph o similar
        
        # Alternativa: usar morphological operations como aproximación
        distancia = self._fast_marching_approx(semillas, peso)
        
        return distancia
    
    def _fast_marching_approx(self, semillas: np.ndarray, peso: np.ndarray) -> np.ndarray:
        """Aproximación de fast marching usando operaciones morfológicas iterativas."""
        from scipy.ndimage import binary_dilation
        
        h, w = peso.shape
        distancia = np.full((h, w), np.inf)
        distancia[semillas] = 0
        
        actual = semillas.copy()
        paso = 1
        
        while actual.any() and paso < max(h, w):
            # Dilatar frente
            nuevo = binary_dilation(actual, structure=np.ones((3,3)))
            nuevo = nuevo & (distancia == np.inf)  # Solo no visitados
            
            if not nuevo.any():
                break
            
            # Actualizar distancias
            distancia[nuevo] = paso * (2.0 - peso[nuevo])  # Costo variable
            
            actual = nuevo
            paso += 1
        
        return distancia

@registrar_en("transformacion")
class TransformadaAbel(TransformadorProyectivo):
    """
        Transformada de Abel para simetría cilíndrica (proyecciones axiales).
        
        Proyecta distribución 2D con simetría rotacional a línea central.
        Inversa: reconstruye distribución 3D desde proyección 2D.
        
        Algoritmo:
            F(y) = 2 ∫_y^∞ f(r) · r / √(r² - y²) dr  (proyección)
            f(r) = -1/π ∫_r^∞ dF/dy / √(y² - r²) dy  (inversa)
        
        Ventajas:
            - Reconstrucción 3D desde imagen 2D (simetría axial),
            - común en espectroscopía y dinámica de fluidos
        Desventajas:
            - Requiere simetría perfecta, sensible a ruido en inversa,
            - singularidad en r = y
        
        Usos microscopía:
            - Reconstrucción de distribución 3D de fluorescencia (confocal),
            - análisis de gotas esféricas, espectroscopía de absorción,
            - dinámica de fluidos (velocimetría)
    """
    nombre = "transformada_abel"
    
    def __init__(self, direccion: Literal['forward', 'inverse'] = 'forward',
                center: Optional[Tuple[int, int]] = None):
        self.direccion = direccion
        self.center = center
    
    def __call__(self, img: np.ndarray) -> np.ndarray:
        self._validar_imagen(img)
        
        try:
            import abel  # PyAbel library
        except ImportError:
            raise ImportError("Se requiere PyAbel: pip install PyAbel")
        
        if self.center is None:
            cy, cx = img.shape[0] // 2, img.shape[1] // 2
        else:
            cy, cx = self.center
        
        if self.direccion == 'forward':
            return abel.Transform(img, direction='forward', center=(cy, cx)).transform
        else:
            return abel.Transform(img, direction='inverse', center=(cy, cx)).transform

@registrar_en("transformacion")
class TransformadaHilbert(TransformadorProyectivo):
    """
        Transformada de Hilbert para análisis de fase y señales analíticas.
        
        Extiende señal real a compleja (señal analítica) donde la parte
        imaginaria es la transformada de Hilbert (desfasaje de 90°).
        
        Algoritmo:
            H{f}(x) = (1/π) P.V. ∫ f(τ)/(x-τ) dτ  (integral principal)
            Señal analítica: f_a(x) = f(x) + i·H{f}(x)
        
        Propiedades:
            - Envoltura instantánea: |f_a(x)| = √(f² + H{f}²)
            - Fase instantánea: φ(x) = arg(f_a(x))
            - Frecuencia instantánea: dφ/dx
        
        Ventajas:
            - Análisis de fase local, detección de envoltura,
            - procesamiento de señales moduladas, análisis de franjas
        Desventajas:
            - Global (no local), sensible a bordes, requiere señal estacionaria
        
        Usos microscopía:
            - Análisis de interferometría (franjas), contraste de fase cuantitativo,
            - procesamiento de imágenes de holografía digital,
            - análisis de oscilaciones en time-lapse (frecuencias instantáneas)
    """
    nombre = "transformada_hilbert"
    
    def __init__(self, eje: Literal[0, 1] = 0):
        self.eje = eje  # 0 = filas, 1 = columnas
    
    def __call__(self, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Aplica transformada de Hilbert.
        
        Returns:
            (señal_analitica, envoltura, fase) donde señal_analitica es compleja
        """
        from scipy.signal import hilbert
        
        self._validar_imagen(img)
        
        # Aplicar a lo largo del eje especificado
        analitica = hilbert(img.astype(float), axis=self.eje)
        
        envoltura = np.abs(analitica)
        fase = np.angle(analitica)
        
        return analitica, envoltura, fase
    
    def get_frecuencia_instantanea(self, fase: np.ndarray) -> np.ndarray:
        """Calcula frecuencia instantánea como derivada de fase."""
        return np.diff(fase, axis=self.eje, prepend=fase[:, :1] if self.eje == 1 else fase[:1, :])

@registrar_en("transformacion")
class IntegralDeLinea(TransformadorProyectivo):
    """
        Integración de intensidad a lo largo de líneas curvas o rectas.
        
        Generalización de proyecciones donde las líneas de integración
        pueden ser curvas (siguiendo estructuras) o múltiples.
        
        Algoritmo:
            I(C) = ∫_C f(x(l), y(l)) dl  donde C es curva parametrizada
        
        Ventajas:
            - Medición a lo largo de estructuras curvas (neuritas, vasos),
            - promediado adaptativo, análisis de perfiles
        Desventajas:
            - Requiere definición de trayectoria, interpolación necesaria,
            - sensibilidad a precisión de la curva
        
        Usos microscopía:
            - Perfiles de intensidad a lo largo de neuritas, medición de señal
            - en estructuras tubulares, análisis de kymographs,
            - integración de señal en ROI curvos
    """
    nombre = "integral_de_linea"
    
    def __init__(self, interpolacion: Literal['linear', 'cubic'] = 'linear'):
        self.interpolacion = interpolacion
    
    def __call__(self, 
                img: np.ndarray,
                curvas: List[np.ndarray],
                ancho: int = 1) -> List[float]:
        """
            Integra intensidad a lo largo de curvas.
            
            Args:
                img: Imagen
                curvas: Lista de arrays (N, 2) con coordenadas (y, x) de cada curva
                ancho: Ancho de la línea de integración (promedio perpendicular)
                
            Returns:
                Lista de valores integrados (uno por curva)
        """
        self._validar_imagen(img)
        
        integrales = []
        
        for curva in curvas:
            if len(curva) < 2:
                integrales.append(0.0)
                continue
            
            # Interpolar valores a lo largo de curva
            valores = self._muestrear_curva(img, curva, ancho)
            
            # Integrar (suma con peso de longitud)
            longitudes = np.sqrt(np.sum(np.diff(curva, axis=0)**2, axis=1))
            longitudes = np.append(longitudes, longitudes[-1])  # Repetir último
            
            integral = np.sum(valores * longitudes)
            integrales.append(integral)
        
        return integrales
    
    def _muestrear_curva(self, img: np.ndarray, curva: np.ndarray, ancho: int) -> np.ndarray:
        """Muestrea intensidad a lo largo de curva con ancho dado."""
        from scipy.ndimage import map_coordinates
        
        if ancho == 1:
            # Línea simple
            coords = [curva[:, 0], curva[:, 1]]
            return map_coordinates(img.astype(float), coords, order=1, mode='nearest')
        
        # Promedio perpendicular
        # Calcular normales
        dy = np.gradient(curva[:, 0])
        dx = np.gradient(curva[:, 1])
        norm = np.sqrt(dx**2 + dy**2) + 1e-10
        ny, nx = -dx / norm, dy / norm  # Normal unitaria
        
        valores = []
        for i, (y, x) in enumerate(curva):
            # Puntos a lo ancho
            offsets = np.arange(-ancho//2, ancho//2 + 1)
            yy = y + offsets * ny[i]
            xx = x + offsets * nx[i]
            
            # Interpolar
            val_ancho = map_coordinates(img.astype(float), [[yy], [xx]], order=1, mode='nearest')
            valores.append(np.mean(val_ancho))
        
        return np.array(valores)