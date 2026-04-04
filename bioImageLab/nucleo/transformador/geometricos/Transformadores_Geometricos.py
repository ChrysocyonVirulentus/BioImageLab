"""
Transformaciones geométricas y topológicas para análisis de forma.

Incluye transformaciones que preservan o extraen propiedades geométricas
y topológicas: distancias, esqueletos, ejes mediales, y operaciones
de deformación espacial (warp, resize, rotación).

Principio fundamental:
Transformar la geometría de la imagen o extraer descriptores geométricos
invariantes que caractericen la forma de objetos.

IMPORTANTE - Separación de responsabilidades:
- NO normalizan imágenes (ese rol es de normalizador.py)
- Algunas requieren máscaras binarias (esqueleto, eje medial)
- Deformaciones espaciales requieren cuidado con interpolación
- Información de escala/resolución debe preservarse en metadatos

Métodos disponibles:
- TransformacionDistancia: Mapa de distancia euclidiana
- Esqueletizacion: Reducción topológica a 1 píxel de ancho
- EjeMedial: Eje central con radio (medial axis transform)
- Deformar: Warping no rígido (registro, corrección de distorsión)
- Redimensionar: Cambio de escala con interpolación
- Rotacion: Rotación con preservación de información
- Remuestreo: Cambio de muestreo espacial (up/down-sampling)
"""

import numpy as np
import cv2
from typing import Optional, Tuple, Literal, Union, List
from scipy import ndimage
from scipy.ndimage import distance_transform_edt, map_coordinates
from skimage import morphology, transform, measure
from skimage.morphology import skeletonize, medial_axis, thin
import warnings


class TransformadorGeometrico:
    """Clase base para transformaciones geométricas y topológicas."""
    nombre = "transformador_geometrico_base"
    
    def __call__(self, img: np.ndarray) -> np.ndarray:
        raise NotImplementedError
    
    def _validar_imagen(self, img: np.ndarray, permitir_binaria: bool = True):
        if img.ndim != 2:
            raise ValueError(f"Imagen debe ser 2D, tiene {img.ndim} dimensiones")

@registrar_en("transformacion")
class TransformacionDistancia(TransformadorGeometrico):
    """
        Transformada de distancia euclidiana y funciones derivadas.
        
        Calcula distancia desde cada píxel de objeto al fondo más cercano.
        Base para segmentación, análisis de morfología y operaciones morfológicas.
        
        Algoritmo (EDT):
            D(x,y) = min{√((x-x')² + (y-y')²) | (x',y') ∈ fondo}
        
        Ventajas:
            - Isotrópico, base para watershed, métrica natural para formas redondas
        Desventajas:
            - Sensible a ruido de borde, no distingue agujeros vs objetos tocantes
        
        Usos microscopía:
            - Segmentación de núcleos (máximos=centros), análisis de packing,
            - grosor de vasos, detección de porosidad
    """
    nombre = "transformacion_distancia"
    
    def __init__(self, metrica: Literal['euclidean', 'squared', 'manhattan'] = 'euclidean',
                return_indices: bool = False, invertir: bool = False):
        self.metrica = metrica
        self.return_indices = return_indices
        self.invertir = invertir
    
    def __call__(self, img: np.ndarray, mascara: Optional[np.ndarray] = None) -> Union[np.ndarray, Tuple]:
        self._validar_imagen(img)
        
        # Preparar binaria
        if mascara is not None:
            binaria = mascara > 0
        elif img.dtype == bool:
            binaria = img
        else:
            from skimage import filters
            binaria = img > filters.threshold_otsu(img)
        
        if self.invertir:
            binaria = ~binaria
        
        # Calcular
        if self.metrica == 'euclidean':
            if self.return_indices:
                return distance_transform_edt(binaria, return_indices=True)
            return distance_transform_edt(binaria)
        elif self.metrica == 'squared':
            return distance_transform_edt(binaria) ** 2
        else:  # manhattan
            return ndimage.distance_transform_cdt(binaria, metric='taxicab')
    
    def get_maximos_locales(self, distancia: np.ndarray, min_distance: int = 5) -> np.ndarray:
        """Detecta máximos locales (centros de objetos)."""
        from skimage.feature import peak_local_max
        coords = peak_local_max(distancia, min_distance=min_distance, exclude_border=False)
        maximos = np.zeros_like(distancia, dtype=bool)
        maximos[coords[:, 0], coords[:, 1]] = True
        return maximos

@registrar_en("transformacion")
class Esqueletizacion(TransformadorGeometrico):
    """
        Reducción topológica a esqueleto de líneas de 1 píxel.
        
        El esqueleto preserva conectividad y forma esencial, eliminando grosor.
        Resultado: líneas de 1 píxel que representan el "esqueleto" del objeto.
        
        Algoritmo (Zhang-Suen):
            Iterativamente remover píxeles de borde que no desconecten el objeto,
            usando condiciones de vecindario 8-conectado.
        
        Ventajas:
            - Compresión morfológica, extracción de topología (ramificaciones),
            - base para análisis de forma (tortuosidad, longitud)
        Desventajas:
            - Sensible a ruido (espuelas), no invariante a rotación exacta,
            - pierde información de grosor
        
        Usos microscopía:
            - Análisis de redes vasculares (bifurcaciones), caracterización de neuritas,
            - fibras de colágeno, trazas de migración celular
    """
    nombre = "esqueletizacion"
    
    def __init__(self, metodo: Literal['zhang_suen', 'lee', 'morphological'] = 'zhang_suen',
                remover_espuelas: bool = False, longitud_espuela: int = 5):
        self.metodo = metodo
        self.remover_espuelas = remover_espuelas
        self.longitud_espuela = longitud_espuela
    
    def __call__(self, img: np.ndarray) -> np.ndarray:
        self._validar_imagen(img)
        
        # Binarizar
        if img.dtype != bool:
            if img.max() > 1:
                from skimage import filters
                binaria = img > filters.threshold_otsu(img)
            else:
                binaria = img > 0
        else:
            binaria = img
        
        # Esqueletizar
        if self.metodo == 'zhang_suen':
            esqueleto = skeletonize(binaria, method='zhang')
        elif self.metodo == 'lee':
            esqueleto = skeletonize(binaria, method='lee')
        else:
            esqueleto, _ = medial_axis(binaria, return_distance=False)
        
        if self.remover_espuelas:
            esqueleto = self._remover_espuelas(esqueleto, self.longitud_espuela)
        
        return esqueleto
    
    def _remover_espuelas(self, esqueleto: np.ndarray, longitud: int) -> np.ndarray:
        """Elimina ramas terminales cortas."""
        from skimage.morphology import remove_small_objects
        # Simplificación: usar remove_small_objects como aproximación
        return remove_small_objects(esqueleto, min_size=longitud, connectivity=2)
    
    def analizar_topologia(self, esqueleto: np.ndarray) -> dict:
        """Extrae información topológica."""
        vecinos = ndimage.convolve(esqueleto.astype(int), np.ones((3,3))) - esqueleto.astype(int)
        endpoints = np.argwhere(esqueleto & (vecinos == 1))
        junctions = np.argwhere(esqueleto & (vecinos >= 3))
        
        return {
            'endpoints': endpoints,
            'junctions': junctions,
            'n_endpoints': len(endpoints),
            'n_junctions': len(junctions),
            'longitud_total': np.sum(esqueleto),
            'n_componentes': ndimage.label(esqueleto)[1]
        }

@registrar_en("transformacion")
class EjeMedial(TransformadorGeometrico):
    """
        Eje Medial (Medial Axis Transform) con radio local.
        
        Conjunto de centros de circunferencias maximales inscritas.
        Cada punto tiene asociado el radio de la circunferencia maximal.
        
        Diferencia con esqueleto:
            - Esqueleto: thinning topológico, 1 píxel ancho
            - Eje medial: centrado geométrico, con radio, puede tener grosor
        
        Algoritmo:
            MA = {x | ∃ y₁, y₂ ∈ borde, y₁≠y₂, ||x-y₁||=||x-y₂||=D(x)}
        
        Ventajas:
            - Información completa (posición+radio), reconstrucción exacta posible,
            - invariante euclidiano, descriptor de forma (shock graphs)
        Desventajas:
            - Costoso, sensible a ruido (ramas espurias), estructura compleja
        
        Usos microscopía:
            - Análisis de grosor de vasos/neuritas, caracterización morfológica,
            - reconstrucción comprimida, estimación de volumen desde 2D
    """
    nombre = "eje_medial"
    
    def __init__(self, return_distance: bool = True, suavizar: bool = False, sigma: float = 1.0):
        self.return_distance = return_distance
        self.suavizar = suavizar
        self.sigma = sigma
    
    def __call__(self, img: np.ndarray) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        self._validar_imagen(img)
        
        # Binarizar
        if img.dtype != bool:
            if img.max() > 1:
                from skimage import filters
                binaria = img > filters.threshold_otsu(img)
            else:
                binaria = img > 0
        else:
            binaria = img
        
        if self.suavizar:
            from skimage.filters import gaussian
            binaria = gaussian(binaria.astype(float), self.sigma) > 0.5
        
        eje, distancia = medial_axis(binaria, return_distance=True)
        
        if self.return_distance:
            return eje, distancia
        return eje
    
    def reconstruir(self, eje: np.ndarray, distancia: np.ndarray) -> np.ndarray:
        """Reconstruye objeto original desde eje medial."""
        from skimage.draw import disk
        h, w = eje.shape
        rec = np.zeros((h, w), dtype=bool)
        
        for y, x in np.argwhere(eje):
            r = int(distancia[y, x])
            if r > 0:
                rr, cc = disk((y, x), r, shape=(h, w))
                rec[rr, cc] = True
        return rec

@registrar_en("transformacion")
class Deformar(TransformadorGeometrico):
    """
        Warping no rígido (deformación espacial libre).
        
        Deforma imagen según campo de vectores de desplazamiento o
        transformación paramétrica (splines, thin-plate, etc.).
        
        Algoritmo:
            Para cada píxel destino (x',y'), encontrar origen (x,y) = T⁻¹(x',y'),
            interpolar valor de imagen en (x,y).
        
        Ventajas:
            - Corrección de distorsiones no lineales (aberraciones ópticas),
            - registro elástico, normalización de formas, augmentación de datos
        Desventajas:
            - Requiere estimación de campo de deformación, interpolación necesaria,
            - puede perder información (compresión) o crear artefactos
        
        Usos microscopía:
            - Corrección de distorsión de lente (barrel/pincushion),
            - registro de imágenes de histología (secciones consecutivas),
            - normalización de morfología celular, stitchting de mosaicos
    """
    nombre = "deformar"
    
    def __init__(self, modo: Literal['grid', 'tps', 'elastic'] = 'grid',
                interpolacion: Literal['linear', 'cubic', 'nearest'] = 'linear'):
        self.modo = modo
        self.interpolacion = interpolacion
    
    def __call__(self, img: np.ndarray, 
                campo_x: Optional[np.ndarray] = None,
                campo_y: Optional[np.ndarray] = None,
                puntos_origen: Optional[np.ndarray] = None,
                puntos_destino: Optional[np.ndarray] = None) -> np.ndarray:
        self._validar_imagen(img)
        
        h, w = img.shape
        
        if self.modo == 'grid' and campo_x is not None and campo_y is not None:
            # Deformación por campo de vectores
            return self._warp_by_field(img, campo_x, campo_y)
        
        elif self.modo == 'tps' and puntos_origen is not None:
            # Thin Plate Spline
            return self._warp_tps(img, puntos_origen, puntos_destino)
        
        else:
            raise ValueError("Parámetros insuficientes para el modo especificado")
    
    def _warp_by_field(self, img: np.ndarray, dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
        """Deforma usando campos de desplazamiento."""
        h, w = img.shape
        y, x = np.mgrid[0:h, 0:w]
        
        # Coordenadas de muestreo
        coords = np.array([y + dy, x + dx])
        
        # Interpolar
        orden = {'nearest': 0, 'linear': 1, 'cubic': 3}[self.interpolacion]
        return map_coordinates(img.astype(float), coords, order=orden, mode='reflect')
    
    def _warp_tps(self, img: np.ndarray, src: np.ndarray, dst: Optional[np.ndarray]) -> np.ndarray:
        """Thin Plate Spline warping."""
        from scipy.interpolate import Rbf
        
        h, w = img.shape
        
        if dst is None:
            # Asumir que src son puntos de control a cuadrícula regular
            dst = src.copy()
            src = self._create_grid_points(src.shape[0], h, w)
        
        # Crear interpolador TPS para cada coordenada
        tps_x = Rbf(src[:, 0], src[:, 1], dst[:, 0], function='thin_plate')
        tps_y = Rbf(src[:, 0], src[:, 1], dst[:, 1], function='thin_plate')
        
        # Aplicar a grid completo
        y, x = np.mgrid[0:h, 0:w]
        x_new = tps_x(y, x)
        y_new = tps_y(y, x)
        
        coords = np.array([y_new, x_new])
        return map_coordinates(img.astype(float), coords, order=1, mode='reflect')
    
    def _create_grid_points(self, n: int, h: int, w: int) -> np.ndarray:
        """Crea puntos de grid regular."""
        y = np.linspace(0, h-1, int(np.sqrt(n)))
        x = np.linspace(0, w-1, int(np.sqrt(n)))
        yy, xx = np.meshgrid(y, x)
        return np.column_stack([yy.ravel(), xx.ravel()])

@registrar_en("transformacion")
class Redimensionar(TransformadorGeometrico):
    """
        Cambio de escala espacial con interpolación controlada.
        
        Modifica resolución espacial preservando información según criterio
        de interpolación (ideal para down/up-sampling controlado).
        
        Algoritmo:
            Escalamiento de coordenadas según factor, interpolación de valores.
        
        Ventajas:
            - Control de calidad vs velocidad, preservación de rangos,
            - anti-aliasing en reducción
        Desventajas:
            - Pérdida de información irreversible (downsampling),
            - artefactos de interpolación (aliasing, ringing)
        
        Usos microscopía:
            - Preparación de pirámides de resolución, ajuste a tamaños de red neuronal,
            - visualización eficiente, corrección de anisotropía de píxel
    """
    nombre = "redimensionar"
    
    def __init__(self, interpolacion: Literal['nearest', 'bilinear', 'bicubic', 'lanczos'] = 'bicubic',
                anti_aliasing: bool = True, preserve_range: bool = True):
        self.interpolacion = interpolacion
        self.anti_aliasing = anti_aliasing
        self.preserve_range = preserve_range
    
    def __call__(self, img: np.ndarray, 
                factor: Optional[float] = None,
                tamanio: Optional[Tuple[int, int]] = None) -> np.ndarray:
        self._validar_imagen(img)
        
        if factor is None and tamanio is None:
            raise ValueError("Debe especificar factor o tamanio")
        
        if factor is not None:
            tamanio = (int(img.shape[0] * factor), int(img.shape[1] * factor))
        
        # Mapeo de interpolaciones
        interp_map = {
            'nearest': cv2.INTER_NEAREST,
            'bilinear': cv2.INTER_LINEAR,
            'bicubic': cv2.INTER_CUBIC,
            'lanczos': cv2.INTER_LANCZOS4
        }
        
        # Anti-aliasing para reducción
        img_proc = img.astype(np.float64)
        if self.anti_aliasing and (tamanio[0] < img.shape[0] or tamanio[1] < img.shape[1]):
            from skimage.filters import gaussian
            sigma = 0.5 * max(img.shape[0] / tamanio[0], img.shape[1] / tamanio[1])
            img_proc = gaussian(img_proc, sigma=sigma, preserve_range=True)
        
        # Redimensionar
        resultado = cv2.resize(img_proc, (tamanio[1], tamanio[0]), 
                            interpolation=interp_map[self.interpolacion])
        
        if self.preserve_range and np.issubdtype(img.dtype, np.integer):
            info = np.iinfo(img.dtype)
            resultado = np.clip(resultado, info.min, info.max).astype(img.dtype)
        
        return resultado

@registrar_en("transformacion")
class Rotacion(TransformadorGeometrico):
    """
        Rotación con preservación de información y tamaño controlado.
        
        Rota imagen alrededor de centro especificado, con opciones para
        manejo de esquinas (recortar o expandir).
        
        Algoritmo:
            Rotación de coordenadas:[x'] = [cosθ -sinθ] [x-cx]
                                    [y'] = [sinθ  cosθ] [y-cy]
        
        Ventajas:
            - Corrección de orientación, augmentación de datos,
            - registro angular, análisis de rotación invariance
        Desventajas:
            - Interpolación necesaria, pérdida de esquinas (si recortar),
            - o aumento de tamaño (si expandir)
        
        Usos microscopía:
            - Corrección de orientación de muestra, alignación de células polarizadas,
            - análisis de simetría rotacional, augmentación para ML
    """
    nombre = "rotacion"
    
    def __init__(self, interpolacion: Literal['nearest', 'bilinear', 'bicubic'] = 'bilinear',
                modo_borde: Literal['constant', 'reflect', 'wrap'] = 'constant',
                valor_constante: float = 0.0):
        self.interpolacion = interpolacion
        self.modo_borde = modo_borde
        self.valor_constante = valor_constante
    
    def __call__(self, img: np.ndarray, 
                angulo: float,
                centro: Optional[Tuple[float, float]] = None,
                escalar: float = 1.0,
                recortar: bool = True) -> np.ndarray:
        self._validar_imagen(img)
        
        h, w = img.shape
        
        if centro is None:
            centro = (w / 2.0, h / 2.0)
        
        # Matriz de rotación
        M = cv2.getRotationMatrix2D(centro, angulo, escalar)
        
        if recortar:
            tamanio = (w, h)
        else:
            # Calcular tamaño para contener imagen rotada completa
            # (simplificado: usar bounding box)
            cos = np.abs(np.cos(np.deg2rad(angulo)))
            sin = np.abs(np.sin(np.deg2rad(angulo)))
            nueva_w = int(h * sin + w * cos)
            nueva_h = int(h * cos + w * sin)
            
            # Ajustar centro en matriz
            M[0, 2] += (nueva_w - w) / 2
            M[1, 2] += (nueva_h - h) / 2
            tamanio = (nueva_w, nueva_h)
        
        # Mapeo de interpolaciones
        interp_map = {
            'nearest': cv2.INTER_NEAREST,
            'bilinear': cv2.INTER_LINEAR,
            'bicubic': cv2.INTER_CUBIC
        }
        
        # Mapeo de bordes
        border_map = {
            'constant': cv2.BORDER_CONSTANT,
            'reflect': cv2.BORDER_REFLECT,
            'wrap': cv2.BORDER_WRAP
        }
        
        return cv2.warpAffine(img.astype(float), M, tamanio,
                            flags=interp_map[self.interpolacion],
                            borderMode=border_map[self.modo_borde],
                            borderValue=self.valor_constante)

@registrar_en("transformacion")
class Remuestreo(TransformadorGeometrico):
    """
        Cambio de muestreo espacial con control de aliasing y preservación.
        
        Operaciones de up-sampling (aumentar resolución) y down-sampling
        (reducir resolución) con filtros anti-aliasing apropiados.
        
        Diferencia con Redimensionar:
            - Remuestreo: énfasis en teoría de muestreo, aliasing, reconstrucción
            - Redimensionar: énfasis en interpolación visual
        
        Algoritmo:
            Down: filtrado paso-bajo (evitar aliasing) + submuestreo
            Up: interpolación + filtrado (suavizar artefactos)
        
        Ventajas:
            - Control teórico de aliasing, preservación de información según Nyquist,
            - reconstrucción óptima, análisis multiresolución
        Desventajas:
            - Más complejo que resize simple, requiere conocimiento de PSF/frecuencias
        
        Usos microscopía:
            - Sub-pixel analysis (super-resolución computacional),
            - cambio de z-spacing en stacks, reconstrucción desde datos sub-muestreados,
            - análisis de frecuencias espaciales
    """
    nombre = "remuestreo"
    
    def __init__(self, orden: int = 3, modo: Literal['decimate', 'interpolate', 'average'] = 'decimate'):
        self.orden = orden  # Orden del filtro de spline
        self.modo = modo
    
    def __call__(self, img: np.ndarray, 
                factor: Union[float, Tuple[float, float]],
                direccion: Literal['up', 'down', 'both'] = 'both') -> np.ndarray:
        self._validar_imagen(img)
        
        if np.isscalar(factor):
            factor = (factor, factor)
        
        if direccion == 'down' or (direccion == 'both' and (factor[0] < 1 or factor[1] < 1)):
            # Down-sampling con anti-aliasing
            return self._downsample(img, factor)
        
        elif direccion == 'up' or (direccion == 'both' and (factor[0] > 1 or factor[1] > 1)):
            # Up-sampling con interpolación
            return self._upsample(img, factor)
        
        return img
    
    def _downsample(self, img: np.ndarray, factor: Tuple[float, float]) -> np.ndarray:
        """Down-sampling con anti-aliasing."""
        from scipy.signal import decimate
        
        # Factor de decimación
        f_y, f_x = factor
        
        # Aplicar filtro paso-bajo antes de submuestrear
        img_filt = img.astype(float)
        
        if f_y < 1:
            n_y = int(1 / f_y)
            # Decimar con filtro anti-aliasing
            for i in range(img.shape[1]):
                img_filt[:, i] = decimate(img_filt[:, i], n_y, ftype='iir', zero_phase=True)
        
        if f_x < 1:
            n_x = int(1 / f_x)
            for i in range(img_filt.shape[0]):
                img_filt[i, :] = decimate(img_filt[i, :], n_x, ftype='iir', zero_phase=True)
        
        return img_filt
    
    def _upsample(self, img: np.ndarray, factor: Tuple[float, float]) -> np.ndarray:
        """Up-sampling con interpolación."""
        from scipy.ndimage import zoom
        
        f_y, f_x = factor
        return zoom(img.astype(float), (f_y, f_x), order=self.orden)