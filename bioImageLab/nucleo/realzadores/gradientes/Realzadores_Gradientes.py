"""
Métodos de detección de gradientes y bordes para análisis de discontinuidades.

Los detectores de gradientes identifican regiones de rápida variación de
intensidad (bordes) mediante operadores diferenciales de primera y segunda
derivada. Son fundamentales para segmentación, medición de morfología y
análisis estructural.

Principio fundamental:
Los bordes corresponden a máximos locales del gradiente (1ª derivada) o
cruces por cero del Laplaciano (2ª derivada). La magnitud indica fuerza
del borde; la dirección indica orientación perpendicular al borde.

IMPORTANTE - Separación de responsabilidades:
- Estos métodos NO normalizan imágenes (ese rol es de normalizador.py)
- Solo realizan conversiones de tipo cuando OpenCV lo requiere estrictamente
- Trabajan con los valores de la imagen tal como vienen
- Detectan bordes, NO segmentan (ese rol es de segmentador.py)
- La normalización previa (si es necesaria) debe hacerse con Normalizador
- Los resultados suelen requerir post-procesamiento (umbralización, esqueletizado)

Tipos de detectores:
- Clásicos: Respuesta directa del operador (Sobel, Scharr, Laplaciano)
- Optimales: Maximizan SNR y localización (Canny, Deriche)
- Multiescala: Detectan bordes en diferentes resoluciones (DoG, LoG)
- Direccionales: Sensibles a orientación específica (Gabor, steerable filters)

Métodos disponibles:
- Laplaciano: Operador isotrópico de segunda derivada
- Canny: Detector óptimo con supresión de no-máximos
- Sobel: Gradiente con suavizado integrado (3x3)
- Scharr: Gradiente mejorado, más isotrópico
- Prewitt: Alternativa simple a Sobel
- Roberts: Operador cruzado de 2x2 (bordes diagonales)
- LaplacianoCero: Detección de cruces por cero con subpixel
"""

import numpy as np
import cv2
from typing import Optional, Tuple, Literal, Union
from scipy.ndimage import gaussian_filter, convolve
from scipy.signal import convolve2d
import warnings


class DetectorGradiente:
    """
        Clase base para detectores de gradientes y bordes.
        
        Los detectores de gradiente identifican discontinuidades de intensidad
        mediante operadores diferenciales espaciales.
        
        Conceptos clave:
            - Gradiente: Vector de primeras derivadas (∂I/∂x, ∂I/∂y)
            - Magnitud: |∇I| = sqrt(Ix² + Iy²) - fuerza del borde
            - Dirección: θ = atan2(Iy, Ix) - orientación perpendicular al borde
            - Laplaciano: ∇²I = ∂²I/∂x² + ∂²I/∂y² - segunda derivada isotrópica
    """
    nombre = "detector_gradiente_base"
    
    def __call__(self, img: np.ndarray) -> np.ndarray:
        """
            Aplica el detector de gradiente a la imagen.
            
            Args:
                img: Array 2D (Y, X) con la imagen a procesar
                
            Returns:
                Mapa de gradientes, bordes o características según el método
        """
        raise NotImplementedError("Subclases deben implementar __call__")
    
    def _validar_imagen(self, img: np.ndarray):
        """Valida que la imagen sea 2D."""
        if img.ndim != 2:
            raise ValueError(f"La imagen debe ser 2D, tiene {img.ndim} dimensiones")
    
    def _calcular_gradientes(self, img: np.ndarray, 
                            tipo: Literal["sobel", "scharr", "prewitt"] = "sobel") -> Tuple[np.ndarray, np.ndarray]:
        """
            Calcula gradientes en x e y usando el operador especificado.
            
            Returns:
                (gx, gy) gradientes en x e y
        """
        img_float = img.astype(np.float64)
        
        if tipo == "sobel":
            gx = cv2.Sobel(img_float, cv2.CV_64F, 1, 0, ksize=3)
            gy = cv2.Sobel(img_float, cv2.CV_64F, 0, 1, ksize=3)
        elif tipo == "scharr":
            gx = cv2.Scharr(img_float, cv2.CV_64F, 1, 0)
            gy = cv2.Scharr(img_float, cv2.CV_64F, 0, 1)
        elif tipo == "prewitt":
            kx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]) / 3.0
            ky = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]]) / 3.0
            gx = convolve2d(img_float, kx, mode='same', boundary='symm')
            gy = convolve2d(img_float, ky, mode='same', boundary='symm')
        else:
            raise ValueError(f"Tipo de gradiente '{tipo}' no soportado")
        
        return gx, gy
    
    def _magnitud_direccion(self, gx: np.ndarray, gy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calcula magnitud y dirección del gradiente."""
        magnitud = np.sqrt(gx**2 + gy**2)
        direccion = np.arctan2(gy, gx)  # [-pi, pi]
        return magnitud, direccion


class Laplaciano(DetectorGradiente):
    """
        Operador Laplaciano para detección de bordes por segunda derivada.
        
        El Laplaciano detecta bordes como cruces por cero de la segunda
        derivada. Es isotrópico (invariante a rotación) pero muy sensible
        al ruido.
        
        Algoritmo:
            1. Calcular segundas derivadas: ∂²I/∂x², ∂²I/∂y²
            2. Sumar: ∇²I = ∂²I/∂x² + ∂²I/∂y²
            3. Opcional: Detectar cruces por cero para localizar bordes
        
        Ecuación continua:
            ∇²I = ∂²I/∂x² + ∂²I/∂y²
        
        Kernels discretos:
            4-vecinos:    8-vecinos (más común):
            [ 0  1  0]        [ 1  1  1]
            [ 1 -4  1]        [ 1 -8  1]
            [ 0  1  0]        [ 1  1  1]
        
        Interpretación:
            - ∇²I ≈ 0: Región plana
            - ∇²I >> 0: Mínimo local (valle)
            - ∇²I << 0: Máximo local (pico)
            - Cruce por cero: Borde (transición claro-oscuro u oscuro-claro)
        
        Ventajas:
            - Isotrópico (detecta bordes en cualquier dirección igual)
            - Localización precisa de bordes (subpixel posible)
            - Simple computacionalmente (una convolución)
            - No requiere cálculo de dirección
        
        Desventajas:
            - Muy sensible al ruido (segunda derivada amplifica alta frecuencia)
            - Doble respuesta en bordes gruesos (pico positivo y negativo)
            - No distingue entre bordes claros→oscuros y viceversa
            - Requiere suavizado previo (Laplaciano de Gaussiana - LoG)
        
        Usos típicos en microscopía:
            - Detección de contornos celulares en imágenes de contraste de fase
            - Identificación de bordes de núcleos en DAPI/Hoechst
            - Segmentación de vesículas con bordes bien definidos
            - Detección de puntos característicos (blobs) en tracking
            - Preprocesamiento para algoritmos de watershed
    """
    nombre = "laplaciano"
    
    def __init__(self, 
                kernel_tipo: Literal["4_vecinos", "8_vecinos"] = "8_vecinos",
                detectar_cruces: bool = False,
                suavizado_sigma: Optional[float] = None):
        """
            Args:
                kernel_tipo: Conectividad del kernel Laplaciano
                            "4_vecinos": Menos sensible a ruido diagonal
                            "8_vecinos": Más isotrópico, estándar
                
                detectar_cruces: Si True, devuelve mapa binario de cruces por cero
                                Si False, devuelve imagen Laplaciana continua
                
                suavizado_sigma: Si se especifica, aplica GaussianBlur antes
                                (equivalente a LoG - Laplacian of Gaussian)
                                Valores típicos: 0.5-2.0
        """
        self.kernel_tipo = kernel_tipo
        self.detectar_cruces = detectar_cruces
        self.suavizado_sigma = suavizado_sigma
        
        # Definir kernel
        if kernel_tipo == "4_vecinos":
            self.kernel = np.array([[0, 1, 0],
                                    [1, -4, 1],
                                    [0, 1, 0]], dtype=np.float64)
        else:
            self.kernel = np.array([[1, 1, 1],
                                    [1, -8, 1],
                                    [1, 1, 1]], dtype=np.float64)
    
    def __call__(self, img: np.ndarray) -> np.ndarray:
        """
            Aplica operador Laplaciano.
            
            Args:
                img: Imagen 2D (cualquier tipo numérico)
                
            Returns:
                Si detectar_cruces=False: Imagen Laplaciana (float64, positiva y negativa)
                Si detectar_cruces=True: Máscara binaria (uint8) de bordes
        """
        self._validar_imagen(img)
        
        img_float = img.astype(np.float64)
        
        # Suavizado opcional (LoG)
        if self.suavizado_sigma is not None:
            img_float = gaussian_filter(img_float, sigma=self.suavizado_sigma)
        
        # Aplicar Laplaciano
        laplacian = convolve(img_float, self.kernel, mode='reflect')
        
        if not self.detectar_cruces:
            return laplacian
        
        # Detectar cruces por cero
        # Un cruce por cero ocurre donde el signo cambia entre píxeles vecinos
        # y la magnitud del cambio es significativa
        
        # Máscaras de signo
        signo = np.sign(laplacian)
        
        # Cambios de signo en x e y
        cruce_x = (signo[:, :-1] * signo[:, 1:] < 0)
        cruce_y = (signo[:-1, :] * signo[1:, :] < 0)
        
        # Expandir a tamaño original
        bordes = np.zeros_like(img, dtype=np.uint8)
        bordes[:, :-1] |= cruce_x.astype(np.uint8) * 255
        bordes[:-1, :] |= cruce_y.astype(np.uint8) * 255
        
        return bordes


class Canny(DetectorGradiente):
    """
        Detector de bordes de Canny - algoritmo óptimo.
        
        Canny diseñó criterios matemáticos para detección óptima de bordes:
        buena detección, buena localización y respuesta única por borde.
        
        Algoritmo:
            1. Suavizado gaussiano para reducir ruido
            2. Cálculo de gradiente (magnitud y dirección)
            3. Supresión de no-máximos: adelgazamiento de bordes anchos
            4. Umbralización con histéresis: conexión de bordes débiles a fuertes
        
        Ecuación - Supresión de no-máximos:
            Para cada píxel, comparar magnitud con vecinos en dirección del gradiente:
            - Si es máximo local: mantener
            - Si no: suprimir (poner a 0)
        
        Umbralización por histéresis:
            - Borde fuerte: magnitud > umbral_alto
            - Borde débil: umbral_bajo < magnitud < umbral_alto
            - Borde débil se mantiene solo si conectado a borde fuerte
        
        Ventajas:
            - Óptimo en teoría (buena detección y localización)
            - Supresión de no-máximos da bordes de 1 píxel de ancho
            - Umbral por histéresis reduce falsos positivos y conecta bordes
            - Detecta bordes en cualquier dirección
            - Estándar de facto en visión por computador
        
        Desventajas:
            - Parámetros sensibles (umbrales, sigma)
            - No detecta bien bordes curvos cerrados (problema de conectividad)
            - Costoso computacionalmente
            - Puede perder bordes débiles no conectados a fuertes
            - No es isotrópico perfecto (usa Sobel)
        
        Usos típicos en microscopía:
            - Segmentación precisa de contornos celulares
            - Detección de bordes de organelos (núcleos, mitocondrias)
            - Preparación de imágenes para análisis morfométrico
            - Extracción de perfiles de intensidad radiales
            - Detección de líneas de división celular (citocinesis)
    """
    nombre = "canny"
    
    def __init__(self,
                umbral_bajo: float = 50,
                umbral_alto: float = 150,
                sigma: float = 1.0,
                aperture_size: int = 3):
        """
            Args:
                umbral_bajo: Umbral inferior para histéresis (0-255 para uint8)
                            Bordes con gradiente < umbral_bajo se descartan
                            Valores típicos: 30-100
                
                umbral_alto: Umbral superior para histéresis
                            Bordes con gradiente > umbral_alto son seguros
                            Valores típicos: 100-200 (típicamente 2-3x umbral_bajo)
                
                sigma: Desviación estándar del gaussiano de suavizado previo
                    Mayor sigma = menos ruido pero bordes más difusos
                    Valores típicos: 0.5-2.0
                
                aperture_size: Tamaño del operador Sobel (3, 5 o 7)
                            Mayor = más suave, menos detalle fino
        """
        if not (0 <= umbral_bajo < umbral_alto <= 255):
            raise ValueError("Se requiere 0 <= umbral_bajo < umbral_alto <= 255")
        if sigma <= 0:
            raise ValueError("sigma debe ser > 0")
        
        self.umbral_bajo = umbral_bajo
        self.umbral_alto = umbral_alto
        self.sigma = sigma
        self.aperture_size = aperture_size
    
    def __call__(self, img: np.ndarray) -> np.ndarray:
        """
            Aplica detector de Canny.
            
            Args:
                img: Imagen 2D (uint8 o convertida a uint8)
                
            Returns:
                Máscara binaria (uint8) con bordes de 1 píxel de ancho
        """
        self._validar_imagen(img)
        
        # Canny requiere uint8 en OpenCV
        if img.dtype != np.uint8:
            # Normalizar a uint8
            img_float = img.astype(np.float64)
            img_norm = (img_float - img_float.min()) / (img_float.max() - img_float.min() + 1e-10)
            img_uint8 = (img_norm * 255).astype(np.uint8)
        else:
            img_uint8 = img
        
        # Aplicar Canny
        bordes = cv2.Canny(img_uint8, 
                            self.umbral_bajo, 
                            self.umbral_alto,
                            apertureSize=self.aperture_size,
                            L2gradient=True)  # Usar norma L2 para magnitud
        
        return bordes


class Sobel(DetectorGradiente):
    """
        Operador Sobel para cálculo de gradientes con suavizado integrado.
        
        El operador de Sobel calcula gradientes usando kernels que incluyen
        suavizado gaussiano, haciéndolo más robusto al ruido que operadores
        puros de diferencias finitas.
        
        Algoritmo:
            1. Aplicar kernel de derivada en x: suavizado + diferenciación
            2. Aplicar kernel de derivada en y: suavizado + diferenciación
            3. Calcular magnitud: |G| = sqrt(Gx² + Gy²)
            4. Opcional: calcular dirección: θ = atan2(Gy, Gx)
        
        Kernels (3x3):
            Gx (vertical)          Gy (horizontal)
            [-1  0  1]              [-1 -2 -1]
            [-2  0  2]      y       [ 0  0  0]
            [-1  0  1]              [ 1  2  1]
            
            (aproximación de derivada con pesos de binomial para suavizado)
        
        Interpretación:
            - Gx responde a bordes verticales (cambios horizontales)
            - Gy responde a bordes horizontales (cambios verticales)
            - Magnitud: fuerza del borde (invariante a rotación)
            - Dirección: perpendicular al borde (0 = vertical, 90 = horizontal)
        
        Ventajas:
            - Simple y computacionalmente eficiente
            - Suavizado integrado reduce ruido vs operadores puros
            - Dirección del borde disponible
            - Bases para detectores más complejos (Canny)
            - Implementación hardware optimizada
        
        Desventajas:
            - No isotrópico perfecto (error de 5-10% en direcciones diagonales)
            - Tamaño fijo (3x3, 5x5) limita adaptabilidad
            - Respuesta proporcional al contraste, no solo al borde
            - Doble respuesta en ciertos patrones de borde
        
        Usos típicos en microscopía:
            - Cálculo rápido de gradientes para segmentación por umbral adaptativo
            - Detección de bordes en tiempo real (procesamiento de video)
            - Orientación de fibras o células alargadas
            - Preprocesamiento para algoritmos de snake/active contours
            - Análisis de textura local (magnitud del gradiente)
    """
    nombre = "sobel"
    
    def __init__(self,
                ksize: int = 3,
                calcular_direccion: bool = False,
                normalizar: bool = True):
        """
            Args:
                ksize: Tamaño del kernel Sobel (1, 3, 5 o 7)
                    1: Usa filtro separable [-1, 0, 1] sin suavizado
                    3: Estándar con suavizado 3x3
                    5, 7: Mayor suavizado, menos detalle fino
                
                calcular_direccion: Si True, también devuelve mapa de direcciones
                
                normalizar: Si True, normaliza magnitud a rango [0, 255] o [0, 1]
        """
        if ksize not in [1, 3, 5, 7]:
            raise ValueError("ksize debe ser 1, 3, 5 o 7")
        
        self.ksize = ksize
        self.calcular_direccion = calcular_direccion
        self.normalizar = normalizar
    
    def __call__(self, img: np.ndarray) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
            Aplica operador Sobel.
            
            Args:
                img: Imagen 2D
                
            Returns:
                Si calcular_direccion=False: Mapa de magnitudes (mismo tipo que entrada)
                Si calcular_direccion=True: (magnitud, direccion) donde direccion en radianes
        """
        self._validar_imagen(img)
        
        tipo_original = img.dtype
        img_float = img.astype(np.float64)
        
        # Calcular gradientes
        gx = cv2.Sobel(img_float, cv2.CV_64F, 1, 0, ksize=self.ksize)
        gy = cv2.Sobel(img_float, cv2.CV_64F, 0, 1, ksize=self.ksize)
        
        # Magnitud
        magnitud = np.sqrt(gx**2 + gy**2)
        
        # Normalizar
        if self.normalizar:
            if magnitud.max() > 0:
                magnitud = magnitud / magnitud.max()
                if np.issubdtype(tipo_original, np.integer):
                    magnitud = magnitud * 255
        
        # Convertir tipo
        if np.issubdtype(tipo_original, np.integer):
            magnitud = np.clip(magnitud, 0, 255).astype(tipo_original)
        else:
            magnitud = magnitud.astype(tipo_original)
        
        if not self.calcular_direccion:
            return magnitud
        
        # Calcular dirección en radianes [-pi, pi]
        direccion = np.arctan2(gy, gx)
        
        return magnitud, direccion


class Scharr(DetectorGradiente):
    """
        Operador Scharr - mejora isotrópica del Sobel.
        
        El operador de Scharr corrige el sesgo direccional del Sobel,
        proporcionando respuesta más uniforme en todas las orientaciones.
        
        Algoritmo:
            Similar a Sobel pero con kernels optimizados para isotropía:
            
            Kernels (3x3):
                Gx                      Gy
            [ -3  0  3]             [ -3 -10 -3]
            [-10  0 10]     y       [  0   0  0]
            [ -3  0  3]             [  3  10  3]
        
        Ecuación de diseño:
            Los coeficientes se eligen para minimizar el error de
            aproximación de la derivada en todas las direcciones,
            resultando en error angular < 0.5° vs ~5° de Sobel.
        
        Interpretación:
            - Igual que Sobel pero con mejor precisión angular
            - Misma complejidad computacional (3x3)
            - Respuesta más uniforme en diagonales
        
        Ventajas:
            - Mejor isotropía que Sobel (casi perfecta)
            - Mismo costo computacional que Sobel 3x3
            - Mejor para análisis de orientación de bordes
            - Recomendado por OpenCV para precisión angular
            - Sin parámetros que ajustar (solo 3x3)
        
        Desventajas:
            - Solo disponible en 3x3 (no hay versiones 5x5, 7x7)
            - Mayor amplificación de ruido que Sobel (coeficientes más grandes)
            - No tan suavizado como Sobel 5x5
        
        Comparación con Sobel:
            - Scharr: Mejor para dirección, peor para ruido fuerte
            - Sobel 3x3: Más suave, peor en diagonales
            - Sobel 5x5: Más suave que ambos, menos detalle fino
        
        Usos típicos en microscopía:
            - Análisis preciso de orientación de fibras colágenas
            - Medición de anisotropía en tejido conectivo
            - Orientación de células en culturas 2D
            - Análisis de flujo en imágenes de PIV (Particle Image Velocimetry)
            - Cualquier aplicación donde la dirección del borde sea crítica
    """
    nombre = "scharr"
    
    def __init__(self,
                calcular_direccion: bool = False,
                normalizar: bool = True):
        """
        Args:
            calcular_direccion: Si True, también devuelve mapa de direcciones
            
            normalizar: Si True, normaliza magnitud a rango de entrada
        """
        self.calcular_direccion = calcular_direccion
        self.normalizar = normalizar
    
    def __call__(self, img: np.ndarray) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
            Aplica operador Scharr.
            
            Args:
                img: Imagen 2D
                
            Returns:
                Mapa de magnitudes o (magnitud, direccion)
        """
        self._validar_imagen(img)
        
        tipo_original = img.dtype
        img_float = img.astype(np.float64)
        
        # Scharr solo existe en 3x3
        gx = cv2.Scharr(img_float, cv2.CV_64F, 1, 0)
        gy = cv2.Scharr(img_float, cv2.CV_64F, 0, 1)
        
        # Magnitud
        magnitud = np.sqrt(gx**2 + gy**2)
        
        # Normalizar
        if self.normalizar and magnitud.max() > 0:
            magnitud = magnitud / magnitud.max()
            if np.issubdtype(tipo_original, np.integer):
                magnitud = magnitud * 255
        
        # Convertir tipo
        if np.issubdtype(tipo_original, np.integer):
            magnitud = np.clip(magnitud, 0, 255).astype(tipo_original)
        else:
            magnitud = magnitud.astype(tipo_original)
        
        if not self.calcular_direccion:
            return magnitud
        
        direccion = np.arctan2(gy, gx)
        return magnitud, direccion


class Prewitt(DetectorGradiente):
    """
        Operador Prewitt - alternativa simple al Sobel.
        
        Similar al Sobel pero con suavizado uniforme en lugar de ponderado
        (binomial), haciéndolo ligeramente más rápido pero más sensible al ruido.
        
        Algoritmo:
            Kernels (3x3):
                        Gx                      Gy
            [-1  0  1]              [-1 -1 -1]
            [-1  0  1]      y       [ 0  0  0]
            [-1  0  1]              [ 1  1  1]
        
        Diferencia con Sobel:
            - Sobel: Pesos [1, 2, 1] en dirección perpendicular (suavizado gaussiano)
            - Prewitt: Pesos [1, 1, 1] (promedio simple)
            - Prewitt ≈ 10% más rápido, Sobel ≈ 10% mejor SNR
        
        Ventajas:
            - Muy simple de implementar
            - Ligeramente más rápido que Sobel
            - Bueno para demostraciones educativas
            - Respuesta lineal en direcciones cardinales
        
        Desventajas:
            - Peor supresión de ruido que Sobel
            - Mismo sesgo direccional que Sobel
            - Raramente usado en producción (Sobel o Scharr preferidos)
        
        Usos típicos:
            - Implementaciones embebidas muy limitadas en recursos
            - Demostraciones didácticas de detección de bordes
            - Algoritmos legacy
    """
    nombre = "prewitt"
    
    def __init__(self, normalizar: bool = True):
        self.normalizar = normalizar
        
        # Kernels Prewitt
        self.kx = np.array([[-1, 0, 1],
                            [-1, 0, 1],
                            [-1, 0, 1]], dtype=np.float64) / 3.0
        self.ky = np.array([[-1, -1, -1],
                            [0, 0, 0],
                            [1, 1, 1]], dtype=np.float64) / 3.0
    
    def __call__(self, img: np.ndarray) -> np.ndarray:
        """Aplica operador Prewitt."""
        self._validar_imagen(img)
        
        tipo_original = img.dtype
        img_float = img.astype(np.float64)
        
        gx = convolve2d(img_float, self.kx, mode='same', boundary='symm')
        gy = convolve2d(img_float, self.ky, mode='same', boundary='symm')
        
        magnitud = np.sqrt(gx**2 + gy**2)
        
        if self.normalizar and magnitud.max() > 0:
            magnitud = magnitud / magnitud.max()
            if np.issubdtype(tipo_original, np.integer):
                magnitud = magnitud * 255
        
        if np.issubdtype(tipo_original, np.integer):
            magnitud = np.clip(magnitud, 0, 255).astype(tipo_original)
        
        return magnitud.astype(tipo_original)


class Roberts(DetectorGradiente):
    """
        Operador de Roberts - detector cruzado de 2x2.
        
        El operador de Roberts usa kernels de 2x2 rotados 45°, siendo
        especialmente sensible a bordes diagonales.
        
        Algoritmo:
            Kernels (2x2):
                    Gx (diagonal \)       Gy (diagonal /)
            [ 1  0]              [ 0  1]
            [ 0 -1]      y       [-1  0]
        
        Ecuación:
            Gx = I(i,j) - I(i+1,j+1)
            Gy = I(i,j+1) - I(i+1,j)
            Magnitud = sqrt(Gx² + Gy²)
        
        Interpretación:
            - Responde a gradientes en direcciones diagonales (45°, 135°)
            - Muy local (solo 4 píxeles) - muy sensible al ruido
            - No hay componente de suavizado
            - Desplazamiento de 0.5 píxeles en la salida
        
        Ventajas:
            - Extremadamente simple y rápido
            - Buena localización de bordes (2x2)
            - Útil para bordes finos y diagonales
        
        Desventajas:
            - Muy sensible al ruido (sin suavizado)
            - Desplazamiento espacial en la detección
            - No isotrópico (favorece diagonales)
            - Obsoleto para la mayoría de aplicaciones
        
        Usos típicos:
            - Detección de bordes en tiempo real muy limitado
            - Análisis de texturas finas diagonales
            - Implementaciones hardware minimalistas
    """
    nombre = "roberts"
    
    def __init__(self, normalizar: bool = True):
        self.normalizar = normalizar
        
        # Kernels Roberts
        self.kx = np.array([[1, 0],
                            [0, -1]], dtype=np.float64)
        self.ky = np.array([[0, 1],
                            [-1, 0]], dtype=np.float64)
    
    def __call__(self, img: np.ndarray) -> np.ndarray:
        """Aplica operador de Roberts."""
        self._validar_imagen(img)
        
        tipo_original = img.dtype
        img_float = img.astype(np.float64)
        
        gx = convolve2d(img_float, self.kx, mode='same', boundary='symm')
        gy = convolve2d(img_float, self.ky, mode='same', boundary='symm')
        
        magnitud = np.sqrt(gx**2 + gy**2)
        
        if self.normalizar and magnitud.max() > 0:
            magnitud = magnitud / magnitud.max()
            if np.issubdtype(tipo_original, np.integer):
                magnitud = magnitud * 255
        
        if np.issubdtype(tipo_original, np.integer):
            magnitud = np.clip(magnitud, 0, 255).astype(tipo_original)
        
        return magnitud.astype(tipo_original)


class LaplacianoCero(DetectorGradiente):
    """
        Detector de bordes por cruces por cero del Laplaciano con subpixel.
        
        Versión avanzada del Laplaciano que detecta cruces por cero con
        interpolación para localización subpixel y estimación de orientación.
        
        Algoritmo:
            1. Calcular Laplaciano de Gaussiana (LoG) en múltiples escalas
            2. Detectar cruces por cero entre píxeles vecinos
            3. Interpolar posición exacta del cruce (subpixel)
            4. Calcular orientación del borde perpendicular al gradiente
        
        Ecuación de interpolación lineal:
            Si L(x) · L(x+1) < 0, el cruce está en:
            x_cruce = x + |L(x)| / (|L(x)| + |L(x+1)|)
        
        Ventajas:
            - Localización subpixel precisa (< 0.1 píxeles)
            - Estimación de orientación del borde
            - Multiescala (robusto a diferentes grosores de borde)
            - Fundamento matemático sólido (2ª derivada)
        
        Desventajas:
            - Complejidad computacional alta
            - Múltiples parámetros (sigmas, umbral de contraste)
            - Sensible a interpolación en bordes fuertes
            - Requiere post-procesamiento de contornos
        
        Usos típicos en microscopía:
            - Medición precisa de diámetros celulares
            - Análisis de deformación de membranas
            - Tracking de bordes con alta precisión
            - Metrología en imágenes microscópicas
            - Calibración de sistemas ópticos
    """
    nombre = "laplaciano_cero"
    
    def __init__(self,
                sigmas: Tuple[float, ...] = (1.0, 2.0),
                umbral_contraste: float = 0.01):
        """
            Args:
                sigmas: Escalas de análisis (tamaños de borde esperados)
                umbral_contraste: Mínimo salto de intensidad para considerar borde
                                (fracción del rango dinámico)
        """
        self.sigmas = sigmas
        self.umbral_contraste = umbral_contraste
    
    def __call__(self, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
            Aplica detección de cruces por cero.
            
            Returns:
                (bordes_x, bordes_y, orientacion) donde:
                - bordes_x, bordes_y: Coordenadas subpixel de bordes
                - orientacion: Ángulo perpendicular al borde en cada punto
        """
        self._validar_imagen(img)
        
        img_float = img.astype(np.float64)
        rango = img_float.max() - img_float.min()
        umbral = rango * self.umbral_contraste
        
        bordes_x = []
        bordes_y = []
        orientaciones = []
        
        for sigma in self.sigmas:
            # LoG
            from scipy.ndimage import gaussian_laplace
            log = gaussian_laplace(img_float, sigma=sigma)
            
            # Detectar cruces por cero en x
            for y in range(log.shape[0]):
                for x in range(log.shape[1] - 1):
                    if log[y, x] * log[y, x+1] < 0:  # Cambio de signo
                        # Interpolar
                        alpha = abs(log[y, x]) / (abs(log[y, x]) + abs(log[y, x+1]))
                        x_sub = x + alpha
                        
                        # Verificar contraste
                        contraste = abs(img_float[y, x] - img_float[y, x+1])
                        if contraste > umbral:
                            bordes_x.append(x_sub)
                            bordes_y.append(y)
                            orientaciones.append(0)  # Borde vertical (grad horizontal)
            
            # Detectar cruces por cero en y
            for y in range(log.shape[0] - 1):
                for x in range(log.shape[1]):
                    if log[y, x] * log[y+1, x] < 0:
                        alpha = abs(log[y, x]) / (abs(log[y, x]) + abs(log[y+1, x]))
                        y_sub = y + alpha
                        
                        contraste = abs(img_float[y, x] - img_float[y+1, x])
                        if contraste > umbral:
                            bordes_x.append(x)
                            bordes_y.append(y_sub)
                            orientaciones.append(np.pi/2)  # Borde horizontal
        
        return (np.array(bordes_x), np.array(bordes_y), np.array(orientaciones))