"""
Métodos de afilación (sharpening) para realce de bordes y detalles finos.

Los métodos de afilación mejoran la definición de bordes y estructuras
de alta frecuencia que pueden aparecer difuminadas por óptica, movimiento
o limitaciones del sensor. A diferencia de los filtros de vesselness que
detectan geometrías, estos realzan transiciones bruscas de intensidad.

Principio fundamental:
Realzar componentes de alta frecuencia (bordes) mediante:
- Detección de discontinuidades (Laplaciano, gradientes)
- Realce de alta frecuencia (High Boost)
- Deconvolución aproximada (unsharp masking)

IMPORTANTE - Separación de responsabilidades:
- Estos métodos NO normalizan imágenes (ese rol es de normalizador.py)
- Solo realizan conversiones de tipo cuando es estrictamente necesario
- Trabajan con los valores de la imagen tal como vienen
- Pueden amplificar ruido de alta frecuencia (usar después de denoising)
- La normalización previa (si es necesaria) debe hacerse con Normalizador

Tipos de afilación:
- Laplaciana: Basada en segundas derivadas (isotrópica)
- Gradientes: Basada en primeras derivadas (direccional)
- Unsharp Masking: Resta versión difuminada (realce de alta frecuencia)
- High Boost: Generalización del unsharp masking con control de ganancia

Métodos disponibles:
- AfilacionLaplaciana: Operador isotrópico de segundo orden
- FiltroHighBoost: Realce de alta frecuencia con control de ganancia
- MascaraEnfoque: Unsharp masking clásico
- AfilacionGradiente: Basada en magnitud del gradiente (Sobel, Scharr)
- AfilacionWavelet: Multiescala usando wavelets
- DeconvolucionLucy: Deconvolución iterativa (restauración óptica)
"""

import numpy as np
import cv2
from typing import Optional, Tuple, Literal, Union
from scipy.ndimage import gaussian_filter, convolve
from skimage import restoration
import warnings


class RealzadorAfilacion:
    """
        Clase base para métodos de afilación de imágenes.
        
        Los realzadores de afilación mejoran la definición de bordes mediante
        realce de componentes de alta frecuencia espacial.
        
        Conceptos clave:
            - Frecuencia espacial: Variación rápida = alta frecuencia (bordes)
            - Kernel de afilación: Matriz de convolución que realza bordes
            - Artefactos: Overshoot/undershoot en bordes fuertes
            - Ruido: La afilación amplifica ruido de alta frecuencia
    """
    nombre = "realzador_afilacion_base"
    
    def __call__(self, img: np.ndarray) -> np.ndarray:
        """
            Aplica el método de afilación a la imagen.
            
            Args:
                img: Array 2D (Y, X) con la imagen a afilar
                
            Returns:
                Imagen afilada del mismo tipo y forma
        """
        raise NotImplementedError("Subclases deben implementar __call__")
    
    def _validar_imagen(self, img: np.ndarray):
        """Valida que la imagen sea 2D."""
        if img.ndim != 2:
            raise ValueError(f"La imagen debe ser 2D, tiene {img.ndim} dimensiones")
    
    def _aplicar_kernel(self, img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """
            Aplica kernel de convolución preservando el tipo de dato.
            
            Args:
                img: Imagen de entrada
                kernel: Kernel 2D de convolución
                
            Returns:
                Imagen filtrada del mismo tipo que la entrada
        """
        tipo_original = img.dtype
        
        # Trabajar en float64 para precisión
        img_float = img.astype(np.float64)
        
        # Convolución (usar cv2 para mejor rendimiento en enteros)
        if img_float.dtype == np.float64:
            resultado = cv2.filter2D(img_float, -1, kernel, borderType=cv2.BORDER_REFLECT)
        else:
            resultado = convolve(img_float, kernel, mode='reflect')
        
        # Clip y conversión al tipo original
        if np.issubdtype(tipo_original, np.integer):
            info = np.iinfo(tipo_original)
            resultado = np.clip(resultado, info.min, info.max)
        
        return resultado.astype(tipo_original)


class AfilacionLaplaciana(RealzadorAfilacion):
    """
        Afilación basada en el operador Laplaciano.
        
        El Laplaciano es un operador isotrópico de segundo orden que detecta
        regiones de rápida variación de intensidad. Al restar el Laplaciano
        a la imagen original, se realzan los bordes.
        
        Algoritmo:
            1. Calcular Laplaciano: ∇²I = ∂²I/∂x² + ∂²I/∂y²
            2. Restar a imagen original: I_afilada = I - α·∇²I
            3. α controla la fuerza del realce
        
        Ecuación:
            I_out(x,y) = I(x,y) - α · ∇²I(x,y)
            
            Kernels comunes (aproximaciones discretas):
            
            4-vecinos:          8-vecinos:
            [ 0 -1  0]        [-1 -1 -1]
            [-1  5 -1]   vs   [-1  9 -1]  (con α=1 incluido)
            [ 0 -1  0]        [-1 -1 -1]
        
        Interpretación:
            - ∇²I > 0: Región más oscura que el entorno (valle)
            - ∇²I < 0: Región más clara que el entorno (pico)
            - ∇²I ≈ 0: Región plana
            - Restar ∇²I: Oscurece picos, aclara valles → aumenta contraste local
        
        Ventajas:
            - Isotrópico (invariante a rotación)
            - Simple y computacionalmente eficiente
            - Realza bordes en todas las direcciones igual
            - No requiere parámetros complejos
        
        Desventajas:
            - Muy sensible al ruido (segunda derivada amplifica ruido)
            - Produce artefactos de doble borde (overshoot)
            - Puede crear valores negativos (requiere clip)
            - No diferencia bordes fuertes de débiles
        
        Usos típicos en microscopía:
            - Realce rápido de detalles finos
            - Preprocesamiento para detección de bordes
            - Mejora de contraste en estructuras pequeñas
            - Compensación de difusión de punto (PSF) leve
            - Post-procesamiento para visualización
    """
    nombre = "afilacion_laplaciana"
    
    def __init__(self, 
                alpha: float = 1.0,
                kernel_tipo: Literal["4_vecinos", "8_vecinos"] = "8_vecinos"):
        """
            Args:
                alpha: Factor de realce (fuerza del filtro)
                    Valores típicos: 0.5-2.0
                    1.0: Resta Laplaciano puro
                    >1: Realce agresivo (puede producir artefactos)
                    <1: Realce suave
                
                kernel_tipo: Tipo de conectividad del kernel
                            "4_vecinos": Más suave, menos ruido
                            "8_vecinos": Más fuerte, mejor realce de esquinas
        """
        if alpha < 0:
            raise ValueError("alpha debe ser >= 0")
        
        self.alpha = alpha
        self.kernel_tipo = kernel_tipo
        
        # Definir kernels (incluyendo α en el kernel)
        if kernel_tipo == "4_vecinos":
            # Kernel: I - α·∇²I → centro = 1 + 4α, vecinos = -α
            self.kernel = np.array([[0, -1, 0],
                                    [-1, 4, -1],
                                    [0, -1, 0]], dtype=np.float64) * alpha
            self.kernel[1, 1] += 1  # Agregar identidad
        else:  # 8_vecinos
            self.kernel = np.array([[-1, -1, -1],
                                    [-1, 8, -1],
                                    [-1, -1, -1]], dtype=np.float64) * alpha
            self.kernel[1, 1] += 1
    
    def __call__(self, img: np.ndarray) -> np.ndarray:
        """
            Aplica afilación Laplaciana.
            
            Args:
                img: Imagen 2D (cualquier tipo numérico)
                
            Returns:
                Imagen afilada del mismo tipo
        """
        self._validar_imagen(img)
        return self._aplicar_kernel(img, self.kernel)


class FiltroHighBoost(RealzadorAfilacion):
    """
        Filtro High-Boost para realce controlado de alta frecuencia.
        
        Generalización del unsharp masking donde se puede controlar
        independientemente la ganancia de bajas frecuencias y el realce
        de altas frecuencias.
        
        Algoritmo:
            1. Crear versión difuminada: I_baja = I * G (G = gaussiano)
            2. Extraer alta frecuencia: I_alta = I - I_baja
            3. Combinar: I_out = A·I - B·I_baja = (A-B)·I + B·I_alta
            
            Cuando A = B+1: I_out = I + B·(I - I_baja) [unsharp masking]
        
        Ecuación:
            I_out = A · I - B · (I * G_σ)
            
            O equivalentemente:
            I_out = I + B · (I - I * G_σ)  cuando A = B + 1
            
            donde:
                A: Ganancia de la imagen original (≥ 1)
                B: Ganancia de la máscara de alta frecuencia (≥ 0)
                G_σ: Filtro gaussiano de desviación σ
        
        Interpretación:
            - A > 1: Amplifica señal original (brillo global)
            - B grande: Realce agresivo de bordes
            - B = 0: Solo amplificación, sin afilación
            - Relación A/B controla balance señal/ruido
        
        Ventajas:
            - Control independiente de ganancia y afilación
            - Más flexible que unsharp masking estándar
            - Reduce ruido respecto a Laplaciano (usa suavizado)
            - Parámetros interpretables físicamente
        
        Desventajas:
            - Requiere ajuste de múltiples parámetros
            - Halo artifacts alrededor de bordes fuertes
            - Puede amplificar ruido si B es muy alto
            - Más costoso que Laplaciano (requiere convolución grande)
        
        Usos típicos en microscopía:
            - Compensación de desenfoque de adquisición
            - Realce de detalles en imágenes de contraste de fase
            - Preparación de imágenes para segmentación manual
            - Mejora de bordes celulares para análisis morfológico
            - Post-procesamiento cuando se requiere control fino
    """
    nombre = "filtro_high_boost"
    
    def __init__(self,
                A: float = 1.5,
                B: float = 0.5,
                sigma: float = 2.0):
        """
            Args:
                A: Ganancia de la imagen original (≥ 1.0)
                1.0: Sin amplificación adicional
                1.5: Incrementa brillo 50%
                Valores típicos: 1.0-2.0
                
                B: Ganancia de la máscara de alta frecuencia (≥ 0)
                0.0: Sin afilación (solo amplificación)
                0.5: Afilación moderada
                1.0+: Afilación fuerte (puede producir halos)
                Valores típicos: 0.3-1.0
                
                sigma: Desviación estándar del filtro gaussiano
                    Controla qué frecuencias se consideran "altas"
                    Valores típicos: 1.0-5.0
                    Menor = realza detalles más finos
        """
        if A < 1:
            raise ValueError("A debe ser >= 1.0")
        if B < 0:
            raise ValueError("B debe ser >= 0")
        if sigma <= 0:
            raise ValueError("sigma debe ser > 0")
        
        self.A = A
        self.B = B
        self.sigma = sigma
    
    def __call__(self, img: np.ndarray) -> np.ndarray:
        """
            Aplica filtro High-Boost.
            
            Args:
                img: Imagen 2D
                
            Returns:
                Imagen procesada del mismo tipo
        """
        self._validar_imagen(img)
        
        tipo_original = img.dtype
        img_float = img.astype(np.float64)
        
        # Crear versión difuminada
        img_baja = gaussian_filter(img_float, sigma=self.sigma)
        
        # Aplicar fórmula: A*I - B*I_baja
        resultado = self.A * img_float - self.B * img_baja
        
        # Clip y conversión
        if np.issubdtype(tipo_original, np.integer):
            info = np.iinfo(tipo_original)
            resultado = np.clip(resultado, info.min, info.max)
        
        return resultado.astype(tipo_original)


class MascaraEnfoque(RealzadorAfilacion):
    """
        Unsharp Masking (Máscara de Enfoque) clásico.
        
        Técnica estándar de afilación usada en fotografía y procesamiento
        de imágenes. Resta una versión difuminada para realzar detalles.
        
        Algoritmo:
            1. Difuminar imagen original: I_suave = I * G_σ
            2. Crear máscara: M = I - I_suave (componentes de alta frecuencia)
            3. Combinar: I_out = I + α · M
        
        Ecuación:
            I_out = I + α · (I - I * G_σ)
            
            Equivalente a High-Boost con A = 1+α, B = α
            
            Componentes:
                - I: Imagen original (bajas + altas frecuencias)
                - I * G_σ: Solo bajas frecuencias
                - I - I*G_σ: Solo altas frecuencias (máscara)
                - α: Factor de realce (amount)
        
        Interpretación:
            - α = 0: Sin cambio
            - α = 0.5-1: Realce natural
            - α = 1-2: Realce fuerte (usado en microscopía)
            - α > 2: Artefactos visibles (halos)
        
        Ventajas:
            - Intuitivo y ampliamente usado
            - Control simple mediante un parámetro
            - Mejor que Laplaciano en suprimir ruido
            - Resultados predecibles y reproducibles
            - Implementación eficiente
        
        Desventajas:
            - Halos alrededor de bordes fuertes (inherentes al método)
            - No adaptativo (mismo realce en toda la imagen)
            - Puede amplificar ruido si σ es pequeño
            - Efecto "plástico" si se exagera
        
        Usos típicos en microscopía:
            - Realce estándar de imágenes de fluorescencia
            - Mejora de contraste en DIC (Contraste de Interferencia Diferencial)
            - Preparación de imágenes para publicación
            - Compensación de desenfoque por aberración cromática
            - Realce de bordes de núcleos y membranas
    """
    nombre = "mascara_enfoque"
    
    def __init__(self,
                sigma: float = 2.0,
                amount: float = 1.5,
                threshold: Optional[int] = None):
        """
            Args:
                sigma: Radio de desenfoque (desviación gaussiana)
                    Determina escala de detalles a realzar
                    Valores típicos: 1.0-5.0
                    Pequeño: Realza texturas finas
                    Grande: Realza estructuras más grandes
                
                amount: Fuerza del realce (factor α)
                    Valores típicos: 0.5-2.0
                    1.0: Realce moderado
                    1.5-2.0: Realce fuerte (común en microscopía)
                
                threshold: Umbral mínimo de diferencia para aplicar realce
                        Si None, se aplica a toda la imagen
                        Si se especifica (0-255), solo realza donde |M| > threshold
                        Útil para evitar amplificar ruido en regiones planas
        """
        if sigma <= 0:
            raise ValueError("sigma debe ser > 0")
        if amount < 0:
            raise ValueError("amount debe ser >= 0")
        if threshold is not None and not (0 <= threshold <= 255):
            raise ValueError("threshold debe estar en [0, 255]")
        
        self.sigma = sigma
        self.amount = amount
        self.threshold = threshold
    
    def __call__(self, img: np.ndarray) -> np.ndarray:
        """
            Aplica unsharp masking.
            
            Args:
                img: Imagen 2D
                
            Returns:
                Imagen afilada del mismo tipo
        """
        self._validar_imagen(img)
        
        tipo_original = img.dtype
        img_float = img.astype(np.float64)
        
        # Suavizar
        img_suave = gaussian_filter(img_float, sigma=self.sigma)
        
        # Máscara de alta frecuencia
        mascara = img_float - img_suave
        
        # Aplicar threshold si se especificó
        if self.threshold is not None:
            mascara = np.where(np.abs(mascara) > self.threshold, mascara, 0)
        
        # Combinar
        resultado = img_float + self.amount * mascara
        
        # Clip y conversión
        if np.issubdtype(tipo_original, np.integer):
            info = np.iinfo(tipo_original)
            resultado = np.clip(resultado, info.min, info.max)
        
        return resultado.astype(tipo_original)


class AfilacionGradiente(RealzadorAfilacion):
    """
        Afilación basada en la magnitud del gradiente (Sobel/Scharr).
        
        Realza bordes usando la magnitud del gradiente como medida de
        discontinuidad. Más robusto al ruido que el Laplaciano.
        
        Algoritmo:
            1. Calcular gradientes: G_x, G_y (Sobel o Scharr)
            2. Magnitud: |∇I| = sqrt(G_x² + G_y²)
            3. Combinar: I_out = I + α · |∇I|
        
        Ecuación:
            I_out = I + α · sqrt((∂I/∂x)² + (∂I/∂y)²)
            
            Operadores:
            - Sobel: Kernel de 3x3 con suavizado integrado
            - Scharr: Mejor isotropía que Sobel
        
        Interpretación:
            - |∇I| es alto en bordes, bajo en regiones planas
            - α controla cuánto se realzan los bordes
            - Direccional: Realza bordes según su orientación relativa al operador
        
        Ventajas:
            - Menos sensible al ruido que Laplaciano (primeras derivadas)
            - Bordes más definidos (un solo pico vs doble pico del Laplaciano)
            - Sobel incluye suavizado integrado
            - Scharr tiene mejor isotropía
        
        Desventajas:
            - No isotrópico perfecto (a menos que use Scharr)
            - Realza solo bordes con componente en ejes del kernel
            - Puede engrosar bordes (depende de α)
            - Menos preciso en localización que Laplaciano
        
        Usos típicos en microscopía:
            - Realce de bordes direccionales
            - Preprocesamiento para detección de contornos
            - Mejora de imágenes con bordes horizontales/verticales predominantes
            - Alternativa al Laplaciano cuando hay ruido moderado
    """
    nombre = "afilacion_gradiente"
    
    def __init__(self,
                alpha: float = 0.5,
                operador: Literal["sobel", "scharr"] = "scharr",
                direccion: Optional[Literal["x", "y"]] = None):
        """
            Args:
                alpha: Fuerza del realce (0.1-2.0)
                    Menor que Laplaciano porque |∇I| tiene rango mayor
                
                operador: "sobel" (más rápido) o "scharr" (más isotrópico)
                
                direccion: Si None, usa magnitud total del gradiente
                        Si "x" o "y", solo realza en esa dirección
        """
        if alpha < 0:
            raise ValueError("alpha debe ser >= 0")
        
        self.alpha = alpha
        self.operador = operador
        self.direccion = direccion
    
    def __call__(self, img: np.ndarray) -> np.ndarray:
        """
            Aplica afilación basada en gradiente.
            
            Args:
                img: Imagen 2D (preferiblemente uint8 o float64)
                
            Returns:
                Imagen afilada del mismo tipo
        """
        self._validar_imagen(img)
        
        tipo_original = img.dtype
        img_float = img.astype(np.float64)
        
        # Calcular gradientes
        if self.operador == "sobel":
            if self.direccion == "x" or self.direccion is None:
                gx = cv2.Sobel(img_float, cv2.CV_64F, 1, 0, ksize=3)
            if self.direccion == "y" or self.direccion is None:
                gy = cv2.Sobel(img_float, cv2.CV_64F, 0, 1, ksize=3)
        else:  # scharr
            if self.direccion == "x" or self.direccion is None:
                gx = cv2.Scharr(img_float, cv2.CV_64F, 1, 0)
            if self.direccion == "y" or self.direccion is None:
                gy = cv2.Scharr(img_float, cv2.CV_64F, 0, 1)
        
        # Calcular magnitud o componente
        if self.direccion is None:
            magnitud = np.sqrt(gx**2 + gy**2)
        elif self.direccion == "x":
            magnitud = np.abs(gx)
        else:
            magnitud = np.abs(gy)
        
        # Normalizar magnitud para estabilidad
        if magnitud.max() > 0:
            magnitud = magnitud / magnitud.max() * (img_float.max() - img_float.min())
        
        # Combinar
        resultado = img_float + self.alpha * magnitud
        
        # Clip y conversión
        if np.issubdtype(tipo_original, np.integer):
            info = np.iinfo(tipo_original)
            resultado = np.clip(resultado, info.min, info.max)
        
        return resultado.astype(tipo_original)


class AfilacionWavelet(RealzadorAfilacion):
    """
        Afilación multiescala usando transformada wavelet.
        
        Descompone la imagen en diferentes escalas de frecuencia y realza
        selectivamente los coeficientes de detalle (bordes) mientras
        preserva la aproximación (bajas frecuencias).
        
        Algoritmo:
            1. Descomposición wavelet: Aproximación + Detalles (H, V, D)
            2. Amplificar coeficientes de detalle: c' = c · (1 + α)
            3. Reconstrucción wavelet inversa
        
        Ecuación:
            I_out = IDWT( LL, α·LH, α·HL, α·HH )
            
            donde:
                LL: Aproximación (bajas frecuencias, sin cambio)
                LH, HL, HH: Detalles horizontales, verticales, diagonales
                α: Factor de amplificación de detalles
        
        Interpretación:
            - Multiescala: Diferentes α por nivel de descomposición
            - Preserva estructuras grandes (LL)
            - Realza detalles en múltiples escalas simultáneamente
            - Menos artefactos que métodos espaciales simples
        
        Ventajas:
            - Control multiescala (diferentes α por nivel)
            - Menos artefactos de halo que unsharp masking
            - Preserva mejor la naturalidad de la imagen
            - Puede eliminar ruido y realzar simultáneamente (umbralización)
        
        Desventajas:
            - Más costoso computacionalmente
            - Artefactos de borde en descomposición
            - Requiere elegir wavelet madre apropiada
            - Parámetros más complejos (múltiples niveles)
        
        Usos típicos en microscopía:
            - Realce de detalles en múltiples escalas (orgánulos + células)
            - Denoising con preservación de bordes (con umbralización)
            - Análisis de texturas en diferentes resoluciones
            - Compresión de imágenes microscópicas con preservación de características
    """
    nombre = "afilacion_wavelet"
    
    def __init__(self,
                wavelet: str = 'db1',
                niveles: int = 2,
                alphas: Optional[Union[float, Tuple[float, ...]]] = None,
                umbral_denoising: Optional[float] = None):
        """
            Args:
                wavelet: Tipo de wavelet ('db1'=Haar, 'db2', 'sym2', etc.)
                        'db1': Simple, bueno para bordes agudos
                        'sym4': Buen balance para imágenes naturales
                
                niveles: Niveles de descomposición (1-4 típicamente)
                        Más niveles = más escalas analizadas
                
                alphas: Factor de amplificación por nivel
                    - float: Mismo α para todos los niveles
                    - tuple: α específico para cada nivel (debe coincidir con niveles)
                    - None: Default 0.5 para todos
                
                umbral_denoising: Si se especifica, umbraliza coeficientes pequeños
                                (elimina ruido antes de realzar)
                                Valor típico: std_noise * 3
        """
        try:
            import pywt
        except ImportError:
            raise ImportError("Se requiere PyWavelets (pywt). Instalar: pip install PyWavelets")
        
        self.wavelet = wavelet
        self.niveles = niveles
        
        if isinstance(alphas, (int, float)):
            self.alphas = [float(alphas)] * niveles
        elif alphas is None:
            self.alphas = [0.5] * niveles
        else:
            if len(alphas) != niveles:
                raise ValueError(f"alphas debe tener {niveles} elementos")
            self.alphas = list(alphas)
        
        self.umbral_denoising = umbral_denoising
    
    def __call__(self, img: np.ndarray) -> np.ndarray:
        """
            Aplica afilación wavelet.
            
            Args:
                img: Imagen 2D
                
            Returns:
                Imagen afilada del mismo tipo
        """
        import pywt
        
        self._validar_imagen(img)
        
        tipo_original = img.dtype
        img_float = img.astype(np.float64)
        
        # Normalizar a [0, 1] para wavelet
        img_min, img_max = img_float.min(), img_float.max()
        img_norm = (img_float - img_min) / (img_max - img_min + 1e-10)
        
        # Descomposición multiescala
        coeffs = pywt.wavedec2(img_norm, self.wavelet, level=self.niveles)
        
        # Procesar coeficientes de detalle por nivel
        coeffs_procesados = [coeffs[0]]  # Aproximación sin cambio
        
        for i, detalles in enumerate(coeffs[1:], 1):
            cH, cV, cD = detalles
            
            # Denoising opcional
            if self.umbral_denoising is not None:
                umbral = self.umbral_denoising
                cH = pywt.threshold(cH, umbral, mode='soft')
                cV = pywt.threshold(cV, umbral, mode='soft')
                cD = pywt.threshold(cD, umbral, mode='soft')
            
            # Amplificar detalles
            alpha = self.alphas[i-1]
            cH *= (1 + alpha)
            cV *= (1 + alpha)
            cD *= (1 + alpha)
            
            coeffs_procesados.append((cH, cV, cD))
        
        # Reconstrucción
        img_reconstruida = pywt.waverec2(coeffs_procesados, self.wavelet)
        
        # Recortar al tamaño original (wavelet puede cambiar dimensiones)
        img_reconstruida = img_reconstruida[:img.shape[0], :img.shape[1]]
        
        # Desnormalizar
        resultado = img_reconstruida * (img_max - img_min) + img_min
        
        # Clip y conversión
        if np.issubdtype(tipo_original, np.integer):
            info = np.iinfo(tipo_original)
            resultado = np.clip(resultado, info.min, info.max)
        
        return resultado.astype(tipo_original)


class DeconvolucionLucy(RealzadorAfilacion):
    """
        Deconvolución de Lucy-Richardson para restauración de imagen.
        
        Método iterativo que restaura una imagen difuminada conociendo
        (o estimando) la función de dispersión de punto (PSF).
        
        Algoritmo:
            1. Inicializar: I⁰ = imagen observada
            2. Iterar:
            - Convolución: C = Iⁿ * PSF
            - Relativa: R = I_observada / C
            - Corrección: Iⁿ⁺¹ = Iⁿ · (R * PSF_rotada)
            3. Normalizar entre iteraciones
        
        Ecuación iterativa:
            I⁽ᵏ⁺¹⁾ = I⁽ᵏ⁾ · [ (I_obs / (I⁽ᵏ⁾ * PSF)) * PSF* ]
            
            donde * es convolución y PSF* es PSF rotada 180°
        
        Interpretación:
            - Maximización de verosimilitud (Poisson)
            - Preserva no-negatividad (importante para intensidades)
            - Converge a máximo likelihood pero puede amplificar ruido
        
        Ventajas:
            - Restauración física (invierte el modelo de adquisición)
            - Preserva no-negatividad de la imagen
            - Mejor que filtros inversos simples (menos amplificación de ruido)
            - Convergencia monótona (estable numéricamente)
        
        Desventajas:
            - Requiere conocer/estimar la PSF
            - Lento (múltiples iteraciones, cada una con convoluciones)
            - Artefactos de ringing si se exceden iteraciones
            - Amplifica ruido si PSF no es precisa o hay mucho ruido
            - Requiere parada temprana (criterio de convergencia)
        
        Usos típicos en microscopía:
            - Corrección de difracción óptica (PSF del microscopio)
            - Restauración de imágenes confocal desenfocadas
            - Deconvolución de imágenes de campo ancho
            - Mejora de resolución en super-resolución computacional
            - Preprocesamiento para cuantificación precisa de intensidad
    """
    nombre = "deconvolucion_lucy"
    
    def __init__(self,
                psf: Optional[np.ndarray] = None,
                sigma_psf: float = 2.0,
                iteraciones: int = 10,
                clip: bool = True):
        """
            Args:
                psf: Kernel de función de dispersión de punto (2D)
                    Si None, se usa gaussiano isotrópico con sigma_psf
                    Debe estar normalizada (suma = 1)
                
                sigma_psf: Desviación estándar de la PSF gaussiana (si psf=None)
                        Valores típicos: 1.0-3.0 (depende de NA y λ)
                
                iteraciones: Número de iteraciones de Richardson-Lucy
                            Valores típicos: 10-50
                            Más iteraciones = más afilación pero más ruido
                
                clip: Si True, fuerza no-negatividad en resultado
        """
        if psf is not None and psf.ndim != 2:
            raise ValueError("PSF debe ser 2D")
        if iteraciones < 1:
            raise ValueError("iteraciones debe ser >= 1")
        
        if psf is None:
            # Crear PSF gaussiana
            size = int(6 * sigma_psf) | 1  # Tamaño impar
            x = np.arange(size) - size // 2
            g = np.exp(-x**2 / (2 * sigma_psf**2))
            psf = np.outer(g, g)
            psf = psf / psf.sum()
        
        self.psf = psf
        self.iteraciones = iteraciones
        self.clip = clip
    
    def __call__(self, img: np.ndarray) -> np.ndarray:
        """
            Aplica deconvolución de Lucy-Richardson.
            
            Args:
                img: Imagen 2D (preferiblemente float o uint16 para precisión)
                
            Returns:
                Imagen restaurada del mismo tipo
        """
        self._validar_imagen(img)
        
        tipo_original = img.dtype
        
        # Usar skimage para deconvolución robusta
        img_float = img.astype(np.float64)
        
        # Normalizar para estabilidad
        img_min = img_float.min()
        img_float = img_float - img_min
        
        # Aplicar deconvolución
        resultado = restoration.richardson_lucy(
            img_float, 
            self.psf, 
            num_iter=self.iteraciones,
            clip=self.clip
        )
        
        # Desnormalizar
        resultado = resultado + img_min
        
        # Clip y conversión
        if np.issubdtype(tipo_original, np.integer):
            info = np.iinfo(tipo_original)
            resultado = np.clip(resultado, info.min, info.max)
        
        return resultado.astype(tipo_original)