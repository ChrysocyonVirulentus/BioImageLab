"""
Filtros multiescala para análisis de imágenes en diferentes niveles de resolución.

Estos filtros operan simultáneamente en múltiples escalas espaciales, permitiendo
detectar estructuras de diferentes tamaños y analizar jerarquías de características.

Características:
- Detectan características a múltiples escalas simultáneamente
- Preservan información de diferentes niveles de detalle
- Útiles para detección de blobs, bordes y texturas
- Permiten análisis jerárquico de la imagen

Ventajas sobre filtros de escala única:
- Detectan objetos de tamaños variables sin ajuste manual
- Más robustos ante variaciones de escala
- Permiten análisis de estructuras jerárquicas

Tipos disponibles:
- DiferenciaGaussiana (DoG): Detección de blobs y aproximación de Laplaciano
- DiferenciaLaplaciana: Detección precisa de estructuras a diferentes escalas
- PiramideLaplaciana: Descomposición multiescala para análisis/reconstrucción
- Wavelet: Análisis tiempo-frecuencia con localización espacial
"""

import numpy as np
import cv2
from typing import List, Tuple, Optional, Literal
import warnings


class FiltroMultiescala:
    """
        Clase base para filtros que operan en múltiples escalas.
        
        Los filtros multiescala analizan la imagen en diferentes niveles de resolución
        simultáneamente, permitiendo detectar estructuras de diferentes tamaños.
    """
    nombre = "filtro_multiescala_base"
    
    def __call__(self, img: np.ndarray) -> np.ndarray:
        """
        Aplica el filtro multiescala a la imagen.
        
        Args:
            img: Array 2D (Y, X) con la imagen a filtrar
            
        Returns:
            Imagen filtrada o descompuesta (puede tener diferente estructura)
        """
        raise NotImplementedError("Subclases deben implementar __call__")
    
    def _validar_imagen(self, img: np.ndarray):
        """Valida que la imagen sea 2D."""
        if img.ndim != 2:
            raise ValueError(f"La imagen debe ser 2D, tiene {img.ndim} dimensiones")


class DiferenciaGaussiana(FiltroMultiescala):
    """
        Filtro de Diferencia de Gaussianas (DoG) para detección de blobs.
        
        Resta dos imágenes gaussianas de diferente sigma, aproximando el operador
        Laplaciano de Gaussiana (LoG). Detecta regiones circulares (blobs) que
        difieren del fondo.
        
        Ecuación: DoG(x,y) = G(x,y,σ₂) - G(x,y,σ₁)
        
        Ventajas:
            - Computacionalmente eficiente (más que LoG directo)
            - Más eficiente que LoG (dos gaussianas vs Laplaciano de gaussiana)
            - Detecta blobs de tamaño específico
            - Robusto ante ruido
            - Invariante a cambios de iluminación
            - Fundamento biológico (modela visión)
        
        Desventajas:
            - Sensible a la elección de sigmas
            - Puede crear artefactos de ringing
            - No detecta estructuras elongadas
            - No es isotrópico perfecto (aproximación)
            - Puede detectar bordes fantasmas (artefactos)
            - Menos preciso que Canny en localización
        
        Usos típicos:
            - Detección de células/blobs de diferentes tamaños
            - Segmentación de núcleos con variabilidad de tamaño
            - Realce de puntos sinápticos (puncta)
            - Identificación de vesículas y puncta
            - Detección de spots de fluorescencia
            - Preprocesamiento para detección de blobs
            - Aproximación rápida de LoG
        
        Relación con LoG:
            DoG ≈ -∇²G cuando σ₂/σ₁ ≈ 1.6 (relación típica en Scale-Space)
    """
    nombre = "diferencia_gaussiana"
    
    def __init__(self, sigma1: float = 1.0, sigma2: float = 2.0, k: Optional[float] = None):
        """
            Args:
                sigma1: Desviación estándar del primer gaussiano (escala fina)
                        Típicamente 0.5-2.0 para microscopía
                sigma2: Desviación estándar del segundo gaussiano (escala gruesa)
                        Típicamente 1.5-4.0 para microscopía
                k: Factor multiplicativo para sigma2 (si se proporciona, sigma2 = sigma1 * k)
                    Valor típico: k=1.6 (teoría de Scale-Space)
            
            Nota: Si se proporciona k, sigma2 se ignora y se calcula como sigma1 * k
        """
        if sigma1 <= 0:
            raise ValueError("sigma1 debe ser > 0")
        
        self.sigma1 = sigma1
        
        if k is not None:
            if k <= 1.0:
                raise ValueError("k debe ser > 1.0 para que sigma2 > sigma1")
            self.sigma2 = sigma1 * k
            self.k = k
        else:
            if sigma2 <= sigma1:
                raise ValueError("sigma2 debe ser > sigma1")
            self.sigma2 = sigma2
            self.k = sigma2 / sigma1
    
    def __call__(self, img: np.ndarray) -> np.ndarray:
        """
            Aplica la diferencia de gaussianas.
            
            Args:
                img: Imagen 2D a filtrar
                
            Returns:
                Imagen DoG (realza blobs del tamaño entre sigma1 y sigma2)
        """
        self._validar_imagen(img)
        
        # Convertir a float para cálculos
        img_float = img.astype(np.float64)
        
        # Aplicar dos filtros gaussianos
        gauss1 = cv2.GaussianBlur(img_float, (0, 0), self.sigma1)
        gauss2 = cv2.GaussianBlur(img_float, (0, 0), self.sigma2)
        
        # Diferencia
        dog = gauss1 - gauss2
        
        # Normalizar a [0, 255] o rango del tipo original
        dog = dog - dog.min()
        if dog.max() > 0:
            dog = dog / dog.max()
        
        # Escalar de vuelta al rango del tipo original
        if np.issubdtype(img.dtype, np.integer):
            info = np.iinfo(img.dtype)
            dog = dog * (info.max - info.min) + info.min
            dog = np.clip(dog, info.min, info.max)
        
        return dog.astype(img.dtype)


class DiferenciaLaplaciana(FiltroMultiescala):
    """
        Filtro de Diferencia de Laplacianas (DoL) para detección multiescala.
        
        Similar a DoG pero usa el operador Laplaciano directamente en lugar de
        gaussianas, proporcionando detección más precisa de estructuras.
        
        Ecuación: DoL(x,y) = ∇²G(x,y,σ₂) - ∇²G(x,y,σ₁)
        
        Ventajas:
            - Detección más precisa que DoG
            - Mejor localización de bordes
            - Respuesta más limpia a estructuras circulares
        
        Desventajas:
            - Más sensible al ruido que DoG
            - Computacionalmente más costoso
            - Requiere preprocesamiento para imágenes ruidosas
        
        Usos típicos:
            - Detección precisa de núcleos
            - Identificación de spots en imágenes de alta calidad
            - Análisis de estructuras circulares
            - Detección de cambios de curvatura
    """
    nombre = "diferencia_laplaciana"
    
    def __init__(self, sigma1: float = 1.0, sigma2: float = 2.0, ksize: int = 3):
        """
            Args:
                sigma1: Escala fina para suavizado previo
                sigma2: Escala gruesa para suavizado previo
                ksize: Tamaño del kernel Laplaciano (debe ser impar: 1, 3, 5, 7)
        """
        if sigma1 <= 0 or sigma2 <= sigma1:
            raise ValueError("sigma1 debe ser > 0 y sigma2 > sigma1")
        if ksize % 2 == 0 or ksize < 1:
            raise ValueError("ksize debe ser impar y >= 1")
        
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.ksize = ksize
    
    def __call__(self, img: np.ndarray) -> np.ndarray:
        """
            Aplica la diferencia de laplacianas.
            
            Args:
                img: Imagen 2D a filtrar
                
            Returns:
                Imagen DoL (detección de estructuras a múltiples escalas)
        """
        self._validar_imagen(img)
        
        img_float = img.astype(np.float64)
        
        # Suavizar a dos escalas
        gauss1 = cv2.GaussianBlur(img_float, (0, 0), self.sigma1)
        gauss2 = cv2.GaussianBlur(img_float, (0, 0), self.sigma2)
        
        # Aplicar Laplaciano a cada escala
        lap1 = cv2.Laplacian(gauss1, cv2.CV_64F, ksize=self.ksize)
        lap2 = cv2.Laplacian(gauss2, cv2.CV_64F, ksize=self.ksize)
        
        # Diferencia
        dol = lap1 - lap2
        
        # Normalizar
        dol = dol - dol.min()
        if dol.max() > 0:
            dol = dol / dol.max()
        
        if np.issubdtype(img.dtype, np.integer):
            info = np.iinfo(img.dtype)
            dol = dol * (info.max - info.min) + info.min
            dol = np.clip(dol, info.min, info.max)
        
        return dol.astype(img.dtype)


class PiramideLaplaciana(FiltroMultiescala):
    """
        Pirámide Laplaciana para descomposición multiescala de la imagen.
        
        Descompone la imagen en una serie de imágenes de diferente resolución,
        donde cada nivel contiene información de frecuencias específicas.
        
        Estructura:
            Nivel 0: Detalles más finos (altas frecuencias)
            Nivel 1-N: Detalles progresivamente más gruesos
            Nivel N+1: Aproximación de baja resolución (residuo)
        
        Ventajas:
            - Separación completa de escalas
            - Reconstrucción perfecta posible
            - Útil para análisis jerárquico
            - Base para fusión de imágenes
        
        Desventajas:
            - Genera múltiples imágenes (mayor uso de memoria)
            - Requiere más procesamiento
            - Necesita cuidado en los bordes
        
        Usos típicos:
            - Análisis multiescala de texturas
            - Fusión de imágenes multiescala
            - Compresión de imágenes
            - Análisis de frecuencias por bandas
            - Detección de características a diferentes escalas
    """
    nombre = "piramide_laplaciana"
    
    def __init__(self, niveles: int = 4):
        """
            Args:
                niveles: Número de niveles de la pirámide (típicamente 3-6)
                        Más niveles = análisis de escalas más gruesas
        """
        if niveles < 1:
            raise ValueError("niveles debe ser >= 1")
        
        self.niveles = niveles
        self.piramide_gaussiana: Optional[List[np.ndarray]] = None
        self.piramide_laplaciana: Optional[List[np.ndarray]] = None
    
    def __call__(self, img: np.ndarray) -> List[np.ndarray]:
        """
            Construye la pirámide laplaciana.
            
            Args:
                img: Imagen 2D a descomponer
                
            Returns:
                Lista de imágenes [L0, L1, ..., LN, Residuo]
                donde Li es el nivel i de la pirámide laplaciana
        """
        self._validar_imagen(img)
        
        img_float = img.astype(np.float64)
        
        # Construir pirámide gaussiana (suavizado + submuestreo)
        self.piramide_gaussiana = [img_float]
        for _ in range(self.niveles):
            # Reducir: suavizar y submuestrear
            img_suavizada = cv2.GaussianBlur(self.piramide_gaussiana[-1], (5, 5), 0)
            img_reducida = cv2.pyrDown(img_suavizada)
            self.piramide_gaussiana.append(img_reducida)
        
        # Construir pirámide laplaciana (diferencias entre niveles)
        self.piramide_laplaciana = []
        for i in range(self.niveles):
            # Expandir nivel superior
            nivel_expandido = cv2.pyrUp(self.piramide_gaussiana[i + 1])
            
            # Ajustar tamaño si no coincide exactamente
            if nivel_expandido.shape != self.piramide_gaussiana[i].shape:
                nivel_expandido = cv2.resize(
                    nivel_expandido,
                    (self.piramide_gaussiana[i].shape[1], self.piramide_gaussiana[i].shape[0])
                )
            
            # Diferencia = detalles de esta escala
            laplaciano = self.piramide_gaussiana[i] - nivel_expandido
            self.piramide_laplaciana.append(laplaciano)
        
        # Agregar el residuo (nivel más grueso)
        self.piramide_laplaciana.append(self.piramide_gaussiana[-1])
        
        return self.piramide_laplaciana
    
    def reconstruir(self) -> np.ndarray:
        """
            Reconstruye la imagen original desde la pirámide laplaciana.
            
            Returns:
                Imagen reconstruida
                
            Raises:
                RuntimeError: Si no se ha construido la pirámide
        """
        if self.piramide_laplaciana is None:
            raise RuntimeError("Primero debes construir la pirámide con __call__()")
        
        # Comenzar desde el nivel más grueso (residuo)
        img_reconstruida = self.piramide_laplaciana[-1]
        
        # Ir agregando detalles de cada nivel
        for i in range(self.niveles - 1, -1, -1):
            # Expandir
            img_expandida = cv2.pyrUp(img_reconstruida)
            
            # Ajustar tamaño
            if img_expandida.shape != self.piramide_laplaciana[i].shape:
                img_expandida = cv2.resize(
                    img_expandida,
                    (self.piramide_laplaciana[i].shape[1], self.piramide_laplaciana[i].shape[0])
                )
            
            # Sumar detalles
            img_reconstruida = img_expandida + self.piramide_laplaciana[i]
        
        return img_reconstruida


class WaveletTransform(FiltroMultiescala):
    """
        Transformada Wavelet para análisis tiempo-frecuencia multiescala.
        
        Descompone la imagen usando wavelets, proporcionando localización tanto
        en espacio como en frecuencia. Superior a Fourier para análisis local.
        
        Ventajas:
            - Localización espacio-frecuencia simultánea
            - Análisis multiescala adaptativo
            - Compresión eficiente
            - Detección de singularidades
        
        Desventajas:
            - Requiere librería especializada (pywt)
            - Elección de wavelet afecta resultados
            - Más complejo de interpretar
        
        Usos típicos:
            - Denoising selectivo por subbandas
            - Compresión de imágenes
            - Análisis de texturas multiescala
            - Detección de bordes multiescala
            - Separación de estructuras por escala
        
        Familias de wavelets comunes:
            - 'db' (Daubechies): Buenas para propósito general
            - 'haar': Simple, buena para bordes
            - 'sym' (Symlets): Simétricas, buenas para análisis
            - 'coif' (Coiflets): Suaves, buenas para imágenes naturales
    """
    nombre = "wavelet"
    
    def __init__(self, 
                wavelet: str = 'db4',
                nivel: int = 3,
                modo: Literal['zero', 'constant', 'symmetric', 'periodic', 'smooth', 'periodization'] = 'symmetric'):
        """
            Args:
                wavelet: Familia de wavelet a usar ('db1'-'db20', 'haar', 'sym2'-'sym20', 'coif1'-'coif5')
                nivel: Número de niveles de descomposición (típicamente 2-5)
                modo: Modo de manejo de bordes
                        'symmetric': Reflexión simétrica (default, bueno para imágenes)
                        'periodic': Asume imagen periódica
                        'zero': Padding con ceros
        """
        try:
            import pywt
            self.pywt = pywt
        except ImportError:
            raise ImportError(
                "Se requiere PyWavelets (pywt). Instalar con: pip install PyWavelets"
            )
        
        # Validar que la wavelet existe
        if wavelet not in self.pywt.wavelist(kind='discrete'):
            raise ValueError(f"Wavelet '{wavelet}' no válida. Usar pywt.wavelist() para ver opciones")
        
        if nivel < 1:
            raise ValueError("nivel debe ser >= 1")
        
        self.wavelet = wavelet
        self.nivel = nivel
        self.modo = modo
        self.coeficientes: Optional[List] = None
    
    def __call__(self, img: np.ndarray) -> Tuple[np.ndarray, List[Tuple[np.ndarray, np.ndarray, np.ndarray]]]:
        """
            Aplica la transformada wavelet 2D.
            
            Args:
                img: Imagen 2D a descomponer
                
            Returns:
                Tupla (cA, [detalles])
                donde:
                    cA: Coeficientes de aproximación (baja frecuencia)
                    detalles: Lista de tuplas (cH, cV, cD) para cada nivel
                        cH: Detalles horizontales
                        cV: Detalles verticales
                        cD: Detalles diagonales
        """
        self._validar_imagen(img)
        
        # Convertir a float
        img_float = img.astype(np.float64)
        
        # Descomposición wavelet multinivel
        self.coeficientes = self.pywt.wavedec2(
            img_float,
            wavelet=self.wavelet,
            level=self.nivel,
            mode=self.modo
        )
        
        # coeficientes[0] = aproximación
        # coeficientes[1:] = (cH, cV, cD) para cada nivel
        return self.coeficientes[0], self.coeficientes[1:]
    
    def reconstruir(self, coeficientes: Optional[List] = None) -> np.ndarray:
        """
            Reconstruye la imagen desde los coeficientes wavelet.
            
            Args:
                coeficientes: Coeficientes wavelet (si None, usa los últimos calculados)
                
            Returns:
                Imagen reconstruida
                
            Raises:
                RuntimeError: Si no hay coeficientes disponibles
        """
        if coeficientes is None:
            coeficientes = self.coeficientes
        
        if coeficientes is None:
            raise RuntimeError("Primero debes calcular los coeficientes con __call__()")
        
        # Reconstrucción
        img_reconstruida = self.pywt.waverec2(
            coeficientes,
            wavelet=self.wavelet,
            mode=self.modo
        )
        
        return img_reconstruida
    
    def denoising(self, img: np.ndarray, umbral: float = 30.0) -> np.ndarray:
        """
            Denoising mediante umbralización de coeficientes wavelet.
            
            Args:
                img: Imagen a limpiar
                umbral: Umbral para eliminar coeficientes pequeños (ruido)
                
            Returns:
                Imagen limpia
        """
        # Descomponer
        cA, detalles = self(img)
        
        # Umbralizar coeficientes de detalle (soft thresholding)
        detalles_umbralizados = []
        for cH, cV, cD in detalles:
            cH_umbral = self.pywt.threshold(cH, umbral, mode='soft')
            cV_umbral = self.pywt.threshold(cV, umbral, mode='soft')
            cD_umbral = self.pywt.threshold(cD, umbral, mode='soft')
            detalles_umbralizados.append((cH_umbral, cV_umbral, cD_umbral))
        
        # Reconstruir
        coeficientes_limpios = [cA] + detalles_umbralizados
        return self.reconstruir(coeficientes_limpios)