"""
Filtros no locales para reducción de ruido avanzada preservando texturas.

Estos filtros operan usando información de toda la imagen (o grandes regiones),
no solo vecindades locales. Comparan parches (bloques) de píxeles para encontrar
estructuras similares y promediarlas de manera ponderada.

Características:
- Usan similitud de parches en toda la imagen
- Preservan texturas y detalles finos excepcionalemente bien
- Eliminan ruido efectivamente sin difuminar bordes
- Computacionalmente intensivos pero muy efectivos

Principio fundamental:
Los píxeles con contextos similares (parches parecidos) deben tener valores
similares, incluso si están muy alejados espacialmente.

Ventajas sobre filtros locales:
- Superior preservación de texturas y detalles
- Mejor eliminación de ruido manteniendo estructuras
- No asume suavidad local (más flexible)
- Excelente para imágenes con patrones repetitivos

Tipos disponibles:
- NLM (Non-Local Means): Promedio ponderado por similitud de parches
- BM3D (Block-Matching 3D): Colaboración de parches similares en 3D
"""

import numpy as np
import cv2
from typing import Optional, Literal
import warnings

@registrar_en("filtrado")
class FiltroNoLocal:
    """
        Clase base para filtros no locales.
        
        Los filtros no locales explotan la auto-similitud en la imagen: regiones
        distantes pero con estructura similar se usan para reducir ruido de manera
        colaborativa.
        
        Concepto clave: Un píxel se denoisea usando información de todos los píxeles
        con vecindades (parches) similares, sin importar la distancia espacial.
    """
    nombre = "filtro_no_local_base"
    
    def __call__(self, img: np.ndarray) -> np.ndarray:
        """
            Aplica el filtro no local a la imagen.
            
            Args:
                img: Array 2D (Y, X) con la imagen a filtrar
                
            Returns:
                Imagen filtrada del mismo tipo y forma
        """
        raise NotImplementedError("Subclases deben implementar __call__")
    
    def _validar_imagen(self, img: np.ndarray):
        """Valida que la imagen sea 2D."""
        if img.ndim != 2:
            raise ValueError(f"La imagen debe ser 2D, tiene {img.ndim} dimensiones")

@registrar_en("filtrado")
class NonLocalMeans(FiltroNoLocal):
    """
        Filtro Non-Local Means (NLM) para denoising preservando texturas.
        
        Algoritmo introducido por Buades et al. (2005). Cada píxel se denoisea
        como un promedio ponderado de píxeles con parches similares en toda la imagen.
        
        Ecuación:
            NL[u](x) = Σ w(x,y) * u(y)
        
        donde w(x,y) = exp(-||N(x) - N(y)||² / h²)
        N(x) es el parche centrado en x
        h controla el decaimiento de los pesos
        
        Ventajas:
            - Excelente preservación de texturas finas
            - No difumina bordes ni estructuras
            - Efectivo en ruido gaussiano moderado
            - Resultados visuales superiores
        
        Desventajas:
            - Computacionalmente costoso (O(n²) naive)
            - Puede crear artefactos en regiones uniformes
            - Sensible a la elección de parámetros
            - Lento en imágenes grandes sin optimización
        
        Usos típicos:
            - Denoising de imágenes de microscopía confocal
            - Limpieza de imágenes con texturas repetitivas
            - Preprocesamiento manteniendo detalles celulares
            - Reducción de ruido en imágenes de alta resolución
            - Mejora de calidad en time-lapse conservando estructuras
        
        Variantes:
            - Fast NLM: Implementación optimizada en OpenCV
            - NLM colored: Para imágenes color
    """
    nombre = "non_local_means"
    
    def __init__(self,
                h: float = 10.0,
                template_window_size: int = 7,
                search_window_size: int = 21):
        """
            Args:
                h: Parámetro de filtrado (fuerza del denoising)
                Valores típicos:
                    - 3-10: Ruido bajo, preservar más detalle
                    - 10-15: Ruido moderado (recomendado para microscopía)
                    - 15-30: Ruido alto, más suavizado
                Mayor h = más suavizado pero más pérdida de detalle
                
                template_window_size: Tamaño del parche de similitud (debe ser impar)
                                    Valores típicos: 5, 7, 9
                                    Mayor tamaño = considera más contexto
                                    Típicamente 7 es bueno para microscopía
                
                search_window_size: Tamaño de la ventana de búsqueda (debe ser impar)
                                Valores típicos: 15, 21, 35
                                Mayor tamaño = busca parches más lejos (más lento)
                                Típicamente 21 es buen balance velocidad/calidad
            
            Nota:
                template_window_size debe ser <= search_window_size
        """
        if h <= 0:
            raise ValueError("h debe ser > 0")
        if template_window_size % 2 == 0 or template_window_size < 1:
            raise ValueError("template_window_size debe ser impar y >= 1")
        if search_window_size % 2 == 0 or search_window_size < 1:
            raise ValueError("search_window_size debe ser impar y >= 1")
        if template_window_size > search_window_size:
            raise ValueError("template_window_size debe ser <= search_window_size")
        
        self.h = h
        self.template_window_size = template_window_size
        self.search_window_size = search_window_size
    
    def __call__(self, img: np.ndarray) -> np.ndarray:
        """
            Aplica el filtro Non-Local Means.
            
            Usa la implementación optimizada de OpenCV (fastNlMeansDenoising)
            que es significativamente más rápida que la implementación naive.
            
            Args:
                img: Imagen 2D a filtrar
                
            Returns:
                Imagen filtrada del mismo tipo
        """
        self._validar_imagen(img)
        
        # OpenCV requiere uint8 o uint16
        tipo_original = img.dtype
        
        if img.dtype == np.uint8:
            # Procesamiento directo
            img_filtrada = cv2.fastNlMeansDenoising(
                img,
                None,
                h=self.h,
                templateWindowSize=self.template_window_size,
                searchWindowSize=self.search_window_size
            )
        elif img.dtype == np.uint16:
            # OpenCV maneja uint16 nativamente
            img_filtrada = cv2.fastNlMeansDenoising(
                img,
                None,
                h=self.h,
                templateWindowSize=self.template_window_size,
                searchWindowSize=self.search_window_size
            )
        else:
            # Convertir a uint8 temporalmente
            warnings.warn(
                f"Convirtiendo de {img.dtype} a uint8 para NLM. "
                "Para mejor calidad, usa uint8 o uint16.",
                RuntimeWarning
            )
            # Normalizar a [0, 255]
            img_normalizada = img.astype(np.float64)
            img_normalizada = (img_normalizada - img_normalizada.min())
            if img_normalizada.max() > 0:
                img_normalizada = img_normalizada / img_normalizada.max() * 255
            img_uint8 = img_normalizada.astype(np.uint8)
            
            # Filtrar
            img_filtrada_uint8 = cv2.fastNlMeansDenoising(
                img_uint8,
                None,
                h=self.h,
                templateWindowSize=self.template_window_size,
                searchWindowSize=self.search_window_size
            )
            
            # Convertir de vuelta
            img_filtrada = img_filtrada_uint8.astype(tipo_original)
        
        return img_filtrada

@registrar_en("filtrado")
class BlockMatching3D(FiltroNoLocal):
    """
        Block-Matching 3D (BM3D) para denoising estado del arte.
        
        Algoritmo introducido por Dabov et al. (2007), considerado uno de los
        mejores métodos de denoising. Usa colaboración de parches similares
        agrupados en estructuras 3D para filtrado conjunto.
        
        Pipeline de BM3D:
            1. Block-matching: Encontrar parches similares
            2. Agrupar en stacks 3D
            3. Transformada 3D (wavelet o DCT)
            4. Hard thresholding (estimación básica)
            5. Wiener filtering (refinamiento)
            6. Agregación de resultados
        
        Ventajas:
            - Estado del arte en denoising (mejor PSNR/SSIM)
            - Preservación excepcional de texturas y detalles
            - Efectivo en alto ruido (σ > 25)
            - No introduce artefactos visuales
            - Basado en principios sólidos (sparse representation)
        
        Desventajas:
            - Muy computacionalmente intensivo
            - Requiere librería externa (bm3d)
            - Muchos parámetros (aunque defaults son buenos)
            - Más lento que NLM
        
        Usos típicos:
            - Denoising de alta calidad para publicaciones
            - Imágenes de microscopía con ruido severo
            - Recuperación de señal en imágenes de bajo SNR
            - Preprocesamiento crítico donde calidad > velocidad
            - Análisis cuantitativo requiriendo máxima fidelidad
        
        Nota:
            Requiere instalación: pip install bm3d
    """
    nombre = "bm3d"
    
    def __init__(self,
                sigma_psd: float = 25.0,
                stage_arg: Literal['hard', 'all'] = 'all'):
        """
            Args:
                sigma_psd: Desviación estándar del ruido estimado
                        Valores típicos:
                            - 10-20: Ruido bajo
                            - 20-40: Ruido moderado (común en microscopía)
                            - 40-80: Ruido alto
                        Puede estimarse automáticamente si se desconoce
                
                stage_arg: Etapa del algoritmo a ejecutar
                        'hard': Solo hard thresholding (más rápido, calidad menor)
                        'all': Hard thresholding + Wiener (completo, mejor calidad)
                        Recomendado: 'all' para denoising final
            
            Nota:
                El parámetro sigma_psd es crítico. Si es muy bajo, no elimina ruido.
                Si es muy alto, sobre-suaviza. Puede estimarse de regiones uniformes.
        """
        if sigma_psd <= 0:
            raise ValueError("sigma_psd debe ser > 0")
        if stage_arg not in ['hard', 'all']:
            raise ValueError("stage_arg debe ser 'hard' o 'all'")
        
        try:
            import bm3d
            self.bm3d = bm3d
        except ImportError:
            raise ImportError(
                "Se requiere bm3d. Instalar con: pip install bm3d\n"
                "Para instalación completa: pip install bm3d[full]"
            )
        
        self.sigma_psd = sigma_psd
        self.stage_arg = stage_arg
    
    def __call__(self, img: np.ndarray) -> np.ndarray:
        """
            Aplica el filtro BM3D.
            
            Args:
                img: Imagen 2D a filtrar
                
            Returns:
                Imagen filtrada del mismo tipo
        """
        self._validar_imagen(img)
        
        # BM3D requiere float en [0, 1] o uint8
        tipo_original = img.dtype
        
        # Convertir a float [0, 1]
        if np.issubdtype(img.dtype, np.integer):
            info = np.iinfo(img.dtype)
            img_float = img.astype(np.float64) / info.max
        else:
            # Ya es float, normalizar a [0, 1]
            img_float = img.astype(np.float64)
            img_min, img_max = img_float.min(), img_float.max()
            if img_max > img_min:
                img_float = (img_float - img_min) / (img_max - img_min)
        
        # Aplicar BM3D
        # sigma_psd debe estar en la escala de la imagen [0, 1]
        sigma_normalizado = self.sigma_psd / 255.0
        
        img_filtrada = self.bm3d.bm3d(
            img_float,
            sigma_psd=sigma_normalizado,
            stage_arg=self.stage_arg
        )
        
        # Convertir de vuelta al tipo original
        if np.issubdtype(tipo_original, np.integer):
            info = np.iinfo(tipo_original)
            img_filtrada = img_filtrada * info.max
            img_filtrada = np.clip(img_filtrada, info.min, info.max)
            img_filtrada = img_filtrada.astype(tipo_original)
        else:
            # Devolver en [0, 1] para float
            img_filtrada = img_filtrada.astype(tipo_original)
        
        return img_filtrada
    
    def estimar_sigma(self, img: np.ndarray, metodo: str = 'mad') -> float:
        """
            Estima la desviación estándar del ruido en la imagen.
            
            Args:
                img: Imagen de la cual estimar el ruido
                metodo: Método de estimación
                    'mad': Median Absolute Deviation (robusto)
                    'std_wavelet': Usando coeficientes wavelet de alta frecuencia
            
            Returns:
                Estimación de sigma (en escala 0-255)
        """
        self._validar_imagen(img)
        
        if metodo == 'mad':
            # MAD de los coeficientes Laplacianos
            # Aproximación rápida del ruido
            laplacian = cv2.Laplacian(img, cv2.CV_64F)
            sigma_estimado = np.median(np.abs(laplacian)) / 0.6745
            
        elif metodo == 'std_wavelet':
            # Usar desviación estándar de subbanda HH (alta-alta) de wavelet
            try:
                import pywt
                # Descomposición wavelet
                coeffs = pywt.dwt2(img.astype(np.float64), 'db1')
                _, (_, _, cD) = coeffs  # cD es la subbanda diagonal (HH)
                sigma_estimado = np.median(np.abs(cD)) / 0.6745
            except ImportError:
                raise ImportError("Método 'std_wavelet' requiere pywt: pip install PyWavelets")
        else:
            raise ValueError(f"Método '{metodo}' no reconocido. Usar 'mad' o 'std_wavelet'")
        
        return float(sigma_estimado)

@registrar_en("filtrado")
class NonLocalMeansMultiescala(FiltroNoLocal):
    """
        Variante multiescala de Non-Local Means para mejor manejo de ruido.
        
        Aplica NLM en múltiples escalas (resoluciones) y combina los resultados,
        mejorando la robustez ante diferentes tipos y niveles de ruido.
        
        Ventajas:
            - Más robusto que NLM de escala única
            - Mejor manejo de ruido variable
            - Preserva mejor estructuras de diferentes tamaños
        
        Desventajas:
            - Más lento que NLM estándar (múltiples procesadas)
            - Más parámetros que ajustar
        
        Usos típicos:
            - Imágenes con ruido no uniforme
            - Mezcla de estructuras finas y gruesas
            - Cuando NLM estándar no es suficiente
    """
    nombre = "non_local_means_multiescala"
    
    def __init__(self,
                escalas: int = 3,
                h_base: float = 10.0,
                template_window_size: int = 7,
                search_window_size: int = 21):
        """
            Args:
                escalas: Número de escalas a procesar (típicamente 2-4)
                h_base: Parámetro h para la escala más fina
                    Se ajusta automáticamente para otras escalas
                template_window_size: Tamaño del parche
                search_window_size: Tamaño de la ventana de búsqueda
        """
        if escalas < 1:
            raise ValueError("escalas debe ser >= 1")
        
        self.escalas = escalas
        self.h_base = h_base
        self.template_window_size = template_window_size
        self.search_window_size = search_window_size
    
    def __call__(self, img: np.ndarray) -> np.ndarray:
        """
            Aplica NLM multiescala.
            
            Args:
                img: Imagen 2D a filtrar
                
            Returns:
                Imagen filtrada combinando múltiples escalas
        """
        self._validar_imagen(img)
        
        # Procesar en múltiples escalas
        resultados_escalas = []
        img_actual = img.copy()
        
        for i in range(self.escalas):
            # Ajustar h según la escala (más suavizado en escalas gruesas)
            h_escala = self.h_base * (1.5 ** i)
            
            # Aplicar NLM en esta escala
            nlm = NonLocalMeans(
                h=h_escala,
                template_window_size=self.template_window_size,
                search_window_size=self.search_window_size
            )
            img_filtrada = nlm(img_actual)
            
            # Expandir de vuelta a tamaño original si no es la primera escala
            if i > 0:
                for _ in range(i):
                    img_filtrada = cv2.pyrUp(img_filtrada)
                    # Ajustar tamaño si no coincide exactamente
                    if img_filtrada.shape != img.shape:
                        img_filtrada = cv2.resize(img_filtrada, (img.shape[1], img.shape[0]))
            
            resultados_escalas.append(img_filtrada)
            
            # Reducir para siguiente escala
            if i < self.escalas - 1:
                img_actual = cv2.pyrDown(img_actual)
        
        # Combinar resultados (promedio ponderado favoreciendo escalas finas)
        pesos = np.array([2.0 ** (self.escalas - i - 1) for i in range(self.escalas)])
        pesos = pesos / pesos.sum()
        
        resultado = np.zeros_like(img, dtype=np.float64)
        for peso, img_escala in zip(pesos, resultados_escalas):
            resultado += peso * img_escala.astype(np.float64)
        
        return resultado.astype(img.dtype)