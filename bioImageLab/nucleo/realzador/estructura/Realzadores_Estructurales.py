"""
Métodos estructurales para realce de filamentos y tubos en imágenes.

Los realzadores estructurales detectan y realzan geometrías específicas
basándose en el análisis de la forma local de las intensidades. A diferencia
de los detectores de bordes que buscan discontinuidades, estos métodos
identifican "tubos" (estructuras alargadas con contraste respecto al fondo).

Principio fundamental:
Analizar la curvatura de la función de intensidad mediante derivadas
segundas (matriz Hessiana) o gradientes (tensor de estructura) para
identificar direcciones principales de variación y clasificar la geometría
local (línea, plano, esfera, ruido).

IMPORTANTE - Separación de responsabilidades:
- Estos métodos NO normalizan imágenes (ese rol es de normalizador.py)
- Solo realizan conversiones de tipo cuando es estrictamente necesario
- Trabajan con los valores de la imagen tal como vienen
- Requieren imagen suavizada previamente (el ruido afecta severamente las derivadas segundas)
- La normalización previa (si es necesaria) debe hacerse con Normalizador

Tipos de análisis estructural:
- Basado en Hessiana: Analiza curvatura (2ª derivada) → ideal para tubos con contraste definido (vasos sanguíneos, neuritas)
- Basado en Tensor de Estructura: Analiza gradientes (1ª derivada) → robusto en presencia de ruido, útil para orientación de filamentos

Métodos disponibles:
- Hessiano: Análisis de curvatura con umbrales de eigenvalores
- Frangi: Vesselness measure basado en razón de eigenvalores Hessianos
- Sato: Mejora del Frangi con mejor discriminación fondo/objeto
- TensorEstructural: Análisis de coherencia de gradientes locales
"""

import numpy as np
import cv2
from typing import Optional, Tuple, List, Union
from scipy.ndimage import gaussian_filter
from skimage.feature import hessian_matrix, hessian_matrix_eigvals
import warnings


class RealzadorEstructural:
    """
        Clase base para métodos de realce estructural de filamentos.
        
        Los realzadores estructurales identifican geometrías tubulares mediante
        análisis del tensor de segunda derivada (Hessiano) o primeras derivadas
        (tensor de estructura).
        
        Conceptos clave:
            - Hessiano: Matriz de segundas derivadas espaciales
            - Eigenvalores: Indican curvatura en direcciones principales
            - Vesselness: Medida de probabilidad de ser un tubo
            - Escala: Determina grosor de estructuras detectables
    """
    nombre = "realzador_estructural_base"
    
    def __call__(self, img: np.ndarray) -> np.ndarray:
        """
            Aplica el realce estructural a la imagen.
            
            Args:
                img: Array 2D (Y, X) con la imagen a procesar
                
            Returns:
                Imagen con medida de vesselness (float64 en [0, 1] o similar)
        """
        raise NotImplementedError("Subclases deben implementar __call__")
    
    def _validar_imagen(self, img: np.ndarray):
        """Valida que la imagen sea 2D."""
        if img.ndim != 2:
            raise ValueError(f"La imagen debe ser 2D, tiene {img.ndim} dimensiones")
    
    def _suavizar_gaussiano(self, img: np.ndarray, sigma: float) -> np.ndarray:
        """
            Suaviza la imagen con filtro gaussiano.
            
            Crítico para métodos basados en derivadas ya que el ruido
            se amplifica en diferenciación.
        """
        return gaussian_filter(img.astype(np.float64), sigma=sigma)

@registrar_en("realzado")
class Hessiano(RealzadorEstructural):
    """
        Análisis de la matriz Hessiana para detección de tubos.
        
        El Hessiano captura la curvatura local de la función de intensidad.
        Para estructuras tubulares, un eigenvalor es cercano a cero (dirección
        del tubo) y el otro es grande negativo (curvatura perpendicular).
        
        Algoritmo:
            1. Suavizar imagen con gaussiano de escala σ
            2. Calcular derivadas segundas: I_xx, I_yy, I_xy
            3. Construir Hessiano: H = [[I_xx, I_xy], [I_xy, I_yy]]
            4. Calcular eigenvalores λ₁, λ₂ (|λ₁| ≤ |λ₂|)
            5. Clasificar geometría según signos y magnitudes
        
        Ecuación de eigenvalores:
            det(H - λI) = 0
            λ = (I_xx + I_yy)/2 ± sqrt((I_xx - I_yy)²/4 + I_xy²)
        
        Interpretación de eigenvalores:
            - λ₂ << λ₁ ≈ 0: Tubo brillante sobre fondo oscuro
            - λ₂ >> λ₁ ≈ 0: Tubo oscuro sobre fondo brillante
            - λ₁ ≈ λ₂ ≈ 0: Región plana (fondo)
            - |λ₁| ≈ |λ₂| grandes: Esquina o punto
        
        Ventajas:
            - Fundamento matemático sólido (invariante a rotación)
            - Detecta tubos en cualquier orientación
            - Distingue entre tubos, planos y puntos
            - Parámetros interpretables físicamente
        
        Desventajas:
            - Muy sensible al ruido (requiere suavizado previo)
            - Escala fija (solo detecta tubos de grosor ~σ)
            - No produce medida continua de "tubularidad"
            - Requiere post-procesamiento para suprimir fondo
        
        Usos típicos en microscopía:
            - Detección de vasos sanguíneos en angiografía
            - Segmentación de neuritas en neurociencia
            - Identificación de filamentos de actina/miosina
            - Análisis de redes vasculares en retina
            - Preprocesamiento para tracking de filamentos
    """
    nombre = "hessiano"
    
    def __init__(self, 
                sigma: float = 2.0,
                umbral_ratio: float = 0.5,
                detectar_oscuros: bool = True):
        """
            Args:
                sigma: Escala del filtro gaussiano (tamaño de suavizado)
                        Determina grosor de tubos detectables (~2-3σ)
                        Valores típicos: 1.0-5.0 para microscopía
                
                umbral_ratio: Ratio mínimo |λ₂/λ₁| para considerar tubo
                            Valores altos = más selectivo (solo tubos "perfectos")
                            Valores típicos: 0.1-0.5
                
                detectar_oscuros: Si True, detecta tubos oscuros sobre fondo claro
                                Si False, detecta tubos claros sobre fondo oscuro
        """
        if sigma <= 0:
            raise ValueError("sigma debe ser > 0")
        if not 0 < umbral_ratio <= 1:
            raise ValueError("umbral_ratio debe estar en (0, 1]")
        
        self.sigma = sigma
        self.umbral_ratio = umbral_ratio
        self.detectar_oscuros = detectar_oscuros
    
    def __call__(self, img: np.ndarray) -> np.ndarray:
        """
            Aplica análisis Hessiano y devuelve máscara de tubos.
            
            Args:
                img: Imagen 2D (cualquier tipo, se convierte a float64)
                
            Returns:
                Imagen binaria (uint8) con tubos detectados
                
            Nota:
                Para obtener eigenvalores crudos, usar calcular_eigenvalores().
        """
        self._validar_imagen(img)
        
        # Calcular eigenvalores
        lambda1, lambda2 = self.calcular_eigenvalores(img)
        
        # Criterio de tubo: |λ₂| >> |λ₁| y λ₂ tiene signo correcto
        ratio = np.abs(lambda2) / (np.abs(lambda1) + 1e-10)
        
        if self.detectar_oscuros:
            # Tubo oscuro: λ₂ positivo grande (curvatura hacia arriba)
            mascara = (ratio > self.umbral_ratio) & (lambda2 > 0)
        else:
            # Tubo claro: λ₂ negativo grande (curvatura hacia abajo)
            mascara = (ratio > self.umbral_ratio) & (lambda2 < 0)
        
        return mascara.astype(np.uint8) * 255
    
    def calcular_eigenvalores(self, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
            Calcula eigenvalores del Hessiano.
            
            Returns:
                (lambda1, lambda2) ordenados por magnitud creciente (|λ₁| ≤ |λ₂|)
        """
        img_float = self._suavizar_gaussiano(img, self.sigma)
        
        # Calcular Hessiano usando skimage
        H_elems = hessian_matrix(img_float, sigma=self.sigma, order='rc')
        lambda1, lambda2 = hessian_matrix_eigvals(H_elems)
        
        # Ordenar por magnitud
        mask = np.abs(lambda1) > np.abs(lambda2)
        lambda1_out = np.where(mask, lambda2, lambda1)
        lambda2_out = np.where(mask, lambda1, lambda2)
        
        return lambda1_out, lambda2_out

@registrar_en("realzado")
class Frangi(RealzadorEstructural):
    """
        Filtro de Frangi para vesselness basado en razón de eigenvalores Hessianos.
        
        El filtro de Frangi mejora el análisis Hessiano proporcionando una medida
        continua de "tubularidad" (vesselness) que suprime el fondo y realza
        estructuras tubulares de forma proporcional a su evidencia geométrica.
        
        Algoritmo:
            1. Calcular Hessiano y eigenvalores λ₁, λ₂ en múltiples escalas
            2. Para cada escala, calcular medidas:
            - R_B = |λ₁| / |λ₂|  (diferencia de curvaturas, idealmente 0 para tubo)
            - S = sqrt(λ₁² + λ₂²)  (energía estructural)
            3. Combinar en vesselness: V = exp(-R_B²/2α²) * (1 - exp(-S²/2β²))
            4. Tomar máximo across escalas
        
        Ecuación de vesselness (2D):
            V(σ) = exp(-R_B²/(2α²)) * (1 - exp(-S²/(2β²)))
            
            donde:
                R_B = |λ₁| / (|λ₂| + ε)  [medida de "tubularidad"]
                S = sqrt(λ₁² + λ₂²)       [magnitud estructural]
                α = controla sensibilidad a desviaciones del tubo perfecto
                β = umbral de fondo vs estructura
        
        Interpretación:
            - R_B ≈ 0: Tubo perfecto (λ₁ ≈ 0, λ₂ grande)
            - S grande: Hay estructura (no es ruido plano)
            - Exp(-R_B²): Alto cuando la forma es tubular
            - (1 - exp(-S²)): Alto cuando hay suficiente contraste
        
        Ventajas:
            - Medida continua de probabilidad tubular [0, 1]
            - Supresión automática de fondo plano
            - Multi-escala (detecta tubos de diferentes grosores)
            - Robustez mediante integración across escalas
            - Estándar de facto en análisis vascular
        
        Desventajas:
            - Costoso computacionalmente (múltiples escalas)
            - Parámetros α, β requieren ajuste por imagen
            - Puede fallar en cruces de vasos (no es estrictamente tubular)
            - Sensible a estructuras similares a tubos (bordes fuertes)
            - No distingue entre vasos entrantes y salientes
        
        Usos típicos en microscopía:
            - Segmentación de redes vasculares 2D/3D
            - Cuantificación de densidad capilar
            - Análisis de tortuosidad vascular
            - Detección de neuritas en cultivos primarios
            - Segmentación de filamentos de actina en células
            - Preprocesamiento para análisis de flujo sanguíneo
    """
    nombre = "frangi"
    
    def __init__(self,
                sigmas: Tuple[float, ...] = (1.0, 2.0, 3.0),
                alpha: float = 0.5,
                beta: float = 0.5,
                gamma: Optional[float] = None,
                black_ridges: bool = True):
        """
            Args:
                sigmas: Escalas a analizar (cada una detecta tubos ~2-3σ de grosor)
                    Más escalas = más completo pero más lento
                    Ejemplo: (1, 2, 4) para tubos finos a gruesos
                
                alpha: Parámetro de sensibilidad a tubularidad
                    Controla cuánto puede desviarse λ₁ de cero
                    Valores típicos: 0.2-0.8
                    Menor = más estricto (solo tubos perfectos)
                
                beta: Umbral de magnitud estructural
                    Descarta regiones de bajo contraste (ruido)
                    Valores típicos: 0.1-1.0 (fracción del máximo de S)
                    Mayor = más selectivo (solo estructuras fuertes)
                
                gamma: Parámetro adicional para versión 3D (no usado en 2D)
                
                black_ridges: Si True, detecta tubos oscuros (negras)
                            Si False, detecta tubos claros (blancas)
        """
        if not all(s > 0 for s in sigmas):
            raise ValueError("Todos los sigmas deben ser > 0")
        if alpha <= 0 or beta <= 0:
            raise ValueError("alpha y beta deben ser > 0")
        
        self.sigmas = sigmas
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.black_ridges = black_ridges
    
    def __call__(self, img: np.ndarray) -> np.ndarray:
        """
            Aplica filtro de Frangi y devuelve imagen de vesselness.
            
            Args:
                img: Imagen 2D (cualquier tipo numérico)
                
            Returns:
                Imagen float64 con valores de vesselness en [0, 1]
                Valores altos indican alta probabilidad de ser tubo
        """
        self._validar_imagen(img)
        img_float = img.astype(np.float64)
        
        # Normalizar para estabilidad numérica
        img_float = (img_float - img_float.min()) / (img_float.max() - img_float.min() + 1e-10)
        
        vesselness_max = np.zeros_like(img_float)
        
        for sigma in self.sigmas:
            # Calcular Hessiano y eigenvalores
            H_elems = hessian_matrix(img_float, sigma=sigma, order='rc')
            lambda1, lambda2 = hessian_matrix_eigvals(H_elems)
            
            # Ordenar por magnitud: |λ₁| ≤ |λ₂|
            mask = np.abs(lambda1) > np.abs(lambda2)
            lambda1, lambda2 = np.where(mask, lambda2, lambda1), np.where(mask, lambda1, lambda2)
            
            # Ajustar signo según tipo de tubo
            if self.black_ridges:
                lambda2 = -lambda2
            
            # Calcular medidas
            rb2 = (lambda1 / (lambda2 + 1e-10)) ** 2  # (|λ₁|/|λ₂|)²
            s2 = lambda1**2 + lambda2**2              # S²
            
            # Calcular vesselness
            vesselness = np.exp(-rb2 / (2 * self.alpha**2))
            vesselness *= (1 - np.exp(-s2 / (2 * self.beta**2)))
            
            # Suprimir valores donde λ₂ > 0 (no es tubo oscuro)
            if self.black_ridges:
                vesselness = np.where(lambda2 < 0, vesselness, 0)
            else:
                vesselness = np.where(lambda2 > 0, vesselness, 0)
            
            # Actualizar máximo across escalas
            vesselness_max = np.maximum(vesselness_max, vesselness)
        
        return vesselness_max

@registrar_en("realzado")
class Sato(RealzadorEstructural):
    """
        Filtro de Sato (mejora del Frangi) para vesselness.
        
        El filtro de Sato mejora el Frangi modificando la función de vesselness
        para tener mejor discriminación entre fondo y estructuras tubulares,
        especialmente en imágenes con ruido heterogéneo.
        
        Algoritmo:
            Similar al Frangi pero con función de vesselness modificada:
            1. Calcular Hessiano y eigenvalores en múltiples escalas
            2. Calcular medidas de tubo y magnitud
            3. Aplicar función de Sato que enfatiza mejor el contraste
        
        Ecuación de vesselness de Sato (2D):
            V(σ) = |λ₂| - α|λ₁|  si λ₂ < 0 (tubo oscuro)
                0             en otro caso
            
            Variante suavizada (usada aquí):
            V = exp(-R_B²/2α²) * (1 - exp(-S²/2β²)) * |λ₂|
            
            La diferencia clave es el factor multiplicativo |λ₂| que da
            mayor peso a tubos con fuerte curvatura (más contrastados).
        
        Interpretación vs Frangi:
            - Similar estructura pero V_sato ~ V_frangi × |λ₂|
            - Da mayor importancia a la magnitud de la curvatura principal
            - Mejor para tubos con contraste variable
            - Más supresión de respuestas débiles en fondo
        
        Ventajas:
            - Mejor discriminación fondo/objeto que Frangi
            - Resalta tubos con alto contraste vs débiles
            - Reduce falsos positivos en regiones de bajo contraste
            - Útil cuando hay variación de intensidad en los tubos
            - Buen rendimiento en imágenes médicas con ruido
        
        Desventajas:
            - Puede perder tubos de bajo contraste (los considera ruido)
            - Más agresivo en suprimir estructuras débiles
            - Parámetros similares al Frangi pero comportamiento diferente
            - Requiere ajuste cuidadoso de α para no sobre-segmentar
            - Computacionalmente equivalente al Frangi (mismo costo)
        
        Usos típicos en microscopía:
            - Segmentación de vasos con contraste heterogéneo
            - Análisis de redes neuronales con señal variable
            - Detección de filamentos gruesos con núcleo oscuro
            - Imágenes con iluminación desigual donde el fondo varía
            - Cuando Frangi produce demasiados falsos positivos débiles
    """
    nombre = "sato"
    
    def __init__(self,
                sigmas: Tuple[float, ...] = (1.0, 2.0, 3.0),
                alpha: float = 0.5,
                beta: float = 0.5,
                black_ridges: bool = True):
        """
            Args:
                sigmas: Escalas espaciales para análisis multi-escala
                    Cada sigma detecta tubos de grosor ~2-3σ
                
                alpha: Controla tolerancia a desviaciones de tubularidad
                    Menor = más estricto con la forma
                    Valores típicos: 0.3-0.7
                
                beta: Umbral de magnitud mínima para considerar estructura
                    Valores típicos: 0.1-0.5 (relativo al rango de intensidad)
                
                black_ridges: True para tubos oscuros, False para claros
        """
        if not all(s > 0 for s in sigmas):
            raise ValueError("Todos los sigmas deben ser > 0")
        if alpha <= 0 or beta <= 0:
            raise ValueError("alpha y beta deben ser > 0")
        
        self.sigmas = sigmas
        self.alpha = alpha
        self.beta = beta
        self.black_ridges = black_ridges
    
    def __call__(self, img: np.ndarray) -> np.ndarray:
        """
            Aplica filtro de Sato y devuelve imagen de vesselness.
            
            Args:
                img: Imagen 2D (cualquier tipo numérico)
                
            Returns:
                Imagen float64 con valores de vesselness mejorados
                Rangos típicos [0, max] donde max depende de la imagen
        """
        self._validar_imagen(img)
        img_float = img.astype(np.float64)
        
        # Normalizar
        img_float = (img_float - img_float.min()) / (img_float.max() - img_float.min() + 1e-10)
        
        vesselness_max = np.zeros_like(img_float)
        
        for sigma in self.sigmas:
            H_elems = hessian_matrix(img_float, sigma=sigma, order='rc')
            lambda1, lambda2 = hessian_matrix_eigvals(H_elems)
            
            # Ordenar
            mask = np.abs(lambda1) > np.abs(lambda2)
            lambda1, lambda2 = np.where(mask, lambda2, lambda1), np.where(mask, lambda1, lambda2)
            
            if self.black_ridges:
                lambda2 = -lambda2
            
            # Medidas
            rb2 = (lambda1 / (lambda2 + 1e-10)) ** 2
            s2 = lambda1**2 + lambda2**2
            
            # Vesselness tipo Frangi base
            v_base = np.exp(-rb2 / (2 * self.alpha**2))
            v_base *= (1 - np.exp(-s2 / (2 * self.beta**2)))
            
            # Factor Sato: multiplicar por |λ₂| para enfatizar contraste
            vesselness = v_base * np.abs(lambda2)
            
            # Suprimir no-tubos
            if self.black_ridges:
                vesselness = np.where(lambda2 < 0, vesselness, 0)
            else:
                vesselness = np.where(lambda2 > 0, vesselness, 0)
            
            vesselness_max = np.maximum(vesselness_max, vesselness)
        
        # Normalizar resultado al rango [0, 1]
        vmax = vesselness_max.max()
        if vmax > 0:
            vesselness_max = vesselness_max / vmax
        
        return vesselness_max

@registrar_en("realzado")
class TensorEstructural(RealzadorEstructural):
    """
        Tensor de estructura (Structure Tensor) para análisis de coherencia local.
        
        A diferencia del Hessiano (2ª derivada), el tensor de estructura usa
        gradientes (1ª derivada) para analizar la coherencia de orientación
        en vecindarios locales. Es más robusto al ruido que el Hessiano.
        
        Algoritmo:
            1. Calcular gradientes: I_x, I_y (derivadas primeras)
            2. Construir tensor en cada pixel (outer product):
            S = [I_x²    I_x*I_y]
                [I_x*I_y   I_y²  ]
            3. Promediar (suavizar) componentes del tensor en vecindario
            4. Calcular eigenvalores μ₁ ≥ μ₂ ≥ 0 del tensor promediado
        
        Ecuación:
            S_ρ = G_ρ * (∇I ⊗ ∇I) = [G_ρ * I_x²  G_ρ * (I_x I_y)]
                                    [G_ρ * (I_x I_y)  G_ρ * I_y²  ]
            
            donde G_ρ es suavizado gaussiano de escala ρ (tamaño de vecindario)
            y ⊗ es producto exterior (outer product).
        
        Interpretación de eigenvalores:
            - μ₁ ≈ μ₂ ≈ 0: Región constante (fondo)
            - μ₁ >> μ₂ ≈ 0: Borde o línea (gradiente dominante en una dirección)
            - μ₁ ≈ μ₂ >> 0: Esquina o textura isotrópica
        
        Medidas derivadas:
            - Coherencia: C = (μ₁ - μ₂)² / (μ₁ + μ₂)² ∈ [0, 1]
            1 = perfectamente orientado (tubo), 0 = isotrópico
            - Energía: E = μ₁ + μ₂ (magnitud total de variación)
        
        Ventajas:
            - Más robusto al ruido que Hessiano (promedia gradientes)
            - Detecta estructuras de borde además de tubos
            - Coherencia da medida directa de "alineación" local
            - Útil para análisis de orientación de filamentos
            - Menor costo computacional que Hessiano (no requiere 2ª derivada)
        
        Desventajas:
            - Menos específico para tubos que Frangi/Sato
            - No distingue bien entre bordes y tubos delgados
            - Requiere escala de integración (ρ) adicional a σ de derivada
            - Menos preciso en localización subpixel de centros de tubos
            - Puede fallar en intersecciones de filamentos
        
        Usos típicos en microscopía:
            - Análisis de orientación de fibras en tejido conectivo
            - Cuantificación de alineación celular
            - Detección de patrones de flujo en imágenes de PIV
            - Segmentación de estructuras laminares
            - Análisis de textura en imágenes histológicas
            - Complemento a Frangi para validar coherencia de tubos detectados
    """
    nombre = "tensor_estructural"
    
    def __init__(self,
                sigma_derivada: float = 1.0,
                rho_integracion: float = 2.0,
                umbral_coherencia: float = 0.5,
                umbral_energia: Optional[float] = None):
        """
            Args:
                sigma_derivada: Escala para cálculo de gradientes (suavizado previo)
                            Valores típicos: 0.5-2.0
                            Mayor = gradientes más suaves, menos ruido
                
                rho_integracion: Escala de integración (tamaño de vecindario)
                                Debe ser >= sigma_derivada
                                Valores típicos: 2-4× sigma_derivada
                                Determina tamaño de estructuras analizadas
                
                umbral_coherencia: Umbral mínimo de coherencia para considerar
                                estructura orientada (0-1)
                                Valores típicos: 0.3-0.7
                
                umbral_energia: Umbral mínimo de energía (contraste)
                            Si None, se calcula automáticamente como percentil 10
        """
        if sigma_derivada <= 0 or rho_integracion <= 0:
            raise ValueError("sigma_derivada y rho_integracion deben ser > 0")
        if rho_integracion < sigma_derivada:
            warnings.warn("rho_integracion debería ser >= sigma_derivada para resultados significativos")
        if not 0 <= umbral_coherencia <= 1:
            raise ValueError("umbral_coherencia debe estar en [0, 1]")
        
        self.sigma_derivada = sigma_derivada
        self.rho_integracion = rho_integracion
        self.umbral_coherencia = umbral_coherencia
        self.umbral_energia = umbral_energia
    
    def __call__(self, img: np.ndarray, 
                modo: str = "coherencia") -> np.ndarray:
        """
            Aplica análisis de tensor de estructura.
            
            Args:
                img: Imagen 2D
                
                modo: Tipo de salida deseada
                    - "coherencia": Mapa de coherencia [0, 1]
                    - "energia": Energía local (magnitud de variación)
                    - "orientacion": Ángulo de orientación principal [-π/2, π/2]
                    - "binario": Máscara de estructuras coherentes
                    
            Returns:
                Imagen según modo especificado (float64 excepto binario que es uint8)
        """
        self._validar_imagen(img)
        
        img_float = img.astype(np.float64)
        
        # Suavizar para cálculo de gradientes
        img_suave = gaussian_filter(img_float, sigma=self.sigma_derivada)
        
        # Calcular gradientes
        gy, gx = np.gradient(img_suave)
        
        # Componentes del tensor (outer product)
        ixx = gx * gx
        ixy = gx * gy
        iyy = gy * gy
        
        # Promediar (integrar) en vecindario
        ixx = gaussian_filter(ixx, sigma=self.rho_integracion)
        ixy = gaussian_filter(ixy, sigma=self.rho_integracion)
        iyy = gaussian_filter(iyy, sigma=self.rho_integracion)
        
        # Calcular eigenvalores del tensor 2x2
        # μ = (ixx + iyy)/2 ± sqrt((ixx - iyy)²/4 + ixy²)
        trace = ixx + iyy
        det = ixx * iyy - ixy * ixy
        discriminant = np.sqrt(np.maximum(trace**2 / 4 - det, 0))
        
        mu1 = trace / 2 + discriminant  # Mayor
        mu2 = trace / 2 - discriminant  # Menor
        
        if modo == "coherencia":
            # C = (μ₁ - μ₂)² / (μ₁ + μ₂)² = 1 - 4det/trace²
            denom = trace**2 + 1e-10
            coherencia = 1 - 4 * det / denom
            return np.clip(coherencia, 0, 1)
        
        elif modo == "energia":
            return trace  # μ₁ + μ₂
        
        elif modo == "orientacion":
            # θ = 0.5 * atan2(2*ixy, ixx - iyy)
            return 0.5 * np.arctan2(2 * ixy, ixx - iyy)
        
        elif modo == "binario":
            coherencia = self(img, modo="coherencia")
            energia = self(img, modo="energia")
            
            if self.umbral_energia is None:
                self.umbral_energia = np.percentile(energia, 10)
            
            mascara = (coherencia > self.umbral_coherencia) & (energia > self.umbral_energia)
            return mascara.astype(np.uint8) * 255
        
        else:
            raise ValueError(f"Modo '{modo}' no reconocido. Usar: coherencia, energia, orientacion, binario")
    
    def obtener_orientacion_principal(self, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
            Obtiene orientación principal y coherencia simultáneamente.
            
            Returns:
                (orientacion, coherencia) útiles para visualización vectorial
        """
        orientacion = self(img, modo="orientacion")
        coherencia = self(img, modo="coherencia")
        return orientacion, coherencia