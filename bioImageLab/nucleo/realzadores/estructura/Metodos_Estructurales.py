"""
Métodos de realce estructural para detección de geometrías en bioimágenes.

Estos métodos se basan en el análisis de las derivadas parciales de segundo orden 
(Matriz Hessiana) o primer orden (Tensor de Estructura) para identificar 
formas tubulares, filamentosas o circulares.

Principio fundamental:
Analizar la curvatura local de la intensidad de los píxeles. En estructuras 
tubulares, la curvatura es máxima en la dirección perpendicular al tubo y 
mínima a lo largo del mismo.

IMPORTANTE - Separación de responsabilidades:
- Estos métodos NO normalizan imágenes.
- Se recomienda normalizar a [0, 1] (float) antes de aplicar, ya que el análisis de autovalores es sensible a la escala de intensidad.
- El resultado suele ser un mapa de "vesselness" o "tubularidad".

Tipos de estructuras:
- Tubulares: Neuritas, vasos, filamentos, cuerpo de C. elegans.
- Blobs: Núcleos celulares, vesículas, punctas de autofagia (LGG-1).
- Bordes: Membranas y transiciones bruscas.
"""

import numpy as np
import cv2
from scipy import ndimage
from typing import Optional, Tuple, List
import warnings


class RealzadorEstructural:
    """
    Clase base para métodos de realce estructural.
    
    Se centra en la extracción de propiedades geométricas basadas en el 
    análisis del espacio de escala (Scale-space).
    """
    nombre = "realzador_estructural_base"
    
    def __call__(self, img: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Subclases deben implementar __call__")
    
    def _validar_imagen(self, img: np.ndarray):
        if img.ndim != 2:
            raise ValueError(f"La imagen debe ser 2D, tiene {img.ndim} dimensiones")

    def _calcular_hessiana(self, img: np.ndarray, sigma: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calcula los componentes de la matriz Hessiana para una escala sigma.
        H = [[Ixx, Ixy], [Ixy, Iyy]]
        """
        # Suavizado gaussiano previo para el análisis en escala sigma
        img_filtered = ndimage.gaussian_filter(img, sigma)
        
        # Derivadas de segundo orden
        Iyy, Iyx = np.gradient(np.gradient(img_filtered, axis=0))
        Ixy, Ixx = np.gradient(np.gradient(img_filtered, axis=1))
        
        return Ixx, Ixy, Iyy

    def _ordenar_autovalores(self, Ixx, Ixy, Iyy) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calcula y ordena los autovalores de la Hessiana (|λ1| <= |λ2|).
        En 2D, λ2 representa la curvatura principal.
        """
        # Traza y determinante
        traza = Ixx + Iyy
        det = Ixx * Iyy - Ixy**2
        
        # Resolución de la ecuación característica: λ² - tr(H)λ + det(H) = 0
        discriminante = np.sqrt(traza**2 - 4 * det + 0.000001)
        l1 = (traza - discriminante) / 2
        l2 = (traza + discriminante) / 2
        
        # Ordenar por magnitud absoluta |λ1| <= |λ2|
        mask = np.abs(l1) > np.abs(l2)
        lam1 = np.where(mask, l2, l1)
        lam2 = np.where(mask, l1, l2)
        
        return lam1, lam2


class FiltroHessiano(RealzadorEstructural):
    """
    Realce básico basado en el determinante de la Hessiana o autovalores puros.
    
    Algoritmo:
        1. Calcula la matriz Hessiana en una escala σ.
        2. Extrae el autovalor de mayor magnitud (λ2).
    
    Ventajas:
        - Extremadamente rápido.
        - Identifica crestas (ridges) de forma directa.
    
    Usos típicos:
        - Detección de centroides en gusanos.
        - Segmentación gruesa de filamentos.
    """
    nombre = "hessiano_puro"

    def __init__(self, sigma: float = 1.0, modo: Literal["l2", "det"] = "l2"):
        self.sigma = sigma
        self.modo = modo

    def __call__(self, img: np.ndarray) -> np.ndarray:
        self._validar_imagen(img)
        Ixx, Ixy, Iyy = self._calcular_hessiana(img, self.sigma)
        
        if self.modo == "det":
            return Ixx * Iyy - Ixy**2
        
        lam1, lam2 = self._ordenar_autovalores(Ixx, Ixy, Iyy)
        return lam2 # Retorna la curvatura principal


class Frangi(RealzadorEstructural):
    """
    Filtro de Vesselness de Frangi.
    
    Diseñado específicamente para realzar estructuras tubulares (vasos, neuritas).
    Analiza la relación entre autovalores para discriminar entre tubos, blobs y ruido.
    
    Ecuación (2D):
        $$V(s) = \begin{cases} 0 & \text{if } \lambda_2 > 0 \\ \exp(-\frac{R_b^2}{2\beta^2}) (1 - \exp(-\frac{S^2}{2c^2})) \end{cases}$$
        donde $R_b = \lambda_1 / \lambda_2$ (medida de blobness) y $S = \sqrt{\lambda_1^2 + \lambda_2^2}$ (estructura).
    
    Ventajas:
        - Excelente supresión de ruido de fondo.
        - Muy específico para líneas (neuritas en C. elegans).
        - Multi-escala permite detectar tubos de diferentes grosores.
    
    Desventajas:
        - Sensible a los parámetros β y c.
        - computacionalmente intensivo en multi-escala.
    """
    nombre = "frangi"

    def __init__(self, 
                sigmas: Tuple[float, ...] = (1.0, 2.0, 3.0), 
                beta: float = 0.5, 
                c: float = 500.0):
        self.sigmas = sigmas
        self.beta = 2 * beta**2
        self.c = 2 * c**2

    def __call__(self, img: np.ndarray) -> np.ndarray:
        self._validar_imagen(img)
        vesselness_final = np.zeros_like(img, dtype=float)

        for s in self.sigmas:
            Ixx, Ixy, Iyy = self._calcular_hessiana(img, s)
            # Escalamiento por s^2 para normalización de escala (Lindeberg)
            lam1, lam2 = self._ordenar_autovalores(Ixx * s**2, Ixy * s**2, Iyy * s**2)
            
            # Rb = ratio de autovalores (0 para tubos perfectos)
            rb = lam1 / (lam2 + 1e-10)
            # S = Magnitud de la estructura (Frobenius norm)
            s_norm = np.sqrt(lam1**2 + lam2**2)
            
            # Cálculo de los términos de la ecuación
            term_rb = np.exp(-(rb**2) / self.beta)
            term_s = 1 - np.exp(-(s_norm**2) / self.c)
            
            v_s = term_rb * term_s
            
            # Solo estructuras más oscuras que el fondo (para λ2 < 0)
            v_s[lam2 > 0] = 0
            
            vesselness_final = np.maximum(vesselness_final, v_s)
            
        return vesselness_final


class Sato(RealzadorEstructural):
    """
    Filtro de Tubularidad de Sato.
    
    Alternativa a Frangi, utiliza una formulación distinta de los autovalores 
    para mejorar la detección de líneas continuas y mitigar el ruido.
    
    Principio:
        Calcula la media geométrica de los autovalores con un factor de peso
        basado en la curvatura local.
    
    Ventajas:
        - Suele dar resultados más limpios en uniones o bifurcaciones.
        - Menos propenso a "romper" la línea si hay variaciones de intensidad.
    """
    nombre = "sato"

    def __init__(self, sigmas: Tuple[float, ...] = (1.0, 2.0, 3.0), alpha: float = 0.25):
        self.sigmas = sigmas
        self.alpha = alpha

    def __call__(self, img: np.ndarray) -> np.ndarray:
        self._validar_imagen(img)
        sato_final = np.zeros_like(img, dtype=float)

        for s in self.sigmas:
            Ixx, Ixy, Iyy = self._calcular_hessiana(img, s)
            lam1, lam2 = self._ordenar_autovalores(Ixx * s**2, Ixy * s**2, Iyy * s**2)
            
            # Sato formula para líneas 2D
            #λc = λ2 * (λ1 / λ2)^α
            v_s = -lam2 * (np.abs(lam1) / (np.abs(lam2) + 1e-10))**self.alpha
            v_s[lam2 > 0] = 0
            
            sato_final = np.maximum(sato_final, v_s)
            
        return sato_final


class TensorEstructura(RealzadorEstructural):
    """
    Tensor de Estructura (Matriz de covarianza del gradiente).
    
    A diferencia de la Hessiana, usa derivadas de primer orden promediadas localmente.
    
    Ecuación:
        $J = K_rho * (nabla I \otimes nabla I)$
        donde $K_rho$ es un kernel gaussiano de integración.
    
    Información extraída:
        - Coherencia: Qué tan orientada está la estructura.
        - Orientación: Ángulo dominante de la textura local.
    
    Ventajas:
        - Muy robusto al ruido.
        - Permite medir la anisotropía local (estiramiento de las células).
    
    Usos:
        - Análisis de alineación de fibras de colágeno.
        - Detección de orientación del cuerpo del gusano.
    """
    nombre = "tensor_estructura"

    def __init__(self, sigma: float = 1.0, rho: float = 3.0):
        """
        Args:
            sigma: Escala de derivación (ruido).
            rho: Escala de integración (tamaño de la ventana de contexto).
        """
        self.sigma = sigma
        self.rho = rho

    def __call__(self, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Retorna (autovalor_menor, autovalor_mayor, coherencia).
        La coherencia mide qué tan 'lineal' es la estructura.
        """
        self._validar_imagen(img)
        
        # Gradientes
        img_f = ndimage.gaussian_filter(img, self.sigma)
        Iy, Ix = np.gradient(img_f)
        
        # Componentes del tensor
        Ixx = ndimage.gaussian_filter(Ix**2, self.rho)
        Ixy = ndimage.gaussian_filter(Ix * Iy, self.rho)
        Iyy = ndimage.gaussian_filter(Iy**2, self.rho)
        
        # Autovalores
        l1, l2 = self._ordenar_autovalores(Ixx, Ixy, Iyy)
        
        # Coherencia: ( (l2-l1) / (l2+l1) )^2
        coherencia = ((l2 - l1) / (l2 + l1 + 1e-10))**2
        
        return l1, l2, coherencia