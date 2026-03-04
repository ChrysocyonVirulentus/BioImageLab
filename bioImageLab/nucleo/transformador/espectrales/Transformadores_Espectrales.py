"""
Transformaciones espectrales para análisis en dominio de frecuencia.

Las transformaciones espectrales descomponen imágenes en componentes
de frecuencia espacial, permitiendo análisis y filtrado basados en
escala/orientación de patrones.

Principio fundamental:
Proyectar f(x,y) → F(u,v) donde u,v son frecuencias espaciales (ciclos/píxel).

IMPORTANTE - Separación de responsabilidades:
- NO normalizan imágenes (ese rol es de normalizador.py)
- Trabajan en float64/complex128 para precisión
- Algunas son invertibles (Fourier, wavelet), otras no
- Reconstrucción requiere conservar fase/coeficientes completos

Métodos disponibles:
- Fourier: Transformada de Fourier 2D (espectro global)
- Wavelet: Descomposición multiescala local (tiempo-frecuencia)
- Gabor : Filtros gaussianos modulados por sinusoides
"""

import numpy as np
from typing import Optional, Tuple, Literal, Union, List
from scipy.fft import fft2, ifft2, fftshift, ifftshift
import warnings


class TransformadorEspectral:
    """Clase base para transformaciones espectrales."""
    nombre = "transformador_espectral_base"
    
    def __call__(self, img: np.ndarray) -> np.ndarray:
        raise NotImplementedError
    
    def _validar_imagen(self, img: np.ndarray):
        if img.ndim != 2:
            raise ValueError(f"Imagen debe ser 2D, tiene {img.ndim} dimensiones")


class Fourier(TransformadorEspectral):
    """
        Transformada de Fourier 2D para análisis espectral global.
        
        Descompone imagen en ondas sinusoidales de diferentes frecuencias.
        Base de filtrado frecuencial, deconvolución y análisis de periodicidad.
        
        Algoritmo:
            F(u,v) = Σ_x Σ_y f[x,y] · e^(-j2π(ux/M + vy/N))
        
        Propiedades:
            - Convolución espacial ↔ Producto frecuencial
            - Energía preservada (Parseval)
            - Información global (no localización espacial de frecuencias)
        
        Ventajas:
            - Filtrado eficiente, análisis de periodicidad, compresión
        Desventajas:
            - No localización espacial, artefactos de borde circular
        
        Usos microscopía:
            - Filtrado de ruido periódico (patrones de escaneo)
            - Corrección de iluminación desigual (suprimir DC)
            - Deconvolución de PSF, análisis de packing celular
    """
    nombre = "fourier"
    
    def __init__(self, shift: bool = True, norm: Optional[str] = 'ortho'):
        self.shift = shift
        self.norm = norm
    
    def __call__(self, img: np.ndarray) -> np.ndarray:
        self._validar_imagen(img)
        espectro = fft2(img.astype(np.complex128), norm=self.norm)
        return fftshift(espectro) if self.shift else espectro
    
    def inversa(self, espectro: np.ndarray, return_real: bool = True) -> np.ndarray:
        """Transformada inversa."""
        if self.shift:
            espectro = ifftshift(espectro)
        img = ifft2(espectro, norm=self.norm)
        return np.real(img) if return_real else img
    
    def get_magnitud_fase(self, espectro: np.ndarray, log_scale: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """Extrae magnitud y fase."""
        magnitud = np.abs(espectro)
        if log_scale:
            magnitud = np.log1p(magnitud)
        return magnitud, np.angle(espectro)
    
    def filtrar(self, espectro: np.ndarray, tipo: Literal['lowpass', 'highpass', 'bandpass'] = 'lowpass',
                corte: Union[float, Tuple[float, float]] = 0.5, orden: int = 1) -> np.ndarray:
        """Filtra frecuencias."""
        h, w = espectro.shape
        cy, cx = h // 2, w // 2
        y, x = np.ogrid[-cy:h-cy, -cx:w-cx]
        r = np.sqrt(x**2 + y**2) / min(cy, cx)
        
        if tipo == 'lowpass':
            mascara = 1 / (1 + (r / corte) ** (2 * orden))
        elif tipo == 'highpass':
            mascara = 1 - 1 / (1 + (r / corte) ** (2 * orden))
        else:  # bandpass
            fmin, fmax = corte
            mascara = ((r > fmin) & (r < fmax)).astype(float)
        
        return espectro * mascara


class Wavelet(TransformadorEspectral):
    """
        Transformada Wavelet 2D para análisis multiescala tiempo-frecuencia.
        
        Descompone imagen en diferentes escalas con localización espacial.
        Ideal para denoising, compresión y análisis de texturas locales.
        
        Algoritmo (DWT):
            Aplicar filtros pasa-baja/alta, submuestrear, recursión en LL.
            Produce: LL (aprox), LH (H), HL (V), HH (D) por nivel.
        
        Ventajas:
            - Localización espacial+frecuencial, denoising adaptativo,
            - compresión eficiente (JPEG 2000), análisis multiescala
        Desventajas:
            - No invariante a rotación, shift-variance en DWT,
            - elección de wavelet crítica
        
        Usos microscopía:
            - Denoising adaptativo (umbralización de coeficientes)
            - compresión con preservación de diagnóstico,
            - detección multiescala (núcleos vs células),
            - análisis de rugosidad de membranas
    """
    nombre = "wavelet"
    
    def __init__(self, wavelet: str = 'db4', niveles: int = 3,
                modo: Literal['dwt', 'swt'] = 'dwt'):
        try:
            import pywt
        except ImportError:
            raise ImportError("Se requiere PyWavelets: pip install PyWavelets")
        
        self.wavelet = wavelet
        self.niveles = niveles
        self.modo = modo
        
        if wavelet not in pywt.wavelist():
            raise ValueError(f"Wavelet '{wavelet}' no disponible")
    
    def __call__(self, img: np.ndarray) -> List:
        import pywt
        self._validar_imagen(img)
        img_f = img.astype(np.float64)
        
        if self.modo == 'swt':
            return pywt.swt2(img_f, self.wavelet, level=self.niveles)
        return pywt.wavedec2(img_f, self.wavelet, level=self.niveles)
    
    def inversa(self, coeffs: List) -> np.ndarray:
        import pywt
        if self.modo == 'swt':
            return pywt.iswt2(coeffs, self.wavelet)
        return pywt.waverec2(coeffs, self.wavelet)
    
    def umbralizar(self, coeffs: List, sigma: Optional[float] = None,
                    modo_umbral: Literal['soft', 'hard'] = 'soft') -> List:
        """Denoising por umbralización."""
        import pywt
        
        if sigma is None:
            # Estimar de coeficientes HH más finos
            c = coeffs[-1][-1] if isinstance(coeffs[-1], tuple) else coeffs[-1]
            sigma = np.median(np.abs(c)) / 0.6745
        
        # Umbral universal
        N = sum(c.size for c in self._flatten(coeffs))
        umbral = sigma * np.sqrt(2 * np.log(N))
        
        def umbralizar_c(c, factor=1.0):
            if np.isscalar(c):
                return c
            return pywt.threshold(c, umbral * factor, mode=modo_umbral)
        
        result = []
        for i, c in enumerate(coeffs):
            if isinstance(c, tuple):
                cH, cV, cD = c
                result.append((umbralizar_c(cH), umbralizar_c(cV), umbralizar_c(cD)))
            else:
                # Aproximación: umbralizar menos o no
                result.append(umbralizar_c(c, 0.5) if i == 0 else umbralizar_c(c))
        return result
    
    def _flatten(self, coeffs):
        flat = []
        for c in coeffs:
            if isinstance(c, tuple):
                flat.extend(c)
            else:
                flat.append(c)
        return flat
    
    def energia_por_nivel(self, coeffs: List) -> List[float]:
        """Energía en cada nivel."""
        energias = []
        for c in coeffs:
            if isinstance(c, tuple):
                energias.append(sum(np.var(sub) for sub in c))
            else:
                energias.append(np.var(c))
        return energias


class Gabor(TransformadorEspectral):
    """
        Transformada de Gabor para análisis de textura direccional.
        
        Filtros gaussianos modulados por sinusoides, optimizados para
        detectar patrones orientados a diferentes escalas.
        
        Algoritmo:
            Gabor(u,v; λ, θ, ψ, σ, γ) = exp(-(u'²+γ²v'²)/(2σ²)) · cos(2πu'/λ + ψ)
            donde (u',v') = rotación de (u,v) por ángulo θ
        
        Ventajas:
            - Selectividad direccional, similitud a V1 visual cortex,
            - robusto a iluminación, bueno para texturas periódicas
        Desventajas:
            - Múltiples parámetros, redundante (no ortogonal),
            - costoso para muchas orientaciones/escalas
        
        Usos microscopía:
            - Análisis de orientación de fibras (colágeno, músculo)
            - detección de patrones periódicos (striaciones),
            - clasificación de texturas tisulares,
            - extracción de features para ML
    """
    nombre = "gabor"
    
    def __init__(self, frecuencias: Tuple[float, ...] = (0.1, 0.2, 0.4),
                orientaciones: int = 8, sigma: float = None, gamma: float = 0.5):
        self.frecuencias = frecuencias
        self.orientaciones = orientaciones
        self.sigma = sigma
        self.gamma = gamma
    
    def __call__(self, img: np.ndarray) -> np.ndarray:
        self._validar_imagen(img)
        
        # Crear banco de filtros
        filtros = self._crear_filtros()
        
        # Aplicar convoluciones
        respuestas = []
        for kernel in filtros:
            resp = cv2.filter2D(img.astype(np.float64), -1, kernel)
            respuestas.append(resp)
        
        # Magnitud de respuesta
        return np.stack(respuestas, axis=-1)  # (H, W, n_filtros)
    
    def _crear_filtros(self) -> List[np.ndarray]:
        """Genera kernels de Gabor."""
        filtros = []
        for theta in np.arange(0, np.pi, np.pi / self.orientaciones):
            for lambd in self.frecuencias:
                sigma = self.sigma if self.sigma else lambd * 2
                kernel = cv2.getGaborKernel(
                    (int(sigma * 6) | 1, int(sigma * 6) | 1),
                    sigma, theta, lambd, self.gamma, 0, cv2.CV_64F
                )
                filtros.append(kernel)
        return filtros
    
    def get_orientacion_dominante(self, respuestas: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Extrae orientación y magnitud dominante por píxel."""
        # respuestas: (H, W, n_filtros)
        magnitud_max = np.max(respuestas, axis=-1)
        indice_max = np.argmax(respuestas, axis=-1)
        
        # Mapear índice a ángulo
        n_freqs = len(self.frecuencias)
        angulos = np.tile(np.arange(0, np.pi, np.pi / self.orientaciones), n_freqs)
        orientacion = angulos[indice_max]
        
        return orientacion, magnitud_max