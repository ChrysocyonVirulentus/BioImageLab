"""
Métodos de deconvolución para restauración de imágenes borrosas.

La deconvolución invierte el proceso de degradación óptica modelado como:
g(x,y) = f(x,y) * h(x,y) + n(x,y)
donde g es la imagen observada, f la original, h la PSF y n el ruido.

Estos métodos recuperan la imagen original f estimada (f̂) conociendo
o estimando la función de dispersión de punto (PSF) del sistema óptico.

IMPORTANTE - Separación de responsabilidades:
- Estos métodos NO normalizan imágenes (ese rol es de normalizador.py)
- Asumen que la PSF está normalizada (suma = 1)
- Trabajan preferentemente en float64 para precisión numérica
- Requieren estimación previa de la PSF (excepto blind deconvolution)
- El ruido debe ser modelado o pre-filtrado para resultados óptimos
- La normalización previa (si es necesaria) debe hacerse con Normalizador

Tipos de deconvolución:
- Directa: Inversión en frecuencia (Wiener, Tikhonov) - rápida pero sensible
- Iterativa: Máxima verosimilitud (Richardson-Lucy) - lenta pero robusta
- Ciega: Estima PSF y imagen simultáneamente - para PSF desconocida

Métodos disponibles:
- Wiener: Filtro en frecuencia con regularización espectral
- RichardsonLucy: Máxima verosimilitud Poisson iterativa
- BlindDeconvolucion: Estimación conjunta de imagen y PSF
- Tikhonov: Regularización de norma L2 en frecuencia
"""

import numpy as np
import cv2
from typing import Optional, Tuple, Literal, Callable
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from scipy.signal import convolve2d
from skimage import restoration
import warnings


class Deconvolucionador:
    """
        Clase base para métodos de deconvolución.
        
        Los deconvolucionadores restauran imágenes borrosas invirtiendo
        el modelo de formación de imagen: convolución con PSF + ruido.
        
        Conceptos clave:
            - PSF (Point Spread Function): Respuesta del sistema a punto fuente
            - OTF (Optical Transfer Function): FFT de la PSF
            - Regularización: Término para estabilizar la inversión
            - Condición de problema mal planteado: División por valores pequeños
    """
    nombre = "deconvolucionador_base"
    
    def __call__(self, img: np.ndarray, psf: Optional[np.ndarray] = None) -> np.ndarray:
        """
            Aplica deconvolución a la imagen.
            
            Args:
                img: Array 2D (Y, X) con la imagen borrosa
                psf: Función de dispersión de punto (si el método la requiere)
                
            Returns:
                Imagen restaurada (generalmente float64)
        """
        raise NotImplementedError("Subclases deben implementar __call__")
    
    def _validar_imagen(self, img: np.ndarray):
        """Valida que la imagen sea 2D."""
        if img.ndim != 2:
            raise ValueError(f"La imagen debe ser 2D, tiene {img.ndim} dimensiones")
    
    def _validar_psf(self, psf: np.ndarray):
        """Valida que la PSF sea 2D y normalizada."""
        if psf.ndim != 2:
            raise ValueError(f"La PSF debe ser 2D, tiene {psf.ndim} dimensiones")
        if not np.isclose(psf.sum(), 1.0, rtol=0.01):
            warnings.warn("La PSF debería estar normalizada (suma = 1). Normalizando automáticamente.")
            psf = psf / psf.sum()
        return psf
    
    def _preparar_psf(self, psf: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
        """
            Prepara la PSF para convolución FFT (centrada y con padding).
            
            Args:
                psf: PSF original
                shape: Forma objetivo (tamaño de la imagen)
                
            Returns:
                PSF preparada para FFT
        """
        psf_padded = np.zeros(shape, dtype=np.float64)
        # Centrar PSF en el array
        h, w = psf.shape
        y_start = (shape[0] - h) // 2
        x_start = (shape[1] - w) // 2
        psf_padded[y_start:y_start+h, x_start:x_start+w] = psf
        # Circular shift para centrar en (0,0) para FFT
        psf_padded = np.fft.ifftshift(psf_padded)
        return psf_padded

@registrar_en("realzado")
class Wiener(Deconvolucionador):
    """
        Filtro de Wiener para deconvolución en dominio de frecuencia.
        
        El filtro de Wiener minimiza el error cuadrático medio entre la
        imagen restaurada y la original, asumiendo conocimiento del
        espectro de potencia del ruido y de la señal.
        
        Algoritmo:
            1. Transformar imagen y PSF al dominio de frecuencia (FFT)
            2. Calcular OTF (Optical Transfer Function) = FFT(PSF)
            3. Aplicar filtro: F̂ = G · H* / (|H|² + K)
            4. Transformar inversa: f̂ = IFFT(F̂)
        
        Ecuación:
            F̂(u,v) = [H*(u,v) · G(u,v)] / [|H(u,v)|² + K]
            
            donde:
                G: FFT de imagen observada
                H: OTF (FFT de PSF)
                H*: Complejo conjugado de H
                K: Constante de regularización (SNR inverso estimado)
        
        Interpretación:
            - K = 0: Deconvolución inversa pura (amplifica ruido)
            - K pequeño: Más agresivo, más detalle pero más ruido
            - K grande: Más suave, menos ruido pero menos detalle
            - Término |H|² en denominador evita división por cero
        
        Ventajas:
            - Solución cerrada, muy rápida (una FFT/IFFT)
            - Óptimo en sentido de mínimo error cuadrático medio
            - Fácil implementación y paralelización
            - Buen balance señal/ruido con K apropiado
        
        Desventajas:
            - Requiere estimar K (relación señal/ruido)
            - Asume estacionariedad (PSF espacialmente invariante)
            - Produce artefactos de Gibbs en bordes fuertes
            - No preserva no-negatividad (puede dar valores negativos)
            - Ringing alrededor de objetos puntiagudos
        
        Usos típicos en microscopía:
            - Restauración rápida de imágenes de campo ancho
            - Corrección de desenfoque de aberración cromática
            - Preprocesamiento cuando el tiempo es crítico
            - Deconvolución de imágenes con SNR moderado/alto
            - Benchmark para métodos iterativos más lentos
    """
    nombre = "wiener"
    
    def __init__(self, K: Optional[float] = None, balance: float = 0.1):
        """
            Args:
                K: Constante de regularización (relación señal/ruido inversa)
                Si None, se estima automáticamente del ruido de fondo
                Valores típicos: 0.001 - 0.1
                Menor = más agresivo, Mayor = más conservador
                
                balance: Factor para estimación automática de K
                        K_auto = varianza_ruido / varianza_señal × balance
        """
        self.K = K
        self.balance = balance
    
    def __call__(self, img: np.ndarray, psf: np.ndarray) -> np.ndarray:
        """
            Aplica filtro de Wiener.
            
            Args:
                img: Imagen borrosa 2D
                psf: Función de dispersión de punto (normalizada)
                
            Returns:
                Imagen restaurada (float64)
        """
        self._validar_imagen(img)
        psf = self._validar_psf(psf)
        
        img_float = img.astype(np.float64)
        
        # Preparar PSF con padding
        psf_prep = self._preparar_psf(psf, img.shape)
        
        # Transformadas de Fourier
        G = fft2(img_float)
        H = fft2(psf_prep)
        H_conj = np.conj(H)
        
        # Estimar K si no se proporcionó
        if self.K is None:
            # Estimación basada en varianza en regiones de fondo
            # Asumimos que las esquinas son fondo
            esquina = img_float[:20, :20]
            var_ruido = np.var(esquina)
            var_señal = np.var(img_float)
            K_est = (var_ruido / (var_señal + 1e-10)) * self.balance
            K = max(K_est, 1e-6)  # Evitar K=0
        else:
            K = self.K
        
        # Filtro de Wiener
        H_mag_sq = np.abs(H) ** 2
        F_hat = (H_conj * G) / (H_mag_sq + K)
        
        # Transformada inversa
        img_restaurada = np.real(ifft2(F_hat))
        
        return img_restaurada

@registrar_en("realzado")
class RichardsonLucy(Deconvolucionador):
    """
        Deconvolución de Richardson-Lucy (máxima verosimilitud Poisson).
        
        Método iterativo que maximiza la verosimilitud de la imagen
        restaurada asumiendo estadística de Poisson (fotones).
        
        Algoritmo:
            1. Inicializar: f⁰ = imagen observada (o uniforme)
            2. Iterar hasta convergencia:
            a. Convolución forward: c = fⁿ ⊗ h
            b. División relativa: r = g / c
            c. Convolución backward: error = r ⊗ h_rotada
            d. Actualización multiplicativa: fⁿ⁺¹ = fⁿ · error
            3. Normalizar si es necesario
        
        Ecuación iterativa:
            f⁽ᵏ⁺¹⁾ = f⁽ᵏ⁾ · [ (g / (f⁽ᵏ⁾ ⊗ h)) ⊗ h* ]
            
            donde:
                ⊗: Convolución
                h*: PSF rotada 180° (adjoint)
                ·: Multiplicación elemento a elemento
        
        Interpretación:
            - Actualización multiplicativa preserva no-negatividad
            - División g/c corrige discrepancias con observación
            - Convolución backward propaga correcciones espacialmente
            - Converge a máximo de verosimilitud (ML) pero puede sobreajustar
        
        Ventajas:
            - Preserva no-negatividad (físicamente correcto para intensidades)
            - Fundamentado estadísticamente (ML para Poisson)
            - Mejor que métodos directos en presencia de ruido de Poisson
            - Convergencia estable (monótona en likelihood)
            - No requiere parámetros de regularización explícitos
        
        Desventajas:
            - Lento (múltiples iteraciones, cada una con 2 convoluciones)
            - Requiere criterio de parada (número de iteraciones)
            - Amplifica ruido si se itera demasiado (sobreajuste)
            - Artefactos de ringing en bordes fuertes
            - Requiere PSF precisa
        
        Usos típicos en microscopía:
            - Deconvolución de imágenes confocal y de fluorescencia
            - Restauración de imágenes con fotocontaje (Poisson)
            - Super-resolución computacional (con PSF sub-píxel)
            - Cuantificación precisa de intensidades
            - Imágenes de bajo SNR donde la estadística importa
    """
    nombre = "richardson_lucy"
    
    def __init__(self, 
                iteraciones: int = 30,
                clip: bool = True,
                filtro_tv: Optional[float] = None):
        """
            Args:
                iteraciones: Número de iteraciones de Richardson-Lucy
                            Valores típicos: 10-50
                            Menos: Sub-restauración
                            Más: Más detalle pero más ruido/artefactos
                
                clip: Si True, fuerza no-negatividad en cada iteración
                
                filtro_tv: Si se especifica, aplica regularización de variación
                        total (Total Variation) para reducir ruido
                        Valor típico: 0.001-0.01 (peso del término TV)
        """
        if iteraciones < 1:
            raise ValueError("iteraciones debe ser >= 1")
        
        self.iteraciones = iteraciones
        self.clip = clip
        self.filtro_tv = filtro_tv
    
    def __call__(self, img: np.ndarray, psf: np.ndarray) -> np.ndarray:
        """
            Aplica deconvolución de Richardson-Lucy.
            
            Args:
                img: Imagen observada 2D
                psf: Función de dispersión de punto (normalizada)
                
            Returns:
                Imagen restaurada (float64, no-negativa)
        """
        self._validar_imagen(img)
        psf = self._validar_psf(psf)
        
        img_float = img.astype(np.float64)
        
        # Asegurar no-negatividad inicial
        img_float = np.maximum(img_float, 0)
        
        # Usar implementación de skimage (optimizada)
        if self.filtro_tv is not None:
            # Con regularización TV
            resultado = restoration.richardson_lucy(
                img_float, 
                psf, 
                num_iter=self.iteraciones,
                clip=self.clip,
                filter_epsilon=self.filtro_tv
            )
        else:
            resultado = restoration.richardson_lucy(
                img_float,
                psf,
                num_iter=self.iteraciones,
                clip=self.clip
            )
        
        return resultado

@registrar_en("realzado")
class BlindDeconvolucion(Deconvolucionador):
    """
        Deconvolución ciega para cuando la PSF es desconocida.
        
        Estima simultáneamente la imagen original y la PSF mediante
        alternancia entre actualización de imagen (con PSF fija) y
        actualización de PSF (con imagen fija).
        
        Algoritmo:
            1. Inicializar: f⁰ = imagen observada, h⁰ = guess inicial (gaussiana)
            2. Iterar:
            a. Actualizar imagen: fⁿ⁺¹ = RL_step(g, hⁿ, fⁿ)
            b. Actualizar PSF: hⁿ⁺¹ = RL_step(g, fⁿ⁺¹, hⁿ)
            c. Normalizar PSF: hⁿ⁺¹ = hⁿ⁺¹ / sum(hⁿ⁺¹)
            d. Aplicar constraints a PSF (simetría, tamaño, etc.)
            3. Retornar imagen y PSF estimadas
        
        Ecuación (esquema alternado):
            f⁽ᵏ⁺¹⁾ = f⁽ᵏ⁾ · [(g / (f⁽ᵏ⁾ ⊗ h⁽ᵏ⁾)) ⊗ h⁽ᵏ⁾*]
            h⁽ᵏ⁺¹⁾ = h⁽ᵏ⁾ · [(g / (f⁽ᵏ⁺¹⁾ ⊗ h⁽ᵏ⁾)) ⊗ f⁽ᵏ⁺¹⁾*]
            
            sujeto a: h(x,y) ≥ 0, ∑h = 1, simetría opcional
        
        Interpretación:
            - Problema altamente sub-determinado (infinitas soluciones)
            - Requiere constraints fuertes en PSF (tamaño, simetría, suavidad)
            - Convergencia lenta y a mínimos locales
            - Útil cuando no se puede calibrar el microscopio
        
        Ventajas:
            - No requiere calibración previa del microscopio
            - Útil para microscopios con aberraciones desconocidas
            - Puede adaptarse a PSF espacialmente variable (slices)
            - Estima PSF específica de cada imagen
        
        Desventajas:
            - Muy lento (doble RL por iteración)
            - Inestable sin buenos constraints
            - Mínimos locales (soluciones triviales: delta, uniforme)
            - Requiere inicialización cuidadosa de PSF
            - Calidad inferior a deconvolución con PSF conocida
            - Puede estimar PSF físicamente imposible
        
        Usos típicos en microscópicos:
            - Microscopios sin calibración de PSF disponible
            - Correción de aberraciones de campo óptico desconocidas
            - Restauración de imágenes históricas sin metadatos
            - Microscopía con medios de montaje desconocidos
            - Cuando la PSF varía significativamente del modelo teórico
    """
    nombre = "blind_deconvolucion"
    
    def __init__(self,
                psf_size: Tuple[int, int] = (15, 15),
                iteraciones: int = 10,
                iteraciones_rl: int = 5,
                simetria: bool = True,
                suavidad_psf: float = 0.01):
        """
            Args:
                psf_size: Tamaño estimado de la PSF (alto, ancho)
                        Debe ser impar para centrar correctamente
                        Típico: (15,15) a (31,31) para microscopía de campo ancho
                
                iteraciones: Número de iteraciones externas (alternancia)
                            Cada una hace RL de imagen y RL de PSF
                
                iteraciones_rl: Iteraciones de RL internas por paso
                            Menos = más rápido pero menos estable
                            Valores típicos: 3-10
                
                simetria: Si True, fuerza simetría en PSF (común en óptica)
                
                suavidad_psf: Regularización de suavidad para PSF
                            Evita PSF con estructuras espurias
                            Valores típicos: 0.001-0.1
        """
        if len(psf_size) != 2 or psf_size[0] % 2 == 0 or psf_size[1] % 2 == 0:
            raise ValueError("psf_size debe ser (impar, impar)")
        if iteraciones < 1 or iteraciones_rl < 1:
            raise ValueError("iteraciones deben ser >= 1")
        
        self.psf_size = psf_size
        self.iteraciones = iteraciones
        self.iteraciones_rl = iteraciones_rl
        self.simetria = simetria
        self.suavidad_psf = suavidad_psf
        
        # Inicializadores RL
        self._rl_img = RichardsonLucy(iteraciones=iteraciones_rl, clip=True)
        self._rl_psf = RichardsonLucy(iteraciones=iteraciones_rl, clip=True)
    
    def __call__(self, img: np.ndarray, psf_inicial: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Aplica deconvolución ciega.
        
        Args:
            img: Imagen observada 2D
            psf_inicial: Estimación inicial de PSF (si None, usa gaussiana)
            
        Returns:
            (imagen_restaurada, psf_estimada) ambas float64
        """
        self._validar_imagen(img)
        
        img_float = img.astype(np.float64)
        img_float = np.maximum(img_float, 0)
        
        # Inicializar PSF
        if psf_inicial is None:
            # Gaussiana isotrópica como inicialización conservadora
            y, x = np.ogrid[-self.psf_size[0]//2:self.psf_size[0]//2+1,
                            -self.psf_size[1]//2:self.psf_size[1]//2+1]
            sigma = min(self.psf_size) / 6.0
            psf = np.exp(-(x**2 + y**2) / (2 * sigma**2))
            psf = psf / psf.sum()
        else:
            psf = self._validar_psf(psf_inicial.copy())
            # Reescalar si es necesario
            if psf.shape != self.psf_size:
                psf = cv2.resize(psf, self.psf_size[::-1])
                psf = psf / psf.sum()
        
        # Inicializar imagen
        img_est = img_float.copy()
        
        # Iteraciones alternadas
        for i in range(self.iteraciones):
            # 1. Actualizar imagen con PSF fija
            img_est = self._rl_img(img_float, psf)
            
            # 2. Actualizar PSF con imagen fija
            # Trasponer para usar RL (la PSF actúa como "imagen")
            psf = self._rl_psf(img_float, img_est)
            
            # 3. Constraints en PSF
            # Normalizar
            psf = np.maximum(psf, 0)
            psf = psf / psf.sum()
            
            # Forzar simetría si se solicita
            if self.simetria:
                psf = (psf + psf[::-1, ::-1]) / 2.0
            
            # Suavidad (regularización)
            if self.suavidad_psf > 0:
                psf = cv2.GaussianBlur(psf, (3, 3), self.suavidad_psf)
                psf = psf / psf.sum()
            
            # Crop a tamaño deseado (evitar crecimiento)
            if psf.shape != self.psf_size:
                cy, cx = psf.shape[0] // 2, psf.shape[1] // 2
                hy, hx = self.psf_size[0] // 2, self.psf_size[1] // 2
                psf = psf[cy-hy:cy+hy+1, cx-hx:cx+hx+1]
                psf = psf / psf.sum()
        
        return img_est, psf

@registrar_en("realzado")
class Tikhonov(Deconvolucionador):
    """
        Deconvolución regularizada de Tikhonov (norma L2).
        
        Método directo en frecuencia que estabiliza la inversión mediante
        regularización de Tikhonov (penalización de energía de la solución).
        
        Algoritmo:
            1. Transformar imagen y PSF a frecuencia
            2. Aplicar filtro regularizado:
            F̂ = H* · G / (|H|² + λ|L|²)
            donde L es operador de regularización (típicamente identidad o Laplaciano)
            3. Transformada inversa
        
        Ecuación:
            F̂(u,v) = [H*(u,v) · G(u,v)] / [|H(u,v)|² + λ · |L(u,v)|²]
            
            Variantes de L:
            - L = 1: Regularización de identidad (suavidad cero)
            - L = 2π√(u²+v²): Regularización de gradiente (Tikhonov generalizado)
            - L = 4π²(u²+v²): Regularización de Laplaciano (suavidad segunda)
        
        Interpretación:
            - λ = 0: Deconvolución inversa (inestable)
            - λ pequeño: Favorece ajuste a datos (más detalle, más ruido)
            - λ grande: Favorece suavidad (menos ruido, menos detalle)
            - Término |L|² penaliza alta frecuencia en la solución
        
        Ventajas:
            - Solución cerrada, rápida de calcular
            - Estable numéricamente (denominador siempre > 0)
            - Control explícito de trade-off señal/ruido via λ
            - Fundamentado en teoría de regularización
            - Generalizable a diferentes operadores L
        
        Desventajas:
            - Asume PSF invariante espacialmente
            - Tiende a sobre-suavizar (preferencia por soluciones de baja energía)
            - No preserva no-negatividad
            - Artefactos de ringing como Wiener
            - Difícil elegir λ óptimo sin conocer estadísticas del ruido
        
        Comparación con Wiener:
            - Wiener usa K constante (estimación de SNR)
            - Tikhonov usa λ|L|² (penalización frecuencia-dependiente)
            - Wiener es óptimo para MSE si se conoce espectro
            - Tikhonov es más general y estable
        
        Usos típicos en microscopía:
            - Restauración cuando se prefiere suavidad controlada
            - Imágenes con ruido gaussiano significativo
            - Deconvolución de grandes volúmenes (rápido)
            - Preprocesamiento antes de segmentación
            - Cuando se requiere control frecuencial explícito
    """
    nombre = "tikhonov"
    
    def __init__(self, 
                lambda_reg: float = 0.01,
                orden: Literal[0, 1, 2] = 0):
        """
            Args:
                lambda_reg: Parámetro de regularización (λ)
                        Valores típicos: 0.001 - 0.1
                        Menor = más fiel a datos, Mayor = más suave
                
                orden: Orden del operador de regularización
                    0: Identidad (Tikhonov estándar)
                    1: Gradiente (penaliza cambios de primer orden)
                    2: Laplaciano (penaliza curvatura)
        """
        if lambda_reg < 0:
            raise ValueError("lambda_reg debe ser >= 0")
        if orden not in [0, 1, 2]:
            raise ValueError("orden debe ser 0, 1 o 2")
        
        self.lambda_reg = lambda_reg
        self.orden = orden
    
    def __call__(self, img: np.ndarray, psf: np.ndarray) -> np.ndarray:
        """
            Aplica deconvolución de Tikhonov.
            
            Args:
                img: Imagen observada 2D
                psf: Función de dispersión de punto (normalizada)
                
            Returns:
                Imagen restaurada (float64)
        """
        self._validar_imagen(img)
        psf = self._validar_psf(psf)
        
        img_float = img.astype(np.float64)
        
        # Preparar PSF
        psf_prep = self._preparar_psf(psf, img.shape)
        
        # Transformadas
        G = fft2(img_float)
        H = fft2(psf_prep)
        H_conj = np.conj(H)
        
        # Construir operador de regularización L en frecuencia
        sy, sx = img.shape
        y, x = np.ogrid[-sy//2:sy//2, -sx//2:sx//2]
        
        if self.orden == 0:
            # Identidad: |L|² = 1
            L_sq = 1.0
        elif self.orden == 1:
            # Gradiente: |L|² ∝ u² + v²
            # Frecuencias normalizadas [-0.5, 0.5]
            u = x / sx
            v = y / sy
            L_sq = (2 * np.pi)**2 * (u**2 + v**2)
            L_sq = fftshift(L_sq)  # Centrar para FFT
        else:  # orden == 2
            # Laplaciano: |L|² ∝ (u² + v²)²
            u = x / sx
            v = y / sy
            L_sq = (2 * np.pi)**4 * (u**2 + v**2)**2
            L_sq = fftshift(L_sq)
        
        # Filtro de Tikhonov
        denominador = np.abs(H)**2 + self.lambda_reg * L_sq
        F_hat = (H_conj * G) / denominador
        
        # Inversa
        img_restaurada = np.real(ifft2(F_hat))
        
        return img_restaurada