"""
Métodos de convolución para filtrado espacial y simulación de adquisición.

La convolución espacial es la operación fundamental de procesamiento de
imágenes: deslizar un kernel sobre la imagen para calcular respuestas
locales ponderadas. Se usa para suavizado, detección de características,
simulación de sistemas ópticos y corrección de aberraciones.

Principio fundamental:
g(x,y) = ∑∑ f(u,v) · h(x-u, y-v) = (f * h)(x,y)

donde f es la imagen, h el kernel y g el resultado. En frecuencia:
G(u,v) = F(u,v) · H(u,v)

IMPORTANTE - Separación de responsabilidades:
- Estos métodos NO normalizan imágenes (ese rol es de normalizador.py)
- Solo realizan conversiones de tipo cuando es estrictamente necesario
- Trabajan con los valores de la imagen tal como vienen
- Los kernels deben estar diseñados/preparados externamente si es necesario
- La normalización previa (si es necesaria) debe hacerse con Normalizador
- Para deconvolución (inversa), usar Metodos_Deconvolucion.py

Tipos de convolución:
- Lineal: Respuesta estándar, bordes tratados con padding
- Circular: Para implementación FFT, asume periodicidad espacial
- Separable: Kernel descompuesto en 1D para eficiencia O(n) vs O(n²)

Métodos disponibles:
- KernelPersonalizado: Convolución con kernel arbitrario definido por usuario
- PSFSimulacion: Generación de PSF teóricas (Airy, Gaussiana, Gibson-Lanni)
- KernelSeparable: Optimización para kernels separables (Gaussian, Sobel, etc.)
- ConvolucionFrecuencia: Implementación FFT para kernels grandes
- CorreccionBordes: Convolución con extrapolación de bordes avanzada
"""

import numpy as np
import cv2
from typing import Optional, Tuple, Literal, Union, Callable
from scipy.ndimage import convolve
from scipy.signal import convolve2d, fftconvolve
from scipy.special import j1
import warnings


class Convolucionador:
    """
        Clase base para métodos de convolución espacial.
        
        Los convolucionadores aplican filtros lineales mediante deslizamiento
        de kernels sobre la imagen.
        
        Conceptos clave:
            - Kernel/Máscara: Matriz de coeficientes de ponderación
            - Padding: Tratamiento de bordes (zero, reflect, constant, etc.)
            - Correlación vs Convolución: Convolución rota kernel 180°
            - Complejidad: O(N·M·K·L) para imagen N×M y kernel K×L
    """
    nombre = "convolucionador_base"
    
    def __call__(self, img: np.ndarray) -> np.ndarray:
        """
            Aplica la operación de convolución a la imagen.
            
            Args:
                img: Array 2D (Y, X) con la imagen a procesar
                
            Returns:
                Imagen convolucionada del mismo tipo o float64
        """
        raise NotImplementedError("Subclases deben implementar __call__")
    
    def _validar_imagen(self, img: np.ndarray):
        """Valida que la imagen sea 2D."""
        if img.ndim != 2:
            raise ValueError(f"La imagen debe ser 2D, tiene {img.ndim} dimensiones")
    
    def _aplicar_convolucion(self, 
                            img: np.ndarray, 
                            kernel: np.ndarray,
                            modo: Literal['auto', 'directo', 'fft'] = 'auto',
                            padding: str = 'reflect') -> np.ndarray:
        """
            Aplica convolución seleccionando el método óptimo.
            
            Args:
                img: Imagen de entrada
                kernel: Kernel de convolución (se asume ya rotado 180° si es necesario)
                modo: 'auto' elige entre directo o FFT según tamaño
                padding: 'reflect', 'constant', 'nearest', 'mirror', 'wrap'
                
            Returns:
                Imagen convolucionada en float64
        """
        img_float = img.astype(np.float64)
        
        # Decidir método
        if modo == 'auto':
            # FFT es más rápido cuando el kernel es > 7x7 aprox
            usar_fft = (kernel.size > 49) or (img.size > 1e6 and kernel.size > 25)
        else:
            usar_fft = (modo == 'fft')
        
        if usar_fft:
            # Usar FFT para eficiencia
            resultado = fftconvolve(img_float, kernel, mode='same')
        else:
            # Convolución directa con manejo de bordes
            if padding == 'reflect':
                boundary = 'symm'
            elif padding == 'constant':
                boundary = 'fill'
            else:
                boundary = padding
            
            resultado = convolve2d(img_float, kernel, mode='same', boundary=boundary)
        
        return resultado
    
    def _normalizar_resultado(self, 
                            resultado: np.ndarray, 
                            tipo_original: np.dtype,
                            clip: bool = True) -> np.ndarray:
        """Convierte resultado al tipo original con clip opcional."""
        if np.issubdtype(tipo_original, np.integer) and clip:
            info = np.iinfo(tipo_original)
            resultado = np.clip(resultado, info.min, info.max)
        
        return resultado.astype(tipo_original)


class KernelPersonalizado(Convolucionador):
    """
        Convolución con kernel arbitrario definido por el usuario.
        
        Permite aplicar cualquier filtro lineal espacial mediante especificación
        directa de la matriz de coeficientes. Base para filtros personalizados
        de suavizado, detección de características o restauración específica.
        
        Algoritmo:
            1. Validar kernel (2D, no vacío)
            2. Opcional: Normalizar kernel (suma = 1 para conservar intensidad)
            3. Aplicar convolución 2D con manejo de bordes
            4. Opcional: Clip y conversión de tipo
        
        Ecuación:
            g(x,y) = ∑_{i=-k}^{k} ∑_{j=-k}^{k} h(i,j) · f(x-i, y-j)
            
            donde h es el kernel de tamaño (2k+1) × (2k+1)
        
        Consideraciones de diseño de kernels:
            - Suma = 1: Conserva intensidad media (filtros de suavizado)
            - Suma = 0: Detectores de borde (respuesta cero en regiones planas)
            - Simétrico: Fase cero, sin desplazamiento espacial
            - Separable: h(x,y) = h_x(x) · h_y(y) para eficiencia computacional
        
        Ventajas:
            - Flexibilidad total en el diseño del filtro
            - Implementación optimizada (selección automática directo/FFT)
            - Múltiples modos de padding para tratamiento de bordes
            - Preservación de tipos de dato
            - Compatible con kernels de cualquier tamaño (impar o par)
        
        Desventajas:
            - Requiere conocimiento del diseño de filtros espaciales
            - Kernels grandes son costosos (O(n²) por píxel)
            - Efectos de borde si no se elige padding apropiado
            - No adaptativo (mismo kernel en toda la imagen)
        
        Usos típicos en microscopía:
            - Aplicación de filtros de suavizado personalizados (anisotrópicos)
            - Convolución con PSF medida experimentalmente
            - Filtros de realce de alta frecuencia específicos
            - Kernels de detección de patrones celulares específicos
            - Corrección de shading iluminación mediante kernel de división
            - Simulación de aberraciones ópticas específicas
    """
    nombre = "kernel_personalizado"
    
    def __init__(self, 
                kernel: np.ndarray,
                normalizar: bool = False,
                modo: Literal['auto', 'directo', 'fft'] = 'auto',
                padding: str = 'reflect'):
        """
            Args:
                kernel: Matriz 2D de coeficientes (se rota 180° si es necesario)
                    Tamaño típico: 3x3 a 31x31 para microscopía
                
                normalizar: Si True, divide kernel por suma (conserva intensidad)
                        Recomendado para filtros de suavizado
                        No usar para detectores de borde (suma=0)
                
                modo: Método de computación
                    'auto': Elige FFT para kernels grandes (>7x7)
                    'directo': Convolución espacial directa
                    'fft': Transformada rápida de Fourier
                
                padding: Modo de tratamiento de bordes
                        'reflect': Refleja la imagen (default, sin artefactos)
                        'constant': Rellena con ceros (pérdida de información en bordes)
                        'nearest': Extiende con valor del borde
                        'wrap': Periódico (para imágenes de fase)
        """
        if kernel.ndim != 2:
            raise ValueError(f"El kernel debe ser 2D, tiene {kernel.ndim} dimensiones")
        if kernel.size == 0:
            raise ValueError("El kernel no puede estar vacío")
        
        self.kernel = kernel.astype(np.float64)
        
        if normalizar and np.abs(self.kernel.sum()) > 1e-10:
            self.kernel = self.kernel / self.kernel.sum()
        
        self.modo = modo
        self.padding = padding
    
    def __call__(self, img: np.ndarray) -> np.ndarray:
        """
            Aplica convolución con el kernel personalizado.
            
            Args:
                img: Imagen 2D (cualquier tipo numérico)
                
            Returns:
                Imagen convolucionada del mismo tipo que la entrada
        """
        self._validar_imagen(img)
        
        # Rotar kernel 180° para convolución verdadera (no correlación)
        # Si el usuario ya proporcionó kernel rotado, esto lo corrige
        kernel_rotado = np.rot90(self.kernel, 2)
        
        resultado = self._aplicar_convolucion(img, kernel_rotado, self.modo, self.padding)
        
        return self._normalizar_resultado(resultado, img.dtype)
    
    def get_kernel(self) -> np.ndarray:
        """Devuelve el kernel actual (útil para inspección)."""
        return self.kernel.copy()


class PSFSimulacion(Convolucionador):
    """
        Simulación de Funciones de Dispersión de Punto (PSF) teóricas.
        
        Genera PSF matemáticamente modeladas para diferentes configuraciones
        de microscopía. Útil para simulación de adquisición, deconvolución
        y análisis de límites de resolución.
        
        Algoritmos disponibles:
        
        1. Gaussiana: Aproximación simple, buena para confocal/Pinhole grande
        PSF(r) = exp(-r²/(2σ²))
        
        2. Airy: Patrón de difracción de apertura circular (campo ancho)
        PSF(r) = [2·J₁(k·NA·r)/(k·NA·r)]²
        donde J₁ es función de Bessel, k=2π/λ, NA apertura numérica
        
        3. Gibson-Lanni: Modelo vectorial 3D para objetivos de inmersión
        Incluye aberraciones de índice de refracción mismatch
        
        4. Born-Wolf: Aproximación escalar de alta apertura
        
        Ecuación general (Gibson-Lanni):
            PSF(z,r) = |∫₀^NA P(θ)·exp[i·k·W(θ,z)]·J₀(k·r·sinθ)·sinθ·dθ|²
            
            donde P es función pupilar, W aberración de fase, J₀ Bessel orden 0
        
        Interpretación:
            - σ (Gauss): Inversamente proporcional a NA
            - Primer mínimo de Airy: r = 0.61·λ/NA (criterio de Rayleigh)
            - Strehl ratio: Pico de PSF real / Pico de PSF ideal (calidad)
        
        Ventajas:
            - No requiere calibración experimental (beads fluorescentes)
            - Parametrizable (NA, λ, medio, etc.)
            - Reproducible y controlable
            - Fundamento físico sólido
            - Generación rápida de PSF para cualquier configuración
        
        Desventajas:
            - Modelos teóricos idealizan condiciones reales
            - No incluyen aberraciones específicas del sistema
            - Gibson-Lanni es computacionalmente costoso
            - Requiere parámetros ópticos precisos
        
        Usos típicos en microscopía:
            - Simulación de adquisición para validación de algoritmos
            - Deconvolución cuando no hay beads de calibración
            - Análisis de resolución teórica vs experimental
            - Diseño de experimentos (elección de objetivo, longitud de onda)
            - Generación de ground truth para benchmarking
            - Corrección de aberraciones de índice de refracción (Gibson-Lanni)
    """
    nombre = "psf_simulacion"
    
    def __init__(self,
                tipo: Literal["gaussiana", "airy", "gibson_lanni", "born_wolf"] = "airy",
                na: float = 1.4,
                lambda_nm: float = 520.0,
                n_medio: float = 1.518,
                resolucion_pixel_nm: float = 100.0,
                tamanio_psf: int = 31,
                **kwargs):
        """
            Args:
                tipo: Modelo de PSF a generar
                    "gaussiana": Aproximación simple, rápida
                    "airy": Difracción circular, campo ancho estándar
                    "gibson_lanni": Modelo 3D con aberraciones de índice
                    "born_wolf": Escalar de alta apertura
                
                na: Apertura numérica del objetivo (0.1-1.6 típico)
                
                lambda_nm: Longitud de onda de emisión (nm)
                        Verde: 520nm, Rojo: 600nm, Azul: 480nm
                
                n_medio: Índice de refracción del medio de montaje
                        1.0: Aire, 1.33: Agua, 1.518: Aceite inmersión
                
                resolucion_pixel_nm: Tamaño de píxel en nanómetros
                
                tamanio_psf: Tamaño del array de salida (impar recomendado)
                
                kwargs adicionales:
                    - para gibson_lanni: n_immersion, espesor_cubeta_um, profundidad_um
                    - para gaussiana: sigma_puntos (opcional, auto-calculado si no se da)
        """
        self.tipo = tipo
        self.na = na
        self.lambda_nm = lambda_nm
        self.n_medio = n_medio
        self.resolucion_pixel_nm = resolucion_pixel_nm
        self.tamanio_psf = tamanio_psf
        self.kwargs = kwargs
        
        # Generar PSF inmediatamente
        self.psf = self._generar_psf()
    
    def _generar_psf(self) -> np.ndarray:
        """Genera la PSF según el tipo especificado."""
        if self.tipo == "gaussiana":
            return self._psf_gaussiana()
        elif self.tipo == "airy":
            return self._psf_airy()
        elif self.tipo == "gibson_lanni":
            return self._psf_gibson_lanni()
        elif self.tipo == "born_wolf":
            return self._psf_born_wolf()
        else:
            raise ValueError(f"Tipo de PSF '{self.tipo}' no reconocido")
    
    def _psf_gaussiana(self) -> np.ndarray:
        """
        PSF Gaussiana 2D.
        
        Aproximación simple donde σ ≈ 0.21·λ/NA (FWHM de Airy)
        """
        # Sigma en micrómetros
        sigma_um = 0.21 * self.lambda_nm / 1000 / self.na
        
        # Convertir a píxeles
        sigma_px = sigma_um * 1000 / self.resolucion_pixel_nm
        
        # Si se especificó sigma en kwargs, usar ese
        if 'sigma_puntos' in self.kwargs:
            sigma_px = self.kwargs['sigma_puntos']
        
        # Crear grid
        size = self.tamanio_psf
        x = np.arange(size) - size // 2
        y = np.arange(size) - size // 2
        X, Y = np.meshgrid(x, y)
        
        # Gaussiana 2D
        psf = np.exp(-(X**2 + Y**2) / (2 * sigma_px**2))
        
        return psf / psf.sum()
    
    def _psf_airy(self) -> np.ndarray:
        """
            Patrón de Airy para apertura circular.
            
            PSF(r) = [2·J₁(v)/v]² donde v = k·NA·r
        """
        # Constantes
        k = 2 * np.pi * self.n_medio / (self.lambda_nm / 1000)  # 1/um
        
        # Grid en micrómetros
        size = self.tamanio_psf
        r_max = size * self.resolucion_pixel_nm / 1000 / 2
        r = np.linspace(-r_max, r_max, size)
        X, Y = np.meshgrid(r, r)
        R = np.sqrt(X**2 + Y**2)
        
        # Variable normalizada
        v = k * self.na * R
        v = np.where(v == 0, 1e-10, v)  # Evitar división por cero
        
        # Patrón de Airy
        psf = (2 * j1(v) / v) ** 2
        
        return psf / psf.sum()
    
    def _psf_gibson_lanni(self) -> np.ndarray:
        """
            Modelo de Gibson-Lanni para objetivos de inmersión.
            
            Incluye aberraciones por mismatch de índice de refracción entre
            medio de diseño (aceite) y medio de muestra (agua/glicerol).
        """
        # Parámetros adicionales con defaults
        n_immersion = self.kwargs.get('n_immersion', 1.518)  # Aceite típico
        espesor_cubeta = self.kwargs.get('espesor_cubeta_um', 170.0)  # Cubeta estándar
        profundidad = self.kwargs.get('profundidad_um', 10.0)  # Profundidad en muestra
        
        # Constantes
        k = 2 * np.pi * self.n_medio / (self.lambda_nm / 1000)
        
        # Grid
        size = self.tamanio_psf
        r_max = size * self.resolucion_pixel_nm / 1000 / 2
        r = np.linspace(0, r_max, size // 2 + 1)  # Solo radio (simetría)
        
        # Ángio de apertura máximo
        theta_max = np.arcsin(min(self.na / self.n_medio, 1.0))
        
        # Integral sobre ángio
        n_puntos = 100
        theta = np.linspace(0, theta_max, n_puntos)
        
        # Aberración de fase (Gibson-Lanni)
        # W = k·(n_medio·cosθ_medio - n_immersion·cosθ_immersion)·d
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        
        # Ley de Snell para ángulos en diferentes medios
        sin_theta_imm = self.n_medio * sin_theta / n_immersion
        cos_theta_imm = np.sqrt(1 - sin_theta_imm**2)
        
        # Aberración defocus y spherical
        defocus = k * espesor_cubeta * (self.n_medio * cos_theta - n_immersion * cos_theta_imm)
        spherical = k * profundidad * self.n_medio * (1 - cos_theta)
        
        aberracion = defocus + spherical
        
        # Función pupilar
        pupila = np.exp(1j * aberracion)
        
        # Integral de Bessel (Hankel transform)
        psf_radial = np.zeros_like(r)
        for i, ri in enumerate(r):
            if ri == 0:
                integrando = pupila * sin_theta * cos_theta
            else:
                integrando = pupila * j1(k * ri * sin_theta) * sin_theta * cos_theta
            psf_radial[i] = np.abs(np.trapz(integrando, theta)) ** 2
        
        # Expandir a 2D (simetría radial)
        psf = np.zeros((size, size))
        center = size // 2
        
        for y in range(size):
            for x in range(size):
                r_px = np.sqrt((x - center)**2 + (y - center)**2)
                r_idx = int(r_px)
                if r_idx < len(psf_radial):
                    psf[y, x] = psf_radial[r_idx]
        
        return psf / psf.sum()
    
    def _psf_born_wolf(self) -> np.ndarray:
        """
            Aproximación de Born-Wolf (escalar de alta apertura).
            
            Similar a Airy pero incluye factor de apodización cos(θ).
        """
        k = 2 * np.pi * self.n_medio / (self.lambda_nm / 1000)
        
        size = self.tamanio_psf
        r_max = size * self.resolucion_pixel_nm / 1000 / 2
        r = np.linspace(0, r_max, size // 2 + 1)
        
        theta_max = np.arcsin(min(self.na / self.n_medio, 1.0))
        n_puntos = 100
        theta = np.linspace(0, theta_max, n_puntos)
        
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        
        psf_radial = np.zeros_like(r)
        for i, ri in enumerate(r):
            if ri == 0:
                integrando = cos_theta**0.5 * sin_theta * cos_theta
            else:
                integrando = (cos_theta**0.5 * j1(k * ri * sin_theta) * 
                             sin_theta * cos_theta)
            psf_radial[i] = np.abs(np.trapz(integrando, theta)) ** 2
        
        # Expandir a 2D
        psf = np.zeros((size, size))
        center = size // 2
        
        for y in range(size):
            for x in range(size):
                r_px = np.sqrt((x - center)**2 + (y - center)**2)
                r_idx = int(r_px)
                if r_idx < len(psf_radial):
                    psf[y, x] = psf_radial[r_idx]
        
        return psf / psf.sum()
    
    def __call__(self, img: Optional[np.ndarray] = None) -> np.ndarray:
        """
            Genera o aplica la PSF simulada.
            
            Args:
                img: Si se proporciona, aplica convolución con la PSF generada.
                    Si None, solo devuelve la PSF.
                
            Returns:
                Si img es None: PSF 2D normalizada (float64)
                Si img es proporcionada: Imagen convolucionada con la PSF
        """
        if img is None:
            return self.psf.copy()
        
        self._validar_imagen(img)
        
        # Aplicar convolución con la PSF generada
        convolucionador = KernelPersonalizado(
            kernel=self.psf,
            normalizar=False,  # Ya está normalizada
            modo='fft',  # PSF suele ser grande
            padding='reflect'
        )
        
        return convolucionador(img)
    
    def get_parametros(self) -> dict:
        """Devuelve diccionario con parámetros de la PSF generada."""
        return {
            'tipo': self.tipo,
            'na': self.na,
            'lambda_nm': self.lambda_nm,
            'n_medio': self.n_medio,
            'resolucion_pixel_nm': self.resolucion_pixel_nm,
            'tamanio_psf': self.tamanio_psf,
            'fwhm_teorico_nm': 0.61 * self.lambda_nm / self.na * 1000 if self.tipo == 'airy' else None,
            **self.kwargs
        }


class KernelSeparable(Convolucionador):
    """
        Optimización de convolución para kernels separables.
        
        Un kernel h(x,y) es separable si puede expresarse como producto
        de dos kernels 1D: h(x,y) = h_x(x) · h_y(y).
        
        Esto reduce la complejidad de O(N·M·K²) a O(N·M·K) para kernel K×K.
        
        Algoritmo:
            1. Verificar/descomponer kernel en componentes 1D
            2. Convolución horizontal: temp = img * h_x
            3. Convolución vertical: resultado = temp * h_y
        
        Ecuación:
            (f * h)(x,y) = ∑_j [∑_i f(x-i,y-j)·h_x(i)] · h_y(j)
                        = ∑_j temp(x,y-j) · h_y(j)
        
        Kernels comúnmente separables:
            - Gaussiano: G(x,y) = G(x)·G(y)
            - Sobel: Sobel_x = [1,0,-1] ⊗ [1,2,1]
            - Caja (box): uniforme en ambas direcciones
            - Derivada de Gaussiana: dG/dx ⊗ G(y)
        
        Ventajas:
            - Velocidad: O(n) vs O(n²) para kernel n×n
            - Memoria: Menor uso de caché, mejor localidad
            - Escalable: Eficiente para kernels grandes (σ > 3)
            - Exactitud: Mismo resultado que convolución 2D directa
        
        Desventajas:
            - Solo aplica a kernels separables
            - Verificación de separabilidad tiene costo
            - Acumulación de error de redondeo en dos pasadas
            - Más complejo de implementar correctamente
        
        Usos típicos en microscopía:
            - Suavizado gaussiano rápido con grandes σ (simulación de PSF)
            - Filtros de derivada para análisis de orientación
            - Procesamiento de imágenes grandes (tiles) en tiempo real
            - Convoluciones iterativas (deconvolución, restauración)
            - Filtrado anisotrópico (diferente σ en x e y)
    """
    nombre = "kernel_separable"
    
    def __init__(self,
                kernel_x: np.ndarray,
                kernel_y: Optional[np.ndarray] = None,
                normalizar: bool = True):
        """
            Args:
                kernel_x: Kernel 1D para dirección X (o kernel 2D a descomponer)
                
                kernel_y: Kernel 1D para dirección Y. 
                        Si None, se asume kernel_x == kernel_y (simetría)
                        Si kernel_x es 2D, se intenta descomponer
                
                normalizar: Normalizar kernels para conservar intensidad
        """
        # Si se pasa kernel 2D, intentar descomponer
        if kernel_x.ndim == 2:
            self.kernel_x, self.kernel_y = self._descomponer_kernel(kernel_x)
            if self.kernel_x is None:
                raise ValueError("El kernel 2D proporcionado no es separable")
        else:
            self.kernel_x = kernel_x.astype(np.float64).flatten()
            self.kernel_y = (kernel_y.astype(np.float64).flatten() 
            if kernel_y is not None 
            else self.kernel_x.copy())
        
        if normalizar:
            sum_x, sum_y = self.kernel_x.sum(), self.kernel_y.sum()
            if abs(sum_x) > 1e-10 and abs(sum_y) > 1e-10:
                # Normalizar para que el producto sume 1
                norm = np.sqrt(abs(sum_x * sum_y))
                self.kernel_x = self.kernel_x / sum_x * norm
                self.kernel_y = self.kernel_y / sum_y * norm
    
    def _descomponer_kernel(self, kernel_2d: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
            Intenta descomponer kernel 2D en producto de dos 1D usando SVD.
            
            Returns:
                (kernel_x, kernel_y) o (None, None) si no es separable
        """
        # SVD: kernel = U · S · V^T
        # Si es rank-1, kernel = (u1·s1) ⊗ v1
        U, S, Vt = np.linalg.svd(kernel_2d)
        
        # Verificar si es efectivamente rank-1
        if S[0] < 1e-10 or S[1] > 1e-10 * S[0]:
            return None, None
        
        kernel_x = U[:, 0] * np.sqrt(S[0])
        kernel_y = Vt[0, :] * np.sqrt(S[0])
        
        return kernel_x, kernel_y
    
    def __call__(self, img: np.ndarray) -> np.ndarray:
        """
            Aplica convolución separable.
            
            Args:
                img: Imagen 2D
                
            Returns:
                Imagen convolucionada del mismo tipo
        """
        self._validar_imagen(img)
        
        tipo_original = img.dtype
        img_float = img.astype(np.float64)
        
        # Convolución horizontal (a lo largo de columnas, eje x)
        temp = convolve(img_float, self.kernel_x[np.newaxis, :], mode='reflect')
        
        # Convolución vertical (a lo largo de filas, eje y)
        resultado = convolve(temp, self.kernel_y[:, np.newaxis], mode='reflect')
        
        return self._normalizar_resultado(resultado, tipo_original)
    
    @staticmethod
    def gaussiano_1d(sigma: float, tamanio: Optional[int] = None) -> np.ndarray:
        """
            Genera kernel gaussiano 1D.
            
            Args:
                sigma: Desviación estándar en píxeles
                tamanio: Longitud del kernel (impar). Si None, calcula automático (6σ+1)
        """
        if tamanio is None:
            tamanio = int(6 * sigma) | 1  # Asegurar impar
        
        x = np.arange(tamanio) - tamanio // 2
        kernel = np.exp(-x**2 / (2 * sigma**2))
        return kernel / kernel.sum()
    
    @staticmethod
    def derivada_gaussiana(sigma: float, orden: int = 1) -> np.ndarray:
        """
            Genera derivada de gaussiana 1D.
            
            Args:
                sigma: Desviación estándar
                orden: 1 para primera derivada, 2 para segunda
        """
        tamanio = int(6 * sigma) | 1
        x = np.arange(tamanio) - tamanio // 2
        
        if orden == 1:
            # dG/dx = -x/sigma^2 · G(x)
            g = np.exp(-x**2 / (2 * sigma**2))
            kernel = -x / sigma**2 * g
        else:
            # d²G/dx² = (x²/sigma^4 - 1/sigma²) · G(x)
            g = np.exp(-x**2 / (2 * sigma**2))
            kernel = (x**2 / sigma**4 - 1 / sigma**2) * g
        
        return kernel


class ConvolucionFrecuencia(Convolucionador):
    """
        Implementación de convolución mediante Transformada Rápida de Fourier.
        
        Para kernels grandes (> 7×7 típicamente), la convolución en frecuencia
        es más eficiente: O(N log N) vs O(N·K²) para imagen N y kernel K.
        
        Algoritmo:
            1. Pad imagen y kernel al mismo tamaño (próximo 2^n para FFT eficiente)
            2. FFT de imagen: F = FFT(f)
            3. FFT de kernel: H = FFT(h) [kernel centrado]
            4. Multiplicación: G = F · H
            5. IFFT: g = IFFT(G)
            6. Crop al tamaño original y corrección de fase
        
        Ecuación:
            g = IFFT( FFT(f, padded) · FFT(h, padded) )
        
        Consideraciones de padding:
            - 'same': Salida mismo tamaño que entrada (circular por defecto)
            - 'valid': Solo donde el kernel cabe completamente
            - 'full': Tamaño imagen + kernel - 1 (convolución completa)
        
        Ventajas:
            - Eficiencia: Mucho más rápido para kernels grandes
            - Complejidad: O(n log n) independiente del tamaño del kernel
            - Precisión: Sin errores de acumulación de convolución directa
            - Regularidad: Bueno para implementación paralela (GPU)
        
        Desventajas:
            - Overhead: Para kernels pequeños, más lento que directo
            - Memoria: Requiere arrays potencia de 2 (padding)
            - Artefactos: Efectos de borde circular si no se paddea correctamente
            - Precisión numérica: Errores de redondeo en FFT para valores grandes
        
        Usos típicos en microscopía:
            - Convolución con PSF experimental grande (ej. 64×64 de beads)
            - Simulación de adquisición con aberraciones complejas
            - Deconvolución frecuencial (Wiener, Tikhonov)
            - Filtrado de imágenes muy grandes (tiling eficiente)
            - Convoluciones iterativas (métodos de optimización)
    """
    nombre = "convolucion_frecuencia"
    
    def __init__(self,
                kernel: np.ndarray,
                modo: Literal['same', 'valid', 'full'] = 'same',
                padding_fft: str = 'reflect'):
        """
            Args:
                kernel: Kernel de convolución 2D
                
                modo: Tamaño de salida
                    'same': Mismo que imagen de entrada (default)
                    'valid': Sin efectos de borde (más pequeña)
                    'full': Convolución completa (imagen + kernel - 1)
                
                padding_fft: Cómo paddear para evitar efectos circulares
                            'reflect': Refleja la imagen (mejor para imágenes naturales)
                            'constant': Ceros (circular puro)
                            'edge': Repite borde
        """
        self.kernel = kernel.astype(np.float64)
        self.modo = modo
        self.padding_fft = padding_fft
    
    def __call__(self, img: np.ndarray) -> np.ndarray:
        """
            Aplica convolución mediante FFT.
            
            Args:
                img: Imagen 2D
                
            Returns:
                Imagen convolucionada (float64)
        """
        self._validar_imagen(img)
        
        from scipy.fft import fft2, ifft2, fftshift, ifftshift
        
        img_float = img.astype(np.float64)
        
        # Determinar tamaño de padding óptimo (próxima potencia de 2)
        if self.modo == 'full':
            shape_out = (img.shape[0] + self.kernel.shape[0] - 1,
                        img.shape[1] + self.kernel.shape[1] - 1)
        elif self.modo == 'valid':
            shape_out = (img.shape[0] - self.kernel.shape[0] + 1,
                        img.shape[1] - self.kernel.shape[1] + 1)
        else:  # 'same'
            shape_out = img.shape
        
        # Tamaño FFT (próxima potencia de 2 para eficiencia)
        shape_fft = (2 ** np.ceil(np.log2(shape_out[0] + self.kernel.shape[0])).astype(int),
                    2 ** np.ceil(np.log2(shape_out[1] + self.kernel.shape[1])).astype(int))
        
        # Pad imagen según modo
        if self.padding_fft == 'reflect':
            img_padded = np.pad(img_float, 
                                ((0, shape_fft[0] - img.shape[0]),
                                (0, shape_fft[1] - img.shape[1])),
                                mode='reflect')
        else:
            img_padded = np.pad(img_float,
                                ((0, shape_fft[0] - img.shape[0]),
                                (0, shape_fft[1] - img.shape[1])),
                                mode='constant')
        
        # Pad kernel (centrado)
        kernel_padded = np.zeros(shape_fft)
        y_start = (shape_fft[0] - self.kernel.shape[0]) // 2
        x_start = (shape_fft[1] - self.kernel.shape[1]) // 2
        kernel_padded[y_start:y_start + self.kernel.shape[0],
                    x_start:x_start + self.kernel.shape[1]] = self.kernel
        
        # FFT
        img_fft = fft2(img_padded)
        kernel_fft = fft2(ifftshift(kernel_padded))  # Centrar kernel en origen
        
        # Multiplicación en frecuencia
        resultado_fft = img_fft * kernel_fft
        
        # IFFT
        resultado = np.real(ifft2(resultado_fft))
        
        # Extraer región de interés según modo
        if self.modo == 'same':
            y_start = (resultado.shape[0] - img.shape[0]) // 2
            x_start = (resultado.shape[1] - img.shape[1]) // 2
            resultado = resultado[y_start:y_start + img.shape[0],
                                x_start:x_start + img.shape[1]]
        elif self.modo == 'valid':
            pad_y = self.kernel.shape[0] - 1
            pad_x = self.kernel.shape[1] - 1
            resultado = resultado[pad_y//2:pad_y//2 + shape_out[0],
                                pad_x//2:pad_x//2 + shape_out[1]]
        
        return resultado


class CorreccionBordes(Convolucionador):
    """
        Convolución con extrapolación avanzada de bordes.
        
        Extiende la imagen más allá de sus límites usando métodos de
        extrapolation que preservan propiedades de la imagen (gradiente,
        curvatura) para reducir artefactos en los bordes de la convolución.
        
        Métodos de extrapolación:
            - 'reflect': Reflejo especular (default, bueno para bordes naturales)
            - 'constant': Valor constante (cero o especificado)
            - 'nearest': Repetición del último valor (continuidad)
            - 'linear': Extrapolación lineal del gradiente local
            - 'cubic': Extrapolación cúbica (suave, preserva curvatura)
            - 'periodic': Imagen periódica (para FFT, imágenes de fase)
        
        Algoritmo:
            1. Calcular región de borde necesaria (radio del kernel)
            2. Extrapolar imagen según método seleccionado
            3. Aplicar convolución estándar en imagen extendida
            4. Recortar al tamaño original
        
        Ventajas:
            - Reduce artefactos de borde vs zero-padding
            - Preserva continuidad de bordes en imágenes naturales
            - Adaptativo: diferentes métodos para diferentes tipos de imagen
            - Esencial para análisis cuantitativo cerca de bordes
        
        Desventajas:
            - Costo computacional de extrapolación
            - Métodos avanzados (cúbico) pueden oscilar
            - No hay información real más allá del borde (adivinanza)
            - Puede introducir sesgos sistemáticos en estadísticas de borde
        
        Usos típicos en microscopía:
            - Análisis de células cerca de los límites del campo de visión
            - Convolución de tiles individuales en procesamiento por bloques
            - Filtrado de imágenes de fase (requieren periodicidad)
            - Corrección de shading en bordes de imagen
            - Convolución con PSF sin pérdida de información en bordes
    """
    nombre = "correccion_bordes"
    
    def __init__(self,
                kernel: np.ndarray,
                modo_borde: Literal['reflect', 'constant', 'nearest', 
                                    'linear', 'cubic', 'periodic'] = 'reflect',
                valor_constante: float = 0.0):
        """
            Args:
                kernel: Kernel de convolución
                
                modo_borde: Método de extrapolación
                        'reflect': Reflejo especular (aaa|abc|cba)
                        'constant': Valor fijo (000|abc|000)
                        'nearest': Último valor (aaa|abc|ccc)
                        'linear': Extrapolación lineal del gradiente
                        'cubic': Splines cúbicos (suave)
                        'periodic': aaa|abc|abc (para imágenes de fase)
                
                valor_constante: Valor para padding si modo='constant'
        """
        self.kernel = kernel.astype(np.float64)
        self.modo_borde = modo_borde
        self.valor_constante = valor_constante
        
        # Radio del kernel
        self.radio_y = kernel.shape[0] // 2
        self.radio_x = kernel.shape[1] // 2
    
    def __call__(self, img: np.ndarray) -> np.ndarray:
        """
            Aplica convolución con extrapolación de bordes.
            
            Args:
                img: Imagen 2D
                
            Returns:
                Imagen convolucionada del mismo tipo
        """
        self._validar_imagen(img)
        
        tipo_original = img.dtype
        img_float = img.astype(np.float64)
        
        # Mapear modos personalizados a scipy/numpy
        if self.modo_borde == 'linear':
            img_pad = self._extrapolar_lineal(img_float)
            modo_conv = 'constant'  # Ya paddeado
        elif self.modo_borde == 'cubic':
            img_pad = self._extrapolar_cubica(img_float)
            modo_conv = 'constant'
        elif self.modo_borde == 'periodic':
            modo_conv = 'wrap'
            img_pad = img_float
        else:
            modo_conv = self.modo_borde
            img_pad = img_float
        
        # Aplicar padding si no se hizo en métodos avanzados
        if self.modo_borde not in ['linear', 'cubic']:
            pad_width = ((self.radio_y, self.radio_y), (self.radio_x, self.radio_x))
            if modo_conv == 'constant':
                img_pad = np.pad(img_pad, pad_width, mode=modo_conv, 
                                constant_values=self.valor_constante)
            else:
                img_pad = np.pad(img_pad, pad_width, mode=modo_conv)
        
        # Convolución
        resultado = convolve2d(img_pad, self.kernel, mode='valid', boundary='fill')
        
        return self._normalizar_resultado(resultado, tipo_original)
    
    def _extrapolar_lineal(self, img: np.ndarray) -> np.ndarray:
        """Extrapolación lineal basada en gradiente local en los bordes."""
        h, w = img.shape
        pad_y, pad_x = self.radio_y, self.radio_x
        
        # Crear imagen extendida
        resultado = np.zeros((h + 2*pad_y, w + 2*pad_x))
        resultado[pad_y:pad_y+h, pad_x:pad_x+w] = img
        
        # Calcular gradientes en bordes
        grad_top = img[1, :] - img[0, :]  # Fila 1 - fila 0
        grad_bottom = img[-1, :] - img[-2, :]
        grad_left = img[:, 1] - img[:, 0]
        grad_right = img[:, -1] - img[:, -2]
        
        # Extrapolar bordes superior e inferior
        for i in range(pad_y):
            factor = pad_y - i
            resultado[i, pad_x:pad_x+w] = img[0, :] - grad_top * factor
            resultado[pad_y+h+i, pad_x:pad_x+w] = img[-1, :] + grad_bottom * (i+1)
        
        # Extrapolar lados
        for i in range(pad_x):
            factor = pad_x - i
            resultado[:, i] = resultado[:, pad_x] - np.interp(
                np.arange(resultado.shape[0]),
                [0, resultado.shape[0]-1],
                [grad_left[0], grad_left[-1]]
            ) * factor
            
            resultado[:, pad_x+w+i] = resultado[:, pad_x+w-1] + np.interp(
                np.arange(resultado.shape[0]),
                [0, resultado.shape[0]-1],
                [grad_right[0], grad_right[-1]]
            ) * (i+1)
        
        return resultado
    
    def _extrapolar_cubica(self, img: np.ndarray) -> np.ndarray:
        """Extrapolación cúbica usando splines."""
        from scipy.interpolate import CubicSpline
        
        h, w = img.shape
        pad_y, pad_x = self.radio_y, self.radio_x
        
        resultado = np.zeros((h + 2*pad_y, w + 2*pad_x))
        resultado[pad_y:pad_y+h, pad_x:pad_x+w] = img
        
        # Interpolar a lo largo de filas (horizontal)
        x = np.arange(w)
        x_pad_left = np.arange(-pad_x, 0)
        x_pad_right = np.arange(w, w + pad_x)
        
        for y in range(h):
            cs = CubicSpline(x, img[y, :], bc_type='natural')
            resultado[pad_y+y, :pad_x] = cs(x_pad_left)
            resultado[pad_y+y, pad_x+w:] = cs(x_pad_right)
        
        # Interpolar a lo largo de columnas (vertical) en regiones paddeadas
        y = np.arange(h)
        y_pad_top = np.arange(-pad_y, 0)
        y_pad_bottom = np.arange(h, h + pad_y)
        
        for x in range(resultado.shape[1]):
            cs = CubicSpline(y, resultado[pad_y:pad_y+h, x], bc_type='natural')
            resultado[:pad_y, x] = cs(y_pad_top)
            resultado[pad_y+h:, x] = cs(y_pad_bottom)
        
        return resultado