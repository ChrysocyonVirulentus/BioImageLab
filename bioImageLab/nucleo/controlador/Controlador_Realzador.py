# === realzador/Controlador_Realzador.py ===
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, replace
from typing import Union, Callable, Optional, Tuple, Dict, Any
from enum import Enum, auto
from pathlib import Path

# Imports de tu sistema
from .Resultado_Either import Resultado, Err, Ok
from .Controlador_BioImagen import BioImagenData, ErrorBioImagen, Dimensiones
from ..gestorLab.Categoria_Operacion import CategoriaOperacion
from ..gestorLab.Operacion import Operacion, TipoSalida

# Imports de métodos específicos (7 submódulos)
from ..realzador.contraste.Realzadores_Constraste import (
    CLAHE,
    Gamma,
    Logaritmico,
    Retinex,
    EcualizacionHistograma
)
from ..realzador.convolucion.Realzadores_Convolucion import (
    KernelPersonalizado,
    PSFSimulacion,
    KernelSeparable,
    ConvolucionFrecuencia,
    CorreccionBordes
)
from ..realzador.deconvolucion.Realzadores_Deconvolucion import (
    Wiener,
    RichardsonLucy,
    BlindDeconvolucion,
    Tikhonov
)
from ..realzador.morfologicos.Realzadores_Morfologicos import (
    Apertura,
    Cierre,
    TopHat,
    BottomHat,
    GradienteMorfologico,
    ReconstruccionMorfologica
)
from ..realzador.afilacion.Realzadores_Afilacion import (
    AfilacionLaplaciana,
    FiltroHighBoost,
    MascaraEnfoque,
    AfilacionGradiente,
    AfilacionWavelet
)
from ..realzador.estructura.Realzadores_Estructurales import (
    Hessiano,
    Frangi,
    Sato,
    TensorEstructural
)
from ..realzador.gradientes.Realzadores_Gradientes import (
    Laplaciano,
    LaplacianoCero,
    Canny,
    Sobel,
    Scharr,
    Prewitt,
    Roberts
)


# ==================== TIPOS ALGEBRAICOS DE REALCE ====================

@dataclass(frozen=True)
class Realce_Global:
    """Aplica el mismo realce a todo el volumen."""
    pass

@dataclass(frozen=True)
class Realce_PorCorteZ:
    """Aplica realce independientemente a cada plano Z."""
    pass

@dataclass(frozen=True)
class Realce_PorTimepoint:
    """Aplica realce independientemente a cada timepoint T."""
    pass

@dataclass(frozen=True)
class Realce_PorCorteEspaciotemporal:
    """Aplica realce a cada corte (t, z) de forma independiente."""
    pass

TipoRealce = Union[
    Realce_Global,
    Realce_PorCorteZ,
    Realce_PorTimepoint,
    Realce_PorCorteEspaciotemporal
]


# ==================== TIPOS DE REALCE POR SUBMÓDULO ====================

class TipoRealceSubmodulo(Enum):
    CONTRASTE = "contraste"           # CLAHE, Gamma, etc.
    CONVOLUCION = "convolucion"       # Kernels, PSF
    DECONVOLUCION = "deconvolucion"   # Wiener, Richardson-Lucy
    MORFOLOGICO = "morfologico"       # Apertura, Cierre, Top-Hat
    AFILACION = "afilacion"           # High-boost, máscaras enfoque
    ESTRUCTURAL = "estructural"       # Frangi, Sato (vesselness)
    GRADIENTE = "gradiente"            # Canny, Sobel, etc.

# Union de todos los métodos de realce disponibles
MetodoRealce = Union[
    # Contraste
    CLAHE, Gamma, Logaritmico, Retinex, EcualizacionHistograma,
    # Convolución
    KernelPersonalizado, PSFSimulacion, KernelSeparable, ConvolucionFrecuencia, CorreccionBordes,
    # Deconvolución
    Wiener, RichardsonLucy, BlindDeconvolucion, Tikhonov,
    # Morfológicos
    Apertura, Cierre, TopHat, BottomHat, GradienteMorfologico, ReconstruccionMorfologica,
    # Afilación
    AfilacionLaplaciana, FiltroHighBoost, MascaraEnfoque, AfilacionGradiente, AfilacionWavelet,
    # Estructurales (vesselness)
    Hessiano, Frangi, Sato, TensorEstructural,
    # Gradientes
    Laplaciano, LaplacianoCero, Canny, Sobel, Scharr, Prewitt, Roberts
]


# ==================== FUNCIONES PURAS ====================

def crear_realce(
    metodo: MetodoRealce,
    tipo: TipoRealce = Realce_Global(),
) -> Callable[[BioImagenData], Resultado[BioImagenData, ErrorBioImagen]]:
    """
    Factory curried que retorna función pura de realce.
    
    Args:
        metodo: Instancia del realzador específico (ej: CLAHE(clip_limit=2.0))
        tipo: Estrategia de aplicación espacio-temporal
        canal: Canal objetivo
    
    Returns:
        Callable para usar con .bind() en pipelines
    """
    def _aplicar_realce(data: BioImagenData, canal_idx: 0) -> Resultado[BioImagenData, ErrorBioImagen]:
        
        # Validación de canal
        if not (0 <= canal_idx < data.dims.C):
            return Err(ErrorBioImagen(
                etapa="realce",
                mensaje=f"Canal {canal_idx} fuera de rango [0, {data.dims.C-1}]",
                ruta=data.ruta_origen
            ))
        
        try:
            T, Z, C, Y, X = data.dims.shape
            
            # Extraer canal objetivo [T, Z, Y, X]
            canal_data = data.datos[:, :, canal_idx, :, :]
            
            # Array resultado manteniendo estructura 5D
            resultado_canal = np.zeros((T, Z, 1, Y, X), dtype=np.float64)
            
            # Aplicar según estrategia
            match tipo:
                case Realce_Global():
                    for t in range(T):
                        for z in range(Z):
                            resultado_canal[t, z, 0, :, :] = metodo(canal_data[t, z, :, :])
                            
                case Realce_PorCorteZ():
                    for z in range(Z):
                        for t in range(T):
                            resultado_canal[t, z, 0, :, :] = metodo(canal_data[t, z, :, :])
                            
                case Realce_PorTimepoint():
                    for t in range(T):
                        for z in range(Z):
                            resultado_canal[t, z, 0, :, :] = metodo(canal_data[t, z, :, :])
                            
                case Realce_PorCorteEspaciotemporal():
                    for t in range(T):
                        for z in range(Z):
                            resultado_canal[t, z, 0, :, :] = metodo(canal_data[t, z, :, :])
            
            # Reconstruir BioImagenData
            nuevos_datos = data.datos.copy().astype(np.float64)
            nuevos_datos[:, :, canal_idx, :, :] = resultado_canal[:, :, 0, :, :]
            
            return Ok(replace(data, datos=nuevos_datos))
            
        except Exception as e:
            nombre_metodo = getattr(metodo, 'nombre', metodo.__class__.__name__)
            return Err(ErrorBioImagen(
                etapa="realce",
                mensaje=f"Fallo en {nombre_metodo}: {str(e)}",
                ruta=data.ruta_origen,
                causa=e
            ))
    
    return _aplicar_realce


def crear_realce_multicanal(
    metodo: MetodoRealce,
    tipo: TipoRealce = Realce_Global()
) -> Callable[[BioImagenData], Resultado[BioImagenData, ErrorBioImagen]]:
    """Versión que aplica el mismo realce a todos los canales."""
    def _aplicar_multicanal(data: BioImagenData) -> Resultado[BioImagenData, ErrorBioImagen]:
        resultado: Resultado[BioImagenData, ErrorBioImagen] = Ok(data)
        
        for c in range(data.dims.C):
            realce_canal = crear_realce(metodo, tipo, canal=c)
            resultado = resultado.bind(realce_canal)
            if resultado.es_err():
                break
        
        return resultado
    
    return _aplicar_multicanal


# ==================== WRAPPER ORIENTADO A OBJETOS ====================

class Controlador_Realzador:
    """
    Wrapper stateful para realce de imágenes.
    Permite crear realzadores configurados y aplicarlos.
    """
    
    def __init__(self):
        self._cache: Optional[Dict[str, Any]] = None
        self._ultimo_metodo: Optional[MetodoRealce] = None
    
    # ===== CONTRASTE =====
    
    def crear_clahe(
        self,
        clip_limit: float = 2.0,
        grid_size: Tuple[int, int] = (8, 8),
        tipo_aplicacion: TipoRealce = Realce_PorCorteEspaciotemporal()
    ) -> Callable[[BioImagenData], Resultado[BioImagenData, ErrorBioImagen]]:
        """CLAHE: Contraste adaptativo limitado por histograma."""
        metodo = CLAHE(clip_limit=clip_limit, grid_size=grid_size)
        self._ultimo_metodo = metodo
        return crear_realce(metodo, tipo_aplicacion)
    
    def crear_gamma(
        self,
        gamma: float = 1.0,
        tipo_aplicacion: TipoRealce = Realce_Global()
    ) -> Callable[[BioImagenData], Resultado[BioImagenData, ErrorBioImagen]]:
        """Corrección gamma (ajuste no lineal de intensidad)."""
        metodo = Gamma(gamma=gamma)
        self._ultimo_metodo = metodo
        return crear_realce(metodo, tipo_aplicacion)
    
    def crear_logaritmico(
        self,
        escala: float = 1.0,
        tipo_aplicacion: TipoRealce = Realce_Global()
    ) -> Callable[[BioImagenData], Resultado[BioImagenData, ErrorBioImagen]]:
        """Transformación logarítmica (comprime rango dinámico)."""
        metodo = Logaritmico(escala=escala)
        self._ultimo_metodo = metodo
        return crear_realce(metodo, tipo_aplicacion)
    
    def crear_retinex(
        self,
        sigma: float = 15.0,
        tipo_retinex: str = "msrcr",
        tipo_aplicacion: TipoRealce = Realce_PorCorteEspaciotemporal()
    ) -> Callable[[BioImagenData], Resultado[BioImagenData, ErrorBioImagen]]:
        """Retinex (mejora contraste simulando percepción humana)."""
        metodo = Retinex(sigma=sigma, tipo=tipo_retinex)
        self._ultimo_metodo = metodo
        return crear_realce(metodo, tipo_aplicacion)
    
    def crear_ecuacion_histograma(
        self,
        metodo_ecuacion: str = "global",
        tipo_aplicacion: TipoRealce = Realce_Global()
    ) -> Callable[[BioImagenData], Resultado[BioImagenData, ErrorBioImagen]]:
        """Ecualización de histograma (global o adaptativa)."""
        metodo = EcuacionHistograma(metodo=metodo_ecuacion)
        self._ultimo_metodo = metodo
        return crear_realce(metodo, tipo_aplicacion)
    
    # ===== CONVOLUCIÓN =====
    
    def crear_kernel_personalizado(
        self,
        kernel: np.ndarray,
        tipo_aplicacion: TipoRealce = Realce_PorCorteEspaciotemporal()
    ) -> Callable[[BioImagenData], Resultado[BioImagenData, ErrorBioImagen]]:
        """Convolución con kernel personalizado."""
        metodo = KernelPersonalizado(kernel=kernel)
        self._ultimo_metodo = metodo
        return crear_realce(metodo, tipo_aplicacion)
    
    def crear_psf_simulacion(
        self,
        tipo_microscopio: str = "confocal",
        na: float = 1.4,
        lambda_nm: float = 550.0,
        tipo_aplicacion: TipoRealce = Realce_Global()
    ) -> Callable[[BioImagenData], Resultado[BioImagenData, ErrorBioImagen]]:
        """Simulación de PSF (Point Spread Function) para convolución."""
        metodo = PSFSimulacion(
            tipo_microscopio=tipo_microscopio,
            na=na,
            lambda_nm=lambda_nm
        )
        self._ultimo_metodo = metodo
        return crear_realce(metodo, tipo_aplicacion)
    
    def crear_kernel_separable(
        self,
        kernel_x: np.ndarray,
        kernel_y: np.ndarray,
        tipo_aplicacion: TipoRealce = Realce_PorCorteEspaciotemporal()
    ) -> Callable[[BioImagenData], Resultado[BioImagenData, ErrorBioImagen]]:
        """Convolución separable (optimización para kernels 2D separables)."""
        metodo = KernelSeparable(kernel_x=kernel_x, kernel_y=kernel_y)
        self._ultimo_metodo = metodo
        return crear_realce(metodo, tipo_aplicacion)
    
    def crear_convolucion_frecuencia(
        self,
        kernel: np.ndarray,
        tipo_aplicacion: TipoRealce = Realce_Global()
    ) -> Callable[[BioImagenData], Resultado[BioImagenData, ErrorBioImagen]]:
        """Convolución vía FFT (más rápida para kernels grandes)."""
        metodo = ConvolucionFrecuencia(kernel=kernel)
        self._ultimo_metodo = metodo
        return crear_realce(metodo, tipo_aplicacion)
    
    def crear_correccion_bordes(
        self,
        metodo: str = "espejo",
        tipo_aplicacion: TipoRealce = Realce_PorCorteEspaciotemporal()
    ) -> Callable[[BioImagenData], Resultado[BioImagenData, ErrorBioImagen]]:
        """Aplica corrección de bordes antes de convolución."""
        metodo = CorreccionBordes(metodo=metodo)
        self._ultimo_metodo = metodo
        return crear_realce(metodo, tipo_aplicacion)
    
    # ===== DECONVOLUCIÓN =====
    
    def crear_wiener(
        self,
        psf: np.ndarray,
        k: float = 0.01,
        balance: float = 0.5,
        iteraciones: int = 1,
        tipo_aplicacion: TipoRealce = Realce_Global()
    ) -> Callable[[BioImagenData], Resultado[BioImagenData, ErrorBioImagen]]:
        """Deconvolución Wiener (restauración con conocimiento de PSF)."""
        metodo = Wiener(psf=psf, k=k, balance=balance, iteraciones=iteraciones)
        self._ultimo_metodo = metodo
        return crear_realce(metodo, tipo_aplicacion)
    
    def crear_richardson_lucy(
        self,
        psf: np.ndarray,
        iteraciones: int = 10,
        tipo_aplicacion: TipoRealce = Realce_Global()
    ) -> Callable[[BioImagenData], Resultado[BioImagenData, ErrorBioImagen]]:
        """Deconvolución Richardson-Lucy (máxima verosimilitud)."""
        metodo = RichardsonLucy(psf=psf, iteraciones=iteraciones)
        self._ultimo_metodo = metodo
        return crear_realce(metodo, tipo_aplicacion)
    
    def crear_blind_deconvolucion(
        self,
        tamano_psf_estimado: Tuple[int, int] = (15, 15),
        iteraciones: int = 10,
        tipo_aplicacion: TipoRealce = Realce_Global()
    ) -> Callable[[BioImagenData], Resultado[BioImagenData, ErrorBioImagen]]:
        """Deconvolución ciega (estima PSF y restaura simultáneamente)."""
        metodo = BlindDeconvolucion(
            tamano_psf_estimado=tamano_psf_estimado,
            iteraciones=iteraciones
        )
        self._ultimo_metodo = metodo
        return crear_realce(metodo, tipo_aplicacion)
    
    def crear_tikhonov(
        self,
        lambda_reg: float = 0.01,
        orden: int = 1,
        tipo_aplicacion: TipoRealce = Realce_Global()
    ) -> Callable[[BioImagenData], Resultado[BioImagenData, ErrorBioImagen]]:
        """Regularización de Tikhonov (suavizada)."""
        metodo = Tikhonov(lambda_reg=lambda_reg, orden=orden)
        self._ultimo_metodo = metodo
        return crear_realce(metodo, tipo_aplicacion)
    
    # ===== MORFOLÓGICOS =====
    
    def crear_apertura(
        self,
        elemento_estructurante: np.ndarray = None,
        iteraciones: int = 1,
        tipo_aplicacion: TipoRealce = Realce_PorCorteEspaciotemporal()
    ) -> Callable[[BioImagenData], Resultado[BioImagenData, ErrorBioImagen]]:
        """Apertura: erosión seguida de dilatación (elimina ruido pequeño)."""
        if elemento_estructurante is None:
            elemento_estructurante = np.ones((3, 3))
        metodo = Apertura(
            elemento_estructurante=elemento_estructurante,
            iteraciones=iteraciones
        )
        self._ultimo_metodo = metodo
        return crear_realce(metodo, tipo_aplicacion)
    
    def crear_cierre(
        self,
        elemento_estructurante: np.ndarray = None,
        iteraciones: int = 1,
        tipo_aplicacion: TipoRealce = Realce_PorCorteEspaciotemporal()
    ) -> Callable[[BioImagenData], Resultado[BioImagenData, ErrorBioImagen]]:
        """Cierre: dilatación seguida de erosión (cierra agujeros pequeños)."""
        if elemento_estructurante is None:
            elemento_estructurante = np.ones((3, 3))
        metodo = Cierre(
            elemento_estructurante=elemento_estructurante,
            iteraciones=iteraciones
        )
        self._ultimo_metodo = metodo
        return crear_realce(metodo, tipo_aplicacion)
    
    def crear_top_hat(
        self,
        elemento_estructurante: np.ndarray = None,
        tipo_aplicacion: TipoRealce = Realce_PorCorteEspaciotemporal()
    ) -> Callable[[BioImagenData], Resultado[BioImagenData, ErrorBioImagen]]:
        """Top-Hat: imagen - apertura (detecta picos brillantes pequeños)."""
        if elemento_estructurante is None:
            elemento_estructurante = np.ones((15, 15))
        metodo = TopHat(elemento_estructurante=elemento_estructurante)
        self._ultimo_metodo = metodo
        return crear_realce(metodo, tipo_aplicacion)
    
    def crear_bottom_hat(
        self,
        elemento_estructurante: np.ndarray = None,
        tipo_aplicacion: TipoRealce = Realce_PorCorteEspaciotemporal()
    ) -> Callable[[BioImagenData], Resultado[BioImagenData, ErrorBioImagen]]:
        """Bottom-Hat: cierre - imagen (detecta valles oscuros)."""
        if elemento_estructurante is None:
            elemento_estructurante = np.ones((15, 15))
        metodo = BottomHat(elemento_estructurante=elemento_estructurante)
        self._ultimo_metodo = metodo
        return crear_realce(metodo, tipo_aplicacion)
    
    def crear_gradiente_morfologico(
        self,
        elemento_estructurante: np.ndarray = None,
        tipo_aplicacion: TipoRealce = Realce_PorCorteEspaciotemporal()
    ) -> Callable[[BioImagenData], Resultado[BioImagenData, ErrorBioImagen]]:
        """Gradiente morfológico: dilatación - erosión (bordes de objetos)."""
        if elemento_estructurante is None:
            elemento_estructurante = np.ones((3, 3))
        metodo = GradienteMorfologico(elemento_estructurante=elemento_estructurante)
        self._ultimo_metodo = metodo
        return crear_realce(metodo, tipo_aplicacion)
    
    def crear_reconstruccion(
        self,
        mascara: np.ndarray = None,
        metodo: str = "dilatacion",
        iteraciones: int = -1,
        tipo_aplicacion: TipoRealce = Realce_PorCorteEspaciotemporal()
    ) -> Callable[[BioImagenData], Resultado[BioImagenData, ErrorBioImagen]]:
        """Reconstrucción morfológica (propagación condicional)."""
        metodo = Reconstruccion(
            mascara=mascara,
            metodo=metodo,
            iteraciones=iteraciones
        )
        self._ultimo_metodo = metodo
        return crear_realce(metodo, tipo_aplicacion)
    
    # ===== AFILACIÓN =====
    
    def crear_afilacion_laplaciana(
        self,
        alpha: float = 1.0,
        tipo_laplaciano: str = "discreto",
        tipo_aplicacion: TipoRealce = Realce_PorCorteEspaciotemporal()
    ) -> Callable[[BioImagenData], Resultado[BioImagenData, ErrorBioImagen]]:
        """Afilación vía Laplaciano (restaura bordes difuminados)."""
        metodo = AfilacionLaplaciana(alpha=alpha, tipo=tipo_laplaciano)
        self._ultimo_metodo = metodo
        return crear_realce(metodo, tipo_aplicacion)
    
    def crear_high_boost(
        self,
        A: float = 1.5,
        tipo_aplicacion: TipoRealce = Realce_PorCorteEspaciotemporal()
    ) -> Callable[[BioImagenData], Resultado[BioImagenData, ErrorBioImagen]]:
        """Filtro High-Boost (generalización de unsharp mask)."""
        metodo = FiltroHighBoost(A=A)
        self._ultimo_metodo = metodo
        return crear_realce(metodo, tipo_aplicacion)
    
    def crear_mascara_enfoque(
        self,
        radio_desenfoque: float = 2.0,
        cantidad: float = 1.0,
        umbral: float = 0.0,
        tipo_aplicacion: TipoRealce = Realce_PorCorteEspaciotemporal()
    ) -> Callable[[BioImagenData], Resultado[BioImagenData, ErrorBioImagen]]:
        """Máscara de enfoque (unsharp mask con umbral)."""
        metodo = MascaraEnfoque(
            radio_desenfoque=radio_desenfoque,
            cantidad=cantidad,
            umbral=umbral
        )
        self._ultimo_metodo = metodo
        return crear_realce(metodo, tipo_aplicacion)
    
    def crear_afilacion_gradiente(
        self,
        operador: str = "sobel",
        escala: float = 1.0,
        tipo_aplicacion: TipoRealce = Realce_PorCorteEspaciotemporal()
    ) -> Callable[[BioImagenData], Resultado[BioImagenData, ErrorBioImagen]]:
        """Afilación basada en gradiente (magnitud del gradiente como máscara)."""
        metodo = AfilacionGradiente(operador=operador, escala=escala)
        self._ultimo_metodo = metodo
        return crear_realce(metodo, tipo_aplicacion)
    
    def crear_afilacion_wavelet(
        self,
        wavelet: str = "db4",
        umbral: float = 0.1,
        tipo_aplicacion: TipoRealce = Realce_Global()
    ) -> Callable[[BioImagenData], Resultado[BioImagenData, ErrorBioImagen]]:
        """Afilación multiescala vía wavelets."""
        metodo = AfilacionWavelet(wavelet=wavelet, umbral=umbral)
        self._ultimo_metodo = metodo
        return crear_realce(metodo, tipo_aplicacion)
    
    # ===== ESTRUCTURALES (VESSELNESS) =====
    
    def crear_hessiano(
        self,
        sigma: float = 1.0,
        tipo_estructura: str = "blob",
        tipo_aplicacion: TipoRealce = Realce_PorCorteEspaciotemporal()
    ) -> Callable[[BioImagenData], Resultado[BioImagenData, ErrorBioImagen]]:
        """Análisis de Hessiano (detecta estructuras por curvatura)."""
        metodo = Hessiano(sigma=sigma, tipo=tipo_estructura)
        self._ultimo_metodo = metodo
        return crear_realce(metodo, tipo_aplicacion)
    
    def crear_frangi(
        self,
        sigmas: Tuple[float, ...] = (1.0, 2.0, 4.0),
        alpha: float = 0.5,
        beta: float = 0.5,
        gamma: float = 15.0,
        tipo_aplicacion: TipoRealce = Realce_PorCorteEspaciotemporal()
    ) -> Callable[[BioImagenData], Resultado[BioImagenData, ErrorBioImagen]]:
        """Vesselness de Frangi (detecta vasos/tubos oscuros sobre claro)."""
        metodo = Frangi(sigmas=sigmas, alpha=alpha, beta=beta, gamma=gamma)
        self._ultimo_metodo = metodo
        return crear_realce(metodo, tipo_aplicacion)
    
    def crear_sato(
        self,
        sigmas: Tuple[float, ...] = (1.0, 2.0, 4.0),
        tipo_aplicacion: TipoRealce = Realce_PorCorteEspaciotemporal()
    ) -> Callable[[BioImagenData], Resultado[BioImagenData, ErrorBioImagen]]:
        """Vesselness de Sato (alternativa a Frangi, más rápida)."""
        metodo = Sato(sigmas=sigmas)
        self._ultimo_metodo = metodo
        return crear_realce(metodo, tipo_aplicacion)
    
    def crear_tensor_estructural(
        self,
        sigma_gradiente: float = 1.0,
        sigma_tension: float = 2.0,
        tipo_aplicacion: TipoRealce = Realce_PorCorteEspaciotemporal()
    ) -> Callable[[BioImagenData], Resultado[BioImagenData, ErrorBioImagen]]:
        """Tensor estructural (análisis de orientación local)."""
        metodo = TensorEstructural(
            sigma_gradiente=sigma_gradiente,
            sigma_tension=sigma_tension
        )
        self._ultimo_metodo = metodo
        return crear_realce(metodo, tipo_aplicacion)
    
    # ===== GRADIENTES =====
    
    def crear_laplaciano(
        self,
        tipo: str = "discreto",
        tipo_aplicacion: TipoRealce = Realce_PorCorteEspaciotemporal()
    ) -> Callable[[BioImagenData], Resultado[BioImagenData, ErrorBioImagen]]:
        """Laplaciano (segunda derivada, detecta puntos de inflexión)."""
        metodo = Laplaciano(tipo=tipo)
        self._ultimo_metodo = metodo
        return crear_realce(metodo, tipo_aplicacion)
    
    def crear_laplaciano_cero(
        self,
        umbral: float = 0.0,
        tipo_aplicacion: TipoRealce = Realce_PorCorteEspaciotemporal()
    ) -> Callable[[BioImagenData], Resultado[BioImagenData, ErrorBioImagen]]:
        """Cruces por cero del Laplaciano (bordes a sub-pixel)."""
        metodo = LaplacianoCero(umbral=umbral)
        self._ultimo_metodo = metodo
        return crear_realce(metodo, tipo_aplicacion)
    
    def crear_canny(
        self,
        sigma: float = 1.0,
        umbral_bajo: float = 0.1,
        umbral_alto: float = 0.3,
        tipo_aplicacion: TipoRealce = Realce_PorCorteEspaciotemporal()
    ) -> Callable[[BioImagenData], Resultado[BioImagenData, ErrorBioImagen]]:
        """Detector de bordes Canny (óptimo, con supresión de no-máximos)."""
        metodo = Canny(
            sigma=sigma,
            umbral_bajo=umbral_bajo,
            umbral_alto=umbral_alto
        )
        self._ultimo_metodo = metodo
        return crear_realce(metodo, tipo_aplicacion)
    
    def crear_sobel(
        self,
        dx: int = 1,
        dy: int = 1,
        ksize: int = 3,
        tipo_aplicacion: TipoRealce = Realce_PorCorteEspaciotemporal()
    ) -> Callable[[BioImagenData], Resultado[BioImagenData, ErrorBioImagen]]:
        """Operador de Sobel (gradiente con suavizado integrado)."""
        metodo = Sobel(dx=dx, dy=dy, ksize=ksize)
        self._ultimo_metodo = metodo
        return crear_realce(metodo, tipo_aplicacion)
    
    def crear_scharr(
        self,
        dx: int = 1,
        dy: int = 1,
        tipo_aplicacion: TipoRealce = Realce_PorCorteEspaciotemporal()
    ) -> Callable[[BioImagenData], Resultado[BioImagenData, ErrorBioImagen]]:
        """Operador de Scharr (mejor rotación-invarianza que Sobel)."""
        metodo = Scharr(dx=dx, dy=dy)
        self._ultimo_metodo = metodo
        return crear_realce(metodo, tipo_aplicacion)
    
    def crear_prewitt(
        self,
        dx: int = 1,
        dy: int = 1,
        tipo_aplicacion: TipoRealce = Realce_PorCorteEspaciotemporal()
    ) -> Callable[[BioImagenData], Resultado[BioImagenData, ErrorBioImagen]]:
        """Operador de Prewitt (gradiente simple)."""
        metodo = Prewitt(dx=dx, dy=dy)
        self._ultimo_metodo = metodo
        return crear_realce(metodo, tipo_aplicacion)
    
    def crear_roberts(
        self,
        tipo_aplicacion: TipoRealce = Realce_PorCorteEspaciotemporal()
    ) -> Callable[[BioImagenData], Resultado[BioImagenData, ErrorBioImagen]]:
        """Operador de Roberts (gradiente cruzado, muy local)."""
        metodo = Roberts()
        self._ultimo_metodo = metodo
        return crear_realce(metodo, tipo_aplicacion)
    
    # ===== MÉTODOS DE APLICACIÓN DIRECTA =====
    
    def aplicar(
        self,
        data: BioImagenData,
        metodo: MetodoRealce,
        tipo: TipoRealce = Realce_Global(),
        canal: int = 0
    ) -> Resultado[BioImagenData, ErrorBioImagen]:
        """Aplica realce específico a datos (versión imperativa)."""
        realce = crear_realce(metodo, tipo, canal)
        return realce(data)
    
    def aplicar_multicanal(
        self,
        data: BioImagenData,
        metodo: MetodoRealce,
        tipo: TipoRealce = Realce_Global()
    ) -> Resultado[BioImagenData, ErrorBioImagen]:
        """Aplica realce a todos los canales."""
        realce = crear_realce_multicanal(metodo, tipo)
        return realce(data)
    
    def reset(self):
        """Limpia caché interno."""
        self._cache = None
        self._ultimo_metodo = None
    
    def __repr__(self) -> str:
        ultimo = getattr(self._ultimo_metodo, 'nombre', 'Ninguno')
        return f"<Controlador_Realzador ultimo={ultimo}>"


# ==================== FACTORIES PARA PIPELINES ====================

def operacion_realce(
    metodo: MetodoRealce,
    tipo: TipoRealce = Realce_Global(),
    canal: int = 0,
    nombre: Optional[str] = None
) -> Operacion:
    """
    Crea Operación de realce lista para PipelineBuilder.
    
    Uso:
        pipeline = (
            PipelineBuilder()
            .realzar("clahe_contraste",
                operacion_realce(CLAHE(clip_limit=3.0), Realce_PorCorteEspaciotemporal(), canal=0))
            .realzar("frangi_vasos",
                operacion_realce(Frangi(sigmas=(1,2,4)), Realce_PorCorteEspaciotemporal(), canal=0))
            .construir()
        )
    """
    nombre_op = nombre or f"realce_{metodo.__class__.__name__}_{tipo.__class__.__name__}"
    
    realce_callable = crear_realce(metodo, tipo)

    return Operacion(
        nombre=nombre_op,
        categoria=CategoriaOperacion.REALZADOR,
        instancia_callable=realce_callable,
        canal_objetivo=canal,
        parametros_originales={
            "metodo": metodo.__class__.__name__,
            "tipo": tipo.__class__.__name__,
            "submodulo": _detectar_submodulo(metodo),
            "params": _extraer_parametros_realce(metodo)
        },
        tipo_salida=TipoSalida.IMAGEN
    )


def _detectar_submodulo(metodo: MetodoRealce) -> str:
    """Detecta a qué submódulo pertenece el método."""
    clase = metodo.__class__.__name__
    mapeo = {
        # Contraste
        'CLAHE': 'contraste', 'Gamma': 'contraste', 'Logaritmico': 'contraste',
        'Retinex': 'contraste', 'EcuacionHistograma': 'contraste',
        # Convolución
        'KernelPersonalizado': 'convolucion', 'PSFSimulacion': 'convolucion',
        'KernelSeparable': 'convolucion', 'ConvolucionFrecuencia': 'convolucion',
        'CorreccionBordes': 'convolucion',
        # Deconvolución
        'Wiener': 'deconvolucion', 'RichardsonLucy': 'deconvolucion',
        'BlindDeconvolucion': 'deconvolucion', 'Tikhonov': 'deconvolucion',
        # Morfológicos
        'Apertura': 'morfologico', 'Cierre': 'morfologico', 'TopHat': 'morfologico',
        'BottomHat': 'morfologico', 'GradienteMorfologico': 'morfologico',
        'Reconstruccion': 'morfologico',
        # Afilación
        'AfilacionLaplaciana': 'afilacion', 'FiltroHighBoost': 'afilacion',
        'MascaraEnfoque': 'afilacion', 'AfilacionGradiente': 'afilacion',
        'AfilacionWavelet': 'afilacion',
        # Estructurales
        'Hessiano': 'estructural', 'Frangi': 'estructural', 'Sato': 'estructural',
        'TensorEstructural': 'estructural',
        # Gradientes
        'Laplaciano': 'gradiente', 'LaplacianoCero': 'gradiente', 'Canny': 'gradiente',
        'Sobel': 'gradiente', 'Scharr': 'gradiente', 'Prewitt': 'gradiente',
        'Roberts': 'gradiente',
    }
    return mapeo.get(clase, 'desconocido')


def _extraer_parametros_realce(metodo: MetodoRealce) -> Dict[str, Any]:
    """Extrae parámetros del método para metadata."""
    params = {}
    atributos_comunes = [
        'sigma', 'radio', 'tamano', 'iteraciones', 'clip_limit', 'gamma',
        'alpha', 'beta', 'A', 'cantidad', 'umbral', 'wavelet', 'sigmas',
        'ksize', 'dx', 'dy', 'lambda_reg', 'k', 'balance'
    ]
    for attr in atributos_comunes:
        if hasattr(metodo, attr):
            params[attr] = getattr(metodo, attr)
    return params


# Factory conveniente desde Controlador_Realzador
def operacion_realce_desde_controlador(
    controlador: Controlador_Realzador,
    metodo_factory: str,
    params: Dict[str, Any],
    canal: int = 0
) -> Resultado[Operacion, ErrorBioImagen]:
    """
    Crea operación usando el controlador (valida que el método exista).
    
    Uso:
        realzador = Controlador_Realzador()
        op_result = operacion_realce_desde_controlador(
            realzador, "crear_clahe", {"clip_limit": 3.0}, canal=0)
        if op_result.es_ok():
            builder.realzar("mi_realce", op_result.unwrap())
    """
    try:
        if not hasattr(controlador, metodo_factory):
            return Err(ErrorBioImagen(
                etapa="configuracion",
                mensaje=f"Controlador_Realzador no tiene método '{metodo_factory}'"
            ))
        
        factory = getattr(controlador, metodo_factory)
        realce_callable = factory(**params)
        
        metodo_subyacente = controlador._ultimo_metodo
        
        return Ok(Operacion(
            nombre=metodo_factory.replace("crear_", ""),
            categoria=CategoriaOperacion.REALZADOR,
            instancia_callable=metodo_subyacente,
            canal_objetivo=canal,
            parametros_originales=params,
            tipo_salida=TipoSalida.IMAGEN
        ))
        
    except Exception as e:
        return Err(ErrorBioImagen(
            etapa="configuracion",
            mensaje=f"Error creando operación desde controlador: {str(e)}",
            causa=e
        ))