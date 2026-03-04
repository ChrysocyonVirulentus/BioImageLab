# === filtrador/Controlador_Filtrador.py ===
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

# Imports de métodos específicos (de diferentes submódulos)
from ..filtrador.locales.Filtros_Locales import (
    CajaBlur,
    Gaussiano,
    Bilateral,
    Mediana,
    DifusionAnisotropica
)
from ..filtrador.espectrales.Filtros_Ffts import (
    FFTPasaBajo,
    FFTPasaAlto,
    FFTPasaBanda,
    FFTBandStop,
    FiltradoNotch
)
from ..filtrador.multiescala.Filtros_Multiescala import (
    DiferenciaLaplaciana,
    DiferenciaGaussiana,
    WaveletTransform,
    PiramideLaplaciana
)
from ..filtrador.noLocales.Filtros_NoLocales import (
    NonLocalMeans,
    BM3D
)


# ==================== TIPOS ALGEBRAICOS DE FILTRADO ====================

@dataclass(frozen=True)
class Filtro_Global:
    """Aplica el mismo filtro a todo el volumen sin distinción de regiones."""
    pass

@dataclass(frozen=True)
class Filtro_PorCorteZ:
    """Aplica filtro independientemente a cada plano Z."""
    pass

@dataclass(frozen=True)
class Filtro_PorTimepoint:
    """Aplica filtro independientemente a cada timepoint T."""
    pass

@dataclass(frozen=True)
class Filtro_PorCorteEspaciotemporal:
    """Aplica filtro a cada corte (t, z) de forma independiente."""
    pass

TipoFiltro = Union[
    Filtro_Global,
    Filtro_PorCorteZ,
    Filtro_PorTimepoint,
    Filtro_PorCorteEspaciotemporal
]


# ==================== TIPOS DE FILTROS POR DOMINIO ====================

class TipoFiltroDominio(Enum):
    ESPACIAL = "espacial"           # Locales: Gaussiano, Mediana, etc.
    ESPECTRAL = "espectral"         # FFTs: Pasa-bajo, Pasa-alto, etc.
    MULTIESCALA = "multiescala"     # Laplacianos, Wavelets, Pirámides
    NO_LOCAL = "no_local"           # NL-means, BM3D

# Union de todos los métodos de filtrado disponibles
MetodoFiltro = Union[
    # Espaciales
    CajaBlur,
    Gaussiano,
    Bilateral,
    Mediana,
    DifusionAnisotropica,
    # Espectrales
    FFTPasaBajo,
    FFTPasaAlto,
    FFTPasaBanda,
    FFTBandStop,
    FiltradoNotch,
    # Multiescala
    DiferenciaLaplaciana,
    DiferenciaGaussiana,
    WaveletTransform,
    PiramideLaplaciana,
    # No locales
    NonLocalMeans,
    BM3D
]


# ==================== FUNCIONES PURAS ====================

def crear_filtro(
    metodo: MetodoFiltro,
    tipo: TipoFiltro = Filtro_Global(),
    canal: int = 0
) -> Callable[[BioImagenData], Resultado[BioImagenData, ErrorBioImagen]]:
    """
    Factory curried que retorna función pura de filtrado.
    
    Args:
        metodo: Instancia del filtro específico (ej: Gaussiano(sigma=2.0))
        tipo: Estrategia de aplicación espacio-temporal
        canal: Canal objetivo (None = todos, pero aquí específico por simplicidad)
    
    Returns:
        Callable para usar con .bind() en pipelines
    """
    def _aplicar_filtro(data: BioImagenData) -> Resultado[BioImagenData, ErrorBioImagen]:
        
        # Validación de canal
        if not (0 <= canal < data.dims.C):
            return Err(ErrorBioImagen(
                etapa="filtracion",
                mensaje=f"Canal {canal} fuera de rango [0, {data.dims.C-1}]",
                ruta=data.ruta_origen
            ))
        
        try:
            T, Z, C, Y, X = data.dims.shape
            
            # Extraer canal objetivo [T, Z, Y, X]
            canal_data = data.datos[:, :, canal, :, :]
            
            # Array resultado manteniendo estructura 5D
            resultado_canal = np.zeros((T, Z, 1, Y, X), dtype=np.float64)
            
            # Aplicar según estrategia
            match tipo:
                case Filtro_Global():
                    # Aplicar a todo el volumen como una secuencia 2D continua
                    for t in range(T):
                        for z in range(Z):
                            resultado_canal[t, z, 0, :, :] = metodo(canal_data[t, z, :, :])
                            
                case Filtro_PorCorteZ():
                    # Cada Z independiente (procesa todo T para cada Z)
                    for z in range(Z):
                        for t in range(T):
                            resultado_canal[t, z, 0, :, :] = metodo(canal_data[t, z, :, :])
                            
                case Filtro_PorTimepoint():
                    # Cada T independiente (procesa todo Z para cada T)
                    for t in range(T):
                        for z in range(Z):
                            resultado_canal[t, z, 0, :, :] = metodo(canal_data[t, z, :, :])
                            
                case Filtro_PorCorteEspaciotemporal():
                    # Máxima granularidad: cada (t,z) independiente
                    for t in range(T):
                        for z in range(Z):
                            resultado_canal[t, z, 0, :, :] = metodo(canal_data[t, z, :, :])
            
            # Reconstruir BioImagenData
            nuevos_datos = data.datos.copy().astype(np.float64)
            nuevos_datos[:, :, canal, :, :] = resultado_canal[:, :, 0, :, :]
            
            return Ok(replace(data, datos=nuevos_datos))
            
        except Exception as e:
            nombre_metodo = getattr(metodo, 'nombre', metodo.__class__.__name__)
            return Err(ErrorBioImagen(
                etapa="filtracion",
                mensaje=f"Fallo en {nombre_metodo}: {str(e)}",
                ruta=data.ruta_origen,
                causa=e
            ))
    
    return _aplicar_filtro


def crear_filtro_multicanal(
    metodo: MetodoFiltro,
    tipo: TipoFiltro = Filtro_Global()
) -> Callable[[BioImagenData], Resultado[BioImagenData, ErrorBioImagen]]:
    """
    Versión que aplica el mismo filtro a todos los canales.
    """
    def _aplicar_multicanal(data: BioImagenData) -> Resultado[BioImagenData, ErrorBioImagen]:
        resultado: Resultado[BioImagenData, ErrorBioImagen] = Ok(data)
        
        for c in range(data.dims.C):
            filtro_canal = crear_filtro(metodo, tipo, canal=c)
            resultado = resultado.bind(filtro_canal)
            if resultado.es_err():
                break
        
        return resultado
    
    return _aplicar_multicanal


# ==================== WRAPPER ORIENTADO A OBJETOS ====================

class Controlador_Filtrador:
    """
    Wrapper stateful para filtrado.
    Permite crear filtros configurados y aplicarlos.
    """
    
    def __init__(self):
        self._cache: Optional[Dict[str, Any]] = None
        self._ultimo_filtro: Optional[MetodoFiltro] = None
    
    # ===== MÉTODOS FACTORY POR DOMINIO =====
    
    # --- Espaciales ---
    def crear_gaussiano(
        self, 
        sigma: float = 1.0,
        tipo_aplicacion: TipoFiltro = Filtro_Global()
    ) -> Callable[[BioImagenData], Resultado[BioImagenData, ErrorBioImagen]]:
        """Crea filtro gaussiano espacial."""
        metodo = Gaussiano(sigma=sigma)
        self._ultimo_filtro = metodo
        return crear_filtro(metodo, tipo_aplicacion)
    
    def crear_mediana(
        self,
        tamano: int = 3,
        tipo_aplicacion: TipoFiltro = Filtro_Global()
    ) -> Callable[[BioImagenData], Resultado[BioImagenData, ErrorBioImagen]]:
        """Crea filtro de mediana."""
        metodo = Mediana(tamano=tamano)
        self._ultimo_filtro = metodo
        return crear_filtro(metodo, tipo_aplicacion)
    
    def crear_bilateral(
        self,
        sigma_color: float = 0.1,
        sigma_spatial: float = 2.0,
        tipo_aplicacion: TipoFiltro = Filtro_PorCorteEspaciotemporal()
    ) -> Callable[[BioImagenData], Resultado[BioImagenData, ErrorBioImagen]]:
        """Crea filtro bilateral (preserva bordes)."""
        metodo = Bilateral(sigma_color=sigma_color, sigma_spatial=sigma_spatial)
        self._ultimo_filtro = metodo
        return crear_filtro(metodo, tipo_aplicacion)
    
    def crear_difusion_anisotropica(
        self,
        iteraciones: int = 10,
        kappa: float = 50.0,
        gamma: float = 0.1,
        tipo_aplicacion: TipoFiltro = Filtro_PorCorteEspaciotemporal()
    ) -> Callable[[BioImagenData], Resultado[BioImagenData, ErrorBioImagen]]:
        """Crea filtro de difusión anisotrópica (preserva bordes fuertes)."""
        metodo = DifusionAnisotropica(iteraciones=iteraciones, kappa=kappa, gamma=gamma)
        self._ultimo_filtro = metodo
        return crear_filtro(metodo, tipo_aplicacion)
    
    # --- Espectrales (FFT) ---
    def crear_pasabajo(
        self,
        radio: int = 30,
        tipo_aplicacion: TipoFiltro = Filtro_Global()
    ) -> Callable[[BioImagenData], Resultado[BioImagenData, ErrorBioImagen]]:
        """Crea filtro pasa-bajo en frecuencia (suavizado)."""
        metodo = FFTPasabajo(radio=radio)
        self._ultimo_filtro = metodo
        return crear_filtro(metodo, tipo_aplicacion)
    
    def crear_pasaalto(
        self,
        radio: int = 30,
        tipo_aplicacion: TipoFiltro = Filtro_Global()
    ) -> Callable[[BioImagenData], Resultado[BioImagenData, ErrorBioImagen]]:
        """Crea filtro pasa-alto en frecuencia (detección de bordes/finos)."""
        metodo = FFTPasaalto(radio=radio)
        self._ultimo_filtro = metodo
        return crear_filtro(metodo, tipo_aplicacion)
    
    def crear_pasa_banda(
        self,
        radio_bajo: int = 10,
        radio_alto: int = 50,
        tipo_aplicacion: TipoFiltro = Filtro_Global()
    ) -> Callable[[BioImagenData], Resultado[BioImagenData, ErrorBioImagen]]:
        """Crea filtro pasa-banda (rechaza frecuencias muy bajas y muy altas)."""
        metodo = FFTPasaBanda(radio_bajo=radio_bajo, radio_alto=radio_alto)
        self._ultimo_filtro = metodo
        return crear_filtro(metodo, tipo_aplicacion)
    
    def crear_bandstop(
        self,
        frecuencia_central: float = 0.25,
        ancho_banda: float = 0.05,
        tipo_aplicacion: TipoFiltro = Filtro_Global()
    ) -> Callable[[BioImagenData], Resultado[BioImagenData, ErrorBioImagen]]:
        """Crea filtro rechaza-banda (elimina frecuencia específica, ej: ruido periódico)."""
        metodo = FFTBandStop(frecuencia_central=frecuencia_central, ancho_banda=ancho_banda)
        self._ultimo_filtro = metodo
        return crear_filtro(metodo, tipo_aplicacion)
    
    def crear_notch(
        self,
        frecuencias_eliminar: list[tuple[float, float]],
        tipo_aplicacion: TipoFiltro = Filtro_Global()
    ) -> Callable[[BioImagenData], Resultado[BioImagenData, ErrorBioImagen]]:
        """Crea filtro notch (elimina frecuencias puntuales, ej: artefactos de cuadrícula)."""
        metodo = FiltradoNotch(frecuencias_eliminar=frecuencias_eliminar)
        self._ultimo_filtro = metodo
        return crear_filtro(metodo, tipo_aplicacion)
    
    # --- Multiescala ---
    def crear_diferencia_gaussiana(
        self,
        sigma1: float = 1.0,
        sigma2: float = 2.0,
        tipo_aplicacion: TipoFiltro = Filtro_PorCorteEspaciotemporal()
    ) -> Callable[[BioImagenData], Resultado[BioImagenData, ErrorBioImagen]]:
        """Crea DoG (detección de blobs por diferencia de escalas)."""
        metodo = DiferenciaGaussiana(sigma1=sigma1, sigma2=sigma2)
        self._ultimo_filtro = metodo
        return crear_filtro(metodo, tipo_aplicacion)
    
    def crear_laplaciano(
        self,
        tipo: str = "discreto",
        tipo_aplicacion: TipoFiltro = Filtro_PorCorteEspaciotemporal()
    ) -> Callable[[BioImagenData], Resultado[BioImagenData, ErrorBioImagen]]:
        """Crea filtro Laplaciano (detección de puntos de inflexión)."""
        metodo = DiferenciaLaplaciana(tipo=tipo)
        self._ultimo_filtro = metodo
        return crear_filtro(metodo, tipo_aplicacion)
    
    def crear_wavelet(
        self,
        wavelet: str = 'db4',
        niveles: int = 3,
        umbral: float = 0.1,
        tipo_aplicacion: TipoFiltro = Filtro_Global()
    ) -> Callable[[BioImagenData], Resultado[BioImagenData, ErrorBioImagen]]:
        """Crea filtro wavelet (denoising multiescala)."""
        metodo = WaveletTransform(wavelet=wavelet, niveles=niveles, umbral=umbral)
        self._ultimo_filtro = metodo
        return crear_filtro(metodo, tipo_aplicacion)
    
    def crear_piramide_laplaciana(
        self,
        niveles: int = 3,
        tipo_aplicacion: TipoFiltro = Filtro_Global()
    ) -> Callable[[BioImagenData], Resultado[BioImagenData, ErrorBioImagen]]:
        """Crea pirámide Laplaciana (representación multiescala reconstruible)."""
        metodo = PiramideLaplaciana(niveles=niveles)
        self._ultimo_filtro = metodo
        return crear_filtro(metodo, tipo_aplicacion)
    
    # --- No locales ---
    def crear_nl_means(
        self,
        h: float = 0.1,
        tamano_patch: int = 7,
        tamano_busqueda: int = 21,
        tipo_aplicacion: TipoFiltro = Filtro_PorCorteEspaciotemporal()
    ) -> Callable[[BioImagenData], Resultado[BioImagenData, ErrorBioImagen]]:
        """Crea filtro Non-Local Means (preserva texturas mientras reduce ruido)."""
        metodo = NonLocalMedians(h=h, tamano_patch=tamano_patch, tamano_busqueda=tamano_busqueda)
        self._ultimo_filtro = metodo
        return crear_filtro(metodo, tipo_aplicacion)
    
    def crear_bm3d(
        self,
        sigma: float = 25.0,
        tipo_aplicacion: TipoFiltro = Filtro_PorCorteEspaciotemporal()
    ) -> Callable[[BioImagenData], Resultado[BioImagenData, ErrorBioImagen]]:
        """Crea BM3D (Block-Matching 3D, state-of-the-art denoising)."""
        metodo = BlockMatching3D(sigma=sigma)
        self._ultimo_filtro = metodo
        return crear_filtro(metodo, tipo_aplicacion)
    
    # ===== MÉTODOS DE APLICACIÓN DIRECTA (compatibilidad) =====
    
    def aplicar(
        self,
        data: BioImagenData,
        metodo: MetodoFiltro,
        tipo: TipoFiltro = Filtro_Global(),
        canal: int = 0
    ) -> Resultado[BioImagenData, ErrorBioImagen]:
        """Aplica filtro específico a datos (versión imperativa)."""
        filtro = crear_filtro(metodo, tipo, canal)
        return filtro(data)
    
    def aplicar_multicanal(
        self,
        data: BioImagenData,
        metodo: MetodoFiltro,
        tipo: TipoFiltro = Filtro_Global()
    ) -> Resultado[BioImagenData, ErrorBioImagen]:
        """Aplica filtro a todos los canales."""
        filtro = crear_filtro_multicanal(metodo, tipo)
        return filtro(data)
    
    def reset(self):
        """Limpia caché interno."""
        self._cache = None
        self._ultimo_filtro = None
    
    def __repr__(self) -> str:
        ultimo = getattr(self._ultimo_filtro, 'nombre', 'Ninguno')
        return f"<Controlador_Filtrador ultimo={ultimo}>"


# ==================== FACTORIES PARA PIPELINES ====================

def operacion_filtro(
    metodo: MetodoFiltro,
    tipo: TipoFiltro = Filtro_Global(),
    canal: int = 0,
    nombre: Optional[str] = None
) -> Operacion:
    """
    Crea Operación de filtración lista para PipelineBuilder.
    
    Uso:
        pipeline = (
            PipelineBuilder()
            .filtrar("gaussiano_suavizado", 
                operacion_filtro(Gaussiano(sigma=2.0), Filtro_Global(), canal=0))
            .filtrar("pasaalto_bordes",
                operacion_filtro(FFTPasaalto(radio=40), Filtro_Global(), canal=0))
            .construir()
        )
    """
    nombre_op = nombre or f"filtrado_{metodo.__class__.__name__}_{tipo.__class__.__name__}"
    
    return Operacion(
        nombre=nombre_op,
        categoria=CategoriaOperacion.FILTRACION,
        instancia_callable=metodo,  # El método ya es callable
        canal_objetivo=canal,
        parametros_originales={
            "metodo": metodo.__class__.__name__,
            "tipo": tipo.__class__.__name__,
            "params": _extraer_parametros(metodo)
        },
        tipo_salida=TipoSalida.IMAGEN
    )


def _extraer_parametros(metodo: MetodoFiltro) -> Dict[str, Any]:
    """Extrae parámetros del método para metadata."""
    # Asume que los métodos tienen atributos públicos documentados
    params = {}
    for attr in ['sigma', 'radio', 'tamano', 'iteraciones', 'h', 'wavelet', 'niveles']:
        if hasattr(metodo, attr):
            params[attr] = getattr(metodo, attr)
    return params


# Factory conveniente desde Controlador_Filtrador
def operacion_filtro_desde_controlador(
    controlador: Controlador_Filtrador,
    metodo_factory: str,  # nombre del método: "crear_gaussiano", "crear_pasaalto", etc.
    params: Dict[str, Any],
    canal: int = 0
) -> Resultado[Operacion, ErrorBioImagen]:
    """
    Crea operación usando el controlador (valida que el método exista).
    
    Uso:
        filtrador = Controlador_Filtrador()
        op_result = operacion_filtro_desde_controlador(
            filtrador, "crear_gaussiano", {"sigma": 2.0}, canal=0)
        if op_result.es_ok():
            builder.filtrar("mi_filtro", op_result.unwrap())
    """
    try:
        if not hasattr(controlador, metodo_factory):
            return Err(ErrorBioImagen(
                etapa="configuracion",
                mensaje=f"Controlador_Filtrador no tiene método '{metodo_factory}'"
            ))
        
        factory = getattr(controlador, metodo_factory)
        filtro_callable = factory(**params)  # Esto retorna el callable curried
        
        # Extraer nombre del método para la operación
        nombre_op = metodo_factory.replace("crear_", "")
        
        # Crear operación con callable interno
        # Necesitamos "desenvolver" el callable para obtener el metodo subyacente
        # Esto es un poco hacky pero funciona para el registro
        metodo_subyacente = controlador._ultimo_filtro
        
        return Ok(Operacion(
            nombre=nombre_op,
            categoria=CategoriaOperacion.FILTRACION,
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