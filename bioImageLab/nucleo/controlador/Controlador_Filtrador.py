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
) -> Callable[[BioImagenData, int], Resultado[BioImagenData, ErrorBioImagen]]:
    """
    Factory curried que retorna función pura de filtrado.
    
    FIRMA CORREGIDA: (BioImagenData, canal_idx) -> Resultado[BioImagenData, ErrorBioImagen]
    Igual que crear_normalizador para compatibilidad con Operacion.ejecutar()
    """
    def _aplicar_filtro(
        data: BioImagenData, 
        canal_idx: int = 0
    ) -> Resultado[BioImagenData, ErrorBioImagen]:
        
        # Validación de canal
        if not (0 <= canal_idx < data.dims.C):
            return Err(ErrorBioImagen(
                etapa="filtracion",
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
                case Filtro_Global():
                    for t in range(T):
                        for z in range(Z):
                            resultado_canal[t, z, 0, :, :] = metodo(canal_data[t, z, :, :])
                            
                case Filtro_PorCorteZ():
                    for z in range(Z):
                        for t in range(T):
                            resultado_canal[t, z, 0, :, :] = metodo(canal_data[t, z, :, :])
                            
                case Filtro_PorTimepoint():
                    for t in range(T):
                        for z in range(Z):
                            resultado_canal[t, z, 0, :, :] = metodo(canal_data[t, z, :, :])
                            
                case Filtro_PorCorteEspaciotemporal():
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
            # crear_filtro ahora retorna callable(BioImagenData, int)
            filtro_canal = crear_filtro(metodo, tipo)
            # Usar bind con lambda que pasa el canal
            resultado = resultado.bind(lambda d, canal=c: filtro_canal(d, canal))
            if resultado.es_err():
                break
        
        return resultado
    
    return _aplicar_multicanal


# ==================== WRAPPER ORIENTADO A OBJETOS ====================

class Controlador_Filtrador:
    """
    Wrapper stateful para filtrado.
    """
    
    def __init__(self):
        self._cache: Optional[Dict[str, Any]] = None
        self._ultimo_filtro: Optional[MetodoFiltro] = None
    
    # ===== MÉTODOS FACTORY POR DOMINIO =====
    # Todos retornan Callable[[BioImagenData], Resultado[...]] para compatibilidad
    # o Callable[[BioImagenData, int], Resultado[...]] según necesidad
    
    def crear_gaussiano(
        self, 
        sigma: float = 1.0,
        tipo_aplicacion: TipoFiltro = Filtro_Global(),
        canal: int = 0
    ) -> Callable[[BioImagenData], Resultado[BioImagenData, ErrorBioImagen]]:
        """Crea filtro gaussiano espacial."""
        metodo = Gaussiano(sigma=sigma)
        self._ultimo_filtro = metodo
        # Usar crear_filtro que retorna callable(BioImagenData, int)
        filtro = crear_filtro(metodo, tipo_aplicacion)
        # Currying: fijar el canal
        return lambda data: filtro(data, canal)
    
    def crear_mediana(
        self,
        tamano: int = 3,
        tipo_aplicacion: TipoFiltro = Filtro_Global(),
        canal: int = 0
    ) -> Callable[[BioImagenData], Resultado[BioImagenData, ErrorBioImagen]]:
        metodo = Mediana(tamano=tamano)
        self._ultimo_filtro = metodo
        filtro = crear_filtro(metodo, tipo_aplicacion)
        return lambda data: filtro(data, canal)
    
    def crear_bilateral(
        self,
        sigma_color: float = 0.1,
        sigma_spatial: float = 2.0,
        tipo_aplicacion: TipoFiltro = Filtro_PorCorteEspaciotemporal(),
        canal: int = 0
    ) -> Callable[[BioImagenData], Resultado[BioImagenData, ErrorBioImagen]]:
        metodo = Bilateral(sigma_color=sigma_color, sigma_spatial=sigma_spatial)
        self._ultimo_filtro = metodo
        filtro = crear_filtro(metodo, tipo_aplicacion)
        return lambda data: filtro(data, canal)
    
    def crear_difusion_anisotropica(
        self,
        iteraciones: int = 10,
        kappa: float = 50.0,
        gamma: float = 0.1,
        tipo_aplicacion: TipoFiltro = Filtro_PorCorteEspaciotemporal(),
        canal: int = 0
    ) -> Callable[[BioImagenData], Resultado[BioImagenData, ErrorBioImagen]]:
        metodo = DifusionAnisotropica(iteraciones=iteraciones, kappa=kappa, gamma=gamma)
        self._ultimo_filtro = metodo
        filtro = crear_filtro(metodo, tipo_aplicacion)
        return lambda data: filtro(data, canal)
    
    def crear_pasabajo(
        self,
        radio: int = 30,
        tipo_aplicacion: TipoFiltro = Filtro_Global(),
        canal: int = 0
    ) -> Callable[[BioImagenData], Resultado[BioImagenData, ErrorBioImagen]]:
        metodo = FFTPasaBajo(radio=radio)
        self._ultimo_filtro = metodo
        filtro = crear_filtro(metodo, tipo_aplicacion)
        return lambda data: filtro(data, canal)
    
    def crear_pasaalto(
        self,
        radio: int = 30,
        tipo_aplicacion: TipoFiltro = Filtro_Global(),
        canal: int = 0
    ) -> Callable[[BioImagenData], Resultado[BioImagenData, ErrorBioImagen]]:
        metodo = FFTPasaAlto(radio=radio)
        self._ultimo_filtro = metodo
        filtro = crear_filtro(metodo, tipo_aplicacion)
        return lambda data: filtro(data, canal)
    
    def crear_pasa_banda(
        self,
        radio_bajo: int = 10,
        radio_alto: int = 50,
        tipo_aplicacion: TipoFiltro = Filtro_Global(),
        canal: int = 0
    ) -> Callable[[BioImagenData], Resultado[BioImagenData, ErrorBioImagen]]:
        metodo = FFTPasaBanda(radio_bajo=radio_bajo, radio_alto=radio_alto)
        self._ultimo_filtro = metodo
        filtro = crear_filtro(metodo, tipo_aplicacion)
        return lambda data: filtro(data, canal)
    
    def crear_bandstop(
        self,
        frecuencia_central: float = 0.25,
        ancho_banda: float = 0.05,
        tipo_aplicacion: TipoFiltro = Filtro_Global(),
        canal: int = 0
    ) -> Callable[[BioImagenData], Resultado[BioImagenData, ErrorBioImagen]]:
        metodo = FFTBandStop(frecuencia_central=frecuencia_central, ancho_banda=ancho_banda)
        self._ultimo_filtro = metodo
        filtro = crear_filtro(metodo, tipo_aplicacion)
        return lambda data: filtro(data, canal)
    
    def crear_notch(
        self,
        frecuencias_eliminar: list[tuple[float, float]],
        tipo_aplicacion: TipoFiltro = Filtro_Global(),
        canal: int = 0
    ) -> Callable[[BioImagenData], Resultado[BioImagenData, ErrorBioImagen]]:
        metodo = FiltradoNotch(frecuencias_eliminar=frecuencias_eliminar)
        self._ultimo_filtro = metodo
        filtro = crear_filtro(metodo, tipo_aplicacion)
        return lambda data: filtro(data, canal)
    
    def crear_diferencia_gaussiana(
        self,
        sigma1: float = 1.0,
        sigma2: float = 2.0,
        tipo_aplicacion: TipoFiltro = Filtro_PorCorteEspaciotemporal(),
        canal: int = 0
    ) -> Callable[[BioImagenData], Resultado[BioImagenData, ErrorBioImagen]]:
        metodo = DiferenciaGaussiana(sigma1=sigma1, sigma2=sigma2)
        self._ultimo_filtro = metodo
        filtro = crear_filtro(metodo, tipo_aplicacion)
        return lambda data: filtro(data, canal)
    
    def crear_laplaciano(
        self,
        tipo: str = "discreto",
        tipo_aplicacion: TipoFiltro = Filtro_PorCorteEspaciotemporal(),
        canal: int = 0
    ) -> Callable[[BioImagenData], Resultado[BioImagenData, ErrorBioImagen]]:
        metodo = DiferenciaLaplaciana(tipo=tipo)
        self._ultimo_filtro = metodo
        filtro = crear_filtro(metodo, tipo_aplicacion)
        return lambda data: filtro(data, canal)
    
    def crear_wavelet(
        self,
        wavelet: str = 'db4',
        niveles: int = 3,
        umbral: float = 0.1,
        tipo_aplicacion: TipoFiltro = Filtro_Global(),
        canal: int = 0
    ) -> Callable[[BioImagenData], Resultado[BioImagenData, ErrorBioImagen]]:
        metodo = WaveletTransform(wavelet=wavelet, niveles=niveles, umbral=umbral)
        self._ultimo_filtro = metodo
        filtro = crear_filtro(metodo, tipo_aplicacion)
        return lambda data: filtro(data, canal)
    
    def crear_piramide_laplaciana(
        self,
        niveles: int = 3,
        tipo_aplicacion: TipoFiltro = Filtro_Global(),
        canal: int = 0
    ) -> Callable[[BioImagenData], Resultado[BioImagenData, ErrorBioImagen]]:
        metodo = PiramideLaplaciana(niveles=niveles)
        self._ultimo_filtro = metodo
        filtro = crear_filtro(metodo, tipo_aplicacion)
        return lambda data: filtro(data, canal)
    
    def crear_nl_means(
        self,
        h: float = 0.1,
        tamano_patch: int = 7,
        tamano_busqueda: int = 21,
        tipo_aplicacion: TipoFiltro = Filtro_PorCorteEspaciotemporal(),
        canal: int = 0
    ) -> Callable[[BioImagenData], Resultado[BioImagenData, ErrorBioImagen]]:
        metodo = NonLocalMedians(h=h, tamano_patch=tamano_patch, tamano_busqueda=tamano_busqueda)
        self._ultimo_filtro = metodo
        filtro = crear_filtro(metodo, tipo_aplicacion)
        return lambda data: filtro(data, canal)
    
    def crear_bm3d(
        self,
        sigma: float = 25.0,
        tipo_aplicacion: TipoFiltro = Filtro_PorCorteEspaciotemporal(),
        canal: int = 0
    ) -> Callable[[BioImagenData], Resultado[BioImagenData, ErrorBioImagen]]:
        metodo = BlockMatching3D(sigma=sigma)
        self._ultimo_filtro = metodo
        filtro = crear_filtro(metodo, tipo_aplicacion)
        return lambda data: filtro(data, canal)
    
    # ===== MÉTODOS DE APLICACIÓN DIRECTA =====
    
    def aplicar(
        self,
        data: BioImagenData,
        metodo: MetodoFiltro,
        tipo: TipoFiltro = Filtro_Global(),
        canal: int = 0
    ) -> Resultado[BioImagenData, ErrorBioImagen]:
        """Aplica filtro específico a datos (versión imperativa)."""
        filtro = crear_filtro(metodo, tipo)
        return filtro(data, canal)
    
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
    
    CORREGIDO: Ahora usa crear_filtro que retorna callable(BioImagenData, int)
    """
    nombre_op = nombre or f"filtrado_{metodo.__class__.__name__}_{tipo.__class__.__name__}"
    
    # CORRECCIÓN CLAVE: Usar crear_filtro que tiene firma (BioImagenData, int) -> Resultado
    filtro_callable = crear_filtro(metodo, tipo)
    
    return Operacion(
        nombre=nombre_op,
        categoria=CategoriaOperacion.FILTRACION,
        instancia_callable=filtro_callable,  # Ahora es callable(data, canal) -> Resultado
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
    params = {}
    for attr in ['sigma', 'radio', 'tamano', 'iteraciones', 'h', 'wavelet', 'niveles', 
                 'radio_bajo', 'radio_alto', 'frecuencia_central', 'ancho_banda']:
        if hasattr(metodo, attr):
            params[attr] = getattr(metodo, attr)
    return params


# Factory desde controlador (opcional, para compatibilidad)
def operacion_filtro_desde_controlador(
    controlador: Controlador_Filtrador,
    nombre_metodo: str,
    params: Dict[str, Any],
    canal: int = 0
) -> Resultado[Operacion, ErrorBioImagen]:
    """
    Crea operación usando método del controlador.
    """
    try:
        if not hasattr(controlador, nombre_metodo):
            return Err(ErrorBioImagen(
                etapa="configuracion",
                mensaje=f"Método '{nombre_metodo}' no existe en Controlador_Filtrador"
            ))
        
        factory = getattr(controlador, nombre_metodo)
        # Llamar factory para obtener el callable curried
        filtro_callable = factory(**params, canal=canal)
        
        # Extraer método subyacente para metadata
        metodo_subyacente = controlador._ultimo_filtro
        
        return Ok(Operacion(
            nombre=nombre_metodo.replace("crear_", ""),
            categoria=CategoriaOperacion.FILTRACION,
            instancia_callable=lambda data, c=canal: filtro_callable(data),  # Adaptador
            canal_objetivo=canal,
            parametros_originales=params,
            tipo_salida=TipoSalida.IMAGEN
        ))
        
    except Exception as e:
        return Err(ErrorBioImagen(
            etapa="configuracion",
            mensaje=f"Error: {str(e)}",
            causa=e
        ))