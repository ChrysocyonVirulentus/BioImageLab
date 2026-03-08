from __future__ import annotations
import numpy as np
from dataclasses import dataclass, replace
from typing import Union, Callable, Optional, Tuple, Dict, Any, List
from enum import Enum, auto
from pathlib import Path

# Imports de tu sistema
from .Resultado_Either import Resultado, Err, Ok
from .Controlador_BioImagen import BioImagenData, ErrorBioImagen, Dimensiones
from ..gestorLab.Categoria_Operacion import CategoriaOperacion
from ..gestorLab.Operacion import Operacion, TipoSalida

# Imports de métodos específicos (3 submódulos)
from ..segmentador.binarizacion.Segmentadores_Binarizacion import (
    Otsu,
    Global,
    Adaptativo,
    Percentil,
    Triangle,
    Isodata,
    Minimum,
    Mean
)
from ..segmentador.instancial.Segmentadores_Instanciales import (
    Watershed,
    WatershedMarcado,
    DistanciaWatershed,
    SplitDistancial,
    WatershedHibrido,
    SplitWatershed
)
from ..segmentador.regional.Segmentadores_Regionales import (
    RegionGrowing,
    RandomWalk,
    CorteGrafico,
    SuperpixelSLIC,
    SuperpixelFelzenszwalb,
    WatershedRegiones,
    MeanShiftSegmentacion
)


# ==================== TIPOS ALGEBRAICOS DE SEGMENTACIÓN ====================

@dataclass(frozen=True)
class Segmentacion_Global:
    """Segmenta todo el volumen con mismos parámetros."""
    pass

@dataclass(frozen=True)
class Segmentacion_PorCorteZ:
    """Segmenta cada plano Z independientemente."""
    pass

@dataclass(frozen=True)
class Segmentacion_PorTimepoint:
    """Segmenta cada timepoint T independientemente."""
    pass

@dataclass(frozen=True)
class Segmentacion_PorCorteEspaciotemporal:
    """Segmenta cada corte (t, z) de forma independiente."""
    pass

TipoSegmentacion = Union[
    Segmentacion_Global,
    Segmentacion_PorCorteZ,
    Segmentacion_PorTimepoint,
    Segmentacion_PorCorteEspaciotemporal
]


# ==================== TIPOS DE SEGMENTACIÓN POR SUBMÓDULO ====================

class TipoSegmentacionSubmodulo(Enum):
    BINARIZACION = "binarizacion"       # Umbralización: Otsu, adaptativo, etc.
    INSTANCIAL = "instancial"           # Watershed, splits, separación de objetos tocándose
    REGIONAL = "regional"               # Crecimiento de regiones, superpixels, clustering espacial

# Union de todos los métodos de segmentación disponibles
MetodoSegmentacion = Union[
    # Binarización
    Otsu, Global, Adaptativo, Percentil,
    Triangle, Isodata, Minimum, Mean,
    # Instanciales
    Watershed, WatershedMarcado, DistanciaWatershed,
    SplitDistancial, WatershedHibrido, SplitWatershed,
    # Regionales
    RegionGrowing, RandomWalk, CorteGrafico,
    SuperpixelSLIC, SuperpixelFelzenszwalb, WatershedRegiones, MeanShiftSegmentacion
]


# ==================== FUNCIONES PURAS ====================

def crear_segmentacion(
    metodo: MetodoSegmentacion,
    tipo: TipoSegmentacion = Segmentacion_PorCorteEspaciotemporal(),
) -> Callable[[BioImagenData], Resultado[BioImagenData, ErrorBioImagen]]:
    """
    Factory curried que retorna función pura de segmentación.
    
    Args:
        metodo: Instancia del segmentador (ej: Otsu(), Watershed(marcadores=...))
        tipo: Estrategia de aplicación espacio-temporal
        canal: Canal objetivo
    
    Returns:
        Callable para usar con .bind() en pipelines
    """
    def _aplicar_segmentacion(data: BioImagenData, canal_idx: 0) -> Resultado[BioImagenData, ErrorBioImagen]:
        
        # Validación de canal
        if not (0 <= canal_idx < data.dims.C):
            return Err(ErrorBioImagen(
                etapa="segmentacion",
                mensaje=f"Canal {canal_idx} fuera de rango [0, {data.dims.C-1}]",
                ruta=data.ruta_origen
            ))
        
        try:
            T, Z, C, Y, X = data.dims.shape
            
            # Extraer canal objetivo [T, Z, Y, X]
            canal_data = data.datos[:, :, canal_idx, :, :]
            
            # Array resultado: máscara binaria o etiquetada (mismo shape)
            resultado_canal = np.zeros((T, Z, 1, Y, X), dtype=np.uint16)  # uint16 para etiquetas
            
            # Aplicar según estrategia
            match tipo:
                case Segmentacion_Global():
                    # Segmentación global (puede usar información de todo el volumen)
                    for t in range(T):
                        for z in range(Z):
                            resultado = metodo(canal_data[t, z, :, :])
                            resultado_canal[t, z, 0, :, :] = _asegurar_shape(resultado, (Y, X))
                            
                case Segmentacion_PorCorteZ():
                    # Cada Z independiente
                    for z in range(Z):
                        for t in range(T):
                            resultado = metodo(canal_data[t, z, :, :])
                            resultado_canal[t, z, 0, :, :] = _asegurar_shape(resultado, (Y, X))
                            
                case Segmentacion_PorTimepoint():
                    # Cada T independiente
                    for t in range(T):
                        for z in range(Z):
                            resultado = metodo(canal_data[t, z, :, :])
                            resultado_canal[t, z, 0, :, :] = _asegurar_shape(resultado, (Y, X))
                            
                case Segmentacion_PorCorteEspaciotemporal():
                    # Máxima granularidad: cada (t,z) independiente
                    for t in range(T):
                        for z in range(Z):
                            resultado = metodo(canal_data[t, z, :, :])
                            resultado_canal[t, z, 0, :, :] = _asegurar_shape(resultado, (Y, X))
            
            # Reconstruir BioImagenData con máscara/etiquetas
            # Preservar metadatos originales pero marcar que es segmentación
            nuevos_datos = np.zeros_like(data.datos, dtype=np.uint16)
            nuevos_datos[:, :, canal_idx, :, :] = resultado_canal[:, :, 0, :, :]
            
            # Copiar otros canales si existen (para mantener contexto)
            for c in range(C):
                if c != canal_idx:
                    # Para canales no segmentados, podemos copiar o poner ceros
                    # Aquí copiamos para mantener referencia
                    nuevos_datos[:, :, c, :, :] = data.datos[:, :, c, :, :]
            
            return Ok(replace(
                data, 
                datos=nuevos_datos,
                metadata={
                    **data.__dict__.get('metadata', {}),
                    'es_segmentacion': True,
                    'canal_segmentado': canal_idx,
                    'metodo_segmentacion': metodo.__class__.__name__,
                    'tipo_segmentacion': tipo.__class__.__name__
                }
            ))
            
        except Exception as e:
            nombre_metodo = getattr(metodo, 'nombre', metodo.__class__.__name__)
            return Err(ErrorBioImagen(
                etapa="segmentacion",
                mensaje=f"Fallo en {nombre_metodo}: {str(e)}",
                ruta=data.ruta_origen,
                causa=e
            ))
    
    return _aplicar_segmentacion


def _asegurar_shape(array: np.ndarray, shape_objetivo: Tuple[int, int]) -> np.ndarray:
    """
    Asegura que el array resultante tenga el shape correcto.
    Los segmentadores pueden retornar binario (Y,X) o etiquetado (Y,X).
    """
    if array.shape == shape_objetivo:
        return array
    
    # Si es 3D con dimensión 1 en algún lado, squeeze
    if array.ndim == 3 and array.shape[2] == 1:
        return array[:, :, 0]
    
    # Si es 1D (lista de etiquetas), reshape (caso raro)
    if array.ndim == 1:
        # Intentar reshape si el tamaño coincide
        if array.size == shape_objetivo[0] * shape_objetivo[1]:
            return array.reshape(shape_objetivo)
    
    raise ValueError(f"Shape inesperado: {array.shape}, esperado: {shape_objetivo}")


def crear_segmentacion_multicanal(
    metodo: MetodoSegmentacion,
    tipo: TipoSegmentacion = Segmentacion_PorCorteEspaciotemporal()
) -> Callable[[BioImagenData], Resultado[BioImagenData, ErrorBioImagen]]:
    """Versión que segmenta todos los canales (útil para multicanal independiente)."""
    def _aplicar_multicanal(data: BioImagenData) -> Resultado[BioImagenData, ErrorBioImagen]:
        resultado: Resultado[BioImagenData, ErrorBioImagen] = Ok(data)
        
        for c in range(data.dims.C):
            seg_canal = crear_segmentacion(metodo, tipo, canal=c)
            resultado = resultado.bind(seg_canal)
            if resultado.es_err():
                break
        
        return resultado
    
    return _aplicar_multicanal


# ==================== WRAPPER ORIENTADO A OBJETOS ====================

class Controlador_Segmentador:
    """
    Wrapper stateful para segmentación de imágenes.
    Permite crear segmentadores configurados y aplicarlos.
    """
    
    def __init__(self):
        self._cache: Optional[Dict[str, Any]] = None
        self._ultimo_metodo: Optional[MetodoSegmentacion] = None
        self._ultima_mascara: Optional[np.ndarray] = None  # Cache de última segmentación
    
    # ===== BINARIZACIÓN =====
    
    def crear_otsu(
        self,
        tipo_aplicacion: TipoSegmentacion = Segmentacion_PorCorteEspaciotemporal()
    ) -> Callable[[BioImagenData], Resultado[BioImagenData, ErrorBioImagen]]:
        """Otsu: Umbral automático por maximización de varianza entre clases."""
        metodo = Otsu()
        self._ultimo_metodo = metodo
        return crear_segmentacion(metodo, tipo_aplicacion)
    
    def crear_umbral_global(
        self,
        umbral: float = 128.0,
        tipo_aplicacion: TipoSegmentacion = Segmentacion_PorCorteEspaciotemporal()
    ) -> Callable[[BioImagenData], Resultado[BioImagenData, ErrorBioImagen]]:
        """Umbral global manual (valor fijo)."""
        metodo = UmbralGlobal(umbral=umbral)
        self._ultimo_metodo = metodo
        return crear_segmentacion(metodo, tipo_aplicacion)
    
    def crear_umbral_adaptativo(
        self,
        tamano_bloque: int = 11,
        C: float = 2.0,
        metodo: str = "gaussian",
        tipo_aplicacion: TipoSegmentacion = Segmentacion_PorCorteEspaciotemporal()
    ) -> Callable[[BioImagenData], Resultado[BioImagenData, ErrorBioImagen]]:
        """Umbral adaptativo local (cambia según vecindario)."""
        metodo = UmbralAdaptativo(
            tamano_bloque=tamano_bloque,
            C=C,
            metodo=metodo
        )
        self._ultimo_metodo = metodo
        return crear_segmentacion(metodo, tipo_aplicacion)
    
    def crear_umbral_percentil(
        self,
        percentil: float = 95.0,
        tipo_aplicacion: TipoSegmentacion = Segmentacion_PorCorteEspaciotemporal()
    ) -> Callable[[BioImagenData], Resultado[BioImagenData, ErrorBioImagen]]:
        """Umbral basado en percentil (útil para señal sobre fondo)."""
        metodo = UmbralPercentil(percentil=percentil)
        self._ultimo_metodo = metodo
        return crear_segmentacion(metodo, tipo_aplicacion)
    
    def crear_triangle(
        self,
        nbins: int = 256,
        tipo_aplicacion: TipoSegmentacion = Segmentacion_PorCorteEspaciotemporal()
    ) -> Callable[[BioImagenData], Resultado[BioImagenData, ErrorBioImagen]]:
        """Triangle: Umbral para histogramas asimétricos (pico + cola larga)."""
        metodo = Triangle(nbins=nbins)
        self._ultimo_metodo = metodo
        return crear_segmentacion(metodo, tipo_aplicacion)
    
    def crear_yen(
        self,
        nbins: int = 256,
        tipo_aplicacion: TipoSegmentacion = Segmentacion_PorCorteEspaciotemporal()
    ) -> Callable[[BioImagenData], Resultado[BioImagenData, ErrorBioImagen]]:
        """Yen: Maximiza entropía de información (bueno para bimodal)."""
        metodo = Yen(nbins=nbins)
        self._ultimo_metodo = metodo
        return crear_segmentacion(metodo, tipo_aplicacion)
    
    def crear_li(
        self,
        tolerancia: float = 0.1,
        tipo_aplicacion: TipoSegmentacion = Segmentacion_PorCorteEspaciotemporal()
    ) -> Callable[[BioImagenData], Resultado[BioImagenData, ErrorBioImagen]]:
        """Li: Umbral mínimo cross-entropy."""
        metodo = Li(tolerancia=tolerancia)
        self._ultimo_metodo = metodo
        return crear_segmentacion(metodo, tipo_aplicacion)
    
    def crear_isodata(
        self,
        nbins: int = 256,
        tipo_aplicacion: TipoSegmentacion = Segmentacion_PorCorteEspaciotemporal()
    ) -> Callable[[BioImagenData], Resultado[BioImagenData, ErrorBioImagen]]:
        """ISODATA: Iterativo, similar a k-means con k=2."""
        metodo = ISODATA(nbins=nbins)
        self._ultimo_metodo = metodo
        return crear_segmentacion(metodo, tipo_aplicacion)
    
    def crear_minimum(
        self,
        nbins: int = 256,
        tipo_aplicacion: TipoSegmentacion = Segmentacion_PorCorteEspaciotemporal()
    ) -> Callable[[BioImagenData], Resultado[BioImagenData, ErrorBioImagen]]:
        """Minimum: Umbral en valle entre dos picos del histograma."""
        metodo = Minimum(nbins=nbins)
        self._ultimo_metodo = metodo
        return crear_segmentacion(metodo, tipo_aplicacion)
    
    def crear_mean(
        self,
        tipo_aplicacion: TipoSegmentacion = Segmentacion_PorCorteEspaciotemporal()
    ) -> Callable[[BioImagenData], Resultado[BioImagenData, ErrorBioImagen]]:
        """Mean: Umbral = media de intensidades."""
        metodo = Mean()
        self._ultimo_metodo = metodo
        return crear_segmentacion(metodo, tipo_aplicacion)
    
    # ===== INSTANCIALES (WATERSHED Y VARIANTES) =====
    
    def crear_watershed(
        self,
        marcadores: Optional[np.ndarray] = None,
        compactness: float = 0.0,
        tipo_aplicacion: TipoSegmentacion = Segmentacion_PorCorteEspaciotemporal()
    ) -> Callable[[BioImagenData], Resultado[BioImagenData, ErrorBioImagen]]:
        """
        Watershed clásico. Si no hay marcadores, usa mínimos locales de gradiente.
        Separa objetos tocándose mediante líneas de watershed.
        """
        metodo = Watershed(marcadores=marcadores, compactness=compactness)
        self._ultimo_metodo = metodo
        return crear_segmentacion(metodo, tipo_aplicacion)
    
    def crear_watershed_marcado(
        self,
        marcadores: np.ndarray,
        tipo_aplicacion: TipoSegmentacion = Segmentacion_PorCorteEspaciotemporal()
    ) -> Callable[[BioImagenData], Resultado[BioImagenData, ErrorBioImagen]]:
        """
        Watershed con marcadores definidos por usuario (semillas).
        Más controlado que automático.
        """
        metodo = WatershedMarcado(marcadores=marcadores)
        self._ultimo_metodo = metodo
        return crear_segmentacion(metodo, tipo_aplicacion)
    
    def crear_watershed_distancia(
        self,
        umbral_distancia: float = 0.5,
        tipo_aplicacion: TipoSegmentacion = Segmentacion_PorCorteEspaciotemporal()
    ) -> Callable[[BioImagenData], Resultado[BioImagenData, ErrorBioImagen]]:
        """
        Watershed en transformada de distancia.
        Útil para separar objetos redondeados que se tocan (células, núcleos).
        """
        metodo = DistanciaWatershed(umbral_distancia=umbral_distancia)
        self._ultimo_metodo = metodo
        return crear_segmentacion(metodo, tipo_aplicacion)
    
    def crear_split_distancial(
        self,
        min_distancia: float = 10.0,
        max_objetos: int = 1000,
        tipo_aplicacion: TipoSegmentacion = Segmentacion_PorCorteEspaciotemporal()
    ) -> Callable[[BioImagenData], Resultado[BioImagenData, ErrorBioImagen]]:
        """
        Split basado en picos de distancia (alternativa a watershed).
        Más rápido para objetos convexos.
        """
        metodo = SplitDistancial(
            min_distancia=min_distancia,
            max_objetos=max_objetos
        )
        self._ultimo_metodo = metodo
        return crear_segmentacion(metodo, tipo_aplicacion)
    
    def crear_watershed_hibrido(
        self,
        umbral_binarizacion: float = 0.5,
        compactness: float = 1.0,
        tipo_aplicacion: TipoSegmentacion = Segmentacion_PorCorteEspaciotemporal()
    ) -> Callable[[BioImagenData], Resultado[BioImagenData, ErrorBioImagen]]:
        """
        Watershed híbrido: combina binarización + distancia + watershed.
        Pipeline completo de separación de objetos en uno solo.
        """
        metodo = WatershedHibrido(
            umbral_binarizacion=umbral_binarizacion,
            compactness=compactness
        )
        self._ultimo_metodo = metodo
        return crear_segmentacion(metodo, tipo_aplicacion)
    
    def crear_split_watershed(
        self,
        metodo_split: str = "distancia",
        tipo_aplicacion: TipoSegmentacion = Segmentacion_PorCorteEspaciotemporal()
    ) -> Callable[[BioImagenData], Resultado[BioImagenData, ErrorBioImagen]]:
        """
        Split + Watershed en cascada (para objetos muy agrupados).
        """
        metodo = SplitWatershed(metodo_split=metodo_split)
        self._ultimo_metodo = metodo
        return crear_segmentacion(metodo, tipo_aplicacion)
    
    # ===== REGIONALES =====
    
    def crear_region_growing(
        self,
        semillas: List[Tuple[int, int]],
        criterio: str = "intensidad",
        tolerancia: float = 10.0,
        tipo_aplicacion: TipoSegmentacion = Segmentacion_PorCorteEspaciotemporal()
    ) -> Callable[[BioImagenData], Resultado[BioImagenData, ErrorBioImagen]]:
        """
        Crecimiento de regiones desde semillas (semiautomático).
        Útil cuando el usuario marca puntos de interés.
        """
        metodo = RegionGrowing(
            semillas=semillas,
            criterio=criterio,
            tolerancia=tolerancia
        )
        self._ultimo_metodo = metodo
        return crear_segmentacion(metodo, tipo_aplicacion)
    
    def crear_random_walk(
        self,
        semillas: np.ndarray,
        beta: float = 130.0,
        modo: str = "bf",
        tipo_aplicacion: TipoSegmentacion = Segmentacion_PorCorteEspaciotemporal()
    ) -> Callable[[BioImagenData], Resultado[BioImagenData, ErrorBioImagen]]:
        """
        Random Walk: propagación probabilística desde semillas.
        Robusto a ruido, bueno para bordes difusos.
        """
        metodo = RandomWalk(semillas=semillas, beta=beta, modo=modo)
        self._ultimo_metodo = metodo
        return crear_segmentacion(metodo, tipo_aplicacion)
    
    def crear_corte_grafico(
        self,
        grafo: Optional[Any] = None,  # Graph cuts necesita grafo de adyacencia
        etiquetas_fuente: Optional[np.ndarray] = None,
        etiquetas_sumidero: Optional[np.ndarray] = None,
        tipo_aplicacion: TipoSegmentacion = Segmentacion_PorCorteEspaciotemporal()
    ) -> Callable[[BioImagenData], Resultado[BioImagenData, ErrorBioImagen]]:
        """
        Corte de grafo (graph cuts) - optimización global.
        Matemáticamente óptimo para ciertas funciones de energía.
        """
        metodo = CorteGrafico(
            grafo=grafo,
            etiquetas_fuente=etiquetas_fuente,
            etiquetas_sumidero=etiquetas_sumidero
        )
        self._ultimo_metodo = metodo
        return crear_segmentacion(metodo, tipo_aplicacion)
    
    def crear_slic(
        self,
        n_segmentos: int = 100,
        compactness: float = 10.0,
        sigma: float = 1.0,
        tipo_aplicacion: TipoSegmentacion = Segmentacion_Global()
    ) -> Callable[[BioImagenData], Resultado[BioImagenData, ErrorBioImagen]]:
        """
        SLIC: Superpixels compactos y uniformes.
        Presegmentación para algoritmos posteriores.
        """
        metodo = SuperpixelSLIC(
            n_segmentos=n_segmentos,
            compactness=compactness,
            sigma=sigma
        )
        self._ultimo_metodo = metodo
        return crear_segmentacion(metodo, tipo_aplicacion)
    
    def crear_felzenszwalb(
        self,
        escala: float = 1.0,
        sigma: float = 0.8,
        min_size: int = 20,
        tipo_aplicacion: TipoSegmentacion = Segmentacion_Global()
    ) -> Callable[[BioImagenData], Resultado[BioImagenData, ErrorBioImagen]]:
        """
        Felzenszwalb: Superpixels por grafos (no necesariamente uniformes).
        Respeta bordes mejor que SLIC en algunos casos.
        """
        metodo = SuperpixelFelzenszwalb(
            escala=escala,
            sigma=sigma,
            min_size=min_size
        )
        self._ultimo_metodo = metodo
        return crear_segmentacion(metodo, tipo_aplicacion)
    
    def crear_watershed_marcas(
        self,
        gradiente: Optional[np.ndarray] = None,
        marcadores: Optional[np.ndarray] = None,
        tipo_aplicacion: TipoSegmentacion = Segmentacion_PorCorteEspaciotemporal()
    ) -> Callable[[BioImagenData], Resultado[BioImagenData, ErrorBioImagen]]:
        """
        Watershed en gradiente con marcas (variante scikit-image).
        Similar a watershed_marcado pero con más control de gradiente.
        """
        metodo = WatershedMarcas(
            gradiente=gradiente,
            marcadores=marcadores
        )
        self._ultimo_metodo = metodo
        return crear_segmentacion(metodo, tipo_aplicacion)
    
    def crear_mean_shift(
        self,
        ancho_banda_espacial: float = 15.0,
        ancho_banda_rango: float = 15.0,
        min_densidad: int = 50,
        tipo_aplicacion: TipoSegmentacion = Segmentacion_Global()
    ) -> Callable[[BioImagenData], Resultado[BioImagenData, ErrorBioImagen]]:
        """
        Mean Shift: Clustering en espacio color-espacial.
        No necesita número de clusters predefinido.
        """
        metodo = MeanShift(
            ancho_banda_espacial=ancho_banda_espacial,
            ancho_banda_rango=ancho_banda_rango,
            min_densidad=min_densidad
        )
        self._ultimo_metodo = metodo
        return crear_segmentacion(metodo, tipo_aplicacion)
    
    # ===== MÉTODOS DE APLICACIÓN DIRECTA =====
    
    def aplicar(
        self,
        data: BioImagenData,
        metodo: MetodoSegmentacion,
        tipo: TipoSegmentacion = Segmentacion_PorCorteEspaciotemporal(),
        canal: int = 0
    ) -> Resultado[BioImagenData, ErrorBioImagen]:
        """Aplica segmentación específica a datos (versión imperativa)."""
        segmentacion = crear_segmentacion(metodo, tipo, canal)
        resultado = segmentacion(data)
        
        # Cachear máscara si es exitosa
        if resultado.es_ok():
            self._ultima_mascara = resultado.unwrap().datos[:, :, canal, :, :]
        
        return resultado
    
    def aplicar_multicanal(
        self,
        data: BioImagenData,
        metodo: MetodoSegmentacion,
        tipo: TipoSegmentacion = Segmentacion_PorCorteEspaciotemporal()
    ) -> Resultado[BioImagenData, ErrorBioImagen]:
        """Segmenta todos los canales."""
        segmentacion = crear_segmentacion_multicanal(metodo, tipo)
        return segmentacion(data)
    
    def obtener_ultima_mascara(self) -> Optional[np.ndarray]:
        """Retorna la última máscara de segmentación calculada."""
        return self._ultima_mascara
    
    def reset(self):
        """Limpia caché interno."""
        self._cache = None
        self._ultimo_metodo = None
        self._ultima_mascara = None
    
    def __repr__(self) -> str:
        ultimo = getattr(self._ultimo_metodo, 'nombre', 'Ninguno')
        return f"<Controlador_Segmentador ultimo={ultimo}>"


# ==================== FACTORIES PARA PIPELINES ====================

def operacion_segmentacion(
    metodo: MetodoSegmentacion,
    tipo: TipoSegmentacion = Segmentacion_PorCorteEspaciotemporal(),
    canal: int = 0,
    nombre: Optional[str] = None
) -> Operacion:
    """
    Crea Operación de segmentación lista para PipelineBuilder.
    
    Uso:
        pipeline = (
            PipelineBuilder()
            .segmentar("otsu_auto",
                operacion_segmentacion(Otsu(), Segmentacion_PorCorteEspaciotemporal(), canal=0))
            .segmentar("watershed_separacion",
                operacion_segmentacion(WatershedMarcado(marcadores=mis_marcas), ...))
            .construir()
        )
    """
    nombre_op = nombre or f"segmentacion_{metodo.__class__.__name__}_{tipo.__class__.__name__}"
    
    # Detectar si es punto de split (instanciales típicamente generan máscaras para cuantificar)
    es_split = metodo.__class__.__name__ in {
        'Watershed', 'WatershedMarcado', 'DistanciaWatershed',
        'SplitDistancial', 'WatershedHibrido', 'SplitWatershed'
    }
    
    segmentacion_callable = crear_segmentacion(metodo, tipo)

    return Operacion(
        nombre=nombre_op,
        categoria=CategoriaOperacion.SEGMENTADOR,
        instancia_callable=segmentacion_callable,
        canal_objetivo=canal,
        parametros_originales={
            "metodo": metodo.__class__.__name__,
            "tipo": tipo.__class__.__name__,
            "submodulo": _detectar_submodulo_segmentacion(metodo),
            "params": _extraer_parametros_segmentacion(metodo)
        },
        tipo_salida=TipoSalida.MASCARA,  # Importante: marca como máscara/etiquetas
        es_operacion_split=es_split  # Marca punto de bifurcación potencial
    )


def _detectar_submodulo_segmentacion(metodo: MetodoSegmentacion) -> str:
    """Detecta a qué submódulo pertenece el método."""
    clase = metodo.__class__.__name__
    mapeo = {
        # Binarización
        'Otsu': 'binarizacion', 'UmbralGlobal': 'binarizacion',
        'UmbralAdaptativo': 'binarizacion', 'UmbralPercentil': 'binarizacion',
        'Triangle': 'binarizacion', 'Yen': 'binarizacion', 'Li': 'binarizacion',
        'ISODATA': 'binarizacion', 'Minimum': 'binarizacion', 'Mean': 'binarizacion',
        # Instanciales
        'Watershed': 'instancial', 'WatershedMarcado': 'instancial',
        'DistanciaWatershed': 'instancial', 'SplitDistancial': 'instancial',
        'WatershedHibrido': 'instancial', 'SplitWatershed': 'instancial',
        # Regionales
        'RegionGrowing': 'regional', 'RandomWalk': 'regional',
        'CorteGrafico': 'regional', 'SuperpixelSLIC': 'regional',
        'SuperpixelFelzenszwalb': 'regional', 'WatershedMarcas': 'regional',
        'MeanShift': 'regional',
    }
    return mapeo.get(clase, 'desconocido')


def _extraer_parametros_segmentacion(metodo: MetodoSegmentacion) -> Dict[str, Any]:
    """Extrae parámetros del método para metadata."""
    params = {}
    atributos_comunes = [
        'umbral', 'nbins', 'tolerancia', 'percentil', 'tamano_bloque',
        'compactness', 'beta', 'n_segmentos', 'sigma', 'escala', 'min_size',
        'ancho_banda_espacial', 'ancho_banda_rango', 'min_densidad',
        'criterio', 'metodo_split', 'min_distancia', 'max_objetos'
    ]
    for attr in atributos_comunes:
        if hasattr(metodo, attr):
            val = getattr(metodo, attr)
            # Convertir arrays a listas para serialización JSON
            if isinstance(val, np.ndarray):
                params[attr] = val.tolist()
            else:
                params[attr] = val
    return params


# Factory conveniente desde Controlador_Segmentador
def operacion_segmentacion_desde_controlador(
    controlador: Controlador_Segmentador,
    metodo_factory: str,
    params: Dict[str, Any],
    canal: int = 0
) -> Resultado[Operacion, ErrorBioImagen]:
    """
    Crea operación usando el controlador (valida que el método exista).
    
    Uso:
        segmentador = Controlador_Segmentador()
        op_result = operacion_segmentacion_desde_controlador(
            segmentador, "crear_otsu", {}, canal=0)
        if op_result.es_ok():
            builder.segmentar("mi_otsu", op_result.unwrap())
    """
    try:
        if not hasattr(controlador, metodo_factory):
            return Err(ErrorBioImagen(
                etapa="configuracion",
                mensaje=f"Controlador_Segmentador no tiene método '{metodo_factory}'"
            ))
        
        factory = getattr(controlador, metodo_factory)
        seg_callable = factory(**params)
        
        metodo_subyacente = controlador._ultimo_metodo
        
        return Ok(Operacion(
            nombre=metodo_factory.replace("crear_", ""),
            categoria=CategoriaOperacion.SEGMENTADOR,
            instancia_callable=metodo_subyacente,
            canal_objetivo=canal,
            parametros_originales=params,
            tipo_salida=TipoSalida.MASCARA,
            es_operacion_split=metodo_factory in {
                'crear_watershed', 'crear_watershed_marcado', 'crear_watershed_distancia',
                'crear_split_distancial', 'crear_watershed_hibrido', 'crear_split_watershed'
            }
        ))
        
    except Exception as e:
        return Err(ErrorBioImagen(
            etapa="configuracion",
            mensaje=f"Error creando operación desde controlador: {str(e)}",
            causa=e
        ))