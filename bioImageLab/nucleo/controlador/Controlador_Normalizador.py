from __future__ import annotations
import numpy as np
from dataclasses import dataclass, replace
from typing import Union, Callable, Optional, Tuple
from enum import Enum, auto

# Imports de tu sistema
from ..preprocesador.normalizador.Metodos_Normalizacion import (
    MetodoNormalizacion, 
    MaxNorm, 
    MinMaxNorm, 
    PercentilNorm, 
    ZScoreNorm
)

from ..preprocesador.normalizador.Metodos_CambioTipos import (
    ToUint8,
    ToUint16,
    ToFloat32,
    ToFloat64
)

from .Resultado_Either import Resultado, Err, Ok
from .Controlador_BioImagen import (
    BioImagenData, 
    ErrorBioImagen
)
from ..gestorLab.Categoria_Operacion import CategoriaOperacion
from ..gestorLab.Operacion import Operacion

# ESTRATEGIAS DE NORMALIZACIÓN (Tipos Algebraicos)

@dataclass(frozen=True)
class Norm_Global:
    """Normaliza todo el volumen espacio-temporal del canal como un único conjunto de datos."""
    pass

@dataclass(frozen=True)
class Norm_PorCorteZ:
    """
    Normaliza cada plano Z independientemente.
    Útil cuando la iluminación varía con la profundidad pero no entre timepoints.
    """
    pass

@dataclass(frozen=True)
class Norm_PorTimepoint:
    """
    Normaliza cada fotograma T independientemente (incluyendo todos sus Z).
    Útil para corregir fotobleaching o cambios de iluminación en el tiempo.
    """
    pass

@dataclass(frozen=True)
class Norm_PorCorteEspaciotemporal:
    """
    Normaliza cada corte (t, z) de forma completamente independiente.
    Máxima granularidad, útil para estabilizar señal en time-lapse largos.
    """
    pass

TipoNormalizacion = Union[
    Norm_Global, 
    Norm_PorCorteZ, 
    Norm_PorTimepoint, 
    Norm_PorCorteEspaciotemporal
]


# FUNCIONES PURAS

def crear_normalizador(
    tipo: TipoNormalizacion = Norm_Global(),
    metodo: MetodoNormalizacion = MaxNorm()
) -> Callable[[BioImagenData, int], Resultado[BioImagenData, ErrorBioImagen]]:
    """
    Factory curried que retorna una función pura de normalización.
    
    Esta es la versión recomendada para pipelines funcionales.
    Retorna una función que puede usarse con .bind() o .map()
    
    Args:
        tipo: Estrategia de normalización espacio-temporal
        metodo: Algoritmo de normalización (MaxNorm, ZScore, etc.)
        
    Returns:
        Callable[[BioImagenData, int], Resultado[BioImagenData, ErrorBioImagen]]
    """
    def _normalizar_canal(
        data: BioImagenData, 
        canal_idx: int = 0
    ) -> Resultado[BioImagenData, ErrorBioImagen]:
        
        # Validaciones
        if not (0 <= canal_idx < data.dims.C):
            return Err(ErrorBioImagen(
                etapa="normalizacion",
                mensaje=f"Canal {canal_idx} fuera de rango [0, {data.dims.C-1}]",
                ruta=data.ruta_origen
            ))
        
        try:
            T, Z, C, Y, X = data.dims.shape
            
            # Extraer canal como [T, Z, Y, X] - quitamos dimensión C
            canal_data = data.datos[:, :, canal_idx, :, :]  # Shape: (T, Z, Y, X)
            
            # Preparar array de resultado manteniendo dimensión C=1
            resultado_canal = np.zeros((T, Z, 1, Y, X), dtype=np.float64)
            
            # Aplicar estrategia según tipo algebraico
            match tipo:
                case Norm_Global():
                    # Aplanar todo el volumen del canal: [T*Z*Y*X]
                    flat = canal_data.reshape(-1)
                    normalizado_flat = metodo(flat)
                    resultado_canal[:, :, 0, :, :] = normalizado_flat.reshape(T, Z, Y, X)
                    
                case Norm_PorCorteZ():
                    # Por cada Z, normalizar todo el tiempo y espacio
                    for z in range(Z):
                        corte_z = canal_data[:, z, :, :]  # [T, Y, X]
                        flat_z = corte_z.reshape(-1)
                        normalizado_z = metodo(flat_z).reshape(T, Y, X)
                        resultado_canal[:, z, 0, :, :] = normalizado_z
                        
                case Norm_PorTimepoint():
                    # Por cada T, normalizar todo Z y espacio
                    for t in range(T):
                        frame_t = canal_data[t, :, :, :]  # [Z, Y, X]
                        flat_t = frame_t.reshape(-1)
                        normalizado_t = metodo(flat_t).reshape(Z, Y, X)
                        resultado_canal[t, :, 0, :, :] = normalizado_t
                        
                case Norm_PorCorteEspaciotemporal():
                    # Cada (t, z) independiente
                    for t in range(T):
                        for z in range(Z):
                            corte = canal_data[t, z, :, :]  # [Y, X]
                            resultado_canal[t, z, 0, :, :] = metodo(corte)
            
            # Reconstruir BioImagenData con canal normalizado reemplazado
            nuevos_datos = data.datos.copy().astype(np.float64)
            nuevos_datos[:, :, canal_idx, :, :] = resultado_canal[:, :, 0, :, :]
            
            # Preservar metadatos, actualizar datos
            return Ok(replace(data, datos=nuevos_datos))
            
        except Exception as e:
            return Err(ErrorBioImagen(
                etapa="normalizacion",
                mensaje=f"Fallo en {metodo.nombre} con {tipo.__class__.__name__}: {str(e)}",
                ruta=data.ruta_origen,
                causa=e
            ))
    
    return _normalizar_canal


def normalizar_todos_canales(
    tipo: TipoNormalizacion = Norm_Global(),
    metodo: MetodoNormalizacion = MaxNorm()
) -> Callable[[BioImagenData], Resultado[BioImagenData, ErrorBioImagen]]:
    """
    Versión que normaliza todos los canales secuencialmente con la misma estrategia.
    Útil cuando quieres aplicar el mismo criterio a todos los fluoróforos.
    """
    def _normalizar_todos(data: BioImagenData) -> Resultado[BioImagenData, ErrorBioImagen]:
        resultado: Resultado[BioImagenData, ErrorBioImagen] = Ok(data)
        
        for c in range(data.dims.C):
            normalizador_canal = crear_normalizador(tipo, metodo)
            resultado = resultado.bind(lambda d, canal=c: normalizador_canal(d, canal))
            if resultado.es_err():
                break
                
        return resultado
    
    return _normalizar_todos


# WRAPPER ORIENTADO A OBJETOS

class Normalizador:
    """
    Wrapper stateful para compatibilidad con código existente.
    Internamente usa funciones puras.
    
    Para código nuevo, preferir usar directamente crear_normalizador() 
    que retorna funciones composables.
    """
    
    def __init__(
        self, 
        tipo: TipoNormalizacion = Norm_Global(),
        metodo: MetodoNormalizacion = MaxNorm()
    ):
        self.tipo = tipo
        self.metodo = metodo
        self._funcion_pura = crear_normalizador(tipo, metodo)
        self._cache: Optional[np.ndarray] = None

    def __call__(
        self,
        img_5d: np.ndarray,
        canal: int = 0,
        z_ref: int = 0,  # Mantenido por compatibilidad, no usado
        t_ref: int = 0,  # Mantenido por compatibilidad, no usado
    ) -> Resultado[np.ndarray, ErrorBioImagen]:
        """
        Versión compatible que trabaja con arrays numpy puros.
        Retorna Resultado explícito (no np.ndarray directo).
        
        Para integración en pipelines funcionales, usar .aplicar() en su lugar.
        """
        # Crear BioImagenData temporal mínimo
        T, Z, C, Y, X = img_5d.shape
        data_temp = BioImagenData(
            datos=img_5d,
            dims=Dimensiones(T, Z, C, Y, X),  # Necesitas importar Dimensiones
            canales=tuple(f"Canal_{i}" for i in range(C)),
            ruta_origen=Path("temporal"),
            es_bioformato=False
        )
        
        resultado = self._funcion_pura(data_temp, canal)
        
        # Extraer array del resultado
        return resultado.map(lambda d: d.datos[:, :, canal, :, :])

    def aplicar(
        self,
        data: BioImagenData,
        canal: int = 0
    ) -> Resultado[BioImagenData, ErrorBioImagen]:
        """
        Versión nativa funcional. Usar esta para pipelines.
        """
        return self._funcion_pura(data, canal)

    def aplicar_todos(
        self,
        data: BioImagenData
    ) -> Resultado[BioImagenData, ErrorBioImagen]:
        """
        Aplica la normalización a todos los canales.
        """
        return normalizar_todos_canales(self.tipo, self.metodo)(data)

    def reset(self):
        """Limpia caché interna."""
        self._cache = None

    def __repr__(self) -> str:
        return f"<Normalizador tipo={self.tipo.__class__.__name__} metodo={self.metodo.nombre}>"


# ==================== FACTORY PARA PIPELINES ====================

def operacion_normalizacion(
    tipo: TipoNormalizacion = Norm_Global(),
    metodo: MetodoNormalizacion = MaxNorm(),
    canal: int = 0,
    nombre: str = ""
) -> Operacion:

    normalizador = crear_normalizador(tipo, metodo)

    return Operacion(
        nombre=nombre or f"normalizacion_{metodo.nombre}_{tipo.__class__.__name__}",
        categoria=CategoriaOperacion.PREPROCESAMIENTO,
        instancia_callable=normalizador,
        canal_objetivo=canal,
        parametros_originales={
            "tipo": tipo.__class__.__name__,
            "metodo": metodo.nombre,
            "canal": canal
        }
    )