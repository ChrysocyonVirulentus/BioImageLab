# === gestorLab/Operacion.py ===
from dataclasses import dataclass, field
from typing import Callable, Dict, Any, Optional, Union
from enum import Enum
import numpy as np

from ..controlador.Resultado_Either import Resultado
from ..controlador.Controlador_BioImagen import BioImagenData, ErrorBioImagen
from .Categoria_Operacion import CategoriaOperacion

class TipoSalida(Enum):
    IMAGEN = "imagen"           # Retorna BioImagenData modificada
    MASCARA = "mascara"         # Retorna BioImagenData binaria/etiquetada
    FEATURES = "features"       # Retorna DataFrame/dict de métricas
    TABLA = "tabla"            # Retorna DataFrame
    MODELO = "modelo"          # Retorna objeto entrenado
    VISUALIZACION = "viz"      # Retorna figura/array para plot
    NINGUNA = "ninguna"        # Side-effect (exportar archivo)

@dataclass(frozen=True)
class Operacion:
    """
    Operación pura adaptada para el pipeline.
    El callable interno viene de tus módulos específicos (ej: Filtros_Ffts.FFTPasabajo)
    """
    nombre: str                      # "fft_pasabajo"
    categoria: CategoriaOperacion    # FILTRACION
    instancia_callable: Callable[[np.ndarray], np.ndarray]  # El objeto callable de tu módulo
    canal_objetivo: Optional[int] = None
    
    # Metadatos
    parametros_originales: Dict[str, Any] = field(default_factory=dict)
    tipo_salida: TipoSalida = TipoSalida.IMAGEN
    descripcion: str = ""
    es_operacion_split: bool = False  # Marca si esta op genera una bifurcación
    requiere_input_especial: Optional[str] = None  # "mascara", "imagen_original", etc.
    
    def ejecutar(self, data: BioImagenData) -> Resultado[BioImagenData, ErrorBioImagen]:
        from dataclasses import replace
        
        try:
            T, Z, C, Y, X = data.dims.shape
            canales = [self.canal_objetivo] if self.canal_objetivo is not None else range(C)
            
            if self.canal_objetivo is not None and not (0 <= self.canal_objetivo < C):
                return Err(ErrorBioImagen(
                    etapa="operacion",
                    mensaje=f"{self.nombre}: canal {self.canal_objetivo} fuera de rango [0, {C-1}]"
                ))
            
            nuevos_datos = data.datos.copy().astype(np.float64)
            
            for c in canales:
                for t in range(T):
                    for z in range(Z):
                        corte = data.datos[t, z, c, :, :]
                        resultado = self.instancia_callable(corte)
                        
                        if resultado.shape != (Y, X):
                            return Err(ErrorBioImagen(
                                etapa="operacion",
                                mensaje=f"{self.nombre} cambió shape: {(Y,X)} -> {resultado.shape}"
                            ))
                        
                        nuevos_datos[t, z, c, :, :] = resultado
            
            return Ok(replace(data, datos=nuevos_datos))
            
        except Exception as e:
            return Err(ErrorBioImagen(
                etapa="operacion",
                mensaje=f"Error en {self.nombre}: {str(e)}",
                causa=e
            ))
    
    def __repr__(self) -> str:
        canal = f"[C{self.canal_objetivo}]" if self.canal_objetivo else "[C*]"
        split = " [SPLIT]" if self.es_operacion_split else ""
        return f"{self.categoria.name}::{self.nombre}{canal}{split}"