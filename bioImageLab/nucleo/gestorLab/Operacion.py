# === gestorLab/Operacion.py ===
from dataclasses import dataclass, field
from typing import Callable, Dict, Any, Optional, Union
from enum import Enum
import numpy as np

from ..controlador.Resultado_Either import Resultado, Err, Ok
from ..controlador.Controlador_BioImagen import BioImagenData, ErrorBioImagen
from .Categoria_Operacion import CategoriaOperacion

class TipoSalida(Enum):
    IMAGEN = "imagen"
    MASCARA = "mascara"
    FEATURES = "features"
    TABLA = "tabla"
    MODELO = "modelo"
    VISUALIZACION = "viz"
    NINGUNA = "ninguna"

@dataclass(frozen=True)
class Operacion:
    """
    Operación pura adaptada para el pipeline.
    El callable interno viene de tus módulos específicos.
    """
    nombre: str
    categoria: CategoriaOperacion
    instancia_callable: Callable[[BioImagenData, int], Resultado[BioImagenData, ErrorBioImagen]]
    canal_objetivo: Optional[int] = None
    
    parametros_originales: Dict[str, Any] = field(default_factory=dict)
    tipo_salida: TipoSalida = TipoSalida.IMAGEN
    descripcion: str = ""
    es_operacion_split: bool = False
    requiere_input_especial: Optional[str] = None
    
    def ejecutar(self, data: BioImagenData) -> Resultado[BioImagenData, ErrorBioImagen]:
        """
        Ejecuta la operación sobre el BioImagenData completo.
        El callable recibe (BioImagenData, canal_idx) y retorna Resultado.
        """
        try:
            # Determinar qué canal procesar
            if self.canal_objetivo is not None:
                if not (0 <= self.canal_objetivo < data.dims.C):
                    return Err(ErrorBioImagen(
                        etapa="operacion",
                        mensaje=f"{self.nombre}: canal {self.canal_objetivo} fuera de rango [0, {data.dims.C-1}]"
                    ))
                canales = [self.canal_objetivo]
            else:
                canales = range(data.dims.C)
            
            # Aplicar operación canal por canal usando el callable
            resultado_actual: Resultado[BioImagenData, ErrorBioImagen] = Ok(data)
            
            for c in canales:
                # El callable recibe BioImagenData y retorna Resultado[BioImagenData, ErrorBioImagen]
                resultado_canal = self.instancia_callable(resultado_actual.unwrap(), c)
                
                if resultado_canal.es_err():
                    return resultado_canal  # Propagar error
                    
                resultado_actual = resultado_canal
            
            return resultado_actual
            
        except Exception as e:
            return Err(ErrorBioImagen(
                etapa="operacion",
                mensaje=f"Error en {self.nombre}: {str(e)}",
                causa=e
            ))
    
    def __repr__(self) -> str:
        canal = f"[C{self.canal_objetivo}]" if self.canal_objetivo is not None else "[C*]"
        split = " [SPLIT]" if self.es_operacion_split else ""
        return f"{self.categoria.name}::{self.nombre}{canal}{split}"