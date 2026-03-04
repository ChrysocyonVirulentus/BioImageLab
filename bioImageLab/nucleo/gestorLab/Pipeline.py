# === gestorLab/Pipeline.py ===
from dataclasses import dataclass, field
from typing import Tuple, List, Callable, Optional, Dict, Any
from enum import Enum

from ..controlador.Resultado_Either import Resultado, Ok, Err 
from ..controlador.Controlador_BioImagen import BioImagenData, ErrorBioImagen
from .Operacion import Operacion, TipoSalida
from .Categoria_Operacion import CategoriaOperacion, RequisitoRama

class EstadoRama(Enum):
    ACTIVA = "activa"           # Ejecutando normalmente
    BLOQUEADA = "bloqueada"     # Esperando merge con otra rama
    COMPLETADA = "completada"   # Finalizó, tiene resultado
    ERROR = "error"             # Falló ejecución

@dataclass
class ContextoRama:
    """
    Estado de una rama de procesamiento.
    Las ramas mantienen su propio historial y pueden esperar datos de otras.
    """
    nombre: str  # "principal", "segmentada", "cuantificacion_intensidad"
    operaciones_pendientes: Tuple[Operacion, ...]
    data_actual: Optional[BioImagenData] = None
    estado: EstadoRama = EstadoRama.ACTIVA
    dependencias_requeridas: Dict[str, Any] = field(default_factory=dict)  # Lo que necesita de otras ramas
    resultados_intermedios: Dict[str, Any] = field(default_factory=dict)   # Lo que exporta para otras ramas
    
    def puede_continuar(self) -> bool:
        """Verifica si todas las dependencias están satisfechas"""
        return self.estado == EstadoRama.ACTIVA and not self.dependencias_requeridas

@dataclass(frozen=True)
class Pipeline:
    """
    Pipeline que puede contener operaciones de bifurcación.
    No ejecuta directamente, solo define la secuencia.
    """
    nombre: str
    operaciones: Tuple[Operacion, ...]
    puntos_split: Tuple[int, ...] = field(default_factory=tuple)  # Índices donde ocurren splits
    
    def __post_init__(self):
        # Detectar automáticamente splits
        splits = tuple(
            i for i, op in enumerate(self.operaciones) 
            if op.es_operacion_split or op.categoria.es_punto_split
        )
        object.__setattr__(self, 'puntos_split', splits)
    
    def obtener_ramas_implicitas(self) -> List['Pipeline']:
        """
        Si este pipeline tiene splits, retorna los sub-pipelines por rama.
        """
        if not self.puntos_split:
            return [self]
        
        # Lógica para dividir en ramas...
        ramas = []
        inicio = 0
        
        for split_idx in self.puntos_split:
            # Rama hasta el split
            rama_principal = Pipeline(
                f"{self.nombre}_hasta_split_{split_idx}",
                self.operaciones[inicio:split_idx+1]
            )
            ramas.append(rama_principal)
            
            # Ramas divergentes después del split
            # Esto requiere análisis de las operaciones post-split
            # para determinar cuántas ramas hay (ej: procesada vs segmentada)
            
        return ramas
    
    def __len__(self) -> int:
        return len(self.operaciones)
    
    def __repr__(self) -> str:
        splits = f" [splits: {self.puntos_split}]" if self.puntos_split else ""
        return f"Pipeline({self.nombre})[{len(self)} ops]{splits}"