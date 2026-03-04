# === gestorLab/Categoria_Operacion.py ===
from enum import Enum, auto
from typing import Set, List, Optional, Dict
from dataclasses import dataclass

class CategoriaOperacion(Enum):
    """
    Categorías con dependencias explícitas y reglas de split/merge.
    """
    PREPROCESAMIENTO = auto()   # 1 - Base, no split
    FILTRACION = auto()         # 2 - Base, no split  
    REALZADOR = auto()          # 3 - Base, no split
    TRANSFORMADOR = auto()      # 4 - Base, no split
    SEGMENTADOR = auto()        # 5 - Genera máscara, punto de split potencial
    CUANTIFICADOR = auto()      # 6 - Requiere imagen + máscara (merge de ramas)
    MODELADOR = auto()          # 7 - Trabaja con features/tablas
    ANALIZADOR = auto()         # 8 - Output final, no split
    
    def orden_ejecucion(self) -> int:
        return list(CategoriaOperacion).index(self) + 1
    
    @property
    def es_punto_split(self) -> bool:
        """Indica si esta categoría típicamente genera ramas divergentes"""
        return self in {
            CategoriaOperacion.SEGMENTADOR,  # Imagen original vs. Máscara
            CategoriaOperacion.CUANTIFICADOR,  # Features → diferentes análisis
        }
    
    @property
    def requiere_merge(self) -> bool:
        """Indica si necesita datos de múltiples ramas"""
        return self == CategoriaOperacion.CUANTIFICADOR
    
    @property
    def dependencias_estrictas(self) -> Set['CategoriaOperacion']:
        deps = {
            CategoriaOperacion.PREPROCESAMIENTO: set(),
            CategoriaOperacion.FILTRACION: {CategoriaOperacion.PREPROCESAMIENTO},
            CategoriaOperacion.REALZADOR: {CategoriaOperacion.PREPROCESAMIENTO},
            CategoriaOperacion.TRANSFORMADOR: {CategoriaOperacion.PREPROCESAMIENTO},
            CategoriaOperacion.SEGMENTADOR: {
                CategoriaOperacion.PREPROCESAMIENTO,
                CategoriaOperacion.FILTRACION,
                CategoriaOperacion.REALZADOR
            },
            CategoriaOperacion.CUANTIFICADOR: {
                CategoriaOperacion.PREPROCESAMIENTO,
                CategoriaOperacion.SEGMENTADOR  # Necesita máscara de segmentación
            },
            CategoriaOperacion.MODELADOR: {
                CategoriaOperacion.CUANTIFICADOR
            },
            CategoriaOperacion.ANALIZADOR: set(),  # Flexible
        }
        return deps.get(self, set())
    
    def validar_dependencias(self, presentes: Set['CategoriaOperacion']) -> tuple[bool, List[str]]:
        errores = []
        faltantes = self.dependencias_estrictas - presentes
        
        if faltantes:
            nombres = [c.name for c in faltantes]
            errores.append(f"{self.name} REQUIERE: {', '.join(nombres)}")
        
        return (not faltantes, errores)


@dataclass(frozen=True)
class RequisitoRama:
    """
    Define qué necesita una rama específica para operar.
    Ejemplo: rama de cuantificación necesita imagen_procesada + mascara_segmentada
    """
    nombre: str  # "imagen_procesada", "mascara_segmentada", "features"
    tipo_dato: str  # "BioImagenData", "np.ndarray", "DataFrame", "List[Region]"
    origen_categoria: Optional[CategoriaOperacion] = None  # De qué etapa viene
    descripcion: str = ""