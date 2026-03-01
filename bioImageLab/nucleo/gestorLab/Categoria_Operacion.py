# === nucleo/pipeline/CategoriaOperacion.py ===
from enum import Enum, auto

class CategoriaOperacion(Enum):
    PREPROCESAMIENTO = auto()   # 1 - Normalización, corrección iluminación
    FILTRACION = auto()         # 2 - Reducción ruido (espacial y espectral)
    REALZADOR = auto()          # 3 - Mejora contraste, bordes
    TRANSFORMADOR = auto()      # 4 - Geométricos, esqueletización
    SEGMENTADOR = auto()        # 5 - Umbralización, watershed
    CUANTIFICADOR = auto()      # 6 - Métricas, conteo
    MODELADOR = auto()          # 7 - PCA, ML
    ANALIZADOR = auto()         # 8 - Visualización, export
    
    def orden_ejecucion(self) -> int:
        return list(CategoriaOperacion).index(self) + 1
    
    def puede_preceder_a(self, otra: 'CategoriaOperacion') -> bool:
        return self.orden_ejecucion() <= otra.orden_ejecucion()