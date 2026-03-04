# === nucleo/pipeline/CategoriaOperacion.py ===
from enum import Enum, auto

class CategoriaOperacion(Enum):
    PREPROCESAMIENTO = auto()   # 1 - Base fundamental (Normalizacion)
    FILTRACION = auto()         # 2 - Requiere: PREPROCESAMIENTO
    REALZADOR = auto()          # 3 - Requiere: PREPROCESAMIENTO, opcional FILTRACION
    TRANSFORMADOR = auto()      # 4 - Requiere: PREPROCESAMIENTO, opcional FILTRACION
    SEGMENTADOR = auto()        # 5 - Requiere: PRE+FILT+REALZ (estricto para calidad)
    CUANTIFICADOR = auto()      # 6 - Requiere: SEGMENTADOR (necesita máscara/objetos)
    MODELADOR = auto()          # 7 - Requiere: CUANTIFICADOR (necesita SEGMENTADOR, opcional CUANTIFICADOR)
    ANALIZADOR = auto()         # 8 - Requiere: cualquier dato previo
    
    def orden_ejecucion(self) -> int:
        return list(CategoriaOperacion).index(self) + 1
    
    def puede_preceder_a(self, otra: 'CategoriaOperacion') -> bool:
        return self.orden_ejecucion() <= otra.orden_ejecucion()