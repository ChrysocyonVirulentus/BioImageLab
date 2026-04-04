# === gestorLab/Categoria_Operacion.py ===

from enum import Enum, auto
from typing import Set, List


# =========================================================
# TIPOS DE DATOS DEL PIPELINE
# =========================================================

class TipoDato(Enum):
    IMAGEN       = "imagen"
    MASCARA      = "mascara"
    FEATURES     = "features"
    TABLA        = "tabla"
    MODELO       = "modelo"
    VISUALIZACION = "viz"
    NINGUNA      = "ninguna"


# =========================================================
# CATEGORIAS (ETAPAS DEL PIPELINE)
# =========================================================

class CategoriaOperacion(Enum):
    """
    Define la ETAPA del pipeline (orden lógico).
    NO contiene lógica de tipos → eso va en Validaciones.
    """

    PREPROCESAMIENTO = auto()
    FILTRACION       = auto()
    REALZADOR        = auto()
    TRANSFORMADOR    = auto()
    SEGMENTADOR      = auto()
    CUANTIFICADOR    = auto()
    MODELADOR        = auto()
    ANALIZADOR       = auto()

    # =========================================================
    # ORDEN
    # =========================================================

    def orden(self) -> int:
        return list(CategoriaOperacion).index(self)

    def puede_preceder_a(self, otra: "CategoriaOperacion") -> bool:
        return self.orden() <= otra.orden()

    # =========================================================
    # PROPIEDADES PIPELINE (DAG)
    # =========================================================

    @property
    def es_punto_split(self) -> bool:
        return self in {
            CategoriaOperacion.SEGMENTADOR,
            CategoriaOperacion.CUANTIFICADOR,
        }

    @property
    def requiere_merge(self) -> bool:
        return self == CategoriaOperacion.CUANTIFICADOR

    # =========================================================
    # DEPENDENCIAS
    # =========================================================

    def dependencias_estrictas(self) -> Set["CategoriaOperacion"]:
        deps = {
            CategoriaOperacion.PREPROCESAMIENTO: set(),

            CategoriaOperacion.FILTRACION: {
                CategoriaOperacion.PREPROCESAMIENTO
            },

            CategoriaOperacion.REALZADOR: {
                CategoriaOperacion.PREPROCESAMIENTO
            },

            CategoriaOperacion.TRANSFORMADOR: {
                CategoriaOperacion.PREPROCESAMIENTO
            },

            CategoriaOperacion.SEGMENTADOR: {
                CategoriaOperacion.PREPROCESAMIENTO,
                CategoriaOperacion.FILTRACION,
            },

            CategoriaOperacion.CUANTIFICADOR: {
                CategoriaOperacion.PREPROCESAMIENTO,
                CategoriaOperacion.SEGMENTADOR,
            },

            CategoriaOperacion.MODELADOR: {
                CategoriaOperacion.CUANTIFICADOR
            },

            CategoriaOperacion.ANALIZADOR: set(),
        }

        return deps.get(self, set())

    def validar_dependencias(
        self,
        presentes: Set["CategoriaOperacion"]
    ) -> tuple[bool, List[str]]:

        faltantes = self.dependencias_estrictas() - presentes

        if not faltantes:
            return True, []

        nombres = [c.name for c in faltantes]
        return False, [f"{self.name} requiere: {', '.join(nombres)}"]

    def __repr__(self) -> str:
        return f"{self.name}({self.orden()})"