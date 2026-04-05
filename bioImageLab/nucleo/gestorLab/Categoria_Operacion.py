# === gestorLab/Categoria_Operacion.py ===

from __future__ import annotations

from enum import Enum, auto
from typing import Set, List, Tuple


class TipoDato(Enum):
    IMAGEN        = "imagen"
    MASCARA       = "mascara"
    FEATURES      = "features"
    TABLA         = "tabla"
    MODELO        = "modelo"
    VISUALIZACION = "viz"
    NINGUNA       = "ninguna"


class CategoriaOperacion(Enum):
    """
    Etapas del pipeline con contratos de dos niveles:

      dependencias_duras  → bloquean ejecución si no se cumplen
      dependencias_blandas → warnings en el log, no bloquean

    Contratos reales del dominio:
      PREPROCESAMIENTO  → sin requisitos (es el inicio)
      FILTRACION        → requiere haber normalizado (PREPROCESAMIENTO)
      REALZADOR         → requiere PREPROCESAMIENTO
      TRANSFORMADOR     → requiere PREPROCESAMIENTO
      SEGMENTADOR       → requiere PREPROCESAMIENTO
                          recomienda FILTRACION o REALZADOR
      CUANTIFICADOR     → requiere PREPROCESAMIENTO + SEGMENTADOR
      MODELADOR         → requiere CUANTIFICADOR
      ANALIZADOR        → sin requisitos duros (puede analizar cualquier cosa)

    Flexibilidad intencional:
      - N filtrados seguidos: OK
      - N realzados seguidos: OK
      - Filtrado + Realzado + Filtrado: OK
      - Saltar filtrado e ir directo a segmentación: WARNING, no error
      - SEGMENTADOR sin FILTRACION: WARNING (puede dar peores resultados)
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
    # ORDEN Y PRECEDENCIA
    # =========================================================

    def orden(self) -> int:
        return list(CategoriaOperacion).index(self)

    def puede_preceder_a(self, otra: "CategoriaOperacion") -> bool:
        """
        Define qué categoría puede conectarse a cuál en el grafo.
        Permisivo dentro del dominio de imagen, estricto en el salto a datos.
        """
        # Dominio imagen: todo puede conectar con todo dentro del dominio
        dominio_imagen = {
            CategoriaOperacion.PREPROCESAMIENTO,
            CategoriaOperacion.FILTRACION,
            CategoriaOperacion.REALZADOR,
            CategoriaOperacion.TRANSFORMADOR,
            CategoriaOperacion.SEGMENTADOR,
        }

        if self in dominio_imagen and otra in dominio_imagen:
            return True

        # Saltos explícitos al dominio tabular/análisis
        reglas_cruce = {
            CategoriaOperacion.SEGMENTADOR:   {CategoriaOperacion.CUANTIFICADOR,
                                               CategoriaOperacion.ANALIZADOR},
            CategoriaOperacion.CUANTIFICADOR: {CategoriaOperacion.MODELADOR,
                                               CategoriaOperacion.ANALIZADOR},
            CategoriaOperacion.MODELADOR:     {CategoriaOperacion.ANALIZADOR},
            CategoriaOperacion.ANALIZADOR:    set(),
        }

        return otra in reglas_cruce.get(self, set())

    # =========================================================
    # DEPENDENCIAS DURAS — bloquean ejecución
    # =========================================================

    def dependencias_duras(self) -> Set["CategoriaOperacion"]:
        """
        Requisitos mínimos sin los cuales la operación no tiene sentido.
        Si no se cumplen → Err, el pipeline no ejecuta.
        """
        deps: dict[CategoriaOperacion, Set[CategoriaOperacion]] = {
            CategoriaOperacion.PREPROCESAMIENTO: set(),

            # Filtrar sin normalizar produce resultados sin sentido
            CategoriaOperacion.FILTRACION: {
                CategoriaOperacion.PREPROCESAMIENTO,
            },

            CategoriaOperacion.REALZADOR: {
                CategoriaOperacion.PREPROCESAMIENTO,
            },

            CategoriaOperacion.TRANSFORMADOR: {
                CategoriaOperacion.PREPROCESAMIENTO,
            },

            CategoriaOperacion.SEGMENTADOR: {
                CategoriaOperacion.PREPROCESAMIENTO,
            },

            # Cuantificar sin segmentar no tiene sentido biológico
            CategoriaOperacion.CUANTIFICADOR: {
                CategoriaOperacion.PREPROCESAMIENTO,
                CategoriaOperacion.SEGMENTADOR,
            },

            # Modelar sin features cuantificadas no tiene sentido
            CategoriaOperacion.MODELADOR: {
                CategoriaOperacion.CUANTIFICADOR,
            },

            # Analizador es libre — puede analizar cualquier cosa
            CategoriaOperacion.ANALIZADOR: set(),
        }
        return deps.get(self, set())

    # =========================================================
    # DEPENDENCIAS BLANDAS — warnings en el log, no bloquean
    # =========================================================

    def dependencias_blandas(self) -> Set["CategoriaOperacion"]:
        """
        Etapas recomendadas para mejores resultados.
        Si no se cumplen → LogEvento WARNING, la ejecución continúa.
        """
        deps: dict[CategoriaOperacion, Set[CategoriaOperacion]] = {
            # Segmentar sin filtrar puede dar peores resultados
            CategoriaOperacion.SEGMENTADOR: {
                CategoriaOperacion.FILTRACION,
            },

            # Cuantificar sin haber realzado puede perder señal
            CategoriaOperacion.CUANTIFICADOR: {
                CategoriaOperacion.FILTRACION,
            },

            # El resto no tiene recomendaciones
            CategoriaOperacion.PREPROCESAMIENTO: set(),
            CategoriaOperacion.FILTRACION:       set(),
            CategoriaOperacion.REALZADOR:        set(),
            CategoriaOperacion.TRANSFORMADOR:    set(),
            CategoriaOperacion.MODELADOR:        set(),
            CategoriaOperacion.ANALIZADOR:       set(),
        }
        return deps.get(self, set())

    # =========================================================
    # VALIDACIÓN CON DOS NIVELES
    # =========================================================

    def validar_dependencias(
        self,
        presentes: Set["CategoriaOperacion"]
    ) -> Tuple[bool, List[str], List[str]]:
        """
        Retorna (es_valido, errores_duros, warnings_blandos).

        errores_duros  → bloquean
        warnings_blandos → van al log, no bloquean
        """
        faltantes_duros = self.dependencias_duras() - presentes
        faltantes_blandos = self.dependencias_blandas() - presentes

        errores = []
        warnings = []

        if faltantes_duros:
            nombres = sorted(c.name for c in faltantes_duros)
            errores.append(
                f"{self.name} requiere obligatoriamente: {', '.join(nombres)}"
            )

        if faltantes_blandos:
            nombres = sorted(c.name for c in faltantes_blandos)
            warnings.append(
                f"{self.name} recomienda haber ejecutado: {', '.join(nombres)} "
                f"(resultados pueden ser subóptimos)"
            )

        return len(errores) == 0, errores, warnings

    # =========================================================
    # PROPIEDADES
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

    def __repr__(self) -> str:
        return f"{self.name}({self.orden()})"