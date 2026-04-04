from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Any, Optional, Generic, TypeVar

from ..controlador.Resultado_Either import Resultado, Err
from ..controlador.Controlador_BioImagen import ErrorBioImagen

from .Categoria_Operacion import CategoriaOperacion, TipoDato
from .Validaciones_Operaciones import (
    es_compatible,
    requiere_adaptador,
    obtener_adaptador,
)

TEntrada = TypeVar("TEntrada")
TSalida  = TypeVar("TSalida")


@dataclass(frozen=True)
class Operacion(Generic[TEntrada, TSalida]):
    """
    Unidad atómica de pipeline.

    RESPONSABILIDADES:
    - Ejecutar callable
    - Componer operaciones (.then)
    - Validación LOCAL (no global)
    - No conoce YAML, ni ramas, ni adaptadores concretos

    NO hace:
    - Resolver adaptadores (eso es del PipelineBuilder)
    - Validación global del pipeline
    """

    nombre: str
    categoria: CategoriaOperacion

    instancia_callable: Callable[[TEntrada], Resultado[TSalida, ErrorBioImagen]]

    # Tipado runtime (flexible)
    tipo_entrada: type = object
    tipo_salida_real: type = object

    tipo_dato_salida: TipoDato = TipoDato.IMAGEN

    parametros_originales: Dict[str, Any] = field(default_factory=dict)
    descripcion: str = ""

    canal_objetivo: Optional[int] = None
    es_operacion_split: bool = False
    requiere_input_especial: Optional[str] = None

    # =========================================================
    # EJECUCIÓN
    # =========================================================

    def ejecutar(self, data: TEntrada) -> Resultado[TSalida, ErrorBioImagen]:
        try:
            return self.instancia_callable(data)
        except Exception as e:
            return Err(ErrorBioImagen(
                etapa="operacion",
                mensaje=f"Error inesperado en '{self.nombre}': {e}",
                causa=e
            ))

    # =========================================================
    # VALIDACIONES
    # =========================================================

    def _validar_runtime(self, salida: Any, siguiente: "Operacion") -> Optional[Resultado]:
        """
        Validación basada en el tipo REAL.
        Nunca rompe ejecución.
        """
        try:
            if not isinstance(salida, siguiente.tipo_entrada):
                return Err(ErrorBioImagen(
                    etapa="pipeline",
                    mensaje=(
                        f"Incompatibilidad runtime: "
                        f"{type(salida).__name__} → "
                        f"{siguiente.tipo_entrada.__name__}"
                    )
                ))
        except TypeError:
            pass  # fallback permisivo

        return None

    def _validar_semantica(self, siguiente: "Operacion") -> Optional[Resultado]:
        """
        Usa Validaciones_Operaciones.
        """

        # 1. Orden de pipeline
        if not self.categoria.puede_preceder_a(siguiente.categoria):
            return Err(ErrorBioImagen(
                etapa="pipeline",
                mensaje=(
                    f"Orden inválido: "
                    f"{self.categoria.name} → {siguiente.categoria.name}"
                )
            ))

        # 2. Compatibilidad de tipos semánticos
        if not es_compatible(self.categoria, siguiente.categoria):

            if requiere_adaptador(self.categoria, siguiente.categoria):
                adaptador = obtener_adaptador(self.categoria, siguiente.categoria)

                return Err(ErrorBioImagen(
                    etapa="pipeline",
                    mensaje=(
                        f"Requiere adaptador: "
                        f"{self.tipo_dato_salida.name} → {siguiente.tipo_dato_salida.name} "
                        f"(sugerido: {adaptador})"
                    )
                ))

            return Err(ErrorBioImagen(
                etapa="pipeline",
                mensaje=(
                    f"Incompatibilidad semántica: "
                    f"{self.categoria.name} → {siguiente.categoria.name}"
                )
            ))

        return None

    # =========================================================
    # COMPOSICIÓN
    # =========================================================

    def then(self, siguiente: "Operacion[TSalida, Any]") -> "Operacion[TEntrada, Any]":

        def _compuesto(data: TEntrada):

            resultado = self.ejecutar(data)

            if resultado.es_err():
                return resultado

            salida = resultado.unwrap()

            # 1. Validación semántica
            error_semantico = self._validar_semantica(siguiente)
            if error_semantico:
                return error_semantico

            # 2. Validación runtime
            error_runtime = self._validar_runtime(salida, siguiente)
            if error_runtime:
                return error_runtime

            # 3. Ejecutar siguiente
            return siguiente.ejecutar(salida)

        return Operacion(
            nombre=f"{self.nombre} >> {siguiente.nombre}",
            categoria=siguiente.categoria,
            instancia_callable=_compuesto,
            tipo_entrada=self.tipo_entrada,
            tipo_salida_real=siguiente.tipo_salida_real,
            tipo_dato_salida=siguiente.tipo_dato_salida,
        )

    # =========================================================
    # DEBUG
    # =========================================================

    def __repr__(self) -> str:
        canal = f"[C{self.canal_objetivo}]" if self.canal_objetivo is not None else "[C*]"
        split = " [SPLIT]" if self.es_operacion_split else ""

        return (
            f"{self.categoria.name}::{self.nombre}"
            f"{canal}{split} → {self.tipo_dato_salida.value}"
        )