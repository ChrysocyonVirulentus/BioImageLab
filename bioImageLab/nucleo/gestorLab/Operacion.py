from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Any, Optional, Generic, TypeVar

from ..controlador.Resultado_Either import Resultado, Err
from ..controlador.Controlador_BioImagen import ErrorBioImagen

from .Categoria_Operacion import CategoriaOperacion, TipoDato
from .Validacion_Operacion import (
    es_compatible,
    requiere_adaptador,
    obtener_adaptador,
)

TEntrada = TypeVar("TEntrada")
TSalida  = TypeVar("TSalida")
EError   = TypeVar("EError") 

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

    instancia_callable: Callable[[TEntrada], Resultado[TSalida, Any]]

    # Tipado runtime (flexible)
    tipo_entrada: type = object
    tipo_salida_real: type = object

    tipo_dato_salida: TipoDato = TipoDato.IMAGEN

    parametros_originales: Dict[str, Any] = field(default_factory=dict)
    descripcion:           str            = ""
    canal_objetivo:        Optional[int]  = None
    es_operacion_split:    bool           = False
    requiere_input_especial: Optional[str] = None

    # =========================================================
    # EJECUCIÓN
    # =========================================================

    def ejecutar(self, data: TEntrada) -> Resultado[TSalida, Any]:
        try:
            return self.instancia_callable(data)
        except Exception as e:
            # Fallback genérico — cada controlador ya captura sus propias excepciones,
            # esto solo atrapa lo que escapó completamente
            return Err(ErrorBioImagen(
                etapa="operacion",
                mensaje=f"Excepción no capturada en '{self.nombre}': {e}",
                causa=e
            ))

    # =========================================================
    # VALIDACIONES
    # =========================================================

    def _validar_runtime(self, salida: Any, siguiente: "Operacion") -> Optional[Resultado]:
        """Validación basada en tipo real. Permisiva — no corta si hay duda."""
        try:
            if siguiente.tipo_entrada is not object:
                if not isinstance(salida, siguiente.tipo_entrada):
                    return Err(ErrorBioImagen(
                        etapa="pipeline",
                        mensaje=(
                            f"Tipo incompatible: "
                            f"{type(salida).__name__} → {siguiente.tipo_entrada.__name__}"
                        )
                    ))
        except TypeError:
            pass
        return None

    def _validar_semantica(self, siguiente: "Operacion") -> Optional[Resultado]:
        if not self.categoria.puede_preceder_a(siguiente.categoria):
            return Err(ErrorBioImagen(
                etapa="pipeline",
                mensaje=f"Orden inválido: {self.categoria.name} → {siguiente.categoria.name}"
            ))

        if not es_compatible(self.categoria, siguiente.categoria):
            if requiere_adaptador(self.categoria, siguiente.categoria):
                adaptador = obtener_adaptador(self.categoria, siguiente.categoria)
                return Err(ErrorBioImagen(
                    etapa="pipeline",
                    mensaje=(
                        f"Requiere adaptador entre "
                        f"{self.tipo_dato_salida.name} → {siguiente.tipo_dato_salida.name} "
                        f"(sugerido: {adaptador})"
                    )
                ))
            return Err(ErrorBioImagen(
                etapa="pipeline",
                mensaje=f"Incompatibilidad semántica: {self.categoria.name} → {siguiente.categoria.name}"
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

            error_sem = self._validar_semantica(siguiente)
            if error_sem:
                return error_sem

            error_rt = self._validar_runtime(salida, siguiente)
            if error_rt:
                return error_rt

            return siguiente.ejecutar(salida)

        # CORREGIDO: eliminado metodo=self.nombre — campo inexistente
        return Operacion(
            nombre               = f"{self.nombre} >> {siguiente.nombre}",
            categoria            = siguiente.categoria,
            instancia_callable   = _compuesto,
            tipo_entrada         = self.tipo_entrada,
            tipo_salida_real     = siguiente.tipo_salida_real,
            tipo_dato_salida     = siguiente.tipo_dato_salida,
            descripcion          = self.descripcion,
        )

    # =========================================================
    # DEBUG
    # =========================================================

    def __repr__(self) -> str:
        canal = f"[C{self.canal_objetivo}]" if self.canal_objetivo is not None else "[C*]"
        split = " [SPLIT]" if self.es_operacion_split else ""
        return f"{self.categoria.name}::{self.nombre}{canal}{split} → {self.tipo_dato_salida.value}"