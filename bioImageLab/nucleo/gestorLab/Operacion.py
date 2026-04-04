from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Any, Optional, Generic, TypeVar
from enum import Enum

from ..controlador.Resultado_Either import Resultado, Err, Ok
from ..controlador.Controlador_BioImagen import BioImagenData, ErrorBioImagen
from .Categoria_Operacion import CategoriaOperacion

# Para tener eficientemente algo como Operacion[TEntrada → TSalida] : handlear un input y output de diferente tipo.
TEntrada = TypeVar("TEntrada")
TSalida  = TypeVar("TSalida")

class TipoSalida(Enum):
    IMAGEN       = "imagen"
    MASCARA      = "mascara"
    FEATURES     = "features"
    TABLA        = "tabla"
    MODELO       = "modelo"
    VISUALIZACION = "viz"
    NINGUNA      = "ninguna"   # procedimientos / side-effects (guardar PNG, etc.)


@dataclass(frozen=True)
class Operacion(Generic[TEntrada, TSalida]):
    """
    Unidad atómica de pipeline. Completamente agnóstica del dominio:

        np.ndarray  → np.ndarray
        BioImagenData → BioImagenData
        BioImagenData → pd.DataFrame
        BioImagenData → Figure  (viz)
        (BioImagenData, BioImagenData) → BioImagenData   (registro, co-loc)
        cualquier TEntrada → Resultado[TSalida, ErrorBioImagen]

    La lógica de canal (si aplica) queda COMPLETAMENTE encapsulada
    dentro de instancia_callable, construida por el Controlador.
    Operacion no sabe nada de canales.
    """

    nombre:     str
    categoria:  CategoriaOperacion

    # ── Callable con canal ya capturado en el cierre (si corresponde) ──────
    instancia_callable: Callable[[TEntrada], Resultado[TSalida, ErrorBioImagen]]

    tipo_salida:           TipoSalida            = TipoSalida.IMAGEN
    parametros_originales: Dict[str, Any]        = field(default_factory=dict)
    descripcion:           str                   = ""

    # ── Metadata de pipeline (no afecta ejecución) ─────────────────────────
    canal_objetivo:        Optional[int]         = None   # sólo informativo
    es_operacion_split:    bool                  = False
    requiere_input_especial: Optional[str]       = None

    # =========================================================
    # EJECUCIÓN
    # =========================================================

    def ejecutar(self, data: TEntrada) -> Resultado[TSalida, ErrorBioImagen]:
        """
        Ejecución limpia: delega TODO al callable.
        No hay canal logic, no hay unwrap, no hay bucles.
        """
        try:
            return self.instancia_callable(data)
        except Exception as e:
            return Err(ErrorBioImagen(
                etapa="operacion",
                mensaje=f"Error inesperado en '{self.nombre}': {e}",
                causa=e
            ))

    # =========================================================
    # DEBUG
    # =========================================================

    #NOTA HAY QUE EN ALGUN MOMENTO HACER: assert tipo_salida_op1 == tipo_entrada_op2
    # Permitira hacer : pipeline = op1.then(op2).then(op3)
    def then(self, siguiente: "Operacion[TSalida, Any]") -> "Operacion[TEntrada, Any]":
        def _compuesto(data: TEntrada):
            return self.ejecutar(data).bind(siguiente.ejecutar)

        return Operacion(
            nombre=f"{self.nombre} >> {siguiente.nombre}",
            categoria=siguiente.categoria,
            instancia_callable=_compuesto,
            tipo_salida=siguiente.tipo_salida
        )

    def __repr__(self) -> str:
        canal = f"[C{self.canal_objetivo}]" if self.canal_objetivo is not None else "[C*]"
        split = " [SPLIT]" if self.es_operacion_split else ""
        return f"{self.categoria.name}::{self.nombre}{canal}{split} → {self.tipo_salida.value}"