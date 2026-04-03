from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Any, Optional, Generic, TypeVar
from enum import Enum

# Sistema
from ..controlador.Resultado_Either import Resultado, Err, Ok
from ..controlador.Controlador_BioImagen import BioImagenData, ErrorBioImagen
from .Categoria_Operacion import CategoriaOperacion


# OUTPUT GENÉRICO
TSalida = TypeVar("TSalida")


class TipoSalida(Enum):
    IMAGEN = "imagen"
    MASCARA = "mascara"
    FEATURES = "features"
    TABLA = "tabla"
    MODELO = "modelo"
    VISUALIZACION = "viz"
    NINGUNA = "ninguna"


@dataclass(frozen=True)
class Operacion(Generic[TSalida]):
    """
    Operación pura para pipelines.

    Ahora soporta:
    - Imagen → Imagen
    - Imagen → máscara
    - Imagen → DataFrame
    - Imagen → modelo
    """

    nombre: str
    categoria: CategoriaOperacion

    # 🔥 CLAVE: ahora genérico
    instancia_callable: Callable[
        [BioImagenData, int],
        Resultado[TSalida, ErrorBioImagen]
    ]

    canal_objetivo: Optional[int] = None

    parametros_originales: Dict[str, Any] = field(default_factory=dict)
    tipo_salida: TipoSalida = TipoSalida.IMAGEN
    descripcion: str = ""

    # Flags de pipeline avanzado
    es_operacion_split: bool = False
    requiere_input_especial: Optional[str] = None

    # =========================================================
    # ===================== EJECUCIÓN ==========================
    # =========================================================

    def ejecutar(
        self,
        data: BioImagenData
    ) -> Resultado[TSalida, ErrorBioImagen]:
        """
        Ejecuta la operación respetando el flujo monádico.

        - Maneja canal único o multicanal
        - Propaga errores correctamente
        - No usa unwrap (seguro)
        """

        try:
            # ===== Determinar canales =====
            if self.canal_objetivo is not None:
                if not (0 <= self.canal_objetivo < data.dims.C):
                    return Err(ErrorBioImagen(
                        etapa="operacion",
                        mensaje=(
                            f"{self.nombre}: canal {self.canal_objetivo} "
                            f"fuera de rango [0, {data.dims.C-1}]"
                        )
                    ))
                canales = [self.canal_objetivo]
            else:
                canales = range(data.dims.C)

            # ===== Ejecución monádica =====
            resultado: Resultado[TSalida, ErrorBioImagen] = Ok(data)  # type: ignore

            for c in canales:
                resultado = resultado.bind(
                    lambda d, canal=c: self.instancia_callable(d, canal)
                )

                if resultado.es_err():
                    return resultado

            return resultado

        except Exception as e:
            return Err(ErrorBioImagen(
                etapa="operacion",
                mensaje=f"Error en {self.nombre}: {str(e)}",
                causa=e
            ))

    # =========================================================
    # ===================== DEBUG ==============================
    # =========================================================

    def __repr__(self) -> str:
        canal = (
            f"[C{self.canal_objetivo}]"
            if self.canal_objetivo is not None
            else "[C*]"
        )

        split = " [SPLIT]" if self.es_operacion_split else ""

        return f"{self.categoria.name}::{self.nombre}{canal}{split}"