# === normalizador/Controlador_Normalizador.py ===
from __future__ import annotations

import numpy as np
from typing import Union, Optional

# Core
from .Controlador_Base import Controlador_Base

# Sistema
from .Resultado_Either import Resultado
from .Controlador_BioImagen import BioImagenData, ErrorBioImagen
from ..gestorLab.Categoria_Operacion import CategoriaOperacion
from ..gestorLab.Operacion import Operacion, TipoDato

# Estrategias — mismas que usa el filtrador, sin override
from .Estrategias_Aplicacion import (
    TipoAplicacion,
    Global,
    PorCorteZ,
    PorTimepoint,
    PorCorteEspaciotemporal,
    PorVolumen3D,
    ConReferencia,
)

# Métodos
from ..preprocesador.normalizador.Metodos_Normalizacion import (
    MetodoNormalizacion, MaxNorm, MinMaxNorm, PercentilNorm, ZScoreNorm,
)
from ..preprocesador.normalizador.Metodos_CambioTipos import (
    ToUint8, ToUint16, ToFloat32, ToFloat64,
)


MetodoNorm = Union[
    MaxNorm, MinMaxNorm, PercentilNorm, ZScoreNorm,
    ToUint8, ToUint16, ToFloat32, ToFloat64,
]


class Controlador_Normalizador(Controlador_Base[BioImagenData, BioImagenData]):
    """
    Controlador de normalización.

    Idéntico al filtrador salvo dos hooks de shape:
      _preprocesar  → devuelve (T,Z,Y,X) en lugar del 2D del base
      _validar_salida → valida shape (T,Z,Y,X)

    Sin Norm_* propios: usa TipoAplicacion de Estrategias_Aplicacion.
    El base llama tipo_aplicacion.estrategia() — sin override necesario.

    Los MetodoNorm hacen el flatten internamente si lo necesitan,
    ya que la estrategia les pasa el bloque con el shape que corresponda:
      Global              → metodo recibe [T, Z, Y, X]
      PorCorteZ           → metodo recibe [T, Y, X]
      PorTimepoint        → metodo recibe [Z, Y, X]
      PorCorteEspaciotemporal → metodo recibe [Y, X]
      PorVolumen3D        → metodo recibe [Z, Y, X]
    """

    def __init__(self):
        super().__init__(etapa="preprocesamiento", dominio="normalizacion")
        self._ultimo_metodo: Optional[MetodoNorm] = None

    # =========================================================
    # HOOKS — solo los de shape, nada más
    # =========================================================

    def _preprocesar(self, data: BioImagenData, canal: int) -> np.ndarray:
        """Devuelve (T, Z, Y, X) — el base devolvería (Y, X) 2D."""
        return data.datos[:, :, canal, :, :].astype(np.float64)

    # Para darle consistencia 4D
    def _postprocesar(self, data, resultado, canal):
        if not isinstance(resultado, np.ndarray):
            return resultado

        nuevos = data.datos.copy()
        nuevos[:, :, canal, :, :] = resultado  # (T,Z,Y,X)
        return replace(data, datos=nuevos)

    def _validar_salida(self, resultado: np.ndarray, canal_data: np.ndarray) -> None:
        """El base compara shape 2D, aquí comparamos shape 4D."""
        if isinstance(resultado, np.ndarray):
            if resultado.shape != canal_data.shape:
                raise ValueError(
                    f"Shape inválido: esperado {canal_data.shape}, "
                    f"obtenido {resultado.shape}"
                )

    # =========================================================
    # HELPER INTERNO — idéntico al filtrador
    # =========================================================

    def _crear(
        self,
        metodo: MetodoNorm,
        tipo: TipoAplicacion,
        canal: int = 0,
    ):
        self._ultimo_metodo = metodo
        return self.crear_operador(metodo=metodo, tipo_aplicacion=tipo, canal=canal)

    def _crear_multicanal(self, metodo: MetodoNorm, tipo: TipoAplicacion):
        self._ultimo_metodo = metodo
        return self.crear_operador_multicanal(metodo=metodo, tipo_aplicacion=tipo)

    # =========================================================
    # FACTORIES YAML / CLI → devuelven Operacion
    # =========================================================

    def crear_operacion_max_norm(
        self,
        tipo: TipoAplicacion = Global(),
        canal: Optional[int] = 0,
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self.crear_operacion(
            nombre_metodo="max_norm",
            categoria=CategoriaOperacion.PREPROCESAMIENTO,
            tipo_aplicacion=tipo,
            canal=canal,
            nombre=nombre,
            params={},
        )

    def crear_operacion_minmax_norm(
        self,
        tipo: TipoAplicacion = Global(),
        canal: Optional[int] = 0,
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self.crear_operacion(
            nombre_metodo="minmax_norm",
            categoria=CategoriaOperacion.PREPROCESAMIENTO,
            tipo_aplicacion=tipo,
            canal=canal,
            nombre=nombre,
            params={},
        )

    def crear_operacion_percentil_norm(
        self,
        percentil_bajo: float = 1.0,
        percentil_alto: float = 99.0,
        tipo: TipoAplicacion = Global(),
        canal: Optional[int] = 0,
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self.crear_operacion(
            nombre_metodo="percentil_norm",
            categoria=CategoriaOperacion.PREPROCESAMIENTO,
            tipo_aplicacion=tipo,
            canal=canal,
            nombre=nombre,
            params={"percentil_bajo": percentil_bajo, "percentil_alto": percentil_alto},
        )

    def crear_operacion_zscore_norm(
        self,
        tipo: TipoAplicacion = Global(),
        canal: Optional[int] = 0,
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self.crear_operacion(
            nombre_metodo="zscore_norm",
            categoria=CategoriaOperacion.PREPROCESAMIENTO,
            tipo_aplicacion=tipo,
            canal=canal,
            nombre=nombre,
            params={},
        )

    def crear_operacion_to_uint8(
        self,
        tipo: TipoAplicacion = Global(),
        canal: Optional[int] = 0,
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self.crear_operacion(
            nombre_metodo="to_uint8",
            categoria=CategoriaOperacion.PREPROCESAMIENTO,
            tipo_aplicacion=tipo,
            canal=canal,
            nombre=nombre,
            params={},
        )

    def crear_operacion_to_uint16(
        self,
        tipo: TipoAplicacion = Global(),
        canal: Optional[int] = 0,
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self.crear_operacion(
            nombre_metodo="to_uint16",
            categoria=CategoriaOperacion.PREPROCESAMIENTO,
            tipo_aplicacion=tipo,
            canal=canal,
            nombre=nombre,
            params={},
        )

    def crear_operacion_to_float32(
        self,
        tipo: TipoAplicacion = Global(),
        canal: Optional[int] = 0,
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self.crear_operacion(
            nombre_metodo="to_float32",
            categoria=CategoriaOperacion.PREPROCESAMIENTO,
            tipo_aplicacion=tipo,
            canal=canal,
            nombre=nombre,
            params={},
        )

    def crear_operacion_to_float64(
        self,
        tipo: TipoAplicacion = Global(),
        canal: Optional[int] = 0,
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self.crear_operacion(
            nombre_metodo="to_float64",
            categoria=CategoriaOperacion.PREPROCESAMIENTO,
            tipo_aplicacion=tipo,
            canal=canal,
            nombre=nombre,
            params={},
        )

    # =========================================================
    # USO IMPERATIVO
    # =========================================================

    def aplicar(
        self,
        data: BioImagenData,
        metodo: MetodoNorm,
        tipo: TipoAplicacion,
        canal: int = 0,
    ) -> Resultado[BioImagenData, ErrorBioImagen]:
        return self._crear(metodo, tipo, canal)(data)

    def aplicar_multicanal(
        self,
        data: BioImagenData,
        metodo: MetodoNorm,
        tipo: TipoAplicacion,
    ) -> Resultado[BioImagenData, ErrorBioImagen]:
        return self._crear_multicanal(metodo, tipo)(data)

    def __repr__(self):
        nombre = getattr(self._ultimo_metodo, "nombre", "None")
        return f"<Controlador_Normalizador ultimo={nombre}>"