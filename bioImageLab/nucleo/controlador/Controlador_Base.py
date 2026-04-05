from __future__ import annotations

import numpy as np
from dataclasses import replace
from typing import Callable, Optional, Dict, Any, TypeVar, Generic

from .Resultado_Either import Resultado, Ok, Err
from .Controlador_BioImagen import BioImagenData, ErrorBioImagen
from ..gestorLab.Registro_Metodos import registro_metodos
from ..gestorLab.Operacion import Operacion
from ..gestorLab.Categoria_Operacion import CategoriaOperacion, TipoDato

TEntrada = TypeVar("TEntrada")
TSalida  = TypeVar("TSalida")


class Controlador_Base(Generic[TEntrada, TSalida]):
    """
    Controlador genérico.

    Responsabilidades:
        - Ejecutar un método sobre datos (hook pipeline)
        - Producir callables con canal ya capturado  ← toda la canal-logic vive acá
        - Crear Operaciones para el pipeline builder

    NO decide qué hace el método: eso es el método.
    NO itera canales desde Operacion: eso es el controlador.
    """

    def __init__(self, etapa: str = "procesamiento", dominio: str = "general"):
        self._etapa   = etapa
        self._dominio = dominio
        self._ultimo_metodo: Optional[Any] = None

    # =========================================================
    # HOOKS — sobreescribibles por subclases
    # =========================================================

    def _preprocesar(self, data: BioImagenData, canal: int) -> np.ndarray:
        """Extrae slice [T,Z,Y,X] y convierte a float64. Punto de extensión."""
        return data.datos[:, :, canal, :, :].astype(np.float64)

    def _postprocesar(self, data: BioImagenData, resultado: Any, canal: int) -> Any:
        """
        Reinserta resultado en BioImagenData si es np.ndarray.
        Para otros tipos (DataFrame, Figure, modelo) devuelve tal cual.
        """
        if not isinstance(resultado, np.ndarray):
            return resultado
        nuevos = data.datos.copy()
        nuevos[:, :, canal, :, :] = resultado
        return replace(data, datos=nuevos)

    def _validar_entrada(self, data: BioImagenData, canal: int) -> None:
        if not (0 <= canal < data.dims.C):
            raise ValueError(f"Canal {canal} fuera de rango [0, {data.dims.C - 1}]")

    def _validar_salida(self, resultado: Any, canal_data: np.ndarray) -> None:
        """Valida shape solo si el resultado es np.ndarray."""
        if isinstance(resultado, np.ndarray):
            if resultado.shape != canal_data.shape:
                raise ValueError(
                    f"Shape inválido: esperado {canal_data.shape}, obtenido {resultado.shape}"
                )

    def _adaptar_metodo(self, metodo) -> Callable:
        from .Estrategias_Aplicacion import adaptar_metodo
        return adaptar_metodo(metodo)

    def _obtener_estrategia(self, tipo_aplicacion) -> Optional[Callable]:
        if tipo_aplicacion is None:
            return None
        return tipo_aplicacion.estrategia()

    def _requiere_estrategia(self) -> bool:
        return True

    # =========================================================
    # INFERENCIA DE TIPOS RUNTIME (para Operacion.tipo_entrada/tipo_salida_real)
    # =========================================================

    def _inferir_tipo_entrada_real(self, categoria: CategoriaOperacion) -> type:
        import pandas as pd
        mapping = {
            CategoriaOperacion.PREPROCESAMIENTO: BioImagenData,
            CategoriaOperacion.FILTRACION:       BioImagenData,
            CategoriaOperacion.REALZADOR:        BioImagenData,
            CategoriaOperacion.TRANSFORMADOR:    BioImagenData,
            CategoriaOperacion.SEGMENTADOR:      BioImagenData,
            CategoriaOperacion.CUANTIFICADOR:    BioImagenData,
            CategoriaOperacion.MODELADOR:        pd.DataFrame,
            CategoriaOperacion.ANALIZADOR:       object,
        }
        return mapping.get(categoria, object)

    def _inferir_tipo_salida_real(self, categoria: CategoriaOperacion) -> type:
        import pandas as pd
        mapping = {
            CategoriaOperacion.PREPROCESAMIENTO: BioImagenData,
            CategoriaOperacion.FILTRACION:       BioImagenData,
            CategoriaOperacion.REALZADOR:        BioImagenData,
            CategoriaOperacion.TRANSFORMADOR:    BioImagenData,
            CategoriaOperacion.SEGMENTADOR:      BioImagenData,
            CategoriaOperacion.CUANTIFICADOR:    pd.DataFrame,
            CategoriaOperacion.MODELADOR:        pd.DataFrame,
            CategoriaOperacion.ANALIZADOR:       object,
        }
        return mapping.get(categoria, object)

    # =========================================================
    # CORE INTERNO
    # =========================================================

    def _ejecutar(
            self,
            data: BioImagenData,
            metodo,
            tipo_aplicacion=None,
            canal: int = 0
        ) -> Resultado[TSalida, ErrorBioImagen]:
            nombre = getattr(metodo, "nombre", metodo.__class__.__name__)
            try:
                self._validar_entrada(data, canal)
                canal_data      = self._preprocesar(data, canal)
                metodo_adaptado = self._adaptar_metodo(metodo)
                estrategia      = (
                    self._obtener_estrategia(tipo_aplicacion)
                    if self._requiere_estrategia() else None
                )
                resultado = (
                    estrategia(canal_data, metodo_adaptado)
                    if estrategia is not None
                    else metodo_adaptado(canal_data)
                )
                self._validar_salida(resultado, canal_data)
                salida = self._postprocesar(data, resultado, canal)
                return Ok(salida)

            except Exception as e:
                return Err(ErrorBioImagen(
                    etapa=self._etapa,
                    mensaje=f"Error en '{nombre}': {e}",
                    ruta=data.ruta_origen,
                    causa=e
                ))

    # =========================================================
    # FACTORIES DE CALLABLE  ← canal capturado en el cierre
    # =========================================================

    def crear_operador(
        self,
        metodo,
        tipo_aplicacion=None,
        canal: int = 0
    ) -> Callable[[BioImagenData], Resultado[TSalida, ErrorBioImagen]]:
        """Canal capturado en el cierre — Operacion recibe un callable de 1 argumento."""
        self._ultimo_metodo = metodo

        def _op(data: BioImagenData) -> Resultado[TSalida, ErrorBioImagen]:
            return self._ejecutar(data, metodo, tipo_aplicacion, canal)

        return _op

    def crear_operador_multicanal(
        self,
        metodo,
        tipo_aplicacion=None
    ) -> Callable[[BioImagenData], Resultado[BioImagenData, ErrorBioImagen]]:
        """Aplica método sobre todos los canales secuencialmente."""
        self._ultimo_metodo = metodo

        def _multi(data: BioImagenData) -> Resultado[BioImagenData, ErrorBioImagen]:
            resultado: Resultado[BioImagenData, ErrorBioImagen] = Ok(data)
            for c in range(data.dims.C):
                resultado = resultado.bind(
                    lambda d, canal=c: self._ejecutar(d, metodo, tipo_aplicacion, canal)
                )
                if resultado.es_err():
                    break
            return resultado

        return _multi

    # =========================================================
    # FACTORY DE OPERACION (para pipeline builder / YAML)
    # =========================================================

    def crear_operacion(
        self,
        nombre_metodo: str,
        categoria: CategoriaOperacion,
        tipo_aplicacion=None,
        canal: Optional[int] = None,
        nombre: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
        tipo_salida: TipoDato = TipoDato.IMAGEN,   # ← este param se usa directamente abajo
    ) -> Operacion:
        """Construye una Operacion lista para el pipeline builder."""
        clase  = registro_metodos.obtener(self._dominio, nombre_metodo)
        metodo = clase(**(params or {}))

        callable_ = (
            self.crear_operador(metodo, tipo_aplicacion, canal)
            if canal is not None
            else self.crear_operador_multicanal(metodo, tipo_aplicacion)
        )

        return Operacion(
            nombre                = nombre or f"{self._etapa}_{nombre_metodo}",
            categoria             = categoria,
            instancia_callable    = callable_,
            tipo_entrada          = self._inferir_tipo_entrada_real(categoria),
            tipo_salida_real      = self._inferir_tipo_salida_real(categoria),
            canal_objetivo        = canal,
            parametros_originales = params or {},
            # CORREGIDO: usar el param 'tipo_salida' directamente, no tipo_salida_cat(categoria)
            # Eso ignoraba MASCARA, TABLA, etc. pasados por subclases
            tipo_dato_salida      = tipo_salida,
        )

    def crear(
        self,
        nombre_metodo: str,
        tipo_aplicacion=None,
        canal: int = 0,
        **params
    ) -> Callable[[BioImagenData], Resultado[TSalida, ErrorBioImagen]]:
        """Resolución dinámica desde YAML — devuelve solo el callable."""
        clase  = registro_metodos.obtener(self._dominio, nombre_metodo)
        metodo = clase(**params)
        return self.crear_operador(metodo, tipo_aplicacion, canal)

    # =========================================================

    def reset(self):
        self._ultimo_metodo = None

    def __repr__(self):
        ultimo = getattr(self._ultimo_metodo, "nombre", "Ninguno")
        return f"<{self.__class__.__name__} dominio={self._dominio} ultimo={ultimo}>"