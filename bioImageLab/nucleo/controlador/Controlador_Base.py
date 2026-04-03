from __future__ import annotations

import numpy as np
from dataclasses import replace
from typing import Callable, Optional, Dict, Any, TypeVar, Generic, Type

# Tipos genéricos
TSalida = TypeVar("TSalida")

# Registro global de métodos (para YAML / factories dinámicos)
REGISTRO_METODOS: Dict[str, Type] = {}


def registrar_metodo(cls):
    nombre = getattr(cls, "nombre", cls.__name__.lower())
    REGISTRO_METODOS[nombre] = cls
    return cls


# Imports de tu sistema
from .Resultado_Either import Resultado, Ok, Err
from .Controlador_BioImagen import BioImagenData, ErrorBioImagen


class Controlador_Base(Generic[TSalida]):
    """
    Core genérico reutilizable para procesamiento de BioImagenData.

    Soporta:
    - Imagen → Imagen
    - Imagen → máscara
    - Imagen → métricas / DataFrame
    """

    def __init__(self, etapa: str = "procesamiento"):
        self._etapa = etapa
        self._ultimo_metodo: Optional[Any] = None
        self._cache: Optional[Dict[str, Any]] = None

    # =========================================================
    # ===================== HOOKS ==============================
    # =========================================================

    def _preprocesar(self, data: BioImagenData, canal_idx: int):
        return data.datos[:, :, canal_idx, :, :]

    def _postprocesar(self, data: BioImagenData, resultado, canal_idx: int):
        if not isinstance(resultado, np.ndarray):
            return resultado  # analizador

        nuevos = data.datos.copy()
        nuevos[:, :, canal_idx, :, :] = resultado
        return replace(data, datos=nuevos)

    def _requiere_estrategia(self) -> bool:
        return True

    def _transformar_dtype(self, arr: np.ndarray) -> np.ndarray:
        return arr.astype(np.float64)

    def _adaptar_metodo(self, metodo):
        from .Estrategias_Aplicacion import adaptar_metodo
        return adaptar_metodo(metodo)

    def _obtener_estrategia(self, tipo_aplicacion):
        if tipo_aplicacion is None:
            return None
        return tipo_aplicacion.estrategia()

    def _validar_entrada(self, data: BioImagenData, canal: int):
        if not (0 <= canal < data.dims.C):
            raise ValueError(f"Canal {canal} fuera de rango [0, {data.dims.C-1}]")

    def _extraer_canal(self, data: BioImagenData, canal: int):
        return data.datos[:, :, canal, :, :]

    def _aplicar_metodo(self, canal_data, metodo, estrategia):
        if estrategia is None:
            return metodo(canal_data)
        return estrategia(canal_data, metodo)

    def _validar_salida(self, resultado, canal_data):
        if isinstance(resultado, np.ndarray):
            if resultado.shape != canal_data.shape:
                raise ValueError(
                    f"Shape inválido: esperado {canal_data.shape}, obtenido {resultado.shape}"
                )

    def _log_exito(self, salida, metodo, canal, tipo_aplicacion):
        return Ok(salida)

    def _log_error(self, error, metodo, canal):
        return Err(error)

    # =========================================================
    # ===================== CORE ===============================
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

            canal_data = self._extraer_canal(data, canal)
            canal_data = self._transformar_dtype(canal_data)

            metodo_adaptado = self._adaptar_metodo(metodo)

            estrategia = (
                self._obtener_estrategia(tipo_aplicacion)
                if self._requiere_estrategia()
                else None
            )

            resultado = self._aplicar_metodo(
                canal_data,
                metodo_adaptado,
                estrategia
            )

            self._validar_salida(resultado, canal_data)

            salida = self._postprocesar(data, resultado, canal)

            return self._log_exito(salida, metodo, canal, tipo_aplicacion)

        except Exception as e:
            return self._log_error(
                ErrorBioImagen(
                    etapa=self._etapa,
                    mensaje=f"Error en {nombre}: {str(e)}",
                    ruta=data.ruta_origen,
                    causa=e
                ),
                metodo,
                canal
            )

    # =========================================================
    # ===================== FACTORIES ==========================
    # =========================================================

    def crear_operador(
        self,
        metodo,
        tipo_aplicacion=None
    ) -> Callable[[BioImagenData, int], Resultado[TSalida, ErrorBioImagen]]:

        self._ultimo_metodo = metodo

        def _op(data: BioImagenData, canal_idx: int = 0):
            return self._ejecutar(
                data,
                metodo,
                tipo_aplicacion,
                canal_idx
            )

        return _op

    # factory dinámico desde string (YAML-friendly)
    def crear(
        self,
        nombre_metodo: str,
        tipo_aplicacion=None,
        **params
    ):
        if nombre_metodo not in REGISTRO_METODOS:
            raise ValueError(f"Método '{nombre_metodo}' no registrado")

        clase = REGISTRO_METODOS[nombre_metodo]
        metodo = clase(**params)

        return self.crear_operador(metodo, tipo_aplicacion)

    # =========================================================
    # ================= MULTICANAL =============================
    # =========================================================

    def crear_operador_multicanal(
        self,
        metodo,
        tipo_aplicacion=None
    ):

        operador = self.crear_operador(metodo, tipo_aplicacion)

        def _multi(data):
            resultado = Ok(data)

            for c in range(data.dims.C):
                resultado = resultado.bind(lambda d, canal=c: operador(d, canal))
                if resultado.es_err():
                    break

            return resultado

        return _multi

    # =========================================================
    # ================= PIPELINE ===============================
    # =========================================================

    def crear_operacion_pipeline(
        self,
        metodo,
        tipo_aplicacion,
        canal: int,
        categoria,
        nombre: Optional[str] = None,
        extra_params: Optional[Dict[str, Any]] = None
    ):

        from ..gestorLab.Operacion import Operacion, TipoSalida

        nombre_op = nombre or f"{self._etapa}_{metodo.__class__.__name__}"

        operador = self.crear_operador(metodo, tipo_aplicacion)

        return Operacion(
            nombre=nombre_op,
            categoria=categoria,
            instancia_callable=operador,
            canal_objetivo=canal,
            parametros_originales=extra_params or {},
            tipo_salida=TipoSalida.IMAGEN
        )

    # =========================================================
    # ================= UTIL ==================================
    # =========================================================

    def reset(self):
        self._cache = None
        self._ultimo_metodo = None

    def __repr__(self):
        ultimo = getattr(self._ultimo_metodo, "nombre", "Ninguno")
        return f"<{self.__class__.__name__} ultimo={ultimo}>"