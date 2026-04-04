# === cuantificador/Controlador_Cuantificador.py ===
from __future__ import annotations

import numpy as np
from typing import Union, Optional

try:
    import pandas as pd
    DataFrame = pd.DataFrame
except ImportError:
    raise ImportError("pandas es requerido para el módulo de cuantificación")

# Core
from .Controlador_Base import Controlador_Base

# Sistema
from .Resultado_Either import Resultado
from .Controlador_BioImagen import BioImagenData, ErrorBioImagen
from ..gestorLab.Categoria_Operacion import CategoriaOperacion
from ..gestorLab.Operacion import Operacion, TipoSalida

# Métodos
from ..cuantificador.intensidad.Cuantificadores_Intensidad import (
    MediaIntensidad, IntensidadIntegrada, MaximoIntensidad, MinimoIntensidad,
    MedianaIntensidad, DesviacionEstandar, CoeficienteVariacion, PercentilIntensidad,
    RelacionSenialRuido, AsimetriaIntensidad, CurtosisIntensidad, PerfilLineal,
)
from ..cuantificador.morfometria.Cuantificadores_Morfometria import (
    GeometricasBasicas, Forma, Escala, Orientacion, IntensidadForma,
)
from ..cuantificador.topologicos.Cuantificadores_Topologicos import (
    Esqueleticas, Ramificacion, Contornos, Conectividad,
    IndiceBetti, GrafoAdyacencia, DistanciaGeodesica,
)
from ..cuantificador.textura.Cuantificadores_Textura import (
    GLCM, CaracteristicasHaralick, LBP, FiltrosGabor, GLRLM, EnergiaLaws,
)
from ..cuantificador.estadisticos.Estadisticos import (
    Estadisticos, Distribuciones, Correlaciones,
)


MetodoIntensidad = Union[
    MediaIntensidad, IntensidadIntegrada, MaximoIntensidad, MinimoIntensidad,
    MedianaIntensidad, DesviacionEstandar, CoeficienteVariacion, PercentilIntensidad,
    RelacionSenialRuido, AsimetriaIntensidad, CurtosisIntensidad, PerfilLineal,
]
MetodoMorfometria = Union[
    GeometricasBasicas, Forma, Escala, Orientacion, IntensidadForma,
]
MetodoTopologico = Union[
    Esqueleticas, Ramificacion, Contornos, Conectividad,
    IndiceBetti, GrafoAdyacencia, DistanciaGeodesica,
]
MetodoTextura = Union[
    GLCM, CaracteristicasHaralick, LBP, FiltrosGabor, GLRLM, EnergiaLaws,
]
MetodoEstadistico = Union[
    Estadisticos, Distribuciones, Correlaciones,
]
MetodoCuantificador = Union[
    MetodoIntensidad, MetodoMorfometria, MetodoTopologico,
    MetodoTextura, MetodoEstadistico,
]


class Controlador_Cuantificador(Controlador_Base):
    """
    Controlador de cuantificación.

    Contrato del método:
      __call__(canal_data: np.ndarray[T,Z,Y,X]) → pd.DataFrame
      El método recibe el volumen completo y decide su propia estructura
      de filas/columnas (por objeto, por (t,z), por canal, etc.)

    Diferencias con los otros controladores:

      Sin TipoAplicacion ni Estrategias_Aplicacion:
        Los métodos son siempre globales — reciben (T,Z,Y,X) completo.
        No tiene sentido iterar por corte Z o timepoint porque las
        métricas estadísticas y morfométricas necesitan el contexto
        completo del volumen para ser significativas.

    Hooks sobreescritos (tres):
      _requiere_estrategia → False  (sin dispatch de estrategia)
      _preprocesar         → (T,Z,Y,X) float64 en lugar de 2D
      _validar_salida      → verifica que el resultado sea DataFrame no vacío

    _postprocesar NO se sobreescribe:
      el base devuelve resultados no-ndarray tal cual → DataFrame pasa directo.
    """

    def __init__(self):
        super().__init__(etapa="cuantificacion", dominio="cuantificacion")
        self._ultimo_metodo: Optional[MetodoCuantificador] = None

    # =========================================================
    # HOOKS SOBREESCRITOS
    # =========================================================

    def _requiere_estrategia(self) -> bool:
        """Sin estrategia: el base llama metodo(canal_data) directamente."""
        return False

    def _preprocesar(self, data: BioImagenData, canal: int) -> np.ndarray:
        """
        Volumen completo (T,Z,Y,X) en float64.
        Sin extracción de slice 2D — el método ve todo el volumen.
        """
        return data.datos[:, :, canal, :, :].astype(np.float64)

    def _validar_salida(self, resultado, canal_data: np.ndarray) -> None:
        """
        Verifica que el método devolvió un DataFrame con datos.
        No hay validación de shape — el DataFrame tiene su propia estructura.
        """
        if not isinstance(resultado, pd.DataFrame):
            raise TypeError(
                f"El método de cuantificación debe devolver un DataFrame, "
                f"recibido: {type(resultado).__name__}"
            )
        if resultado.empty:
            raise ValueError(
                "El método de cuantificación devolvió un DataFrame vacío"
            )

    # =========================================================
    # HELPER INTERNO
    # =========================================================

    def _crear(
        self,
        metodo: MetodoCuantificador,
        canal: int = 0,
    ):
        """
        Sin tipo_aplicacion — _requiere_estrategia=False hace que el base
        llame directamente a metodo(canal_data) sin pasar por ninguna estrategia.
        """
        self._ultimo_metodo = metodo
        return self.crear_operador(
            metodo=metodo,
            tipo_aplicacion=None,
            canal=canal,
        )

    def _operacion(
        self,
        nombre_metodo: str,
        canal: Optional[int],
        nombre: Optional[str],
        params: dict,
    ) -> Operacion:
        """
        Wrapper interno para crear_operacion.
        Fija tipo_aplicacion=None y tipo_salida=TABLA en todas las operaciones.
        Evita repetir estos dos argumentos en cada factory.
        """
        return self.crear_operacion(
            nombre_metodo=nombre_metodo,
            categoria=CategoriaOperacion.CUANTIFICACION,
            tipo_aplicacion=None,
            canal=canal,
            nombre=nombre,
            params=params,
            tipo_salida=TipoSalida.TABLA,
        )

    # =========================================================
    # FACTORIES YAML / CLI — devuelven Operacion
    # Sin parámetro `tipo` — siempre global
    # =========================================================

    # ── INTENSIDAD ────────────────────────────────────────────

    def crear_operacion_media_intensidad(
        self,
        canal: Optional[int] = 0,
        nombre: Optional[str] = None,
        # TODO: params
    ) -> Operacion:
        return self._operacion("media_intensidad", canal, nombre, {})

    def crear_operacion_intensidad_integrada(
        self,
        canal: Optional[int] = 0,
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self._operacion("intensidad_integrada", canal, nombre, {})

    def crear_operacion_maximo_intensidad(
        self,
        canal: Optional[int] = 0,
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self._operacion("maximo_intensidad", canal, nombre, {})

    def crear_operacion_minimo_intensidad(
        self,
        canal: Optional[int] = 0,
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self._operacion("minimo_intensidad", canal, nombre, {})

    def crear_operacion_mediana_intensidad(
        self,
        canal: Optional[int] = 0,
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self._operacion("mediana_intensidad", canal, nombre, {})

    def crear_operacion_desviacion_estandar(
        self,
        canal: Optional[int] = 0,
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self._operacion("desviacion_estandar", canal, nombre, {})

    def crear_operacion_coeficiente_variacion(
        self,
        canal: Optional[int] = 0,
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self._operacion("coeficiente_variacion", canal, nombre, {})

    def crear_operacion_percentil_intensidad(
        self,
        # TODO: params (percentil: float = 95.0)
        canal: Optional[int] = 0,
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self._operacion("percentil_intensidad", canal, nombre, {})

    def crear_operacion_relacion_senal_ruido(
        self,
        canal: Optional[int] = 0,
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self._operacion("relacion_senal_ruido", canal, nombre, {})

    def crear_operacion_asimetria_intensidad(
        self,
        canal: Optional[int] = 0,
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self._operacion("asimetria_intensidad", canal, nombre, {})

    def crear_operacion_curtosis_intensidad(
        self,
        canal: Optional[int] = 0,
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self._operacion("curtosis_intensidad", canal, nombre, {})

    def crear_operacion_perfil_lineal(
        self,
        # TODO: params (punto_inicio, punto_fin, n_puntos)
        canal: Optional[int] = 0,
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self._operacion("perfil_lineal", canal, nombre, {})

    # ── MORFOMETRÍA ───────────────────────────────────────────

    def crear_operacion_geometricas_basicas(
        self,
        canal: Optional[int] = 0,
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self._operacion("geometricas_basicas", canal, nombre, {})

    def crear_operacion_forma(
        self,
        canal: Optional[int] = 0,
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self._operacion("forma", canal, nombre, {})

    def crear_operacion_escala(
        self,
        # TODO: params (pixel_size_xy, pixel_size_z, unidad)
        canal: Optional[int] = 0,
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self._operacion("escala", canal, nombre, {})

    def crear_operacion_orientacion(
        self,
        canal: Optional[int] = 0,
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self._operacion("orientacion", canal, nombre, {})

    def crear_operacion_intensidad_forma(
        self,
        canal: Optional[int] = 0,
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self._operacion("intensidad_forma", canal, nombre, {})

    # ── TOPOLÓGICOS ───────────────────────────────────────────

    def crear_operacion_esqueleticas(
        self,
        canal: Optional[int] = 0,
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self._operacion("esqueleticas", canal, nombre, {})

    def crear_operacion_ramificacion(
        self,
        canal: Optional[int] = 0,
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self._operacion("ramificacion", canal, nombre, {})

    def crear_operacion_contornos(
        self,
        # TODO: params (nivel)
        canal: Optional[int] = 0,
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self._operacion("contornos", canal, nombre, {})

    def crear_operacion_conectividad(
        self,
        canal: Optional[int] = 0,
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self._operacion("conectividad", canal, nombre, {})

    def crear_operacion_indice_betti(
        self,
        canal: Optional[int] = 0,
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self._operacion("indice_betti", canal, nombre, {})

    def crear_operacion_grafo_adyacencia(
        self,
        canal: Optional[int] = 0,
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self._operacion("grafo_adyacencia", canal, nombre, {})

    def crear_operacion_distancia_geodesica(
        self,
        canal: Optional[int] = 0,
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self._operacion("distancia_geodesica", canal, nombre, {})

    # ── TEXTURA ───────────────────────────────────────────────

    def crear_operacion_glcm(
        self,
        # TODO: params (distancias, angulos, niveles)
        canal: Optional[int] = 0,
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self._operacion("glcm", canal, nombre, {})

    def crear_operacion_haralick(
        self,
        canal: Optional[int] = 0,
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self._operacion("caracteristicas_haralick", canal, nombre, {})

    def crear_operacion_lbp(
        self,
        # TODO: params (radio, n_puntos, metodo)
        canal: Optional[int] = 0,
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self._operacion("lbp", canal, nombre, {})

    def crear_operacion_filtros_gabor(
        self,
        # TODO: params (frecuencias, thetas)
        canal: Optional[int] = 0,
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self._operacion("filtros_gabor", canal, nombre, {})

    def crear_operacion_glrlm(
        self,
        # TODO: params (angulos)
        canal: Optional[int] = 0,
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self._operacion("glrlm", canal, nombre, {})

    def crear_operacion_energia_laws(
        self,
        canal: Optional[int] = 0,
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self._operacion("energia_laws", canal, nombre, {})

    # ── ESTADÍSTICOS ─────────────────────────────────────────

    def crear_operacion_estadisticos(
        self,
        canal: Optional[int] = 0,
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self._operacion("estadisticos", canal, nombre, {})

    def crear_operacion_distribuciones(
        self,
        # TODO: params (distribuciones_a_ajustar)
        canal: Optional[int] = 0,
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self._operacion("distribuciones", canal, nombre, {})

    def crear_operacion_correlaciones(
        self,
        # TODO: params (metodo: pearson/spearman/kendall)
        canal: Optional[int] = 0,
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self._operacion("correlaciones", canal, nombre, {})

    # =========================================================
    # USO IMPERATIVO
    # =========================================================

    def aplicar(
        self,
        data: BioImagenData,
        metodo: MetodoCuantificador,
        canal: int = 0,
    ) -> Resultado[pd.DataFrame, ErrorBioImagen]:
        return self._crear(metodo, canal)(data)

    def __repr__(self):
        nombre = getattr(self._ultimo_metodo, "nombre", "None")
        return f"<Controlador_Cuantificador ultimo={nombre}>"