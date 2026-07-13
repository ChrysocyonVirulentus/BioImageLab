# === cuantificador/Controlador_Cuantificador.py ===
from __future__ import annotations

import numpy as np
from typing import Union, Optional, Dict, Any

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
from ..gestorLab.Operacion import Operacion, TipoDato

# Estrategias — sin override (usa las del base)
from .Estrategias_Aplicacion import (
    TipoAplicacion,
    Global,
    PorCorteZ,
    PorTimepoint,
    PorCorteEspaciotemporal,
)

# ── INTENSIDAD ──────────────────────────────────────────────
from ..cuantificador.intensidad.Cuantificadores_Intensidad import (
    MediaIntensidad,
    IntensidadIntegrada,
    MaximoIntensidad,
    MinimoIntensidad,
    MedianaIntensidad,
    DesviacionEstandar,
    CoeficienteVariacion,
    PercentilIntensidad,
    RelacionSenialRuido,
    AsimetriaIntensidad,
    CurtosisIntensidad,
    PerfilLineal,
)

# ── MORFOMETRÍA ───────────────────────────────────────────
from ..cuantificador.morfometria.Cuantificadores_Morfometricos import (
    Area,
    Perimetro,
    Centroide,
    CajaFrontera,
    DiametroEquivalente,
    Excentricidad,
    Circularidad,
    Compactacion,
    Orientacion,
    Convexidad,
    Concavidad,
    Forma,
)

# ── TOPOLÓGICOS ───────────────────────────────────────────
from ..cuantificador.topologicos.Cuantificadores_Topologicos import (
    MetricasEsqueleticas,
    Branching,
    Contornos,
    Conectividad,
    DistanciaGeodesica,
)

# ── TEXTURA ────────────────────────────────────────────────
from ..cuantificador.texturas.Cuantificadores_Texturas import (
    GrayLevelCoocurrenceMatrix,
    CaracteristicasHaralick,
    LocalBinaryPattern,
    FiltrosGabor,
    GrayLevelRunLengthMatrix,
    EnergiaLaws,
)

# ── ESTADÍSTICOS ─────────────────────────────────────────
from ..cuantificador.estadisticos.Estadisticos import (
    EstadisticosDescriptivos,
    Distribuciones,
    Correlaciones,
)


# ── Type aliases ───────────────────────────────────────────

MetodoIntensidad = Union[
    MediaIntensidad, IntensidadIntegrada, MaximoIntensidad, MinimoIntensidad,
    MedianaIntensidad, DesviacionEstandar, CoeficienteVariacion,
    PercentilIntensidad, RelacionSenialRuido, AsimetriaIntensidad,
    CurtosisIntensidad, PerfilLineal,
]

MetodoMorfometria = Union[
    Area, Perimetro, Centroide, CajaFrontera, DiametroEquivalente,
    Excentricidad, Circularidad, Compactacion, Orientacion,
    Convexidad, Concavidad, Forma,
]

MetodoTopologico = Union[
    MetricasEsqueleticas, Branching, Contornos,
    Conectividad, DistanciaGeodesica,
]

MetodoTextura = Union[
    GrayLevelCoocurrenceMatrix, CaracteristicasHaralick,
    LocalBinaryPattern, FiltrosGabor,
    GrayLevelRunLengthMatrix, EnergiaLaws,
]

MetodoEstadistico = Union[
    EstadisticosDescriptivos, Distribuciones, Correlaciones,
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

    def _validar_salida(self, resultado: Any, canal_data: np.ndarray) -> None:
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
            tipo_salida=TipoDato.TABLA,
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
        percentil: float = 95.0,
        canal: Optional[int] = 0,
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self._operacion(
            "percentil_intensidad", canal, nombre, {"percentil": percentil}
        )

    def crear_operacion_relacion_senal_ruido(
        self,
        usar_media_fondo: bool = False,
        canal: Optional[int] = 0,
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self._operacion(
            "relacion_senal_ruido", canal, nombre,
            {"usar_media_fondo": usar_media_fondo},
        )

    def crear_operacion_asimetria_intensidad(
        self,
        canal: Optional[int] = 0,
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self._operacion("asimetria_intensidad", canal, nombre, {})

    def crear_operacion_curtosis_intensidad(
        self,
        exceso: bool = True,
        canal: Optional[int] = 0,
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self._operacion(
            "curtosis_intensidad", canal, nombre, {"exceso": exceso}
        )

    def crear_operacion_perfil_lineal(
        self,
        punto_inicio: tuple = (0, 0),
        punto_fin: tuple = (10, 10),
        num_puntos: int = 100,
        solo_en_mascara: bool = True,
        orden_interpolacion: int = 1,
        canal: Optional[int] = 0,
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self._operacion(
            "perfil_lineal", canal, nombre,
            {
                "punto_inicio": punto_inicio,
                "punto_fin": punto_fin,
                "num_puntos": num_puntos,
                "solo_en_mascara": solo_en_mascara,
                "orden_interpolacion": orden_interpolacion,
            },
        )

    # ── MORFOMETRÍA ───────────────────────────────────────────

    def crear_operacion_area(
        self,
        por_region: bool = False,
        canal: Optional[int] = 0,
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self._operacion(
            "area", canal, nombre, {"por_region": por_region}
        )

    def crear_operacion_perimetro(
        self,
        por_region: bool = False,
        metodo: str = "crofton",
        canal: Optional[int] = 0,
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self._operacion(
            "perimetro", canal, nombre,
            {"por_region": por_region, "metodo": metodo},
        )

    def crear_operacion_centroide(
        self,
        por_region: bool = True,
        canal: Optional[int] = 0,
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self._operacion(
            "centroide", canal, nombre, {"por_region": por_region}
        )

    def crear_operacion_caja_frontera(
        self,
        por_region: bool = True,
        canal: Optional[int] = 0,
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self._operacion(
            "caja_frontera", canal, nombre, {"por_region": por_region}
        )

    def crear_operacion_diametro_equivalente(
        self,
        por_region: bool = False,
        canal: Optional[int] = 0,
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self._operacion(
            "diametro_equivalente", canal, nombre, {"por_region": por_region}
        )

    def crear_operacion_excentricidad(
        self,
        por_region: bool = False,
        canal: Optional[int] = 0,
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self._operacion(
            "excentricidad", canal, nombre, {"por_region": por_region}
        )

    def crear_operacion_circularidad(
        self,
        por_region: bool = False,
        suavizar: bool = True,
        canal: Optional[int] = 0,
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self._operacion(
            "circularidad", canal, nombre,
            {"por_region": por_region, "suavizar": suavizar},
        )

    def crear_operacion_compactacion(
        self,
        por_region: bool = False,
        normalizada: bool = True,
        canal: Optional[int] = 0,
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self._operacion(
            "compactacion", canal, nombre,
            {"por_region": por_region, "normalizada": normalizada},
        )

    def crear_operacion_orientacion(
        self,
        por_region: bool = False,
        modo: str = "degrees",
        canal: Optional[int] = 0,
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self._operacion(
            "orientacion", canal, nombre,
            {"por_region": por_region, "modo": modo},
        )

    def crear_operacion_convexidad(
        self,
        por_region: bool = False,
        canal: Optional[int] = 0,
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self._operacion(
            "convexidad", canal, nombre, {"por_region": por_region}
        )

    def crear_operacion_concavidad(
        self,
        por_region: bool = False,
        modo: str = "area",
        canal: Optional[int] = 0,
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self._operacion(
            "concavidad", canal, nombre,
            {"por_region": por_region, "modo": modo},
        )

    def crear_operacion_forma(
        self,
        por_region: bool = False,
        log_transform: bool = True,
        canal: Optional[int] = 0,
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self._operacion(
            "forma", canal, nombre,
            {"por_region": por_region, "log_transform": log_transform},
        )

    # ── TOPOLÓGICOS ───────────────────────────────────────────

    def crear_operacion_esqueleticas(
        self,
        conectividad: int = 8,
        pruning: bool = False,
        longitud_min_rama: int = 5,
        canal: Optional[int] = 0,
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self._operacion(
            "metricas_esqueleticas", canal, nombre,
            {
                "conectividad": conectividad,
                "pruning": pruning,
                "longitud_min_rama": longitud_min_rama,
            },
        )

    def crear_operacion_ramificacion(
        self,
        orden_metodo: str = "strahler",
        min_longitud_rama: float = 5.0,
        canal: Optional[int] = 0,
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self._operacion(
            "branching", canal, nombre,
            {
                "orden_metodo": orden_metodo,
                "min_longitud_rama": min_longitud_rama,
            },
        )

    def crear_operacion_contornos(
        self,
        por_region: bool = False,
        suavizado: float = 0.0,
        canal: Optional[int] = 0,
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self._operacion(
            "contornos", canal, nombre,
            {"por_region": por_region, "suavizado": suavizado},
        )

    def crear_operacion_conectividad(
        self,
        conectividad: int = 8,
        canal: Optional[int] = 0,
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self._operacion(
            "conectividad", canal, nombre, {"conectividad": conectividad}
        )

    def crear_operacion_distancia_geodesica(
        self,
        conectividad: int = 8,
        canal: Optional[int] = 0,
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self._operacion(
            "distancia_geodesica", canal, nombre, {"conectividad": conectividad}
        )

    # ── TEXTURA ───────────────────────────────────────────────

    def crear_operacion_glcm(
        self,
        distancias: Optional[list] = None,
        angulos: Optional[list] = None,
        num_niveles: int = 32,
        simetrica: bool = True,
        promediar_angulos: bool = False,
        canal: Optional[int] = 0,
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self._operacion(
            "glcm", canal, nombre,
            {
                "distancias": distancias or [1],
                "angulos": angulos or [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                "num_niveles": num_niveles,
                "simetrica": simetrica,
                "promediar_angulos": promediar_angulos,
            },
        )

    def crear_operacion_haralick(
        self,
        distancias: Optional[list] = None,
        angulos: Optional[list] = None,
        num_niveles: int = 32,
        promediar_angulos: bool = True,
        features: Optional[list] = None,
        canal: Optional[int] = 0,
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self._operacion(
            "haralick", canal, nombre,
            {
                "distancias": distancias or [1],
                "angulos": angulos or [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                "num_niveles": num_niveles,
                "promediar_angulos": promediar_angulos,
                "features": features,
            },
        )

    def crear_operacion_lbp(
        self,
        radio: int = 1,
        num_puntos: int = 8,
        metodo: str = "uniform",
        canal: Optional[int] = 0,
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self._operacion(
            "lbp", canal, nombre,
            {
                "radio": radio,
                "num_puntos": num_puntos,
                "metodo": metodo,
            },
        )

    def crear_operacion_filtros_gabor(
        self,
        frecuencias: Optional[list] = None,
        orientaciones: Optional[list] = None,
        sigma_x: float = 1.0,
        sigma_y: float = 1.0,
        canal: Optional[int] = 0,
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self._operacion(
            "filtros_gabor", canal, nombre,
            {
                "frecuencias": frecuencias or [0.1, 0.2, 0.3],
                "orientaciones": orientaciones or [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                "sigma_x": sigma_x,
                "sigma_y": sigma_y,
            },
        )

    def crear_operacion_glrlm(
        self,
        angulos: Optional[list] = None,
        num_niveles: int = 16,
        promediar_angulos: bool = True,
        canal: Optional[int] = 0,
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self._operacion(
            "glrlm", canal, nombre,
            {
                "angulos": angulos or [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                "num_niveles": num_niveles,
                "promediar_angulos": promediar_angulos,
            },
        )

    def crear_operacion_energia_laws(
        self,
        filtros: Optional[list] = None,
        tamaño_ventana_energia: int = 15,
        combinar_transpuestos: bool = True,
        estadisticas: Optional[list] = None,
        canal: Optional[int] = 0,
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self._operacion(
            "energia_laws", canal, nombre,
            {
                "filtros": filtros,
                "tamaño_ventana_energia": tamaño_ventana_energia,
                "combinar_transpuestos": combinar_transpuestos,
                "estadisticas": estadisticas or ["media", "desviacion"],
            },
        )

    # ── ESTADÍSTICOS ─────────────────────────────────────────

    def crear_operacion_estadisticos_descriptivos(
        self,
        estadisticos: Optional[list] = None,
        percentiles: Optional[list] = None,
        ignorar_nan: bool = True,
        canal: Optional[int] = 0,
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self._operacion(
            "estadisticos_descriptivos", canal, nombre,
            {
                "estadisticos": estadisticos,
                "percentiles": percentiles,
                "ignorar_nan": ignorar_nan,
            },
        )

    def crear_operacion_distribuciones(
        self,
        distribuciones: Optional[list] = None,
        test_normalidad: bool = True,
        bins_histograma: int = 50,
        canal: Optional[int] = 0,
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self._operacion(
            "distribuciones", canal, nombre,
            {
                "distribuciones": distribuciones,
                "test_normalidad": test_normalidad,
                "bins_histograma": bins_histograma,
            },
        )

    def crear_operacion_correlaciones(
        self,
        metodo: str = "pearson",
        umbral_significancia: float = 0.05,
        corregir_multiple: bool = True,
        canal: Optional[int] = 0,
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self._operacion(
            "correlaciones", canal, nombre,
            {
                "metodo": metodo,
                "umbral_significancia": umbral_significancia,
                "corregir_multiple": corregir_multiple,
            },
        )

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