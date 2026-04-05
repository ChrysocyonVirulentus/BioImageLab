# === modelador/Controlador_Modelador.py ===
from __future__ import annotations

from dataclasses import dataclass
from typing import Union, Optional, Callable

import pandas as pd

# Sistema — solo Resultado_Either, sin BioImagenData
from .Resultado_Either import Resultado, Ok, Err

# Operacion sí se reutiliza — es agnóstica del dominio
from ..gestorLab.Categoria_Operacion import CategoriaOperacion
from ..gestorLab.Operacion import Operacion, TipoDato
from ..gestorLab.Registro_Metodos import registro_metodos

# Métodos
from ..modelador.dimensionalidad.Modelador_Dimensionalidad import (
    PCADimensional,
    UMAPDimensional,
    tSNE,
)
from ..modelador.clustering.Modelador_Clustering import (
    KMeans,
    DBSCANClustering,
    HDBSCANClustering,
)
from ..modelador.clasificacion.Modelador_Clasificacion import (
    SVMClasificador,
    LogisticRegressionClasificador,
    RandomForestClasificador,
)


# ==================== ERROR PROPIO ====================
# Sin BioImagenData ni ruta de archivo — el dominio es tabular.

@dataclass(frozen=True)
class ErrorModelado:
    """
    Error del dominio de modelado.
    Análogo a ErrorBioImagen pero para datos tabulares.
    """
    etapa:   str
    mensaje: str
    causa:   Optional[Exception] = None

    def con_contexto(self, nueva_etapa: str) -> ErrorModelado:
        return ErrorModelado(
            etapa=f"{nueva_etapa} -> {self.etapa}",
            mensaje=self.mensaje,
            causa=self.causa,
        )


# ==================== TIPOS ====================

MetodoDimensionalidad = Union[PCADimensional, UMAPDimensional, tSNE]
MetodoClustering      = Union[KMeans, DBSCANClustering, HDBSCANClustering]
MetodoClasificacion   = Union[SVMClasificador, LogisticRegressionClasificador, RandomForestClasificador]
MetodoModelador       = Union[
    MetodoDimensionalidad,
    MetodoClustering,
    MetodoClasificacion,
]


# ==================== CONTROLADOR ====================

class Controlador_Modelador:
    """
    Controlador de modelado estadístico/ML.

    Dominio: DataFrame → DataFrame.
    Sin BioImagenData, sin Estrategias_Aplicacion, sin Controlador_Base.

    El flujo es el más simple posible:
        DataFrame → metodo(df) → DataFrame

    Cada método es responsable de:
        - Validar las columnas que necesita
        - Devolver un DataFrame estructurado con sus resultados
            (coordenadas reducidas, etiquetas de cluster, probabilidades, etc.)

    El controlador es responsable de:
        - Capturar excepciones y envolverlas en Err(ErrorModelado)
        - Producir Operacion para el pipeline builder (YAML / CLI)
        - Mantener registro del último método ejecutado
    """

    def __init__(self):
        self._dominio = "modelado"
        self._ultimo_metodo: Optional[MetodoModelador] = None

    # =========================================================
    # CORE INTERNO
    # =========================================================

    def _ejecutar(
        self,
        df: pd.DataFrame,
        metodo: MetodoModelador,
    ) -> Resultado[pd.DataFrame, ErrorModelado]:
        """
        Ejecuta el método sobre el DataFrame.
        Única responsabilidad: capturar excepciones → Err.
        """
        nombre = getattr(metodo, "nombre", metodo.__class__.__name__)
        try:
            self._validar_entrada(df)
            resultado = metodo(df)
            self._validar_salida(resultado, nombre)
            return Ok(resultado)
        except Exception as e:
            return Err(ErrorModelado(
                etapa=self._dominio,
                mensaje=f"Error en '{nombre}': {e}",
                causa=e,
            ))

    def _validar_entrada(self, df: pd.DataFrame) -> None:
        if not isinstance(df, pd.DataFrame):
            raise TypeError(
                f"El input debe ser un DataFrame, recibido: {type(df).__name__}"
            )
        if df.empty:
            raise ValueError("El DataFrame de entrada está vacío")

    def _validar_salida(self, resultado, nombre: str) -> None:
        if not isinstance(resultado, pd.DataFrame):
            raise TypeError(
                f"'{nombre}' debe devolver un DataFrame, "
                f"recibido: {type(resultado).__name__}"
            )
        if resultado.empty:
            raise ValueError(f"'{nombre}' devolvió un DataFrame vacío")

    # =========================================================
    # FACTORIES DE CALLABLE
    # =========================================================

    def crear_operador(
        self,
        metodo: MetodoModelador,
    ) -> Callable[[pd.DataFrame], Resultado[pd.DataFrame, ErrorModelado]]:
        """
        Devuelve callable: (DataFrame) → Resultado[DataFrame, ErrorModelado].
        Análogo a crear_operador del Controlador_Base.
        """
        self._ultimo_metodo = metodo

        def _op(df: pd.DataFrame) -> Resultado[pd.DataFrame, ErrorModelado]:
            return self._ejecutar(df, metodo)

        return _op

    def crear(
        self,
        nombre_metodo: str,
        **params,
    ) -> Callable[[pd.DataFrame], Resultado[pd.DataFrame, ErrorModelado]]:
        """Para uso dinámico desde YAML — resuelve clase vía registro."""
        clase  = registro_metodos.obtener(self._dominio, nombre_metodo)
        metodo = clase(**params)
        return self.crear_operador(metodo)

    def crear_operacion(
        self,
        nombre_metodo: str,
        categoria: CategoriaOperacion,
        nombre: Optional[str] = None,
        params: Optional[dict] = None,
        tipo_salida: TipoDato = TipoDato.TABLA,
        # ── kwargs de compatibilidad con Constructor_Flujo_Trabajo ──
        canal: Optional[int] = None,            # ignorado — dominio tabular sin canales
        tipo_aplicacion = None,                 # ignorado — siempre global
    ) -> Operacion:
        """
        Construye una Operacion lista para el pipeline builder.
        Sin canal_objetivo — el dominio es tabular, no hay canales.
        """
        clase  = registro_metodos.obtener(self._dominio, nombre_metodo)
        metodo = clase(**(params or {}))
        callable_ = self.crear_operador(metodo)

        return Operacion(
            nombre               = nombre or f"modelado_{nombre_metodo}",
            categoria            = categoria,
            instancia_callable   = callable_,
            canal_objetivo       = None,     # sin canales
            parametros_originales= params or {},
            tipo_salida          = tipo_salida,
        )

    def _operacion(
        self,
        nombre_metodo: str,
        nombre: Optional[str],
        params: dict,
        tipo_salida: TipoDato = TipoDato.TABLA,
    ) -> Operacion:
        """Helper interno — fija categoría y evita repetición en factories."""
        return self.crear_operacion(
            nombre_metodo=nombre_metodo,
            categoria=CategoriaOperacion.MODELADO,
            nombre=nombre,
            params=params,
            tipo_salida=tipo_salida,
        )

    # =========================================================
    # FACTORIES YAML / CLI — devuelven Operacion
    # =========================================================

    # ── DIMENSIONALIDAD ───────────────────────────────────────

    def crear_operacion_pca(
        self,
        # TODO: params (n_componentes, ...)
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self._operacion("pca", nombre, {})

    def crear_operacion_umap(
        self,
        # TODO: params (n_componentes, n_vecinos, min_dist, ...)
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self._operacion("umap", nombre, {})

    def crear_operacion_tsne(
        self,
        # TODO: params (n_componentes, perplejidad, iteraciones, ...)
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self._operacion("tsne", nombre, {})

    # ── CLUSTERING ────────────────────────────────────────────

    def crear_operacion_kmeans(
        self,
        # TODO: params (n_clusters, init, n_init, ...)
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self._operacion("kmeans", nombre, {})

    def crear_operacion_dbscan(
        self,
        # TODO: params (eps, min_samples, metrica, ...)
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self._operacion("dbscan", nombre, {})

    def crear_operacion_hdbscan(
        self,
        # TODO: params (min_cluster_size, min_samples, ...)
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self._operacion("hdbscan", nombre, {})

    # ── CLASIFICACIÓN ─────────────────────────────────────────

    def crear_operacion_svm(
        self,
        # TODO: params (kernel, C, gamma, ...)
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self._operacion(
            "svm", nombre, {},
            tipo_salida=TipoDato.MODELO,   # SVM produce modelo + predicciones
        )

    def crear_operacion_regresion_logistica(
        self,
        # TODO: params (C, max_iter, solver, ...)
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self._operacion(
            "regresion_logistica", nombre, {},
            tipo_salida=TipoDato.MODELO,
        )

    def crear_operacion_random_forest(
        self,
        # TODO: params (n_estimadores, max_depth, ...)
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self._operacion(
            "random_forest", nombre, {},
            tipo_salida=TipoDato.MODELO,
        )

    # =========================================================
    # USO IMPERATIVO
    # =========================================================

    def aplicar(
        self,
        df: pd.DataFrame,
        metodo: MetodoModelador,
    ) -> Resultado[pd.DataFrame, ErrorModelado]:
        """Aplica directamente. Para scripts y tests."""
        return self.crear_operador(metodo)(df)

    def reset(self):
        self._ultimo_metodo = None

    def __repr__(self):
        nombre = getattr(self._ultimo_metodo, "nombre", "None")
        return f"<Controlador_Modelador ultimo={nombre}>"