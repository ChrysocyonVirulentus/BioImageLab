# === analizador/Controlador_Analizador.py ===
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Union, Optional, Callable, Any

import pandas as pd

try:
    import matplotlib.figure
    Figure = matplotlib.figure.Figure
except ImportError:
    Figure = Any  # se valida en runtime

# Sistema — solo Resultado_Either
from .Resultado_Either import Resultado, Ok, Err

# Operacion es agnóstica del dominio
from ..gestorLab.Categoria_Operacion import CategoriaOperacion
from ..gestorLab.Operacion import Operacion, TipoDato
from ..gestorLab.Registro_Metodos import registro_metodos

# Plots
from ..analizador.plots.Plots_Estadisticos import (
    HistogramaIntensidad,
    BoxplotCanales,
    ViolinDistribucion,
    ScatterFeatures,
    HeatmapCorrelacion,
    CurvaROC,
    MatrizConfusion,
)
from ..analizador.plots.Plots_Imagen import (
    MIPProyeccion,
    OrthoView,
    StackViewer,
    OverlayMascara,
    PerfilIntensidad,
    ComparacionCanales,
)
from ..analizador.plots.Plots_Modelos import (
    BiplotPCA,
    UMAPScatter,
    ClusterMap,
    ImportanciaFeatures,
    SeparabilidadClases,
)
from ..analizador.plots.Estetica import Estetica

# Exportación
from ..analizador.exportacion.csv import ExportadorCSV
from ..analizador.exportacion.parquet import ExportadorParquet
from ..analizador.exportacion.figures import ExportadorFiguras


# ==================== ERROR PROPIO ====================

@dataclass(frozen=True)
class ErrorAnalizador:
    """
    Error del dominio de análisis/visualización/exportación.
    Sin ruta de imagen ni canal — el dominio es tabular o gráfico.
    """
    etapa:   str            # "plot", "exportacion", "qc"
    mensaje: str
    causa:   Optional[Exception] = None

    def con_contexto(self, nueva_etapa: str) -> ErrorAnalizador:
        return ErrorAnalizador(
            etapa=f"{nueva_etapa} -> {self.etapa}",
            mensaje=self.mensaje,
            causa=self.causa,
        )


# ==================== TIPOS ====================

# Plots — input puede ser DataFrame, BioImagenData, o ambos
MetodoPlotEstadistico = Union[
    HistogramaIntensidad, BoxplotCanales, ViolinDistribucion,
    ScatterFeatures, HeatmapCorrelacion, CurvaROC, MatrizConfusion,
]
MetodoPlotImagen = Union[
    MIPProyeccion, OrthoView, StackViewer,
    OverlayMascara, PerfilIntensidad, ComparacionCanales,
]
MetodoPlotModelo = Union[
    BiplotPCA, UMAPScatter, ClusterMap,
    ImportanciaFeatures, SeparabilidadClases,
]
MetodoPlot = Union[MetodoPlotEstadistico, MetodoPlotImagen, MetodoPlotModelo]

# Exportación — procedimientos, devuelven Path del archivo escrito
MetodoExportacion = Union[ExportadorCSV, ExportadorParquet, ExportadorFiguras]

# Input genérico para plots
InputPlot = Union[pd.DataFrame, Any]  # Any cubre BioImagenData sin importarlo


# ==================== CONTROLADOR ====================

class Controlador_Analizador:
    """
    Controlador de análisis, visualización y exportación.

    Tres subdominios con contratos distintos:

        plots/
            input:  DataFrame | BioImagenData  (según el método)
            output: Figure
            TipoDato.VISUALIZACION

        exportacion/
            input:  DataFrame | Figure
            output: Path  (archivo escrito a disco — side effect puro)
            TipoDato.NINGUNA

        qc/
            NO IMPLEMENTADO — stub con NotImplementedError explícito.

    Sin Controlador_Base: no hay BioImagenData como contrato central,
    no hay canales, no hay estrategias de iteración.
    Sin BioImagenData en imports: plots de imagen lo importan
    internamente si lo necesitan.
    """

    def __init__(self, estetica: Optional[Estetica] = None):
        self._dominio        = "analizador"
        self._estetica       = estetica or Estetica()
        self._ultimo_metodo: Optional[Any] = None

    # =========================================================
    # CORE INTERNO — dos ejecutores según tipo de output
    # =========================================================

    def _ejecutar_plot(
        self,
        data: InputPlot,
        metodo: MetodoPlot,
    ) -> Resultado[Figure, ErrorAnalizador]:
        nombre = getattr(metodo, "nombre", metodo.__class__.__name__)
        try:
            self._validar_input_no_vacio(data, nombre)
            figura = metodo(data, estetica=self._estetica)
            self._validar_figura(figura, nombre)
            return Ok(figura)
        except Exception as e:
            return Err(ErrorAnalizador(
                etapa="plot",
                mensaje=f"Error en '{nombre}': {e}",
                causa=e,
            ))

    def _ejecutar_exportacion(
        self,
        data: Union[pd.DataFrame, Figure],
        metodo: MetodoExportacion,
        ruta_salida: Path,
    ) -> Resultado[Path, ErrorAnalizador]:
        nombre = getattr(metodo, "nombre", metodo.__class__.__name__)
        try:
            self._validar_input_no_vacio(data, nombre)
            ruta_escrita = metodo(data, ruta_salida)
            return Ok(ruta_escrita)
        except Exception as e:
            return Err(ErrorAnalizador(
                etapa="exportacion",
                mensaje=f"Error en '{nombre}': {e}",
                causa=e,
            ))

    def _validar_input_no_vacio(self, data: Any, nombre: str) -> None:
        if data is None:
            raise ValueError(f"'{nombre}' recibió input None")
        if isinstance(data, pd.DataFrame) and data.empty:
            raise ValueError(f"'{nombre}' recibió un DataFrame vacío")

    def _validar_figura(self, figura: Any, nombre: str) -> None:
        # Evitamos importar matplotlib en el nivel de módulo
        if figura is None:
            raise ValueError(f"'{nombre}' devolvió None en lugar de una Figure")

    # =========================================================
    # FACTORIES DE CALLABLE
    # =========================================================

    def crear_operador_plot(
        self,
        metodo: MetodoPlot,
    ) -> Callable[[InputPlot], Resultado[Figure, ErrorAnalizador]]:
        """(DataFrame | BioImagenData) → Resultado[Figure, ErrorAnalizador]"""
        self._ultimo_metodo = metodo

        def _op(data: InputPlot) -> Resultado[Figure, ErrorAnalizador]:
            return self._ejecutar_plot(data, metodo)

        return _op

    def crear_operador_exportacion(
        self,
        metodo: MetodoExportacion,
        ruta_salida: Path,
    ) -> Callable[[Union[pd.DataFrame, Figure]], Resultado[Path, ErrorAnalizador]]:
        """(DataFrame | Figure) → Resultado[Path, ErrorAnalizador]"""
        self._ultimo_metodo = metodo

        def _op(data: Union[pd.DataFrame, Figure]) -> Resultado[Path, ErrorAnalizador]:
            return self._ejecutar_exportacion(data, metodo, ruta_salida)

        return _op

    def crear(
        self,
        nombre_metodo: str,
        **params,
    ) -> Callable:
        """Resolución dinámica desde YAML vía registro."""
        clase  = registro_metodos.obtener(self._dominio, nombre_metodo)
        metodo = clase(**params)
        # El tipo de operador se infiere del submódulo vía isinstance
        if isinstance(metodo, MetodoExportacion.__args__):
            raise ValueError(
                "Para exportación usar crear_operador_exportacion directamente "
                "(requiere ruta_salida en tiempo de construcción)"
            )
        return self.crear_operador_plot(metodo)

    def _operacion_plot(
        self,
        nombre_metodo: str,
        nombre: Optional[str],
        params: dict,
    ) -> Operacion:
        """Helper interno para factories de plot."""
        clase    = registro_metodos.obtener(self._dominio, nombre_metodo)
        metodo   = clase(**params)
        callable_ = self.crear_operador_plot(metodo)

        return Operacion(
            nombre               = nombre or f"plot_{nombre_metodo}",
            categoria            = CategoriaOperacion.ANALISIS,
            instancia_callable   = callable_,
            canal_objetivo       = None,
            parametros_originales= params,
            tipo_salida          = TipoDato.VISUALIZACION,
        )

    def _operacion_exportacion(
        self,
        nombre_metodo: str,
        nombre: Optional[str],
        params: dict,
        ruta_salida: Path,
    ) -> Operacion:
        """Helper interno para factories de exportación."""
        clase    = registro_metodos.obtener(self._dominio, nombre_metodo)
        metodo   = clase(**params)
        callable_ = self.crear_operador_exportacion(metodo, ruta_salida)

        return Operacion(
            nombre               = nombre or f"exportar_{nombre_metodo}",
            categoria            = CategoriaOperacion.ANALISIS,
            instancia_callable   = callable_,
            canal_objetivo       = None,
            parametros_originales= {**params, "ruta_salida": str(ruta_salida)},
            tipo_salida          = TipoDato.NINGUNA,  # side effect puro
        )

    # =========================================================
    # FACTORIES YAML / CLI — devuelven Operacion
    # =========================================================

    # ── PLOTS ESTADÍSTICOS ────────────────────────────────────

    def crear_operacion_histograma_intensidad(
        self,
        # TODO: params (bins, canal, ...)
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self._operacion_plot("histograma_intensidad", nombre, {})

    def crear_operacion_boxplot_canales(
        self,
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self._operacion_plot("boxplot_canales", nombre, {})

    def crear_operacion_violin_distribucion(
        self,
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self._operacion_plot("violin_distribucion", nombre, {})

    def crear_operacion_scatter_features(
        self,
        # TODO: params (feature_x, feature_y, hue, ...)
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self._operacion_plot("scatter_features", nombre, {})

    def crear_operacion_heatmap_correlacion(
        self,
        # TODO: params (metodo: pearson/spearman, ...)
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self._operacion_plot("heatmap_correlacion", nombre, {})

    def crear_operacion_curva_roc(
        self,
        # TODO: params (columna_scores, columna_etiquetas, ...)
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self._operacion_plot("curva_roc", nombre, {})

    def crear_operacion_matriz_confusion(
        self,
        # TODO: params (columna_pred, columna_real, ...)
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self._operacion_plot("matriz_confusion", nombre, {})

    # ── PLOTS DE IMAGEN ───────────────────────────────────────

    def crear_operacion_mip_proyeccion(
        self,
        # TODO: params (eje, canal, ...)
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self._operacion_plot("mip_proyeccion", nombre, {})

    def crear_operacion_ortho_view(
        self,
        # TODO: params (z_ref, t_ref, ...)
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self._operacion_plot("ortho_view", nombre, {})

    def crear_operacion_stack_viewer(
        self,
        # TODO: params (canal, t, ...)
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self._operacion_plot("stack_viewer", nombre, {})

    def crear_operacion_overlay_mascara(
        self,
        # TODO: params (alpha, color_mascara, ...)
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self._operacion_plot("overlay_mascara", nombre, {})

    def crear_operacion_perfil_intensidad(
        self,
        # TODO: params (punto_inicio, punto_fin, ...)
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self._operacion_plot("perfil_intensidad", nombre, {})

    def crear_operacion_comparacion_canales(
        self,
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self._operacion_plot("comparacion_canales", nombre, {})

    # ── PLOTS DE MODELOS ──────────────────────────────────────

    def crear_operacion_biplot_pca(
        self,
        # TODO: params (componentes, hue, ...)
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self._operacion_plot("biplot_pca", nombre, {})

    def crear_operacion_umap_scatter(
        self,
        # TODO: params (hue, tamaño_punto, ...)
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self._operacion_plot("umap_scatter", nombre, {})

    def crear_operacion_cluster_map(
        self,
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self._operacion_plot("cluster_map", nombre, {})

    def crear_operacion_importancia_features(
        self,
        # TODO: params (top_n, ...)
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self._operacion_plot("importancia_features", nombre, {})

    def crear_operacion_separabilidad_clases(
        self,
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self._operacion_plot("separabilidad_clases", nombre, {})

    # ── EXPORTACIÓN ───────────────────────────────────────────

    def crear_operacion_exportar_csv(
        self,
        ruta_salida: Path,
        # TODO: params (separador, encoding, index, ...)
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self._operacion_exportacion("exportador_csv", nombre, {}, ruta_salida)

    def crear_operacion_exportar_parquet(
        self,
        ruta_salida: Path,
        # TODO: params (compresion, engine, ...)
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self._operacion_exportacion("exportador_parquet", nombre, {}, ruta_salida)

    def crear_operacion_exportar_figura(
        self,
        ruta_salida: Path,
        # TODO: params (formato: png/svg/pdf, dpi, ...)
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self._operacion_exportacion("exportador_figuras", nombre, {}, ruta_salida)

    # ── QC — NO IMPLEMENTADO ──────────────────────────────────

    def crear_operacion_overlay_qc(self, *args, **kwargs):
        raise NotImplementedError(
            "QC overlays no implementado aún. "
            "Pendiente: analizador/qc/overlays.py"
        )

    def crear_operacion_sanity_check(self, *args, **kwargs):
        raise NotImplementedError(
            "Sanity checks no implementado aún. "
            "Pendiente: analizador/qc/sanity_checks.py"
        )

    # =========================================================
    # USO IMPERATIVO
    # =========================================================

    def aplicar_plot(
        self,
        data: InputPlot,
        metodo: MetodoPlot,
    ) -> Resultado[Figure, ErrorAnalizador]:
        return self.crear_operador_plot(metodo)(data)

    def aplicar_exportacion(
        self,
        data: Union[pd.DataFrame, Figure],
        metodo: MetodoExportacion,
        ruta_salida: Path,
    ) -> Resultado[Path, ErrorAnalizador]:
        return self.crear_operador_exportacion(metodo, ruta_salida)(data)

    def cambiar_estetica(self, estetica: Estetica) -> None:
        """Reemplaza la estética global para todos los plots siguientes."""
        self._estetica = estetica

    def reset(self):
        self._ultimo_metodo = None

    def __repr__(self):
        nombre = getattr(self._ultimo_metodo, "nombre", "None")
        return f"<Controlador_Analizador ultimo={nombre}>"