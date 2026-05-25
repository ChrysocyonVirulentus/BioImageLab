# === analizador/Controlador_Analizador.py ===
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Union, Optional, Callable, Any, List, Tuple

import pandas as pd
import numpy as np

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
from ..analizador.exportacion.tsv import ExportadorTSV
from ..analizador.exportacion.parquet import ExportadorParquet
from ..analizador.exportacion.figures import ExportadorFiguras

# QC
from ..analizador.qc.QC_Visualizacion import (
    visualizar_pasos,
    comparar_antes_despues,
    reporte_qc_completo,
    ConfigQC,
    ErrorQC,
)


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

    def con_contexto(self, nueva_etapa: str) -> "ErrorAnalizador":
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
MetodoExportacion = Union[ExportadorCSV, ExportadorTSV, ExportadorParquet, ExportadorFiguras]

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
            Visualización de pasos del pipeline con plots de imagen.

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
        columna: Optional[str] = None,
        bins: int = 64,
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self._operacion_plot("histograma_intensidad", nombre,
                                    {"columna": columna, "bins": bins})

    def crear_operacion_boxplot_canales(
        self,
        columnas: Optional[List[str]] = None,
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self._operacion_plot("boxplot_canales", nombre,
                                    {"columnas": columnas})

    def crear_operacion_violin_distribucion(
        self,
        columna_x: Optional[str] = None,
        columna_y: Optional[str] = None,
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self._operacion_plot("violin_distribucion", nombre,
                                    {"columna_x": columna_x, "columna_y": columna_y})

    def crear_operacion_scatter_features(
        self,
        x: str = "feature_0",
        y: str = "feature_1",
        hue: Optional[str] = None,
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self._operacion_plot("scatter_features", nombre,
                                    {"x": x, "y": y, "hue": hue})

    def crear_operacion_heatmap_correlacion(
        self,
        metodo: str = "pearson",
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self._operacion_plot("heatmap_correlacion", nombre,
                                    {"metodo": metodo})

    def crear_operacion_curva_roc(
        self,
        columna_scores: str = "score",
        columna_etiquetas: str = "label",
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self._operacion_plot("curva_roc", nombre,
                                    {"columna_scores": columna_scores,
                                     "columna_etiquetas": columna_etiquetas})

    def crear_operacion_matriz_confusion(
        self,
        columna_pred: str = "pred",
        columna_real: str = "real",
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self._operacion_plot("matriz_confusion", nombre,
                                    {"columna_pred": columna_pred,
                                     "columna_real": columna_real})

    # ── PLOTS DE IMAGEN ───────────────────────────────────────

    def crear_operacion_mip_proyeccion(
        self,
        eje: str = "Z",
        canal: int = 0,
        t: int = 0,
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self._operacion_plot("mip_proyeccion", nombre,
                                    {"eje": eje, "canal": canal, "t": t})

    def crear_operacion_ortho_view(
        self,
        z_ref: Optional[int] = None,
        t_ref: int = 0,
        canal: int = 0,
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self._operacion_plot("ortho_view", nombre,
                                    {"z_ref": z_ref, "t_ref": t_ref, "canal": canal})

    def crear_operacion_stack_viewer(
        self,
        canal: int = 0,
        t: int = 0,
        ncols: int = 4,
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self._operacion_plot("stack_viewer", nombre,
                                    {"canal": canal, "t": t, "ncols": ncols})

    def crear_operacion_overlay_mascara(
        self,
        mascara_datos: Optional[np.ndarray] = None,
        canal: int = 0,
        t: int = 0,
        z: int = 0,
        alpha: float = 0.35,
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self._operacion_plot("overlay_mascara", nombre,
                                    {"mascara_datos": mascara_datos, "canal": canal,
                                     "t": t, "z": z, "alpha": alpha})

    def crear_operacion_perfil_intensidad(
        self,
        punto_inicio: Tuple[int, int] = (0, 0),
        punto_fin: Optional[Tuple[int, int]] = None,
        canal: int = 0,
        t: int = 0,
        z: int = 0,
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self._operacion_plot("perfil_intensidad", nombre,
                                    {"punto_inicio": punto_inicio,
                                     "punto_fin": punto_fin,
                                     "canal": canal, "t": t, "z": z})

    def crear_operacion_comparacion_canales(
        self,
        t: int = 0,
        z: int = 0,
        canales: Optional[List[int]] = None,
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self._operacion_plot("comparacion_canales", nombre,
                                    {"t": t, "z": z, "canales": canales})

    # ── PLOTS DE MODELOS ──────────────────────────────────────

    def crear_operacion_biplot_pca(
        self,
        componentes: Tuple[int, int] = (0, 1),
        hue: Optional[str] = None,
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self._operacion_plot("biplot_pca", nombre,
                                    {"componentes": componentes, "hue": hue})

    def crear_operacion_umap_scatter(
        self,
        hue: Optional[str] = None,
        tamano_punto: float = 30,
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self._operacion_plot("umap_scatter", nombre,
                                    {"hue": hue, "tamano_punto": tamano_punto})

    def crear_operacion_cluster_map(
        self,
        metrica: str = "euclidean",
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self._operacion_plot("cluster_map", nombre,
                                    {"metrica": metrica})

    def crear_operacion_importancia_features(
        self,
        columna_feature: str = "feature",
        columna_importancia: str = "importancia",
        top_n: int = 20,
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self._operacion_plot("importancia_features", nombre,
                                    {"columna_feature": columna_feature,
                                     "columna_importancia": columna_importancia,
                                     "top_n": top_n})

    def crear_operacion_separabilidad_clases(
        self,
        columna_clase: str = "clase",
        tipo: str = "pairplot",
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self._operacion_plot("separabilidad_clases", nombre,
                                    {"columna_clase": columna_clase, "tipo": tipo})

    # ── EXPORTACIÓN ───────────────────────────────────────────

    def crear_operacion_exportar_csv(
        self,
        ruta_salida: Path,
        separador: str = ",",
        index: bool = False,
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self._operacion_exportacion("exportador_csv", nombre,
                                           {"separador": separador, "index": index},
                                           ruta_salida)

    def crear_operacion_exportar_tsv(
        self,
        ruta_salida: Path,
        index: bool = False,
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self._operacion_exportacion("exportador_tsv", nombre,
                                           {"index": index}, ruta_salida)

    def crear_operacion_exportar_parquet(
        self,
        ruta_salida: Path,
        compresion: str = "snappy",
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self._operacion_exportacion("exportador_parquet", nombre,
                                           {"compresion": compresion}, ruta_salida)

    def crear_operacion_exportar_figura(
        self,
        ruta_salida: Path,
        formato: str = "png",
        dpi: Optional[int] = None,
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self._operacion_exportacion("exportador_figuras", nombre,
                                           {"formato": formato, "dpi": dpi}, ruta_salida)

    # ── QC — VISUALIZACIÓN DE PASOS ───────────────────────────

    def qc_visualizar_pasos(
        self,
        secuencia: List[Tuple[str, Any]],
        config: Optional[ConfigQC] = None,
        ruta: Optional[Path] = None,
        titulo: str = "Control de Calidad — Pipeline",
    ) -> Resultado[Figure, Union[ErrorAnalizador, ErrorQC]]:
        """
        Visualiza pasos del pipeline con plots de imagen.
        Wrapper sobre analizador.qc.QC_Visualizacion.visualizar_pasos.
        """
        try:
            res = visualizar_pasos(secuencia, config=config, ruta=ruta, titulo=titulo)
            return res.map_err(lambda e: ErrorAnalizador("qc", e.mensaje, e.causa))
        except Exception as e:
            return Err(ErrorAnalizador("qc", f"Error en QC: {e}", causa=e))

    def qc_comparar_antes_despues(
        self,
        data_antes: Any,
        data_despues: Any,
        nombre_op: str = "Operación",
        config: Optional[ConfigQC] = None,
        ruta: Optional[Path] = None,
    ) -> Resultado[Figure, Union[ErrorAnalizador, ErrorQC]]:
        """Comparación lado a lado antes/después de una operación."""
        try:
            res = comparar_antes_despues(data_antes, data_despues,
                                          nombre_op=nombre_op,
                                          config=config, ruta=ruta)
            return res.map_err(lambda e: ErrorAnalizador("qc", e.mensaje, e.causa))
        except Exception as e:
            return Err(ErrorAnalizador("qc", f"Error en comparación: {e}", causa=e))

    def qc_reporte_completo(
        self,
        secuencia: List[Tuple[str, Any]],
        config: Optional[ConfigQC] = None,
        ruta_salida: Optional[Path] = None,
    ) -> Resultado[dict, Union[ErrorAnalizador, ErrorQC]]:
        """Reporte QC completo con figuras y estadísticas."""
        try:
            res = reporte_qc_completo(secuencia, config=config,
                                      ruta_salida=ruta_salida)
            return res.map_err(lambda e: ErrorAnalizador("qc", e.mensaje, e.causa))
        except Exception as e:
            return Err(ErrorAnalizador("qc", f"Error en reporte: {e}", causa=e))

    # ── QC — TESTING / SANITY CHECKS ──────────────────────────

    def qc_sanity_check_intensidad(
        self,
        data: Any,
        canal: int = 0,
        t: int = 0,
        z: int = 0,
    ) -> Resultado[dict, ErrorAnalizador]:
        """
        Sanity check de intensidad: detecta valores nulos, NaN, Inf,
        y rangos fuera de lo esperado.
        """
        try:
            slice_2d = data.datos[t, z, canal, :, :]
            stats = {
                "shape": slice_2d.shape,
                "dtype": str(slice_2d.dtype),
                "min": float(slice_2d.min()),
                "max": float(slice_2d.max()),
                "mean": float(slice_2d.mean()),
                "std": float(slice_2d.std()),
                "has_nan": bool(np.isnan(slice_2d).any()),
                "has_inf": bool(np.isinf(slice_2d).any()),
                "all_zeros": bool(np.all(slice_2d == 0)),
                "all_same": bool(np.all(slice_2d == slice_2d.flat[0])),
            }
            # Validaciones
            alertas = []
            if stats["has_nan"]:
                alertas.append("Valores NaN detectados")
            if stats["has_inf"]:
                alertas.append("Valores Inf detectados")
            if stats["all_zeros"]:
                alertas.append("Imagen completamente negra")
            if stats["all_same"]:
                alertas.append("Imagen constante (sin variación)")
            if stats["max"] == 0:
                alertas.append("Máxima intensidad es cero")

            stats["alertas"] = alertas
            stats["ok"] = len(alertas) == 0
            return Ok(stats)
        except Exception as e:
            return Err(ErrorAnalizador("qc_sanity", f"Error en sanity check: {e}", causa=e))

    def qc_sanity_check_dimensiones(
        self,
        data: Any,
    ) -> Resultado[dict, ErrorAnalizador]:
        """
        Sanity check de dimensiones: valida coherencia shape vs dims.
        """
        try:
            dims = data.dims
            shape_real = data.datos.shape
            shape_esperado = (dims.T, dims.Z, dims.C, dims.Y, dims.X)

            ok = shape_real == shape_esperado
            stats = {
                "shape_real": shape_real,
                "shape_esperado": shape_esperado,
                "dims": {
                    "T": dims.T, "Z": dims.Z, "C": dims.C,
                    "Y": dims.Y, "X": dims.X,
                },
                "coherente": ok,
                "alertas": [] if ok else [f"Shape {shape_real} != esperado {shape_esperado}"],
            }
            return Ok(stats)
        except Exception as e:
            return Err(ErrorAnalizador("qc_sanity", f"Error en check dimensiones: {e}", causa=e))

    def qc_histograma_por_paso(
        self,
        secuencia: List[Tuple[str, Any]],
        canal: int = 0,
        t: int = 0,
        z: int = 0,
        bins: int = 64,
    ) -> Resultado[Figure, ErrorAnalizador]:
        """
        Genera histograma comparativo de intensidades para cada paso.
        """
        try:
            import matplotlib.pyplot as plt
            n = len(secuencia)
            fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), dpi=self._estetica.layout.dpi)
            self._estetica.aplicar_tema()
            if n == 1:
                axes = [axes]
            else:
                axes = axes.flatten()

            for ax, (nombre_op, data) in zip(axes, secuencia):
                t_idx = min(t, data.dims.T - 1)
                z_idx = min(z, data.dims.Z - 1)
                c_idx = min(canal, data.dims.C - 1)
                slice_2d = data.datos[t_idx, z_idx, c_idx, :, :].flatten()

                ax.hist(slice_2d, bins=bins, color=self._estetica.color(0),
                        alpha=0.7, edgecolor="white", linewidth=0.3)
                ax.set_title(nombre_op, fontsize=self._estetica.fuentes.tamano_subtitulo)
                ax.set_xlabel("Intensidad", fontsize=self._estetica.fuentes.tamano_tick)
                ax.set_ylabel("Frecuencia", fontsize=self._estetica.fuentes.tamano_tick)
                ax.grid(True, alpha=0.3)

            fig.suptitle("Histogramas por Paso", fontsize=self._estetica.fuentes.tamano_titulo,
                        fontweight=self._estetica.fuentes.peso_titulo)
            fig.tight_layout(pad=self._estetica.layout.padding)
            return Ok(fig)
        except Exception as e:
            return Err(ErrorAnalizador("qc_histograma", f"Error: {e}", causa=e))

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
