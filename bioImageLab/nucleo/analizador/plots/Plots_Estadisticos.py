# === analizador/plots/Plots_Estadisticos.py ===
"""
Plots estadísticos para análisis de features/tablas.
Todos reciben pd.DataFrame y devuelven matplotlib.figure.Figure.
"""
from __future__ import annotations

from typing import Optional, List, Tuple, Any
from pathlib import Path

import numpy as np
import pandas as pd

from .Estetica import Estetica


def _setup_fig(estetica: Estetica, figsize_key: str = "default"):
    import matplotlib.pyplot as plt
    estetica.aplicar_tema()
    fig, ax = plt.subplots(figsize=estetica.figsize(figsize_key), dpi=estetica.layout.dpi)
    return fig, ax


def _titulo(ax, titulo: Optional[str], estetica: Estetica, subtitulo: Optional[str] = None):
    if titulo:
        ax.set_title(titulo, fontsize=estetica.fuentes.tamano_titulo,
                     fontweight=estetica.fuentes.peso_titulo)
    if subtitulo:
        ax.set_xlabel(subtitulo, fontsize=estetica.fuentes.tamano_subtitulo)


def _guardar_o_mostrar(fig, estetica: Estetica, ruta: Optional[Path] = None):
    if estetica.layout.tight_layout:
        fig.tight_layout(pad=estetica.layout.padding)
    if ruta:
        fig.savefig(ruta, dpi=estetica.layout.dpi, bbox_inches="tight")
    return fig


# =========================================================
# CLASES PLOT ESTADÍSTICO (patrón callable)
# =========================================================

class HistogramaIntensidad:
    """Histograma de intensidades de una o más columnas numéricas."""
    nombre = "histograma_intensidad"

    def __init__(self, columna: Optional[str] = None, bins: int = 64,
                 rango: Optional[Tuple[float, float]] = None,
                 kde: bool = True, log_scale: bool = False):
        self.columna = columna
        self.bins = bins
        self.rango = rango
        self.kde = kde
        self.log_scale = log_scale

    def __call__(self, data: pd.DataFrame, estetica: Optional[Estetica] = None) -> Any:
        import matplotlib.pyplot as plt
        import seaborn as sns
        est = estetica or Estetica()
        fig, ax = _setup_fig(est, "default")

        cols = [self.columna] if self.columna else data.select_dtypes(include=[np.number]).columns.tolist()
        if not cols:
            raise ValueError("No hay columnas numéricas para histograma")

        for i, col in enumerate(cols[:6]):  # máx 6 series
            color = est.color(i)
            subset = data[col].dropna()
            if self.log_scale:
                subset = np.log1p(subset[subset > 0])
            ax.hist(subset, bins=self.bins, range=self.rango, alpha=0.6,
                    color=color, label=col, edgecolor="white", linewidth=0.3)

        _titulo(ax, "Histograma de Intensidad", est)
        ax.set_xlabel("Intensidad" + (" (log)" if self.log_scale else ""), fontsize=est.fuentes.tamano_etiqueta)
        ax.set_ylabel("Frecuencia", fontsize=est.fuentes.tamano_etiqueta)
        if len(cols) > 1:
            ax.legend(fontsize=est.fuentes.tamano_leyenda)
        ax.grid(True, alpha=0.3)
        return _guardar_o_mostrar(fig, est)


class BoxplotCanales:
    """Boxplot comparativo entre canales/columnas."""
    nombre = "boxplot_canales"

    def __init__(self, columnas: Optional[List[str]] = None,
                 orient: str = "v", mostrar_outliers: bool = True):
        self.columnas = columnas
        self.orient = orient
        self.mostrar_outliers = mostrar_outliers

    def __call__(self, data: pd.DataFrame, estetica: Optional[Estetica] = None) -> Any:
        import matplotlib.pyplot as plt
        import seaborn as sns
        est = estetica or Estetica()
        fig, ax = _setup_fig(est, "ancho" if self.orient == "h" else "default")

        cols = self.columnas or data.select_dtypes(include=[np.number]).columns.tolist()
        plot_data = data[cols].dropna()

        if self.orient == "h":
            sns.boxplot(data=plot_data, orient="h", ax=ax,
                        palette=est.paleta.categorico[:len(cols)],
                        showfliers=self.mostrar_outliers)
        else:
            sns.boxplot(data=plot_data, ax=ax,
                        palette=est.paleta.categorico[:len(cols)],
                        showfliers=self.mostrar_outliers)

        _titulo(ax, "Distribución por Canal", est)
        ax.set_ylabel("Canal" if self.orient == "h" else "Intensidad", fontsize=est.fuentes.tamano_etiqueta)
        ax.set_xlabel("Intensidad" if self.orient == "h" else "Canal", fontsize=est.fuentes.tamano_etiqueta)
        ax.grid(True, alpha=0.3, axis="y" if self.orient == "v" else "x")
        return _guardar_o_mostrar(fig, est)


class ViolinDistribucion:
    """Violin plot para distribuciones."""
    nombre = "violin_distribucion"

    def __init__(self, columna_x: Optional[str] = None, columna_y: Optional[str] = None,
                 split: bool = False, inner: str = "box"):
        self.columna_x = columna_x
        self.columna_y = columna_y
        self.split = split
        self.inner = inner

    def __call__(self, data: pd.DataFrame, estetica: Optional[Estetica] = None) -> Any:
        import matplotlib.pyplot as plt
        import seaborn as sns
        est = estetica or Estetica()
        fig, ax = _setup_fig(est, "ancho")

        if self.columna_x and self.columna_y:
            sns.violinplot(data=data, x=self.columna_x, y=self.columna_y,
                           split=self.split, inner=self.inner, ax=ax,
                           palette=est.paleta.categorico)
        else:
            cols = data.select_dtypes(include=[np.number]).columns[:4]
            sns.violinplot(data=data[cols], ax=ax, palette=est.paleta.categorico,
                           inner=self.inner)

        _titulo(ax, "Distribución (Violin)", est)
        ax.grid(True, alpha=0.3, axis="y")
        return _guardar_o_mostrar(fig, est)


class ScatterFeatures:
    """Scatter plot de dos features con posible color por categoría."""
    nombre = "scatter_features"

    def __init__(self, x: str = "feature_0", y: str = "feature_1",
                 hue: Optional[str] = None, size: Optional[str] = None,
                 alpha: float = 0.7, tamano_punto: float = 30):
        self.x = x
        self.y = y
        self.hue = hue
        self.size = size
        self.alpha = alpha
        self.tamano_punto = tamano_punto

    def __call__(self, data: pd.DataFrame, estetica: Optional[Estetica] = None) -> Any:
        import matplotlib.pyplot as plt
        import seaborn as sns
        est = estetica or Estetica()
        fig, ax = _setup_fig(est, "cuadrado")

        sns.scatterplot(data=data, x=self.x, y=self.y, hue=self.hue,
                        size=self.size, alpha=self.alpha,
                        s=self.tamano_punto, ax=ax, palette=est.paleta.categorico,
                        edgecolor="black", linewidth=0.2)

        _titulo(ax, f"{self.x} vs {self.y}", est)
        ax.grid(True, alpha=0.3)
        if self.hue:
            ax.legend(fontsize=est.fuentes.tamano_leyenda, loc="best")
        return _guardar_o_mostrar(fig, est)


class HeatmapCorrelacion:
    """Heatmap de matriz de correlación."""
    nombre = "heatmap_correlacion"

    def __init__(self, metodo: str = "pearson", annot: bool = True,
                 tamano_fig: Tuple[float, float] = (10, 8)):
        self.metodo = metodo
        self.annot = annot
        self.tamano_fig = tamano_fig

    def __call__(self, data: pd.DataFrame, estetica: Optional[Estetica] = None) -> Any:
        import matplotlib.pyplot as plt
        import seaborn as sns
        est = estetica or Estetica()
        fig, ax = plt.subplots(figsize=self.tamano_fig, dpi=est.layout.dpi)
        est.aplicar_tema()

        numeric = data.select_dtypes(include=[np.number])
        corr = numeric.corr(method=self.metodo)

        mask = np.triu(np.ones_like(corr, dtype=bool), k=1) if not self.annot else None
        sns.heatmap(corr, mask=mask, annot=self.annot, fmt=".2f",
                    cmap=est.cmap("divergente"), center=0,
                    square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
                    ax=ax, vmin=-1, vmax=1)

        _titulo(ax, f"Correlación ({self.metodo})", est)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right",
                           fontsize=est.fuentes.tamano_tick)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0,
                           fontsize=est.fuentes.tamano_tick)
        return _guardar_o_mostrar(fig, est)


class CurvaROC:
    """Curva ROC para clasificación binaria."""
    nombre = "curva_roc"

    def __init__(self, columna_scores: str = "score", columna_etiquetas: str = "label",
                 titulo: Optional[str] = None):
        self.columna_scores = columna_scores
        self.columna_etiquetas = columna_etiquetas
        self.titulo = titulo

    def __call__(self, data: pd.DataFrame, estetica: Optional[Estetica] = None) -> Any:
        from sklearn.metrics import roc_curve, auc
        import matplotlib.pyplot as plt
        est = estetica or Estetica()
        fig, ax = _setup_fig(est, "cuadrado")

        y_true = data[self.columna_etiquetas].values
        y_score = data[self.columna_scores].values

        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)

        ax.plot(fpr, tpr, color=est.paleta.primario, lw=est.linea.ancho * 2,
                label=f"AUC = {roc_auc:.3f}")
        ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--", alpha=0.7)
        ax.fill_between(fpr, tpr, alpha=0.2, color=est.paleta.primario)

        _titulo(ax, self.titulo or "Curva ROC", est)
        ax.set_xlabel("Tasa de Falsos Positivos", fontsize=est.fuentes.tamano_etiqueta)
        ax.set_ylabel("Tasa de Verdaderos Positivos", fontsize=est.fuentes.tamano_etiqueta)
        ax.legend(loc="lower right", fontsize=est.fuentes.tamano_leyenda)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.05])
        return _guardar_o_mostrar(fig, est)


class MatrizConfusion:
    """Matriz de confusión visual."""
    nombre = "matriz_confusion"

    def __init__(self, columna_pred: str = "pred", columna_real: str = "real",
                 normalizar: Optional[str] = None,  # "true", "pred", "all", None
                 titulo: Optional[str] = None):
        self.columna_pred = columna_pred
        self.columna_real = columna_real
        self.normalizar = normalizar
        self.titulo = titulo

    def __call__(self, data: pd.DataFrame, estetica: Optional[Estetica] = None) -> Any:
        from sklearn.metrics import confusion_matrix
        import matplotlib.pyplot as plt
        import seaborn as sns
        est = estetica or Estetica()
        fig, ax = _setup_fig(est, "cuadrado")

        y_true = data[self.columna_real].values
        y_pred = data[self.columna_pred].values
        cm = confusion_matrix(y_true, y_pred, normalize=self.normalizar)

        fmt = ".2%" if self.normalizar else "d"
        sns.heatmap(cm, annot=True, fmt=fmt, cmap=est.cmap("secuencial"),
                    square=True, linewidths=0.5, ax=ax,
                    cbar_kws={"shrink": 0.8})

        _titulo(ax, self.titulo or "Matriz de Confusión", est)
        ax.set_xlabel("Predicción", fontsize=est.fuentes.tamano_etiqueta)
        ax.set_ylabel("Real", fontsize=est.fuentes.tamano_etiqueta)
        return _guardar_o_mostrar(fig, est)
