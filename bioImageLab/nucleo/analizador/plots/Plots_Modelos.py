# === analizador/plots/Plots_Modelos.py ===
"""
Plots para visualización de resultados de modelos de machine learning.
Reciben pd.DataFrame y devuelven matplotlib.figure.Figure.
"""
from __future__ import annotations

from typing import Optional, List, Tuple, Any

import numpy as np
import pandas as pd

from .Estetica import Estetica


def _setup_fig(estetica: Estetica, figsize_key: str = "default"):
    import matplotlib.pyplot as plt
    estetica.aplicar_tema()
    fig, ax = plt.subplots(figsize=estetica.figsize(figsize_key), dpi=estetica.layout.dpi)
    return fig, ax


def _setup_fig_multi(nrows: int, ncols: int, estetica: Estetica,
                     figsize_key: str = "default"):
    import matplotlib.pyplot as plt
    estetica.aplicar_tema()
    figsize = estetica.figsize(figsize_key)
    figsize = (figsize[0] * ncols, figsize[1] * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, dpi=estetica.layout.dpi)
    return fig, axes


def _titulo(ax, titulo: Optional[str], estetica: Estetica):
    if titulo:
        ax.set_title(titulo, fontsize=estetica.fuentes.tamano_titulo,
                     fontweight=estetica.fuentes.peso_titulo)


def _guardar_o_mostrar(fig, estetica: Estetica, ruta=None):
    if estetica.layout.tight_layout:
        fig.tight_layout(pad=estetica.layout.padding)
    if ruta:
        fig.savefig(ruta, dpi=estetica.layout.dpi, bbox_inches="tight")
    return fig


# =========================================================
# CLASES PLOT MODELOS
# =========================================================

class BiplotPCA:
    """Biplot PCA: proyección de muestras + loadings de features."""
    nombre = "biplot_pca"

    def __init__(self, componentes: Tuple[int, int] = (0, 1),
                 hue: Optional[str] = None, escala_flechas: float = 1.0,
                 mostrar_labels: bool = True, tamano_punto: float = 40):
        self.componentes = componentes
        self.hue = hue
        self.escala_flechas = escala_flechas
        self.mostrar_labels = mostrar_labels
        self.tamano_punto = tamano_punto

    def __call__(self, data: pd.DataFrame, estetica: Optional[Estetica] = None) -> Any:
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        import matplotlib.pyplot as plt
        est = estetica or Estetica()
        fig, ax = _setup_fig(est, "cuadrado")

        # Separar features y target
        features = data.select_dtypes(include=[np.number])
        if self.hue and self.hue in data.columns:
            features = features.drop(columns=[self.hue], errors="ignore")

        X = StandardScaler().fit_transform(features)
        pca = PCA(n_components=max(self.componentes) + 1)
        proyeccion = pca.fit_transform(X)

        pc_x, pc_y = self.componentes
        var_x = pca.explained_variance_ratio_[pc_x] * 100
        var_y = pca.explained_variance_ratio_[pc_y] * 100

        # Scatter de muestras
        hue_vals = data[self.hue].values if self.hue and self.hue in data.columns else None
        if hue_vals is not None:
            categorias = np.unique(hue_vals)
            for i, cat in enumerate(categorias):
                mask = hue_vals == cat
                ax.scatter(proyeccion[mask, pc_x], proyeccion[mask, pc_y],
                           c=est.color(i), s=self.tamano_punto,
                           label=str(cat), alpha=0.8, edgecolors="black", linewidth=0.3)
            ax.legend(fontsize=est.fuentes.tamano_leyenda, title=self.hue)
        else:
            ax.scatter(proyeccion[:, pc_x], proyeccion[:, pc_y],
                       c=est.paleta.primario, s=self.tamano_punto,
                       alpha=0.8, edgecolors="black", linewidth=0.3)

        # Loadings (flechas)
        loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
        n_features = min(len(features.columns), 15)
        for i, feature in enumerate(features.columns[:n_features]):
            lx, ly = loadings[i, pc_x] * self.escala_flechas, loadings[i, pc_y] * self.escala_flechas
            ax.arrow(0, 0, lx, ly, color=est.paleta.cuaternario, alpha=0.7,
                     width=0.002, head_width=0.05)
            if self.mostrar_labels:
                ax.text(lx * 1.1, ly * 1.1, feature, color=est.paleta.cuaternario,
                        fontsize=est.fuentes.tamano_anotacion, ha="center")

        ax.axhline(0, color="gray", lw=0.5, alpha=0.5)
        ax.axvline(0, color="gray", lw=0.5, alpha=0.5)
        ax.set_xlabel(f"PC{pc_x + 1} ({var_x:.1f}%)", fontsize=est.fuentes.tamano_etiqueta)
        ax.set_ylabel(f"PC{pc_y + 1} ({var_y:.1f}%)", fontsize=est.fuentes.tamano_etiqueta)
        ax.grid(True, alpha=0.3)
        _titulo(ax, "Biplot PCA", est)
        return _guardar_o_mostrar(fig, est)


class UMAPScatter:
    """Scatter plot de embedding UMAP."""
    nombre = "umap_scatter"

    def __init__(self, hue: Optional[str] = None, tamano_punto: float = 30,
                 alpha: float = 0.7, n_neighbors: int = 15, min_dist: float = 0.1):
        self.hue = hue
        self.tamano_punto = tamano_punto
        self.alpha = alpha
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist

    def __call__(self, data: pd.DataFrame, estetica: Optional[Estetica] = None) -> Any:
        try:
            import umap
        except ImportError:
            raise ImportError("umap-learn es requerido. pip install umap-learn")
        from sklearn.preprocessing import StandardScaler
        import matplotlib.pyplot as plt
        est = estetica or Estetica()
        fig, ax = _setup_fig(est, "cuadrado")

        features = data.select_dtypes(include=[np.number])
        if self.hue and self.hue in data.columns:
            features = features.drop(columns=[self.hue], errors="ignore")

        X = StandardScaler().fit_transform(features)
        reducer = umap.UMAP(n_neighbors=self.n_neighbors, min_dist=self.min_dist, random_state=42)
        embedding = reducer.fit_transform(X)

        hue_vals = data[self.hue].values if self.hue and self.hue in data.columns else None
        if hue_vals is not None:
            categorias = np.unique(hue_vals)
            for i, cat in enumerate(categorias):
                mask = hue_vals == cat
                ax.scatter(embedding[mask, 0], embedding[mask, 1],
                           c=est.color(i), s=self.tamano_punto,
                           label=str(cat), alpha=self.alpha,
                           edgecolors="black", linewidth=0.2)
            ax.legend(fontsize=est.fuentes.tamano_leyenda, title=self.hue)
        else:
            ax.scatter(embedding[:, 0], embedding[:, 1],
                       c=est.paleta.primario, s=self.tamano_punto,
                       alpha=self.alpha, edgecolors="black", linewidth=0.2)

        ax.set_xlabel("UMAP 1", fontsize=est.fuentes.tamano_etiqueta)
        ax.set_ylabel("UMAP 2", fontsize=est.fuentes.tamano_etiqueta)
        ax.grid(True, alpha=0.3)
        _titulo(ax, "UMAP Embedding", est)
        return _guardar_o_mostrar(fig, est)


class ClusterMap:
    """Heatmap con clustering jerárquico (clustermap)."""
    nombre = "cluster_map"

    def __init__(self, metrica: str = "euclidean", metodo: str = "ward",
                 estandarizar: bool = True, tamano_fig: Tuple[float, float] = (10, 10)):
        self.metrica = metrica
        self.metodo = metodo
        self.estandarizar = estandarizar
        self.tamano_fig = tamano_fig

    def __call__(self, data: pd.DataFrame, estetica: Optional[Estetica] = None) -> Any:
        import seaborn as sns
        import matplotlib.pyplot as plt
        est = estetica or Estetica()
        est.aplicar_tema()

        numeric = data.select_dtypes(include=[np.number])
        if self.estandarizar:
            numeric = (numeric - numeric.mean()) / numeric.std()

        g = sns.clustermap(numeric, metric=self.metrica, method=self.metodo,
                           cmap=est.cmap("divergente"), figsize=self.tamano_fig,
                           dendrogram_ratio=0.2, cbar_pos=(0.02, 0.8, 0.03, 0.15),
                           linewidths=0.5, linecolor="white")
        g.fig.suptitle("Cluster Map", fontsize=est.fuentes.tamano_titulo,
                       fontweight=est.fuentes.peso_titulo, y=1.02)
        if est.layout.tight_layout:
            g.fig.tight_layout(pad=est.layout.padding)
        return g.fig


class ImportanciaFeatures:
    """Barplot de importancia de features (ej: Random Forest)."""
    nombre = "importancia_features"

    def __init__(self, columna_feature: str = "feature", columna_importancia: str = "importancia",
                 top_n: int = 20, horizontal: bool = True):
        self.columna_feature = columna_feature
        self.columna_importancia = columna_importancia
        self.top_n = top_n
        self.horizontal = horizontal

    def __call__(self, data: pd.DataFrame, estetica: Optional[Estetica] = None) -> Any:
        import matplotlib.pyplot as plt
        est = estetica or Estetica()
        fig, ax = _setup_fig(est, "ancho" if self.horizontal else "alto")

        df = data.sort_values(self.columna_importancia, ascending=True).tail(self.top_n)
        colors = [est.color(i) for i in range(len(df))]

        if self.horizontal:
            ax.barh(df[self.columna_feature], df[self.columna_importancia], color=colors)
            ax.set_xlabel("Importancia", fontsize=est.fuentes.tamano_etiqueta)
        else:
            ax.bar(df[self.columna_feature], df[self.columna_importancia], color=colors)
            ax.set_ylabel("Importancia", fontsize=est.fuentes.tamano_etiqueta)
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

        ax.grid(True, alpha=0.3, axis="x" if self.horizontal else "y")
        _titulo(ax, f"Top {self.top_n} Features", est)
        return _guardar_o_mostrar(fig, est)


class SeparabilidadClases:
    """Visualización de separabilidad entre clases (pairplot reducido o distribuciones)."""
    nombre = "separabilidad_clases"

    def __init__(self, columna_clase: str = "clase", features: Optional[List[str]] = None,
                 tipo: str = "pairplot",  # "pairplot", "kde", "box"
                 max_features: int = 4):
        self.columna_clase = columna_clase
        self.features = features
        self.tipo = tipo
        self.max_features = max_features

    def __call__(self, data: pd.DataFrame, estetica: Optional[Estetica] = None) -> Any:
        import seaborn as sns
        import matplotlib.pyplot as plt
        est = estetica or Estetica()

        cols = self.features if self.features else data.select_dtypes(include=[np.number]).columns.tolist()
        cols = [c for c in cols if c != self.columna_clase][:self.max_features]

        if self.tipo == "pairplot":
            g = sns.pairplot(data, vars=cols, hue=self.columna_clase,
                             palette=est.paleta.categorico[:len(data[self.columna_clase].unique())],
                             diag_kind="kde", plot_kws={"alpha": 0.6, "s": 30, "edgecolor": "k", "linewidth": 0.2},
                             diag_kws={"alpha": 0.5}, height=2.5)
            g.fig.suptitle("Separabilidad de Clases", fontsize=est.fuentes.tamano_titulo,
                           fontweight=est.fuentes.peso_titulo, y=1.02)
            if est.layout.tight_layout:
                g.fig.tight_layout(pad=est.layout.padding)
            return g.fig

        elif self.tipo == "kde":
            fig, axes = _setup_fig_multi(1, len(cols), est, "ancho")
            if len(cols) == 1:
                axes = [axes]
            else:
                axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
            for i, col in enumerate(cols):
                ax = axes[i]
                for j, clase in enumerate(data[self.columna_clase].unique()):
                    subset = data[data[self.columna_clase] == clase][col].dropna()
                    sns.kdeplot(subset, ax=ax, color=est.color(j), label=str(clase),
                                fill=True, alpha=0.3, linewidth=est.linea.ancho)
                ax.set_title(col, fontsize=est.fuentes.tamano_subtitulo)
                ax.legend(fontsize=est.fuentes.tamano_leyenda)
                ax.grid(True, alpha=0.3)
            fig.suptitle("Densidad por Clase", fontsize=est.fuentes.tamano_titulo,
                        fontweight=est.fuentes.peso_titulo)
            return _guardar_o_mostrar(fig, est)

        elif self.tipo == "box":
            fig, axes = _setup_fig_multi(1, len(cols), est, "ancho")
            if len(cols) == 1:
                axes = [axes]
            else:
                axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
            for i, col in enumerate(cols):
                ax = axes[i]
                sns.boxplot(data=data, x=self.columna_clase, y=col, ax=ax,
                            palette=est.paleta.categorico)
                ax.set_title(col, fontsize=est.fuentes.tamano_subtitulo)
                ax.grid(True, alpha=0.3, axis="y")
            fig.suptitle("Distribución por Clase", fontsize=est.fuentes.tamano_titulo,
                        fontweight=est.fuentes.peso_titulo)
            return _guardar_o_mostrar(fig, est)

        else:
            raise ValueError(f"Tipo '{self.tipo}' no válido. Use 'pairplot', 'kde' o 'box'.")
