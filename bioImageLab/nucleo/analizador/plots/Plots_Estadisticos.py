"""
Gráficos estadísticos generales para análisis de datos de microscopía.

INPUT  : pandas DataFrame (una fila por imagen / experimento).
OUTPUT : Tupla (fig, ax) o (fig, axes) de matplotlib, lista para mostrar,
         retocar o guardar.

Todos los gráficos aceptan un parámetro `estetica: EsteticaGrafico` opcional
que centraliza paletas, fuentes, tamaños y transparencias. Si no se provee,
se usa EsteticaGrafico() con valores por defecto.

Funciones disponibles:
    histograma()        Histograma de una o varias columnas con KDE opcional.
    scatter()           Scatter de dos variables con color y forma por grupo.
    boxplot_violinplot()Boxplot y/o violinplot (flags independientes) combinados.
    heatmap_correlacion()Heatmap de matriz de correlación (Pearson o Spearman).
    pairplot()          Matriz de scatter y distribuciones marginales (pairplot).

IMPORTANTE — Separación de responsabilidades:
    Estas funciones NO hacen cuantificación, normalización ni modelado.
    Reciben DataFrames ya construidos por etapas anteriores del pipeline.
    La elección de qué columnas comparar es responsabilidad del usuario.
"""

import warnings
from typing import Dict, List, Literal, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from Estetica import EsteticaGrafico, ESTILOS_PREDEFINIDOS

_EST_DEFAULT = EsteticaGrafico()


# ─────────────────────────────────────────────────────────────
# Histograma
# ─────────────────────────────────────────────────────────────

def histograma(
    df: pd.DataFrame,
    columnas: Union[str, List[str]],
    columna_grupo: Optional[str] = None,
    mostrar_kde: bool = True,
    mostrar_media: bool = False,
    mostrar_mediana: bool = False,
    n_bins: Union[int, str] = 'auto',
    escala_x: Literal['lineal', 'log'] = 'lineal',
    titulo: str = 'Distribución de Intensidades',
    xlabel: Optional[str] = None,
    ylabel: str = 'Frecuencia',
    estetica: Optional[EsteticaGrafico] = None,
    mostrar: bool = True,
) -> Tuple[Figure, Union[Axes, np.ndarray]]:
    """
    Histograma de una o varias columnas numéricas con KDE y líneas de referencia.

    Si se pasa una sola columna y columna_grupo, superpone un histograma
    por grupo en el mismo eje. Si se pasan varias columnas, crea subplots.

    Args:
        df: DataFrame con los datos.
        columnas: Nombre de columna (str) o lista de columnas a graficar.
        columna_grupo: Si se especifica, colorea por grupo dentro de cada columna.
        mostrar_kde: Si True, superpone la curva de densidad KDE.
        mostrar_media: Si True, traza una línea vertical en la media.
        mostrar_mediana: Si True, traza una línea vertical en la mediana.
        n_bins: Número de bins o estrategia ('auto', 'fd', 'sturges', 'rice').
        escala_x: Escala del eje X: 'lineal' o 'log'.
        titulo: Título principal del gráfico.
        xlabel: Etiqueta eje X. None → usa el nombre de la columna.
        ylabel: Etiqueta eje Y.
        estetica: Objeto EsteticaGrafico. None → valores por defecto.
        mostrar: Si True, llama plt.show() al final.

    Returns:
        Tupla (fig, ax) si columnas es str, (fig, axes_array) si es lista.
    """
    est = estetica or _EST_DEFAULT
    est.aplicar()

    if isinstance(columnas, str):
        columnas = [columnas]

    n_cols = len(columnas)
    fig, axes = plt.subplots(
        1, n_cols,
        figsize=(est.figsize[0] * n_cols, est.figsize[1]),
        squeeze=False,
    )

    paleta = sns.color_palette(est.paleta)

    for col_idx, col in enumerate(columnas):
        ax = axes[0, col_idx]

        if columna_grupo is not None:
            grupos = df[columna_grupo].unique()
            for gi, grupo in enumerate(grupos):
                datos = df[df[columna_grupo] == grupo][col].dropna()
                color = paleta[gi % len(paleta)]
                ax.hist(datos, bins=n_bins, density=mostrar_kde,
                        alpha=est.alpha_relleno, color=color, label=str(grupo),
                        edgecolor='white', linewidth=0.5)
                if mostrar_kde:
                    from scipy.stats import gaussian_kde
                    if len(datos) > 3:
                        xmin, xmax = datos.min(), datos.max()
                        xs = np.linspace(xmin, xmax, 300)
                        kde = gaussian_kde(datos)
                        ax.plot(xs, kde(xs), color=color, linewidth=est.grosor_linea)
        else:
            datos = df[col].dropna()
            ax.hist(datos, bins=n_bins, density=mostrar_kde,
                    alpha=est.alpha_relleno + 0.2,
                    color=paleta[0], edgecolor='white', linewidth=0.5)
            if mostrar_kde:
                from scipy.stats import gaussian_kde
                if len(datos) > 3:
                    xmin, xmax = datos.min(), datos.max()
                    xs = np.linspace(xmin, xmax, 300)
                    kde = gaussian_kde(datos)
                    ax.plot(xs, kde(xs), color=paleta[0],
                            linewidth=est.grosor_linea, label='KDE')
            datos_ref = datos

        datos_ref = df[col].dropna()
        if mostrar_media:
            ax.axvline(datos_ref.mean(), color=est.color_media, linestyle='--',
                       linewidth=est.grosor_linea, label=f'Media={datos_ref.mean():.2f}')
        if mostrar_mediana:
            ax.axvline(datos_ref.median(), color=est.color_mediana, linestyle='-.',
                       linewidth=est.grosor_linea, label=f'Mediana={datos_ref.median():.2f}')

        if escala_x == 'log':
            ax.set_xscale('log')

        est.aplicar_a_ax(ax,
                         titulo=titulo if n_cols == 1 else col,
                         xlabel=xlabel or col,
                         ylabel=ylabel if col_idx == 0 else '')

        if columna_grupo is not None or mostrar_media or mostrar_mediana or mostrar_kde:
            ax.legend(**est.kwargs_leyenda())

    fig.suptitle(titulo if n_cols > 1 else '', fontsize=est.fuente_titulo + 2,
                 fontweight='bold' if est.negrita_titulo else 'normal',
                 fontfamily=est.fuente_familia)
    plt.tight_layout()
    if mostrar:
        plt.show()

    return fig, (axes[0, 0] if len(columnas) == 1 else axes[0])


# ─────────────────────────────────────────────────────────────
# Scatter
# ─────────────────────────────────────────────────────────────

def scatter(
    df: pd.DataFrame,
    columna_x: str,
    columna_y: str,
    columna_color: Optional[str] = None,
    columna_forma: Optional[str] = None,
    columna_tamaño: Optional[str] = None,
    mapa_etiquetas: Optional[Dict] = None,
    mostrar_regresion: bool = False,
    mostrar_identidad: bool = False,
    titulo: str = 'Scatter',
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    titulo_leyenda: str = '',
    estetica: Optional[EsteticaGrafico] = None,
    mostrar: bool = True,
) -> Tuple[Figure, Axes]:
    """
    Scatter plot de dos variables con codificación opcional por color, forma y tamaño.

    Args:
        df: DataFrame con los datos.
        columna_x: Columna para el eje X.
        columna_y: Columna para el eje Y.
        columna_color: Columna para el color de los puntos (categórica o continua).
        columna_forma: Columna para el estilo de marcador (categórica).
        columna_tamaño: Columna para el tamaño proporcional de los puntos (numérica).
        mapa_etiquetas: Dict para renombrar categorías de columna_color.
        mostrar_regresion: Si True, superpone línea de regresión lineal con IC.
        mostrar_identidad: Si True, traza la diagonal y=x como referencia.
        titulo: Título del gráfico.
        xlabel: Etiqueta eje X. None → usa nombre de columna_x.
        ylabel: Etiqueta eje Y. None → usa nombre de columna_y.
        titulo_leyenda: Título de la leyenda.
        estetica: Objeto EsteticaGrafico.
        mostrar: Si True, llama plt.show().

    Returns:
        Tupla (fig, ax)
    """
    est = estetica or _EST_DEFAULT
    est.aplicar()

    df_plot = df.copy()
    if mapa_etiquetas and columna_color:
        df_plot[columna_color] = df_plot[columna_color].map(mapa_etiquetas)

    fig, ax = plt.subplots(figsize=est.figsize)

    # Tamaño proporcional
    sizes = None
    if columna_tamaño is not None:
        vals = df_plot[columna_tamaño].values
        rango = vals.max() - vals.min()
        if rango > 0:
            sizes = 20 + 200 * (vals - vals.min()) / rango
        else:
            sizes = np.full(len(vals), est.tamaño_punto)

    sns.scatterplot(
        data=df_plot,
        x=columna_x, y=columna_y,
        hue=columna_color,
        style=columna_forma,
        size=columna_tamaño if columna_tamaño else None,
        sizes=(20, 300) if columna_tamaño else None,
        palette=est.paleta,
        s=est.tamaño_punto if sizes is None else None,
        alpha=est.alpha_puntos,
        legend='full', ax=ax,
    )

    if mostrar_regresion:
        sns.regplot(data=df_plot, x=columna_x, y=columna_y,
                    scatter=False, ax=ax,
                    line_kws={'color': est.color_media,
                               'linewidth': est.grosor_linea},
                    ci=95)

    if mostrar_identidad:
        lims = [
            min(ax.get_xlim()[0], ax.get_ylim()[0]),
            max(ax.get_xlim()[1], ax.get_ylim()[1]),
        ]
        ax.plot(lims, lims, linestyle='--',
                color=est.color_referencia, linewidth=est.grosor_linea,
                label='y = x', zorder=0)

    est.aplicar_a_ax(ax, titulo=titulo,
                     xlabel=xlabel or columna_x,
                     ylabel=ylabel or columna_y)

    if columna_color or columna_forma or mostrar_identidad:
        ax.legend(**est.kwargs_leyenda(titulo_leyenda))

    plt.tight_layout()
    if mostrar:
        plt.show()
    return fig, ax


# ─────────────────────────────────────────────────────────────
# Boxplot + Violinplot combinados
# ─────────────────────────────────────────────────────────────

def boxplot_violinplot(
    df: pd.DataFrame,
    columna_y: str,
    columna_x: Optional[str] = None,
    columna_hue: Optional[str] = None,
    mostrar_violin: bool = True,
    mostrar_box: bool = True,
    mostrar_strip: bool = False,
    mostrar_media: bool = False,
    mapa_etiquetas: Optional[Dict] = None,
    orientacion: Literal['vertical', 'horizontal'] = 'vertical',
    titulo: str = 'Distribución por Grupo',
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    titulo_leyenda: str = '',
    estetica: Optional[EsteticaGrafico] = None,
    mostrar: bool = True,
) -> Tuple[Figure, Axes]:
    """
    Violin y/o boxplot combinados con stripplot opcional.

    Las tres capas son independientes: se pueden activar/desactivar con
    mostrar_violin, mostrar_box y mostrar_strip.

    Cuando se combinan violin + box, el boxplot se superpone dentro del violín
    con relleno semitransparente para no ocultarlo.
    Cuando solo se usa una capa, ocupa todo el ancho disponible.

    Args:
        df: DataFrame con los datos.
        columna_y: Variable numérica a comparar.
        columna_x: Variable categórica del eje X. None → una sola distribución.
        columna_hue: Variable adicional para subdividir por color.
        mostrar_violin: Si True, muestra la capa de violín.
        mostrar_box: Si True, muestra la capa de boxplot.
        mostrar_strip: Si True, superpone los puntos individuales.
        mostrar_media: Si True, añade un marcador de media sobre cada grupo.
        mapa_etiquetas: Dict para renombrar categorías de columna_hue.
        orientacion: 'vertical' (y arriba) o 'horizontal' (x arriba).
        titulo: Título del gráfico.
        xlabel: Etiqueta eje X.
        ylabel: Etiqueta eje Y.
        titulo_leyenda: Título de la leyenda.
        estetica: Objeto EsteticaGrafico.
        mostrar: Si True, llama plt.show().

    Returns:
        Tupla (fig, ax)
    """
    est = estetica or _EST_DEFAULT
    est.aplicar()

    df_plot = df.copy()
    if mapa_etiquetas and columna_hue:
        df_plot[columna_hue] = df_plot[columna_hue].map(mapa_etiquetas)

    # Intercambiar ejes si es horizontal
    x_col = columna_x if orientacion == 'vertical' else columna_y
    y_col = columna_y if orientacion == 'vertical' else columna_x

    fig, ax = plt.subplots(figsize=est.figsize)
    paleta = est.paleta

    kwargs_comun = dict(data=df_plot, x=x_col, y=y_col,
                        hue=columna_hue, palette=paleta, ax=ax)

    if mostrar_violin:
        sns.violinplot(
            **kwargs_comun,
            inner=None,
            width=0.8,
            alpha=est.alpha_relleno,
            saturation=0.85,
            linewidth=0.8,
            split=False,
        )

    if mostrar_box:
        ancho_box = 0.35 if mostrar_violin else 0.6
        sns.boxplot(
            **kwargs_comun,
            width=ancho_box,
            boxprops={'zorder': 2, 'alpha': est.alpha_puntos,
                      'edgecolor': 'black', 'linewidth': est.grosor_linea},
            medianprops={'color': est.color_mediana, 'linewidth': 2.5},
            whiskerprops={'color': 'black', 'linewidth': est.grosor_linea},
            capprops={'color': 'black', 'linewidth': est.grosor_linea},
            showfliers=not mostrar_strip,
            flierprops={'marker': 'o', 'markersize': 4,
                        'markerfacecolor': est.color_outlier, 'alpha': 0.5},
        )

    if mostrar_strip:
        sns.stripplot(
            **kwargs_comun,
            dodge=columna_hue is not None,
            s=max(3, est.tamaño_punto // 15),
            alpha=est.alpha_puntos,
            linewidth=0,
            jitter=0.15,
        )

    if mostrar_media and x_col is not None:
        grupos = df_plot[x_col].unique() if x_col else [None]
        hue_grupos = df_plot[columna_hue].unique() if columna_hue else [None]
        for gi, g in enumerate(grupos):
            for hi, h in enumerate(hue_grupos):
                subset = df_plot
                if g is not None:
                    subset = subset[subset[x_col] == g]
                if h is not None:
                    subset = subset[subset[columna_hue] == h]
                if subset.empty:
                    continue
                media_val = subset[y_col].mean() if orientacion == 'vertical' \
                    else subset[x_col].mean()
                pos = gi if h is None else gi + (hi - (len(hue_grupos)-1)/2) * 0.25
                if orientacion == 'vertical':
                    ax.plot(pos, media_val, 'D',
                            color=est.color_media, markersize=6, zorder=5)
                else:
                    ax.plot(media_val, pos, 'D',
                            color=est.color_media, markersize=6, zorder=5)

    est.aplicar_a_ax(ax, titulo=titulo,
                     xlabel=xlabel or (x_col or ''),
                     ylabel=ylabel or (y_col or ''))

    if columna_hue:
        handles, labels = ax.get_legend_handles_labels()
        n_hue = len(df_plot[columna_hue].unique())
        ax.legend(handles[:n_hue], labels[:n_hue],
                  **est.kwargs_leyenda(titulo_leyenda))

    plt.tight_layout()
    if mostrar:
        plt.show()
    return fig, ax


# ─────────────────────────────────────────────────────────────
# Heatmap de correlación
# ─────────────────────────────────────────────────────────────

def heatmap_correlacion(
    df: pd.DataFrame,
    columnas: Optional[List[str]] = None,
    metodo: Literal['pearson', 'spearman', 'kendall'] = 'pearson',
    mostrar_valores: bool = True,
    mascara_triangulo: bool = True,
    vmin: float = -1.0,
    vmax: float = 1.0,
    titulo: Optional[str] = None,
    estetica: Optional[EsteticaGrafico] = None,
    mostrar: bool = True,
) -> Tuple[Figure, Axes]:
    """
    Heatmap de la matriz de correlación entre variables numéricas del DataFrame.

    Args:
        df: DataFrame con los datos. Solo se usan columnas numéricas.
        columnas: Subconjunto de columnas a correlacionar. None → todas las numéricas.
        metodo: Método de correlación: 'pearson', 'spearman' o 'kendall'.
                'pearson'  → correlación lineal (asume normalidad).
                'spearman' → correlación de rangos (robusta, no lineal).
                'kendall'  → correlación de concordancia de rangos.
        mostrar_valores: Si True, anota el coeficiente en cada celda.
        mascara_triangulo: Si True, oculta el triángulo superior (simétrico).
        vmin: Mínimo de la escala de color.
        vmax: Máximo de la escala de color.
        titulo: Título del gráfico. None → se genera automáticamente.
        estetica: Objeto EsteticaGrafico.
        mostrar: Si True, llama plt.show().

    Returns:
        Tupla (fig, ax)
    """
    est = estetica or _EST_DEFAULT
    est.aplicar()

    df_num = df[columnas] if columnas else df.select_dtypes(include=np.number)
    if df_num.empty:
        raise ValueError("No hay columnas numéricas en el DataFrame.")

    corr = df_num.corr(method=metodo)

    mask = None
    if mascara_triangulo:
        mask = np.triu(np.ones_like(corr, dtype=bool))

    n = len(corr)
    tam = max(6, n * 0.8)
    fig, ax = plt.subplots(figsize=(tam, tam * 0.85))

    sns.heatmap(
        corr,
        mask=mask,
        cmap=est.paleta_divergente,
        vmin=vmin, vmax=vmax,
        center=0,
        annot=mostrar_valores,
        fmt='.2f',
        linewidths=0.5,
        linecolor='white',
        square=True,
        ax=ax,
        annot_kws={'size': est.fuente_anotaciones},
    )

    titulo_final = titulo or f'Correlación ({metodo.capitalize()})'
    est.aplicar_a_ax(ax, titulo=titulo_final, xlabel='', ylabel='')
    ax.tick_params(axis='x', rotation=45, labelsize=est.fuente_ticks)
    ax.tick_params(axis='y', rotation=0,  labelsize=est.fuente_ticks)

    plt.tight_layout()
    if mostrar:
        plt.show()
    return fig, ax


# ─────────────────────────────────────────────────────────────
# Pairplot
# ─────────────────────────────────────────────────────────────

def pairplot(
    df: pd.DataFrame,
    columnas: Optional[List[str]] = None,
    columna_color: Optional[str] = None,
    tipo_diagonal: Literal['hist', 'kde'] = 'kde',
    tipo_offdiagonal: Literal['scatter', 'reg', 'kde'] = 'scatter',
    titulo: str = 'Matriz de Scatter',
    estetica: Optional[EsteticaGrafico] = None,
    mostrar: bool = True,
) -> Figure:
    """
    Matriz de scatter y distribuciones marginales (seaborn pairplot).

    Muestra simultáneamente todas las relaciones por pares entre variables
    numéricas. La diagonal muestra la distribución marginal de cada variable.
    Útil para detectar correlaciones, clusters y distribuciones no gaussianas
    antes de aplicar reductores de dimensionalidad.

    Args:
        df: DataFrame con los datos.
        columnas: Subconjunto de columnas a incluir. None → todas las numéricas.
        columna_color: Columna para colorear por grupo (categórica).
        tipo_diagonal: Tipo de gráfico en la diagonal: 'hist' o 'kde'.
        tipo_offdiagonal: Tipo de gráfico fuera de la diagonal:
                         'scatter': puntos individuales.
                         'reg'    : scatter + línea de regresión.
                         'kde'    : densidad 2D.
        titulo: Título superpuesto en la figura.
        estetica: Objeto EsteticaGrafico.
        mostrar: Si True, llama plt.show().

    Returns:
        Figura del pairplot (seaborn PairGrid).
    """
    est = estetica or _EST_DEFAULT
    est.aplicar()

    df_plot = df.copy()
    vars_plot = columnas or list(df_plot.select_dtypes(include=np.number).columns)

    if len(vars_plot) > 8:
        warnings.warn(
            f"Pairplot con {len(vars_plot)} variables puede ser muy lento. "
            "Considerar reducir el número de columnas.",
            UserWarning, stacklevel=2,
        )

    kwargs_pp: Dict = dict(
        data=df_plot,
        vars=vars_plot,
        hue=columna_color,
        palette=est.paleta,
        diag_kind=tipo_diagonal,
        plot_kws={'alpha': est.alpha_puntos, 's': est.tamaño_punto // 3},
        diag_kws={'alpha': est.alpha_relleno + 0.2},
        corner=False,
    )

    if tipo_offdiagonal == 'reg':
        g = sns.pairplot(**{k: v for k, v in kwargs_pp.items() if k != 'plot_kws'},
                         kind='reg',
                         plot_kws={'scatter_kws': {'alpha': est.alpha_puntos,
                                                    's': est.tamaño_punto // 3}})
    elif tipo_offdiagonal == 'kde':
        g = sns.pairplot(**{k: v for k, v in kwargs_pp.items() if k != 'plot_kws'},
                         kind='kde',
                         plot_kws={'alpha': est.alpha_relleno + 0.1})
    else:
        g = sns.pairplot(**kwargs_pp)

    g.figure.suptitle(titulo, y=1.01,
                       fontsize=est.fuente_titulo + 2,
                       fontweight='bold' if est.negrita_titulo else 'normal',
                       fontfamily=est.fuente_familia)
    plt.tight_layout()
    if mostrar:
        plt.show()
    return g.figure