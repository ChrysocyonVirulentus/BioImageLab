"""
Gráficos de diagnóstico para modelos de aprendizaje automático.

INPUT  : objetos de modelo sklearn ya ajustados (fit) + np.ndarray o DataFrame.
OUTPUT : Tupla (fig, ax) o (fig, axes).

A diferencia de Plots_Estadisticos.py (solo DataFrames), estas funciones
requieren objetos de modelo para extraer parámetros internos:
centroides, hiperplanos de decisión, importancias de features, etc.

Modelos cubiertos:
    Clustering:
        elbow_kmeans()          Curva de inercia vs. K (elbow method)
        silhouette_kmeans()     Silhouette plot por muestra y cluster
        scatter_clusters()      Scatter 2D coloreado por cluster (KMeans o DBSCAN)
        reachability_dbscan()   Gráfico de alcanzabilidad (DBSCAN/OPTICS)

    Clasificación / Regresión:
        matriz_confusion()      Heatmap de la matriz de confusión
        curva_roc()             Curva ROC con AUC (binaria o multiclase OvR)
        importancia_features()  Importancias de RandomForest / GradientBoosting
        frontera_decision_2d()  Frontera de decisión en el plano PCA/UMAP (SVM, RF, etc.)

IMPORTANTE — Separación de responsabilidades:
    Estos gráficos NO entrenan los modelos.
    El entrenamiento se hace en los archivos kmeans.py, dbscan.py, svm.py,
    random_forest.py, etc. Este módulo solo visualiza modelos ya ajustados.
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
# Clustering — KMeans
# ─────────────────────────────────────────────────────────────

def elbow_kmeans(
    inercias: List[float],
    rango_k: Optional[List[int]] = None,
    k_optimo: Optional[int] = None,
    titulo: str = 'Método del Codo — KMeans',
    xlabel: str = 'Número de Clusters (K)',
    ylabel: str = 'Inercia (WCSS)',
    estetica: Optional[EsteticaGrafico] = None,
    mostrar: bool = True,
) -> Tuple[Figure, Axes]:
    """
    Curva de inercia (Within-Cluster Sum of Squares) vs. número de clusters K.

    Permite identificar visualmente el "codo": el punto donde agregar más
    clusters aporta rendimientos decrecientes. Se usa para elegir K antes
    de ajustar el modelo final de KMeans.

    Uso típico:
        from sklearn.cluster import KMeans
        inercias = []
        for k in range(2, 12):
            km = KMeans(n_clusters=k, random_state=42).fit(X_scaled)
            inercias.append(km.inertia_)
        fig, ax = elbow_kmeans(inercias, rango_k=list(range(2, 12)), k_optimo=4)

    Args:
        inercias: Lista de valores de inercia (km.inertia_) para cada K.
        rango_k: Lista de valores de K correspondientes a las inercias.
                None → se asume K = [1, 2, ..., len(inercias)].
        k_optimo: Si se especifica, traza una línea vertical en ese K.
        titulo: Título del gráfico.
        xlabel: Etiqueta eje X.
        ylabel: Etiqueta eje Y.
        estetica: Objeto EsteticaGrafico.
        mostrar: Si True, llama plt.show().

    Returns:
        Tupla (fig, ax)
    """
    est = estetica or _EST_DEFAULT
    est.aplicar()

    ks = rango_k if rango_k is not None else list(range(1, len(inercias) + 1))

    fig, ax = plt.subplots(figsize=est.figsize)
    paleta  = sns.color_palette(est.paleta)

    ax.plot(ks, inercias, 'o-', color=paleta[0], linewidth=est.grosor_linea + 0.5,
            markersize=8, markerfacecolor='white', markeredgewidth=2,
            markeredgecolor=paleta[0])

    for k, inercia in zip(ks, inercias):
        ax.annotate(f'{inercia:.0f}', (k, inercia),
                    textcoords='offset points', xytext=(0, 8),
                    ha='center', fontsize=est.fuente_anotaciones,
                    color='dimgray')

    if k_optimo is not None:
        ax.axvline(k_optimo, linestyle='--', color=est.color_outlier,
                   linewidth=est.grosor_linea,
                   label=f'K óptimo = {k_optimo}')
        ax.legend(**est.kwargs_leyenda())

    ax.set_xticks(ks)
    est.aplicar_a_ax(ax, titulo=titulo, xlabel=xlabel, ylabel=ylabel)
    plt.tight_layout()
    if mostrar:
        plt.show()
    return fig, ax


def silhouette_kmeans(
    X: np.ndarray,
    etiquetas: np.ndarray,
    titulo: str = 'Silhouette Plot — KMeans',
    estetica: Optional[EsteticaGrafico] = None,
    mostrar: bool = True,
) -> Tuple[Figure, Axes]:
    """
    Silhouette plot: coeficiente de silueta por muestra, agrupado por cluster.

    El coeficiente de silueta s(i) para la muestra i mide qué tan bien
    asignada está comparada con otras muestras de su cluster:
        s(i) = (b(i) - a(i)) / max(a(i), b(i))
        a(i) = distancia media intra-cluster
        b(i) = distancia media al cluster más cercano

    Interpretación:
        s ≈  1 : muestra bien asignada, cluster compacto y separado.
        s ≈  0 : muestra en el borde entre clusters.
        s ≈ -1 : muestra probablemente mal asignada.

    La línea punteada vertical indica el coeficiente medio global.

    Args:
        X: Matriz (N, D) de features estandarizadas.
        etiquetas: Array (N,) con etiquetas de cluster (km.labels_).
        titulo: Título del gráfico.
        estetica: Objeto EsteticaGrafico.
        mostrar: Si True, llama plt.show().

    Returns:
        Tupla (fig, ax)
    """
    from sklearn.metrics import silhouette_samples, silhouette_score

    est = estetica or _EST_DEFAULT
    est.aplicar()

    coefs   = silhouette_samples(X, etiquetas)
    media   = silhouette_score(X, etiquetas)
    clusters = np.unique(etiquetas)
    n_clus  = len(clusters)

    paleta = sns.color_palette(est.paleta, n_clus)

    fig, ax = plt.subplots(figsize=est.figsize)
    y_lower = 10

    for ci, cluster in enumerate(sorted(clusters)):
        vals = np.sort(coefs[etiquetas == cluster])
        size_c = len(vals)
        y_upper = y_lower + size_c

        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, vals,
                         facecolor=paleta[ci], edgecolor=paleta[ci],
                         alpha=est.alpha_relleno + 0.3)
        ax.text(-0.05, y_lower + 0.5 * size_c, f'Cluster {cluster}',
                fontsize=est.fuente_anotaciones, color=paleta[ci])
        y_lower = y_upper + 10

    ax.axvline(media, linestyle='--', color=est.color_media,
               linewidth=est.grosor_linea,
               label=f'Silhouette media = {media:.3f}')

    est.aplicar_a_ax(ax, titulo=titulo,
                     xlabel='Coeficiente de Silueta',
                     ylabel='Muestra (agrupada por cluster)',
                     grid=False)
    ax.set_yticks([])
    ax.set_xlim(-0.2, 1.0)
    ax.legend(**est.kwargs_leyenda())
    plt.tight_layout()
    if mostrar:
        plt.show()
    return fig, ax


def scatter_clusters(
    Z: np.ndarray,
    etiquetas: np.ndarray,
    columna_forma: Optional[np.ndarray] = None,
    nombres_clusters: Optional[Dict[int, str]] = None,
    centroides: Optional[np.ndarray] = None,
    titulo: str = 'Clusters en Espacio 2D',
    nombre_eje_x: str = 'Eje 1',
    nombre_eje_y: str = 'Eje 2',
    marcar_ruido: bool = True,
    estetica: Optional[EsteticaGrafico] = None,
    mostrar: bool = True,
) -> Tuple[Figure, Axes]:
    """
    Scatter 2D coloreado por cluster. Compatible con KMeans y DBSCAN.

    Para DBSCAN, el cluster −1 (ruido) se muestra en gris si marcar_ruido=True.
    Si se proveen centroides (KMeans), se marcan con una estrella.

    Args:
        Z: Array (N, 2+). Se usan las primeras dos columnas.
        etiquetas: Array (N,) con etiquetas de cluster.
        columna_forma: Array (N,) para estilo de marcador (p.ej. genotipo).
        nombres_clusters: Dict {id_cluster: nombre_legible}.
        centroides: Array (K, 2) con coordenadas de centroides (KMeans).
        titulo: Título del gráfico.
        nombre_eje_x: Etiqueta eje X.
        nombre_eje_y: Etiqueta eje Y.
        marcar_ruido: Si True, pinta el cluster −1 (ruido DBSCAN) en gris.
        estetica: Objeto EsteticaGrafico.
        mostrar: Si True, llama plt.show().

    Returns:
        Tupla (fig, ax)
    """
    est = estetica or _EST_DEFAULT
    est.aplicar()

    fig, ax = plt.subplots(figsize=est.figsize)

    labels_unicos = np.unique(etiquetas)
    paleta = sns.color_palette(est.paleta, sum(l >= 0 for l in labels_unicos))
    color_idx = 0

    for lab in sorted(labels_unicos):
        mask  = etiquetas == lab
        color = 'lightgray' if (lab == -1 and marcar_ruido) else paleta[color_idx]
        label = (nombres_clusters.get(lab, f'Cluster {lab}') if nombres_clusters
                 else ('Ruido' if lab == -1 else f'Cluster {lab}'))

        if lab >= 0:
            color_idx += 1

        ax.scatter(Z[mask, 0], Z[mask, 1],
                   c=[color], s=est.tamaño_punto, alpha=est.alpha_puntos,
                   label=label, zorder=3,
                   edgecolors='none' if lab == -1 else 'white',
                   linewidths=0.3)

    if centroides is not None:
        ax.scatter(centroides[:, 0], centroides[:, 1],
                   marker='*', s=300, c='black', zorder=5,
                   label='Centroides', edgecolors='white', linewidths=0.5)

    ax.legend(**est.kwargs_leyenda('Cluster'))
    est.aplicar_a_ax(ax, titulo=titulo,
                     xlabel=nombre_eje_x, ylabel=nombre_eje_y)
    plt.tight_layout()
    if mostrar:
        plt.show()
    return fig, ax


# ─────────────────────────────────────────────────────────────
# Clustering — DBSCAN / OPTICS
# ─────────────────────────────────────────────────────────────

def reachability_dbscan(
    modelo,
    titulo: str = 'Gráfico de Alcanzabilidad — OPTICS/DBSCAN',
    xlabel: str = 'Orden de procesamiento',
    ylabel: str = 'Distancia de alcanzabilidad',
    estetica: Optional[EsteticaGrafico] = None,
    mostrar: bool = True,
) -> Tuple[Figure, Axes]:
    """
    Gráfico de alcanzabilidad (reachability plot) para OPTICS o DBSCAN.

    Muestra la distancia de alcanzabilidad de cada punto en el orden en que
    fue procesado por OPTICS. Los valles en el gráfico corresponden a clusters:
    puntos dentro de un cluster tienen distancias bajas; los picos entre
    valles indican separación entre clusters.

    Uso típico:
        from sklearn.cluster import OPTICS
        op = OPTICS(min_samples=5, xi=0.05).fit(X_scaled)
        fig, ax = reachability_dbscan(op)

    Args:
        modelo: Modelo OPTICS ajustado (debe tener reachability_ y labels_).
        titulo: Título del gráfico.
        xlabel: Etiqueta eje X.
        ylabel: Etiqueta eje Y.
        estetica: Objeto EsteticaGrafico.
        mostrar: Si True, llama plt.show().

    Returns:
        Tupla (fig, ax)
    """
    est = estetica or _EST_DEFAULT
    est.aplicar()

    if not hasattr(modelo, 'reachability_'):
        raise ValueError(
            "El modelo debe ser sklearn OPTICS con atributo reachability_. "
            "Para DBSCAN puro, usar scatter_clusters() directamente."
        )

    reach   = modelo.reachability_[modelo.ordering_]
    labels  = modelo.labels_[modelo.ordering_]
    n       = len(reach)

    clusters = np.unique(labels[labels >= 0])
    paleta   = sns.color_palette(est.paleta, len(clusters))

    fig, ax = plt.subplots(figsize=est.figsize)

    # Colorear cada punto según su cluster
    for i in range(n):
        lab   = labels[i]
        color = 'lightgray' if lab == -1 else paleta[lab % len(paleta)]
        ax.bar(i, reach[i], color=color, width=1.0, alpha=0.85, linewidth=0)

    # Leyenda manual
    handles = [plt.Rectangle((0, 0), 1, 1, fc=paleta[ci % len(paleta)])
               for ci in range(len(clusters))]
    handles.append(plt.Rectangle((0, 0), 1, 1, fc='lightgray'))
    labels_leg = [f'Cluster {c}' for c in clusters] + ['Ruido']
    ax.legend(handles, labels_leg, **est.kwargs_leyenda('Cluster'))

    est.aplicar_a_ax(ax, titulo=titulo, xlabel=xlabel, ylabel=ylabel)
    plt.tight_layout()
    if mostrar:
        plt.show()
    return fig, ax


# ─────────────────────────────────────────────────────────────
# Clasificación — Matriz de confusión
# ─────────────────────────────────────────────────────────────

def matriz_confusion(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series],
    nombres_clases: Optional[List[str]] = None,
    normalizar: Literal['true', 'pred', 'all', None] = 'true',
    titulo: str = 'Matriz de Confusión',
    estetica: Optional[EsteticaGrafico] = None,
    mostrar: bool = True,
) -> Tuple[Figure, Axes]:
    """
    Heatmap de la matriz de confusión para clasificadores.

    La normalización controla qué se muestra en cada celda:
        'true' : fracción de la clase verdadera (sensibilidad por fila).
        'pred' : fracción de la clase predicha (precisión por columna).
        'all'  : fracción del total de muestras.
        None   : conteos absolutos.

    Args:
        y_true: Etiquetas verdaderas.
        y_pred: Etiquetas predichas por el modelo.
        nombres_clases: Nombres de las clases para los ejes.
        normalizar: Tipo de normalización.
        titulo: Título del gráfico.
        estetica: Objeto EsteticaGrafico.
        mostrar: Si True, llama plt.show().

    Returns:
        Tupla (fig, ax)
    """
    from sklearn.metrics import confusion_matrix

    est = estetica or _EST_DEFAULT
    est.aplicar()

    cm = confusion_matrix(y_true, y_pred, normalize=normalizar)

    fmt    = '.2f' if normalizar else 'd'
    vmax   = 1.0   if normalizar else None
    n      = cm.shape[0]
    tam    = max(6, n * 1.1)

    fig, ax = plt.subplots(figsize=(tam, tam * 0.9))
    sns.heatmap(
        cm,
        annot=True, fmt=fmt,
        cmap=est.paleta_continua,
        vmin=0, vmax=vmax,
        xticklabels=nombres_clases or 'auto',
        yticklabels=nombres_clases or 'auto',
        linewidths=0.5, linecolor='white',
        ax=ax,
        annot_kws={'size': est.fuente_anotaciones + 1},
    )
    est.aplicar_a_ax(ax, titulo=titulo,
                     xlabel='Predicho', ylabel='Verdadero')
    ax.tick_params(axis='x', rotation=45, labelsize=est.fuente_ticks)
    ax.tick_params(axis='y', rotation=0,  labelsize=est.fuente_ticks)
    plt.tight_layout()
    if mostrar:
        plt.show()
    return fig, ax


# ─────────────────────────────────────────────────────────────
# Clasificación — Curva ROC
# ─────────────────────────────────────────────────────────────

def curva_roc(
    y_true: Union[np.ndarray, pd.Series],
    y_scores: np.ndarray,
    nombres_clases: Optional[List[str]] = None,
    titulo: str = 'Curva ROC',
    estetica: Optional[EsteticaGrafico] = None,
    mostrar: bool = True,
) -> Tuple[Figure, Axes]:
    """
    Curva ROC con área bajo la curva (AUC) para clasificadores binarios
    o multiclase (estrategia One-vs-Rest).

    Para clasificación binaria:
        y_scores debe ser array 1D con la probabilidad de la clase positiva.

    Para clasificación multiclase:
        y_scores debe ser array (N, K) con probabilidades por clase.
        Se grafican K curvas ROC (una por clase) más la micro-average.

    Args:
        y_true: Etiquetas verdaderas.
        y_scores: Probabilidades predichas (modelo.predict_proba()).
        nombres_clases: Nombres de las clases para la leyenda.
        titulo: Título del gráfico.
        estetica: Objeto EsteticaGrafico.
        mostrar: Si True, llama plt.show().

    Returns:
        Tupla (fig, ax)
    """
    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import label_binarize

    est = estetica or _EST_DEFAULT
    est.aplicar()

    fig, ax = plt.subplots(figsize=est.figsize)
    paleta = sns.color_palette(est.paleta)

    y_arr = np.asarray(y_true)
    clases = np.unique(y_arr)

    if y_scores.ndim == 1 or len(clases) == 2:
        # Binario
        scores_1d = y_scores if y_scores.ndim == 1 else y_scores[:, 1]
        fpr, tpr, _ = roc_curve(y_arr, scores_1d)
        roc_auc     = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=paleta[0], linewidth=est.grosor_linea + 0.5,
                label=f'AUC = {roc_auc:.3f}')
    else:
        # Multiclase OvR
        y_bin   = label_binarize(y_arr, classes=clases)
        nombres = nombres_clases or [str(c) for c in clases]

        for i, nombre in enumerate(nombres):
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_scores[:, i])
            roc_auc     = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=paleta[i % len(paleta)],
                    linewidth=est.grosor_linea,
                    label=f'{nombre} (AUC={roc_auc:.3f})')

        # Micro-average
        fpr_micro, tpr_micro, _ = roc_curve(y_bin.ravel(), y_scores.ravel())
        auc_micro = auc(fpr_micro, tpr_micro)
        ax.plot(fpr_micro, tpr_micro, color='black',
                linewidth=est.grosor_linea + 0.5, linestyle='-.',
                label=f'Micro-average (AUC={auc_micro:.3f})')

    # Línea de clasificador aleatorio
    ax.plot([0, 1], [0, 1], linestyle='--', linewidth=est.grosor_linea,
            color=est.color_referencia, label='Aleatorio (AUC=0.5)')

    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])
    ax.legend(**est.kwargs_leyenda())
    est.aplicar_a_ax(ax, titulo=titulo,
                     xlabel='Tasa de Falsos Positivos (FPR)',
                     ylabel='Tasa de Verdaderos Positivos (TPR)')
    plt.tight_layout()
    if mostrar:
        plt.show()
    return fig, ax


# ─────────────────────────────────────────────────────────────
# RandomForest / GradientBoosting — Importancia de features
# ─────────────────────────────────────────────────────────────

def importancia_features(
    modelo,
    nombres_features: List[str],
    top_n: int = 20,
    tipo: Literal['impureza', 'permutacion'] = 'impureza',
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    titulo: str = 'Importancia de Features',
    estetica: Optional[EsteticaGrafico] = None,
    mostrar: bool = True,
) -> Tuple[Figure, Axes]:
    """
    Gráfico horizontal de importancia de features para RandomForest o
    GradientBoosting.

    Dos tipos de importancia:
        'impureza'    : feature_importances_ (MDI, Mean Decrease Impurity).
                       Rápido, integrado en el modelo. Puede sobreestimar
                       features de alta cardinalidad.
        'permutacion' : sklearn PermutationImportance sobre un set de validación.
                       Más costoso pero más robusto. Requiere X_val, y_val.

    Args:
        modelo: Modelo con atributo feature_importances_ (RF, GBM, etc.).
        nombres_features: Nombres de las D features del modelo.
        top_n: Número de features más importantes a mostrar.
        tipo: 'impureza' o 'permutacion'.
        X_val: Features de validación (requerido si tipo='permutacion').
        y_val: Etiquetas de validación (requerido si tipo='permutacion').
        titulo: Título del gráfico.
        estetica: Objeto EsteticaGrafico.
        mostrar: Si True, llama plt.show().

    Returns:
        Tupla (fig, ax)
    """
    est = estetica or _EST_DEFAULT
    est.aplicar()

    if tipo == 'permutacion':
        if X_val is None or y_val is None:
            raise ValueError(
                "tipo='permutacion' requiere X_val e y_val."
            )
        from sklearn.inspection import permutation_importance
        result      = permutation_importance(modelo, X_val, y_val,
                                             n_repeats=10, random_state=42)
        importancias = result.importances_mean
        errores      = result.importances_std
    else:
        if not hasattr(modelo, 'feature_importances_'):
            raise ValueError(
                "El modelo no tiene atributo feature_importances_. "
                "Usar tipo='permutacion' para otros modelos."
            )
        importancias = modelo.feature_importances_
        errores      = None

    indices    = np.argsort(importancias)[-top_n:]
    nombres_ok = np.array(nombres_features)[indices]
    imp_ok     = importancias[indices]
    err_ok     = errores[indices] if errores is not None else None

    fig, ax = plt.subplots(
        figsize=(est.figsize[0], max(5, top_n * 0.35))
    )
    paleta = sns.color_palette(est.paleta_continua, top_n)[::-1]

    barras = ax.barh(range(len(imp_ok)), imp_ok,
                     color=paleta, alpha=est.alpha_relleno + 0.4,
                     edgecolor='none')

    if err_ok is not None:
        ax.barh(range(len(imp_ok)), imp_ok,
                xerr=err_ok, color='none',
                ecolor='black', capsize=3, linewidth=0.8)

    ax.set_yticks(range(len(imp_ok)))
    ax.set_yticklabels(nombres_ok, fontsize=est.fuente_ticks)

    # Anotar valor al final de cada barra
    for i, v in enumerate(imp_ok):
        ax.text(v + 0.001, i, f'{v:.4f}',
                va='center', fontsize=est.fuente_anotaciones, color='dimgray')

    tipo_label = 'MDI (Mean Decrease Impurity)' if tipo == 'impureza' \
        else 'Importancia por Permutación'
    est.aplicar_a_ax(ax, titulo=titulo,
                     xlabel=tipo_label, ylabel='Feature')
    plt.tight_layout()
    if mostrar:
        plt.show()
    return fig, ax


# ─────────────────────────────────────────────────────────────
# Frontera de decisión 2D
# ─────────────────────────────────────────────────────────────

def frontera_decision_2d(
    modelo,
    X_2d: np.ndarray,
    etiquetas: np.ndarray,
    resolucion: int = 200,
    nombres_clases: Optional[List[str]] = None,
    titulo: str = 'Frontera de Decisión (2D)',
    nombre_eje_x: str = 'Eje 1',
    nombre_eje_y: str = 'Eje 2',
    estetica: Optional[EsteticaGrafico] = None,
    mostrar: bool = True,
) -> Tuple[Figure, Axes]:
    """
    Frontera de decisión del clasificador en el plano de dos dimensiones.

    Dibuja el fondo del gráfico coloreado según la clase predicha por el
    modelo en una malla de puntos, y superpone los datos reales. Funciona
    con cualquier clasificador de sklearn.

    Advertencia:
        X_2d debe ser la representación 2D de los datos (p.ej. salida de PCA
        o UMAP de 2 componentes). El modelo debe haber sido entrenado sobre
        esas mismas 2 features para que las fronteras sean coherentes.
        Si el modelo fue entrenado en alta dimensión, ver nota de uso.

    Nota de uso para modelos entrenados en alta dimensión:
        Entrenar un modelo auxiliar sobre X_2d para visualizar aproximadamente
        las fronteras: clf_2d.fit(X_2d, etiquetas); frontera_decision_2d(clf_2d, ...).

    Args:
        modelo: Clasificador sklearn con predict() ajustado sobre X_2d.
        X_2d: Array (N, 2) con coordenadas en el espacio 2D.
        etiquetas: Array (N,) con etiquetas verdaderas de clase.
        resolucion: Resolución de la malla de decisión (píxeles por eje).
        nombres_clases: Nombres de las clases para la leyenda.
        titulo: Título del gráfico.
        nombre_eje_x: Etiqueta eje X.
        nombre_eje_y: Etiqueta eje Y.
        estetica: Objeto EsteticaGrafico.
        mostrar: Si True, llama plt.show().

    Returns:
        Tupla (fig, ax)
    """
    est = estetica or _EST_DEFAULT
    est.aplicar()

    clases = np.unique(etiquetas)
    n_cl   = len(clases)
    paleta = sns.color_palette(est.paleta, n_cl)

    # Malla de decisión
    margen = 0.05
    x_min, x_max = X_2d[:, 0].min(), X_2d[:, 0].max()
    y_min, y_max = X_2d[:, 1].min(), X_2d[:, 1].max()
    dx = (x_max - x_min) * margen
    dy = (y_max - y_min) * margen

    xx, yy = np.meshgrid(
        np.linspace(x_min - dx, x_max + dx, resolucion),
        np.linspace(y_min - dy, y_max + dy, resolucion),
    )
    Z = modelo.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Mapa de colores discreto para las regiones
    from matplotlib.colors import ListedColormap
    cmap_fondo = ListedColormap([(*c, 0.25) for c in paleta])

    fig, ax = plt.subplots(figsize=est.figsize)
    ax.contourf(xx, yy, Z, alpha=0.3, cmap=cmap_fondo)
    ax.contour(xx, yy, Z, colors='gray', linewidths=0.5, alpha=0.6)

    for ci, clase in enumerate(clases):
        mask  = etiquetas == clase
        label = nombres_clases[ci] if nombres_clases else f'Clase {clase}'
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                   c=[paleta[ci]], s=est.tamaño_punto,
                   alpha=est.alpha_puntos,
                   edgecolors='white', linewidths=0.4,
                   label=label, zorder=3)

    ax.legend(**est.kwargs_leyenda('Clase'))
    est.aplicar_a_ax(ax, titulo=titulo,
                     xlabel=nombre_eje_x, ylabel=nombre_eje_y)
    plt.tight_layout()
    if mostrar:
        plt.show()
    return fig, ax