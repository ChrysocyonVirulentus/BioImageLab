"""
Control de calidad visual del pipeline de procesamiento de bioimágenes.

Este módulo genera figuras de diagnóstico que muestran cada transformación
aplicada a una imagen de microscopía, desde la carga hasta la segmentación.
Su propósito es exclusivamente de QC (Quality Control): verificar que cada
etapa produjo el resultado esperado antes de continuar con la cuantificación.

Diseño:
    Cada función recibe una o más imágenes (np.ndarray 2D) y sus etiquetas
    descriptivas, y genera un panel de subplots comparativo con:
        - Título de cada subplot indicando la etapa del pipeline.
        - Histograma de intensidades opcional en cada panel.
        - Barra de color compartida o independiente.
        - Métricas estadísticas anotadas (min, max, media, std).

    La figura resultante puede mostrarse inline (Colab/Jupyter) o guardarse
    a disco usando figures.guardar_figura() o figures.guardar_desde_funcion().

Funciones disponibles:
    panel_transformaciones()    N imágenes de distintas etapas en una fila.
    panel_antes_despues()       Comparación 1×2 o 2×2 con diferencia.
    panel_segmentacion()        Imagen original + máscara + overlay + histograma.
    panel_canales()             Imagen multicanal desglosada por canal + merge.
    panel_zstack()              Proyecciones de un z-stack (max, sum, mean, std).

IMPORTANTE — Separación de responsabilidades:
    Este módulo NO aplica transformaciones a las imágenes.
    Recibe imágenes ya procesadas por las etapas anteriores del pipeline
    (normalización, filtrado, binarización, etc.) y solo las visualiza.
    El guardado del panel se hace con figures.guardar_figura().
"""

import warnings
from typing import Dict, List, Literal, Optional, Tuple, Union

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from Estetica import EsteticaGrafico, ESTILOS_PREDEFINIDOS

_EST_DEFAULT = EsteticaGrafico(
    figsize=(6, 5),
    paleta_continua='gray',
    fuente_titulo=11,
    fuente_ejes=9,
    fuente_ticks=8,
    fuente_anotaciones=8,
)


# ─────────────────────────────────────────────────────────────
# Helpers internos
# ─────────────────────────────────────────────────────────────

def _normalizar_display(
    img: np.ndarray,
    modo: Literal['minmax', 'percentil', 'log', 'sqrt', 'ninguno'] = 'percentil',
    p_low: float = 1.0,
    p_high: float = 99.0,
) -> np.ndarray:
    """Normaliza una imagen a [0, 1] para display."""
    img = img.astype(np.float64)
    if modo == 'log':
        img = np.log1p(np.maximum(img, 0))
    elif modo == 'sqrt':
        img = np.sqrt(np.maximum(img, 0))

    if modo == 'percentil':
        vmin = np.percentile(img, p_low)
        vmax = np.percentile(img, p_high)
    elif modo == 'minmax':
        vmin, vmax = img.min(), img.max()
    elif modo == 'ninguno':
        return img
    else:
        vmin, vmax = img.min(), img.max()

    rango = vmax - vmin
    return np.clip((img - vmin) / rango if rango > 0 else img * 0, 0, 1)


def _anotar_estadisticos(
    ax: Axes,
    img: np.ndarray,
    fuente: int = 8,
    color: str = 'white',
    posicion: Literal['esquina', 'titulo'] = 'esquina',
) -> None:
    """Anota min/max/media/std sobre un Axes de imagen."""
    texto = (
        f"min={img.min():.1f}  max={img.max():.1f}\n"
        f"μ={img.mean():.1f}  σ={img.std():.1f}"
    )
    if posicion == 'esquina':
        ax.text(
            0.02, 0.02, texto,
            transform=ax.transAxes,
            fontsize=fuente, color=color,
            verticalalignment='bottom',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.5),
        )
    else:
        titulo_actual = ax.get_title()
        ax.set_title(f"{titulo_actual}\n{texto}", fontsize=fuente)


def _subplot_imagen(
    ax: Axes,
    img: np.ndarray,
    titulo: str,
    cmap: str = 'gray',
    normalizar: Literal['minmax', 'percentil', 'log', 'sqrt', 'ninguno'] = 'percentil',
    mostrar_stats: bool = True,
    colorbar: bool = True,
    fuente_titulo: int = 10,
    fuente_stats: int = 8,
) -> None:
    """Dibuja una sola imagen en un Axes con título, stats y colorbar."""
    img_display = _normalizar_display(img, modo=normalizar)
    im = ax.imshow(img_display, cmap=cmap, origin='lower', vmin=0, vmax=1)
    ax.set_title(titulo, fontsize=fuente_titulo, pad=4)
    ax.set_xticks([])
    ax.set_yticks([])
    if colorbar:
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if mostrar_stats:
        _anotar_estadisticos(ax, img, fuente=fuente_stats)


# ─────────────────────────────────────────────────────────────
# 1. Panel de transformaciones en cadena
# ─────────────────────────────────────────────────────────────

def panel_transformaciones(
    imagenes: List[np.ndarray],
    etiquetas: List[str],
    titulo_general: str = 'QC — Pipeline de Procesamiento',
    cmap: str = 'gray',
    normalizar: Literal['minmax', 'percentil', 'log', 'sqrt', 'ninguno'] = 'percentil',
    mostrar_histograma: bool = True,
    mostrar_stats: bool = True,
    colorbar: bool = True,
    ancho_por_imagen: float = 4.0,
    alto: float = 4.5,
    fuente_titulo_general: int = 13,
    fuente_subtitulo: int = 10,
    estetica: Optional[EsteticaGrafico] = None,
    mostrar: bool = True,
) -> Tuple[Figure, np.ndarray]:
    """
    Panel con N imágenes de distintas etapas del pipeline en una fila horizontal.

    Muestra la progresión de transformaciones: carga → normalización →
    filtrado → realce → segmentación, con el título de cada etapa sobre
    cada subplot.

    Si mostrar_histograma=True, añade una segunda fila con el histograma
    de intensidades de cada imagen, permitiendo detectar cambios en la
    distribución de píxeles entre etapas.

    Args:
        imagenes: Lista de arrays 2D, una por etapa del pipeline.
                 Deben tener la misma forma (H, W) para comparación directa.
                 No es obligatorio (pueden ser versiones recortadas).

        etiquetas: Nombre de cada etapa, uno por imagen.
                  Ejemplos: ['Cruda', 'Normalizada', 'Filtro Gaussiano',
                             'CLAHE', 'Binarizada (Otsu)']

        titulo_general: Título del panel completo (suptitle).

        cmap: Mapa de color para todas las imágenes.
             'gray'    : escala de grises (por defecto, microscopía).
             'inferno' : buen contraste para fluorescencia.
             'hot'     : resalta valores altos.

        normalizar: Normalización de display para cada imagen.
                   'percentil': clip a [p1, p99] — recomendado para fluorescencia.
                   'minmax':    usa min y max de cada imagen individualmente.
                   'log':       escala logarítmica — útil para alto rango dinámico.
                   'sqrt':      raíz cuadrada — compresión suave.
                   'ninguno':   muestra valores crudos.

        mostrar_histograma: Si True, añade fila inferior con histogramas.

        mostrar_stats: Si True, anota min/max/media/std en cada imagen.

        colorbar: Si True, añade barra de color a cada subplot de imagen.

        ancho_por_imagen: Ancho en pulgadas de cada columna.

        alto: Alto en pulgadas de cada fila de imágenes.

        fuente_titulo_general: Tamaño de fuente del título principal.

        fuente_subtitulo: Tamaño de fuente del subtítulo de cada subplot.

        estetica: Objeto EsteticaGrafico. None → valores por defecto de QC.

        mostrar: Si True, llama plt.show(). False para guardar con figures.py.

    Returns:
        Tupla (fig, axes):
            axes tiene forma (1, N) si mostrar_histograma=False,
            o (2, N) si mostrar_histograma=True.

    Raises:
        ValueError: Si len(imagenes) != len(etiquetas) o listas vacías.
    """
    if len(imagenes) != len(etiquetas):
        raise ValueError(
            f"imagenes ({len(imagenes)}) y etiquetas ({len(etiquetas)}) "
            "deben tener la misma longitud."
        )
    if not imagenes:
        raise ValueError("La lista de imágenes está vacía.")

    est    = estetica or _EST_DEFAULT
    n      = len(imagenes)
    n_filas = 2 if mostrar_histograma else 1

    fig, axes = plt.subplots(
        n_filas, n,
        figsize=(ancho_por_imagen * n, alto * n_filas),
        squeeze=False,
    )

    for col, (img, label) in enumerate(zip(imagenes, etiquetas)):
        _subplot_imagen(
            ax=axes[0, col],
            img=img,
            titulo=f'[{col}] {label}',
            cmap=cmap,
            normalizar=normalizar,
            mostrar_stats=mostrar_stats,
            colorbar=colorbar,
            fuente_titulo=fuente_subtitulo,
        )

        if mostrar_histograma:
            ax_h = axes[1, col]
            datos = img.ravel().astype(np.float64)
            ax_h.hist(datos, bins=80, color='steelblue',
                      alpha=0.8, edgecolor='none', density=True)
            ax_h.set_title(f'Histograma — {label}',
                           fontsize=fuente_subtitulo - 1, pad=3)
            ax_h.set_xlabel('Intensidad', fontsize=est.fuente_ticks)
            ax_h.set_ylabel('Densidad',   fontsize=est.fuente_ticks)
            ax_h.tick_params(labelsize=est.fuente_ticks - 1)
            ax_h.spines['top'].set_visible(False)
            ax_h.spines['right'].set_visible(False)
            ax_h.grid(True, linestyle='--', alpha=0.4, axis='y')

    fig.suptitle(titulo_general, fontsize=fuente_titulo_general,
                 fontweight='bold', y=1.01)
    plt.tight_layout()
    if mostrar:
        plt.show()
    return fig, axes


# ─────────────────────────────────────────────────────────────
# 2. Panel antes / después
# ─────────────────────────────────────────────────────────────

def panel_antes_despues(
    img_antes: np.ndarray,
    img_despues: np.ndarray,
    etiqueta_antes: str = 'Antes',
    etiqueta_despues: str = 'Después',
    mostrar_diferencia: bool = True,
    titulo_general: str = 'QC — Antes / Después',
    cmap: str = 'gray',
    cmap_diferencia: str = 'RdBu_r',
    normalizar: Literal['minmax', 'percentil', 'log', 'sqrt', 'ninguno'] = 'percentil',
    mostrar_stats: bool = True,
    estetica: Optional[EsteticaGrafico] = None,
    mostrar: bool = True,
) -> Tuple[Figure, np.ndarray]:
    """
    Panel comparativo de una imagen antes y después de una transformación.

    Útil para verificar el efecto de una sola etapa: normalización,
    filtro de ruido, corrección de fondo, realce de contraste, etc.

    Crea un panel de 2 o 3 subplots:
        [Antes] [Después] [Diferencia (opcional)]

    La diferencia (img_despues - img_antes) se muestra con una paleta
    divergente para resaltar zonas donde la transformación aumentó (rojo)
    o disminuyó (azul) la intensidad.

    Args:
        img_antes: Array 2D de la imagen original (antes de la transformación).
        img_despues: Array 2D de la imagen transformada.
        etiqueta_antes: Nombre de la etapa de entrada.
        etiqueta_despues: Nombre de la etapa de salida.
        mostrar_diferencia: Si True, añade un tercer subplot con la diferencia.
        titulo_general: Título del panel.
        cmap: Mapa de color para las imágenes antes/después.
        cmap_diferencia: Mapa de color para el panel de diferencia.
        normalizar: Modo de normalización de display.
        mostrar_stats: Si True, anota estadísticas en cada imagen.
        estetica: Objeto EsteticaGrafico.
        mostrar: Si True, llama plt.show().

    Returns:
        Tupla (fig, axes) — array (1, 2) o (1, 3).
    """
    est    = estetica or _EST_DEFAULT
    n_cols = 3 if mostrar_diferencia else 2

    fig, axes = plt.subplots(
        1, n_cols,
        figsize=(4.5 * n_cols, 4.5),
        squeeze=False,
    )

    for ax, img, label in [
        (axes[0, 0], img_antes,   etiqueta_antes),
        (axes[0, 1], img_despues, etiqueta_despues),
    ]:
        _subplot_imagen(ax, img, titulo=label, cmap=cmap,
                        normalizar=normalizar, mostrar_stats=mostrar_stats,
                        fuente_titulo=est.fuente_ejes)

    if mostrar_diferencia:
        diff = img_despues.astype(np.float64) - img_antes.astype(np.float64)
        lim  = max(abs(diff.min()), abs(diff.max()))
        lim  = lim if lim > 0 else 1.0

        im = axes[0, 2].imshow(diff, cmap=cmap_diferencia, origin='lower',
                                vmin=-lim, vmax=lim)
        axes[0, 2].set_title('Diferencia (despues − antes)',
                              fontsize=est.fuente_ejes, pad=4)
        axes[0, 2].set_xticks([])
        axes[0, 2].set_yticks([])
        plt.colorbar(im, ax=axes[0, 2], fraction=0.046, pad=0.04)

        if mostrar_stats:
            _anotar_estadisticos(axes[0, 2], diff,
                                  fuente=est.fuente_anotaciones)

    fig.suptitle(titulo_general, fontsize=est.fuente_titulo,
                 fontweight='bold', y=1.01)
    plt.tight_layout()
    if mostrar:
        plt.show()
    return fig, axes


# ─────────────────────────────────────────────────────────────
# 3. Panel de segmentación
# ─────────────────────────────────────────────────────────────

def panel_segmentacion(
    img_original: np.ndarray,
    mascara: np.ndarray,
    img_procesada: Optional[np.ndarray] = None,
    etiqueta_original: str = 'Original',
    etiqueta_procesada: str = 'Procesada (pre-seg)',
    etiqueta_mascara: str = 'Máscara',
    etiqueta_overlay: str = 'Overlay',
    titulo_general: str = 'QC — Segmentación',
    cmap_imagen: str = 'gray',
    cmap_mascara: str = 'hot',
    alpha_overlay: float = 0.45,
    normalizar: Literal['minmax', 'percentil', 'log', 'sqrt', 'ninguno'] = 'percentil',
    mostrar_histograma_mascara: bool = True,
    mostrar_stats: bool = True,
    estetica: Optional[EsteticaGrafico] = None,
    mostrar: bool = True,
) -> Tuple[Figure, np.ndarray]:
    """
    Panel de QC específico para verificar la calidad de la segmentación.

    Crea un panel de 4–5 subplots:
        [Original] [Procesada*] [Máscara] [Overlay] [Histograma máscara*]
        (* opcionales)

    Permite verificar:
        - Si la máscara cubre correctamente los objetos de interés.
        - Si hay sobre/sub-segmentación.
        - Si el umbral de binarización fue adecuado (histograma de la máscara).
        - La calidad del realce previo a la segmentación.

    Args:
        img_original: Imagen sin procesar (resultado de la carga).
        mascara: Máscara binaria (0/255 o 0/1) o etiquetada (0,1,2,...).
        img_procesada: Imagen procesada usada como entrada para segmentación.
                      None → no se muestra (solo 4 subplots).
        etiqueta_original: Nombre para el subplot de imagen original.
        etiqueta_procesada: Nombre para el subplot de imagen procesada.
        etiqueta_mascara: Nombre para el subplot de máscara.
        etiqueta_overlay: Nombre para el subplot de overlay.
        titulo_general: Título del panel.
        cmap_imagen: Paleta para imágenes de intensidad.
        cmap_mascara: Paleta para la máscara.
        alpha_overlay: Transparencia de la máscara en el overlay.
        normalizar: Modo de normalización de display para imágenes de intensidad.
        mostrar_histograma_mascara: Si True, añade histograma de la máscara
                                   para verificar la distribución de píxeles.
        mostrar_stats: Si True, anota estadísticas en los subplots de imagen.
        estetica: Objeto EsteticaGrafico.
        mostrar: Si True, llama plt.show().

    Returns:
        Tupla (fig, axes).
    """
    est = estetica or _EST_DEFAULT

    tiene_procesada = img_procesada is not None
    n_imagen_cols   = 2 + int(tiene_procesada)  # original, [procesada,] overlay
    n_cols_total    = n_imagen_cols + 1 + int(mostrar_histograma_mascara)
    # +1 por máscara, +1 por histograma

    fig = plt.figure(
        figsize=(4.2 * n_cols_total, 4.5),
        tight_layout=True,
    )
    gs = gridspec.GridSpec(1, n_cols_total, figure=fig)

    axes = []
    col = 0

    # Original
    ax0 = fig.add_subplot(gs[0, col]); col += 1
    _subplot_imagen(ax0, img_original, titulo=etiqueta_original,
                    cmap=cmap_imagen, normalizar=normalizar,
                    mostrar_stats=mostrar_stats,
                    fuente_titulo=est.fuente_ejes)
    axes.append(ax0)

    # Procesada (opcional)
    if tiene_procesada:
        ax1 = fig.add_subplot(gs[0, col]); col += 1
        _subplot_imagen(ax1, img_procesada, titulo=etiqueta_procesada,
                        cmap=cmap_imagen, normalizar=normalizar,
                        mostrar_stats=mostrar_stats,
                        fuente_titulo=est.fuente_ejes)
        axes.append(ax1)

    # Máscara
    ax2 = fig.add_subplot(gs[0, col]); col += 1
    mascara_display = (mascara > 0).astype(np.uint8) * 255
    im_mask = ax2.imshow(mascara_display, cmap=cmap_mascara, origin='lower')
    ax2.set_title(etiqueta_mascara, fontsize=est.fuente_ejes, pad=4)
    ax2.set_xticks([]); ax2.set_yticks([])
    plt.colorbar(im_mask, ax=ax2, fraction=0.046, pad=0.04)
    n_positivos = int((mascara > 0).sum())
    pct         = 100 * n_positivos / mascara.size
    ax2.text(0.02, 0.02, f'pixels positivos: {n_positivos}\n({pct:.1f}%)',
             transform=ax2.transAxes, fontsize=est.fuente_anotaciones,
             color='yellow',
             bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.5))
    axes.append(ax2)

    # Overlay
    ax3 = fig.add_subplot(gs[0, col]); col += 1
    img_ref = img_procesada if tiene_procesada else img_original
    img_norm = _normalizar_display(img_ref, modo=normalizar)
    ax3.imshow(img_norm, cmap='gray', origin='lower')
    mascara_rgba = np.zeros((*mascara_display.shape, 4), dtype=np.float32)
    mask_bool    = mascara_display > 0
    mascara_rgba[mask_bool] = [1.0, 0.3, 0.0, alpha_overlay]  # naranja semitransparente
    ax3.imshow(mascara_rgba, origin='lower')
    ax3.set_title(etiqueta_overlay, fontsize=est.fuente_ejes, pad=4)
    ax3.set_xticks([]); ax3.set_yticks([])
    axes.append(ax3)

    # Histograma de la máscara
    if mostrar_histograma_mascara:
        ax4 = fig.add_subplot(gs[0, col])
        img_dentro  = img_ref[mask_bool].astype(np.float64)
        img_fuera   = img_ref[~mask_bool].astype(np.float64)
        bins = min(80, max(20, int(np.sqrt(len(img_dentro)))))
        ax4.hist(img_fuera,  bins=bins, alpha=0.5, color='steelblue',
                 label='Fondo',  density=True, edgecolor='none')
        ax4.hist(img_dentro, bins=bins, alpha=0.7, color='tomato',
                 label='Objeto', density=True, edgecolor='none')
        ax4.set_title('Intensidades: objeto vs fondo',
                      fontsize=est.fuente_ejes - 1, pad=3)
        ax4.set_xlabel('Intensidad', fontsize=est.fuente_ticks)
        ax4.set_ylabel('Densidad',   fontsize=est.fuente_ticks)
        ax4.tick_params(labelsize=est.fuente_ticks - 1)
        ax4.legend(fontsize=est.fuente_anotaciones, frameon=False)
        ax4.spines['top'].set_visible(False)
        ax4.spines['right'].set_visible(False)
        ax4.grid(True, linestyle='--', alpha=0.4, axis='y')
        axes.append(ax4)

    fig.suptitle(titulo_general, fontsize=est.fuente_titulo,
                 fontweight='bold', y=1.01)
    plt.tight_layout()
    if mostrar:
        plt.show()
    return fig, np.array(axes)


# ─────────────────────────────────────────────────────────────
# 4. Panel multicanal
# ─────────────────────────────────────────────────────────────

def panel_canales(
    canales: List[np.ndarray],
    nombres_canales: List[str],
    imagen_merge: Optional[np.ndarray] = None,
    colormaps: Optional[List[str]] = None,
    titulo_general: str = 'QC — Canales de Fluorescencia',
    normalizar: Literal['minmax', 'percentil', 'log', 'sqrt', 'ninguno'] = 'percentil',
    mostrar_stats: bool = True,
    estetica: Optional[EsteticaGrafico] = None,
    mostrar: bool = True,
) -> Tuple[Figure, np.ndarray]:
    """
    Panel con cada canal de una imagen multicanal y su composición.

    Útil para microscopía de fluorescencia multicanal (DAPI + GFP + mCherry)
    para verificar que cada canal tiene la señal esperada antes de cuantificar.

    Args:
        canales: Lista de arrays 2D, uno por canal.
        nombres_canales: Nombre de cada canal (p.ej. ['DAPI', 'GFP', 'mCherry']).
        imagen_merge: Array 2D o (H, W, 3) con la imagen merge de todos los canales.
                     None → no se muestra.
        colormaps: Lista de colormaps, uno por canal.
                  None → usa ['Blues', 'Greens', 'Reds', 'Purples', ...].
        titulo_general: Título del panel.
        normalizar: Modo de normalización de display.
        mostrar_stats: Si True, anota estadísticas.
        estetica: Objeto EsteticaGrafico.
        mostrar: Si True, llama plt.show().

    Returns:
        Tupla (fig, axes).
    """
    if len(canales) != len(nombres_canales):
        raise ValueError("canales y nombres_canales deben tener la misma longitud.")

    est = estetica or _EST_DEFAULT

    cmaps_default = ['Blues', 'Greens', 'Reds', 'Purples', 'Oranges', 'YlOrBr']
    cmaps = colormaps or [cmaps_default[i % len(cmaps_default)]
                          for i in range(len(canales))]

    n_extra = 1 if imagen_merge is not None else 0
    n_cols  = len(canales) + n_extra

    fig, axes = plt.subplots(
        1, n_cols,
        figsize=(4.2 * n_cols, 4.5),
        squeeze=False,
    )

    for i, (canal, nombre, cmap_c) in enumerate(zip(canales, nombres_canales, cmaps)):
        _subplot_imagen(
            axes[0, i], canal,
            titulo=f'Canal: {nombre}',
            cmap=cmap_c,
            normalizar=normalizar,
            mostrar_stats=mostrar_stats,
            fuente_titulo=est.fuente_ejes,
        )

    if imagen_merge is not None:
        ax_merge = axes[0, -1]
        if imagen_merge.ndim == 2:
            ax_merge.imshow(_normalizar_display(imagen_merge, modo=normalizar),
                            cmap='gray', origin='lower')
        else:
            ax_merge.imshow(
                np.clip(imagen_merge.astype(np.float32) /
                        imagen_merge.max() if imagen_merge.max() > 0 else imagen_merge,
                        0, 1),
                origin='lower',
            )
        ax_merge.set_title('Merge', fontsize=est.fuente_ejes, pad=4)
        ax_merge.set_xticks([]); ax_merge.set_yticks([])

    fig.suptitle(titulo_general, fontsize=est.fuente_titulo,
                 fontweight='bold', y=1.01)
    plt.tight_layout()
    if mostrar:
        plt.show()
    return fig, axes


# ─────────────────────────────────────────────────────────────
# 5. Panel de z-stack
# ─────────────────────────────────────────────────────────────

def panel_zstack(
    zstack: np.ndarray,
    titulo_general: str = 'QC — Proyecciones Z-Stack',
    cmap: str = 'gray',
    normalizar: Literal['minmax', 'percentil', 'log', 'sqrt', 'ninguno'] = 'percentil',
    mostrar_stats: bool = True,
    mostrar_planos_muestra: int = 3,
    estetica: Optional[EsteticaGrafico] = None,
    mostrar: bool = True,
) -> Tuple[Figure, np.ndarray]:
    """
    Panel de proyecciones de un z-stack 3D para control de calidad.

    Genera automáticamente las proyecciones estándar del z-stack y
    opcionalmente muestra planos individuales de muestra.

    Proyecciones calculadas:
        MIP (Maximum Intensity Projection) : max a lo largo del eje Z.
        SUM (Sum Projection)               : suma normalizada.
        MEAN (Mean Projection)             : promedio.
        STD (Std Projection)               : desviación estándar — resalta
                                             zonas de alta variabilidad Z.

    Args:
        zstack: Array 3D de forma (Z, H, W) o (H, W, Z).
               Si la última dimensión es la más pequeña, se infiere como Z.

        titulo_general: Título del panel.

        cmap: Mapa de color para todas las proyecciones.

        normalizar: Modo de normalización de display.

        mostrar_stats: Si True, anota estadísticas en cada proyección.

        mostrar_planos_muestra: Número de planos Z individuales a mostrar
                               en la segunda fila (0 → no se muestra segunda fila).

        estetica: Objeto EsteticaGrafico.

        mostrar: Si True, llama plt.show().

    Returns:
        Tupla (fig, axes).
    """
    est = estetica or _EST_DEFAULT

    # Normalizar orientación del z-stack a (Z, H, W)
    arr = np.asarray(zstack, dtype=np.float64)
    if arr.ndim != 3:
        raise ValueError(f"zstack debe ser 3D. Forma recibida: {arr.shape}")
    if arr.shape[2] < arr.shape[0] and arr.shape[2] < arr.shape[1]:
        arr = np.moveaxis(arr, -1, 0)  # (H, W, Z) → (Z, H, W)

    n_z = arr.shape[0]

    proyecciones = {
        'MIP (máximo)':      arr.max(axis=0),
        'SUM (suma)':        arr.sum(axis=0),
        'MEAN (promedio)':   arr.mean(axis=0),
        'STD (variabilidad)': arr.std(axis=0),
    }

    n_filas = 1 + (1 if mostrar_planos_muestra > 0 else 0)
    n_cols_fila1 = len(proyecciones)
    n_cols_fila2 = min(mostrar_planos_muestra, n_z)
    n_cols       = max(n_cols_fila1, n_cols_fila2)

    fig, axes = plt.subplots(
        n_filas, n_cols,
        figsize=(4.0 * n_cols, 4.2 * n_filas),
        squeeze=False,
    )

    for col, (nombre, proj) in enumerate(proyecciones.items()):
        _subplot_imagen(
            axes[0, col], proj,
            titulo=nombre, cmap=cmap,
            normalizar=normalizar,
            mostrar_stats=mostrar_stats,
            fuente_titulo=est.fuente_ejes,
        )

    # Ocultar subplots vacíos de fila 1
    for col in range(len(proyecciones), n_cols):
        axes[0, col].set_visible(False)

    # Planos de muestra
    if mostrar_planos_muestra > 0 and n_filas > 1:
        indices_z = np.linspace(0, n_z - 1, n_cols_fila2, dtype=int)
        for col, iz in enumerate(indices_z):
            _subplot_imagen(
                axes[1, col], arr[iz],
                titulo=f'Plano Z={iz} / {n_z-1}',
                cmap=cmap, normalizar=normalizar,
                mostrar_stats=False,
                fuente_titulo=est.fuente_ejes - 1,
            )
        for col in range(n_cols_fila2, n_cols):
            axes[1, col].set_visible(False)

    fig.suptitle(
        f'{titulo_general}  ({n_z} planos Z | {arr.shape[1]}×{arr.shape[2]} px)',
        fontsize=est.fuente_titulo, fontweight='bold', y=1.01,
    )
    plt.tight_layout()
    if mostrar:
        plt.show()
    return fig, axes