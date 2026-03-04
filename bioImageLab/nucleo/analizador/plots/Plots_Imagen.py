"""
Gráficos de diagnóstico para datos de imagen y ajuste de modelos físicos.

INPUT  : np.ndarray 2D/3D (imágenes, datos de ajuste, máscaras de segmentación).
OUTPUT : Tupla (fig, ax) o (fig, axes).

A diferencia de Plots_Estadisticos.py (DataFrames) y Plots_Modelos.py
(objetos sklearn), estas funciones operan directamente sobre arrays de imagen.
Son específicas para diagnóstico óptico y análisis de microscopía:

    PSF y óptica:
        ajuste_psf()            Datos vs. modelo PSF ajustado (corte radial + 2D)
        perfil_lineal_imagen()  Perfil de intensidad a lo largo de una línea

    Análisis de superficie:
        ajuste_superficie()     Datos de imagen vs. superficie ajustada en 3D y residuos

    Multi-objeto y segmentación:
        multi_objeto_overlay()  Superposición de máscara de segmentación sobre imagen
        mapa_propiedades()      Mapa 2D coloreado por una propiedad cuantificada por objeto

IMPORTANTE — Separación de responsabilidades:
    Estas funciones NO realizan el ajuste PSF ni de superficie.
    Reciben los resultados de ajuste_psf.py y ajuste_superficie.py ya calculados.
    No realizan segmentación: reciben máscaras ya generadas por la etapa de
    binarización/segmentación del pipeline.
"""

import warnings
from typing import Dict, List, Literal, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib import colors as mcolors

from Estetica import EsteticaGrafico, ESTILOS_PREDEFINIDOS

_EST_DEFAULT = EsteticaGrafico()


# ─────────────────────────────────────────────────────────────
# PSF — Función de Dispersión de Punto
# ─────────────────────────────────────────────────────────────

def ajuste_psf(
    img_datos: np.ndarray,
    img_modelo: np.ndarray,
    centro: Optional[Tuple[int, int]] = None,
    titulo: str = 'Diagnóstico de Ajuste PSF',
    escala_pixeles: float = 1.0,
    unidad_escala: str = 'px',
    escala_color: Literal['lineal', 'log', 'sqrt'] = 'lineal',
    estetica: Optional[EsteticaGrafico] = None,
    mostrar: bool = True,
) -> Tuple[Figure, np.ndarray]:
    """
    Diagnóstico visual completo del ajuste de la Función de Dispersión de Punto.

    Crea un panel de 4 subplots:
        [0,0] Datos crudos           : imagen experimental de la PSF
        [0,1] Modelo ajustado        : imagen reconstruida del modelo
        [1,0] Residuos               : diferencia (datos - modelo) normalizada
        [1,1] Perfil radial           : corte radial comparando datos y modelo

    La PSF (Point Spread Function) describe la respuesta del sistema óptico
    a una fuente puntual. Su ajuste permite extraer parámetros como FWHM
    (resolución lateral), astigmatismo y aberraciones.

    Modelos comunes de PSF:
        Gaussiana 2D:  I(r) = A · exp(-r²/(2σ²)) + B
        Airy pattern:  I(r) = A · [2J₁(πr/r₀)/(πr/r₀)]² + B
        Gaussiana 2D elíptica: σₓ ≠ σᵧ (astigmatismo)

    Args:
        img_datos: Array 2D con la imagen experimental de la PSF (recortada).
        img_modelo: Array 2D con la imagen del modelo ajustado (misma forma).
        centro: (fila, col) del centro de la PSF. None → usa el máximo de img_datos.
        titulo: Título principal del panel.
        escala_pixeles: Factor de conversión píxel → unidades físicas.
        unidad_escala: Unidad de la escala espacial ('px', 'nm', 'μm').
        escala_color: Escala del mapa de color: 'lineal', 'log' o 'sqrt'.
        estetica: Objeto EsteticaGrafico.
        mostrar: Si True, llama plt.show().

    Returns:
        Tupla (fig, axes) donde axes es un array (2, 2) de Axes.
    """
    est = estetica or _EST_DEFAULT
    est.aplicar()

    if img_datos.shape != img_modelo.shape:
        raise ValueError(
            f"img_datos {img_datos.shape} e img_modelo {img_modelo.shape} "
            "deben tener la misma forma."
        )

    if centro is None:
        idx_max = np.unravel_index(np.argmax(img_datos), img_datos.shape)
        centro  = (int(idx_max[0]), int(idx_max[1]))

    H, W   = img_datos.shape
    residuos = img_datos - img_modelo

    # Normalización de residuos por sigma de fondo
    sigma_fondo = np.std(img_datos[:5, :5])
    if sigma_fondo > 0:
        residuos_norm = residuos / sigma_fondo
    else:
        residuos_norm = residuos

    # Transformación de escala de color
    def _transform(arr: np.ndarray) -> np.ndarray:
        if escala_color == 'log':
            return np.log1p(np.maximum(arr, 0))
        elif escala_color == 'sqrt':
            return np.sqrt(np.maximum(arr, 0))
        return arr

    # Coordenadas físicas
    escala_y = np.arange(H) * escala_pixeles
    escala_x = np.arange(W) * escala_pixeles

    fig, axes = plt.subplots(2, 2, figsize=(est.figsize[0] * 1.4, est.figsize[1] * 1.3))
    cmap = est.paleta_continua

    # ── [0,0] Datos ──────────────────────────────────────────
    im0 = axes[0, 0].imshow(
        _transform(img_datos),
        cmap=cmap, origin='lower',
        extent=[0, W * escala_pixeles, 0, H * escala_pixeles],
    )
    axes[0, 0].plot(centro[1] * escala_pixeles, centro[0] * escala_pixeles,
                    '+', color='red', markersize=12, markeredgewidth=1.5)
    plt.colorbar(im0, ax=axes[0, 0], shrink=0.85, label='Intensidad')
    axes[0, 0].set_title('Datos (experimento)', fontsize=est.fuente_ejes,
                          fontfamily=est.fuente_familia)
    axes[0, 0].set_xlabel(f'X [{unidad_escala}]')
    axes[0, 0].set_ylabel(f'Y [{unidad_escala}]')

    # ── [0,1] Modelo ─────────────────────────────────────────
    im1 = axes[0, 1].imshow(
        _transform(img_modelo),
        cmap=cmap, origin='lower',
        extent=[0, W * escala_pixeles, 0, H * escala_pixeles],
        vmin=im0.get_clim()[0], vmax=im0.get_clim()[1],
    )
    plt.colorbar(im1, ax=axes[0, 1], shrink=0.85, label='Intensidad')
    axes[0, 1].set_title('Modelo ajustado', fontsize=est.fuente_ejes,
                          fontfamily=est.fuente_familia)
    axes[0, 1].set_xlabel(f'X [{unidad_escala}]')
    axes[0, 1].set_ylabel(f'Y [{unidad_escala}]')

    # ── [1,0] Residuos ────────────────────────────────────────
    lim_res = max(abs(residuos_norm.min()), abs(residuos_norm.max()))
    im2 = axes[1, 0].imshow(
        residuos_norm,
        cmap=est.paleta_divergente, origin='lower',
        extent=[0, W * escala_pixeles, 0, H * escala_pixeles],
        vmin=-lim_res, vmax=lim_res,
    )
    plt.colorbar(im2, ax=axes[1, 0], shrink=0.85, label='Residuo / σ_fondo')
    axes[1, 0].set_title('Residuos normalizados', fontsize=est.fuente_ejes,
                          fontfamily=est.fuente_familia)
    axes[1, 0].set_xlabel(f'X [{unidad_escala}]')
    axes[1, 0].set_ylabel(f'Y [{unidad_escala}]')

    # ── [1,1] Perfil radial ───────────────────────────────────
    ax_perf = axes[1, 1]
    y_c, x_c = centro

    # Extraer perfiles horizontal y vertical
    perfil_datos_h  = img_datos[y_c, :]
    perfil_modelo_h = img_modelo[y_c, :]
    perfil_datos_v  = img_datos[:, x_c]
    perfil_modelo_v = img_modelo[:, x_c]

    ax_perf.plot(escala_x, perfil_datos_h,  'o', markersize=4, alpha=0.7,
                 color='steelblue', label='Datos (H)')
    ax_perf.plot(escala_x, perfil_modelo_h, '-', linewidth=est.grosor_linea,
                 color='steelblue', label='Modelo (H)')
    ax_perf.plot(escala_y, perfil_datos_v,  's', markersize=4, alpha=0.7,
                 color='tomato', label='Datos (V)')
    ax_perf.plot(escala_y, perfil_modelo_v, '--', linewidth=est.grosor_linea,
                 color='tomato', label='Modelo (V)')

    ax_perf.set_title('Perfil a través del centro', fontsize=est.fuente_ejes,
                       fontfamily=est.fuente_familia)
    ax_perf.set_xlabel(f'Posición [{unidad_escala}]')
    ax_perf.set_ylabel('Intensidad')
    ax_perf.legend(fontsize=est.fuente_leyenda, frameon=est.leyenda_marco)
    ax_perf.grid(True, linestyle=est.grid_linestyle, alpha=est.grid_alpha)
    ax_perf.spines['top'].set_visible(False)
    ax_perf.spines['right'].set_visible(False)

    fig.suptitle(titulo, fontsize=est.fuente_titulo + 1,
                 fontweight='bold' if est.negrita_titulo else 'normal',
                 fontfamily=est.fuente_familia)
    plt.tight_layout()
    if mostrar:
        plt.show()
    return fig, axes


def perfil_lineal_imagen(
    img: np.ndarray,
    punto_inicio: Tuple[int, int],
    punto_fin: Tuple[int, int],
    num_puntos: int = 100,
    escala_pixeles: float = 1.0,
    unidad_escala: str = 'px',
    titulo: str = 'Perfil de Intensidad Lineal',
    mostrar_linea_en_imagen: bool = True,
    estetica: Optional[EsteticaGrafico] = None,
    mostrar: bool = True,
) -> Tuple[Figure, np.ndarray]:
    """
    Perfil de intensidad a lo largo de una línea en la imagen.

    Crea un panel de dos subplots:
        Izquierda: imagen con la línea de perfil marcada.
        Derecha:   gráfico del perfil de intensidad vs. distancia.

    Usa interpolación bilineal (scipy.ndimage.map_coordinates) para muestrear
    la intensidad en posiciones sub-pixel a lo largo de la línea.

    Args:
        img: Array 2D de intensidades.
        punto_inicio: (col, fila) del inicio de la línea.
        punto_fin:    (col, fila) del final de la línea.
        num_puntos:   Puntos de muestreo a lo largo de la línea.
        escala_pixeles: Factor de conversión píxel → unidades físicas.
        unidad_escala:  Unidad de la escala espacial.
        titulo: Título del panel.
        mostrar_linea_en_imagen: Si True, dibuja la línea sobre la imagen.
        estetica: Objeto EsteticaGrafico.
        mostrar: Si True, llama plt.show().

    Returns:
        Tupla (fig, axes) — array (1, 2) de Axes.
    """
    from scipy.ndimage import map_coordinates

    est = estetica or _EST_DEFAULT
    est.aplicar()

    x0, y0 = punto_inicio
    x1, y1 = punto_fin
    cols    = np.linspace(x0, x1, num_puntos)
    filas   = np.linspace(y0, y1, num_puntos)
    longitud = np.sqrt((x1 - x0)**2 + (y1 - y0)**2) * escala_pixeles
    distancias = np.linspace(0, longitud, num_puntos)

    intensidades = map_coordinates(
        img.astype(np.float64), [filas, cols], order=1, mode='nearest'
    )

    fig, axes = plt.subplots(1, 2, figsize=(est.figsize[0] * 1.5, est.figsize[1]))

    # Imagen con línea
    axes[0].imshow(img, cmap=est.paleta_continua, origin='lower')
    if mostrar_linea_en_imagen:
        axes[0].plot([x0, x1], [y0, y1], '-',
                     color='red', linewidth=est.grosor_linea + 0.5)
        axes[0].plot([x0, x1], [y0, y1], 'o',
                     color='yellow', markersize=6, markeredgecolor='red')
    axes[0].set_title('Imagen con línea de perfil',
                       fontsize=est.fuente_ejes, fontfamily=est.fuente_familia)
    axes[0].set_xlabel(f'X [px]')
    axes[0].set_ylabel(f'Y [px]')

    # Perfil
    paleta = __import__('seaborn').color_palette(est.paleta)
    axes[1].plot(distancias, intensidades, '-o',
                 color=paleta[0], linewidth=est.grosor_linea,
                 markersize=3, alpha=est.alpha_puntos)
    axes[1].fill_between(distancias, intensidades,
                          alpha=est.alpha_relleno, color=paleta[0])
    axes[1].set_title('Perfil de intensidad',
                       fontsize=est.fuente_ejes, fontfamily=est.fuente_familia)
    axes[1].set_xlabel(f'Distancia [{unidad_escala}]')
    axes[1].set_ylabel('Intensidad')
    axes[1].grid(True, linestyle=est.grid_linestyle, alpha=est.grid_alpha)
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)

    fig.suptitle(titulo, fontsize=est.fuente_titulo,
                 fontweight='bold' if est.negrita_titulo else 'normal',
                 fontfamily=est.fuente_familia)
    plt.tight_layout()
    if mostrar:
        plt.show()
    return fig, axes


# ─────────────────────────────────────────────────────────────
# Ajuste de superficie
# ─────────────────────────────────────────────────────────────

def ajuste_superficie(
    img_datos: np.ndarray,
    img_modelo: np.ndarray,
    escala_pixeles: float = 1.0,
    unidad_escala: str = 'px',
    titulo: str = 'Diagnóstico de Ajuste de Superficie',
    mostrar_3d: bool = True,
    paso_malla: int = 3,
    estetica: Optional[EsteticaGrafico] = None,
    mostrar: bool = True,
) -> Tuple[Figure, np.ndarray]:
    """
    Diagnóstico visual del ajuste de una superficie 2D a datos de imagen.

    Útil para:
        - Corrección de fondo (shading correction): ajuste de la iluminación
          no uniforme como superficie polinómica o gaussiana.
        - Ajuste de z-stack: verificar la calidad del ajuste focal.
        - Ajuste de superficie de activación: mapas de concentración.

    Crea un panel de 3 o 4 subplots:
        Imagen: datos originales (heatmap 2D)
        Modelo: superficie ajustada (heatmap 2D)
        Residuos: diferencia normalizada (heatmap 2D)
        3D (opcional): superficie en perspectiva 3D si mostrar_3d=True

    Args:
        img_datos: Array 2D con los datos de imagen originales.
        img_modelo: Array 2D con la superficie ajustada (misma forma).
        escala_pixeles: Factor de conversión píxel → unidades físicas.
        unidad_escala: Unidad de escala.
        titulo: Título principal del panel.
        mostrar_3d: Si True, añade un subplot de superficie en perspectiva 3D.
        paso_malla: Submuestreo de la malla 3D (1 = full, 3 = cada 3 píxeles).
                   Aumentar para imágenes grandes (reduce tiempo de render).
        estetica: Objeto EsteticaGrafico.
        mostrar: Si True, llama plt.show().

    Returns:
        Tupla (fig, axes) — array de Axes.
    """
    est = estetica or _EST_DEFAULT
    est.aplicar()

    if img_datos.shape != img_modelo.shape:
        raise ValueError("img_datos e img_modelo deben tener la misma forma.")

    H, W      = img_datos.shape
    residuos  = img_datos.astype(np.float64) - img_modelo.astype(np.float64)
    sigma_res = np.std(residuos)
    res_norm  = residuos / sigma_res if sigma_res > 0 else residuos

    n_cols = 4 if mostrar_3d else 3
    fig    = plt.figure(figsize=(est.figsize[0] * n_cols / 2.5,
                                  est.figsize[1] * 1.1))

    escala_y = np.arange(H) * escala_pixeles
    escala_x = np.arange(W) * escala_pixeles
    extent   = [0, W * escala_pixeles, 0, H * escala_pixeles]

    # Colorbar compartido para datos y modelo
    vmin = min(img_datos.min(), img_modelo.min())
    vmax = max(img_datos.max(), img_modelo.max())

    ax0 = fig.add_subplot(1, n_cols, 1)
    ax1 = fig.add_subplot(1, n_cols, 2)
    ax2 = fig.add_subplot(1, n_cols, 3)

    for ax, data, titulo_sub in [
        (ax0, img_datos,  'Datos'),
        (ax1, img_modelo, 'Modelo'),
    ]:
        im = ax.imshow(data, cmap=est.paleta_continua, origin='lower',
                       extent=extent, vmin=vmin, vmax=vmax)
        plt.colorbar(im, ax=ax, shrink=0.8)
        ax.set_title(titulo_sub, fontsize=est.fuente_ejes,
                     fontfamily=est.fuente_familia)
        ax.set_xlabel(f'X [{unidad_escala}]')
        ax.set_ylabel(f'Y [{unidad_escala}]')

    lim_r = max(abs(res_norm.min()), abs(res_norm.max()))
    im2   = ax2.imshow(res_norm, cmap=est.paleta_divergente, origin='lower',
                        extent=extent, vmin=-lim_r, vmax=lim_r)
    plt.colorbar(im2, ax=ax2, shrink=0.8, label='Residuo / σ')
    ax2.set_title('Residuos normalizados', fontsize=est.fuente_ejes,
                   fontfamily=est.fuente_familia)
    ax2.set_xlabel(f'X [{unidad_escala}]')
    ax2.set_ylabel(f'Y [{unidad_escala}]')

    # Subplot 3D opcional
    if mostrar_3d:
        ax3 = fig.add_subplot(1, n_cols, 4, projection='3d')
        s   = paso_malla
        XX, YY = np.meshgrid(escala_x[::s], escala_y[::s])
        ax3.plot_surface(XX, YY, img_datos[::s, ::s],
                         cmap=est.paleta_continua, alpha=0.6, linewidth=0)
        ax3.plot_surface(XX, YY, img_modelo[::s, ::s],
                         cmap='Reds', alpha=0.4, linewidth=0)
        ax3.set_xlabel(f'X [{unidad_escala}]', fontsize=est.fuente_ticks)
        ax3.set_ylabel(f'Y [{unidad_escala}]', fontsize=est.fuente_ticks)
        ax3.set_zlabel('Intensidad', fontsize=est.fuente_ticks)
        ax3.set_title('Vista 3D (azul=datos, rojo=modelo)',
                       fontsize=est.fuente_ejes, fontfamily=est.fuente_familia)

    fig.suptitle(titulo, fontsize=est.fuente_titulo,
                 fontweight='bold' if est.negrita_titulo else 'normal',
                 fontfamily=est.fuente_familia)
    plt.tight_layout()
    if mostrar:
        plt.show()
    return fig, np.array([ax0, ax1, ax2])


# ─────────────────────────────────────────────────────────────
# Multi-objeto y segmentación
# ─────────────────────────────────────────────────────────────

def multi_objeto_overlay(
    img: np.ndarray,
    mascara: np.ndarray,
    etiquetas_objeto: Optional[np.ndarray] = None,
    mostrar_contorno: bool = True,
    mostrar_relleno: bool = True,
    mostrar_ids: bool = False,
    escala_color_mascara: str = 'hsv',
    alpha_relleno: Optional[float] = None,
    escala_imagen: Literal['lineal', 'log', 'sqrt', 'percentil'] = 'percentil',
    percentil_max: float = 99.5,
    titulo: str = 'Segmentación Multi-objeto',
    estetica: Optional[EsteticaGrafico] = None,
    mostrar: bool = True,
) -> Tuple[Figure, Axes]:
    """
    Superposición de la máscara de segmentación sobre la imagen de fluorescencia.

    Muestra simultáneamente:
        - La imagen de intensidades (escala de grises o falso color).
        - Los contornos de los objetos segmentados.
        - El relleno semitransparente de cada objeto (coloreado por ID o clase).
        - Los IDs numéricos en el centroide de cada objeto (opcional).

    Acepta dos tipos de máscara:
        - Binaria   (0/255 o 0/1): todos los objetos en el mismo color.
        - Etiquetada (0=fondo, 1=obj_1, 2=obj_2, ...): cada objeto tiene su color.

    Args:
        img: Array 2D de intensidades (imagen de fluorescencia o campo claro).
        mascara: Array 2D. Binaria (0/no-0) o etiquetada (0, 1, 2, ...).
        etiquetas_objeto: Si se provee, colorea el relleno según esta clasificación
                         (p.ej. cluster asignado). Debe tener N_objetos valores.
        mostrar_contorno: Si True, dibuja el contorno de cada objeto.
        mostrar_relleno: Si True, rellena cada objeto con color semitransparente.
        mostrar_ids: Si True, anota el ID numérico en el centroide de cada objeto.
        escala_color_mascara: Paleta para colorear objetos. 'hsv' produce colores
                             muy distintos entre objetos adyacentes.
        alpha_relleno: Transparencia del relleno. None → usa estetica.alpha_relleno.
        escala_imagen: Transformación de la imagen de fondo:
                       'percentil': clip en percentil_max (recomendado para fluorescencia)
                       'log':       log(1+I), útil para rango dinámico alto
                       'sqrt':      sqrt(I)
                       'lineal':    sin transformación
        percentil_max: Percentil para el clip de intensidad (solo si escala='percentil').
        titulo: Título del gráfico.
        estetica: Objeto EsteticaGrafico.
        mostrar: Si True, llama plt.show().

    Returns:
        Tupla (fig, ax)
    """
    from scipy import ndimage as ndi

    est  = estetica or _EST_DEFAULT
    alpha = alpha_relleno if alpha_relleno is not None else est.alpha_relleno

    # Transformación de la imagen de fondo
    img_float = img.astype(np.float64)
    if escala_imagen == 'log':
        img_display = np.log1p(np.maximum(img_float, 0))
    elif escala_imagen == 'sqrt':
        img_display = np.sqrt(np.maximum(img_float, 0))
    elif escala_imagen == 'percentil':
        vmax = np.percentile(img_float, percentil_max)
        img_display = np.clip(img_float, 0, vmax)
    else:
        img_display = img_float

    # Normalizar a [0, 1] para mostrar
    rango = img_display.max() - img_display.min()
    img_norm = (img_display - img_display.min()) / rango if rango > 0 else img_display * 0

    fig, ax = plt.subplots(figsize=est.figsize)
    ax.imshow(img_norm, cmap='gray', origin='lower', vmin=0, vmax=1)

    # Determinar IDs de objetos
    mascara_arr = np.asarray(mascara)
    if mascara_arr.max() <= 1:
        # Máscara binaria → etiquetar con scipy
        mascara_etiq, n_obj = ndi.label(mascara_arr > 0)
    else:
        mascara_etiq = mascara_arr.astype(int)
        n_obj        = mascara_etiq.max()

    if n_obj == 0:
        warnings.warn("La máscara no contiene objetos.", UserWarning, stacklevel=2)
        est.aplicar_a_ax(ax, titulo=titulo)
        if mostrar:
            plt.show()
        return fig, ax

    # Paleta de colores para objetos
    import matplotlib.cm as cm
    cmap_obj = cm.get_cmap(escala_color_mascara, n_obj + 1)

    # Construir imagen RGBA del overlay
    overlay = np.zeros((*img_norm.shape, 4), dtype=np.float32)

    for obj_id in range(1, n_obj + 1):
        mask_obj = mascara_etiq == obj_id
        if not np.any(mask_obj):
            continue

        # Color según etiqueta externa o por ID
        if etiquetas_objeto is not None and obj_id - 1 < len(etiquetas_objeto):
            n_clases = len(np.unique(etiquetas_objeto))
            import seaborn as sns
            paleta_cls = sns.color_palette(est.paleta, n_clases)
            cls = etiquetas_objeto[obj_id - 1]
            cls_idx = list(np.unique(etiquetas_objeto)).index(cls)
            color = (*paleta_cls[cls_idx % len(paleta_cls)], alpha)
        else:
            color = cmap_obj(obj_id / (n_obj + 1))
            color = (color[0], color[1], color[2], alpha)

        if mostrar_relleno:
            overlay[mask_obj] = color

        if mostrar_contorno:
            # Contorno: borde exterior del objeto
            erosionado = ndi.binary_erosion(mask_obj)
            contorno   = mask_obj & ~erosionado
            overlay[contorno] = (color[0], color[1], color[2], 1.0)

        if mostrar_ids:
            centroide = ndi.center_of_mass(mask_obj)
            ax.text(centroide[1], centroide[0], str(obj_id),
                    ha='center', va='center',
                    fontsize=est.fuente_anotaciones, color='white',
                    fontweight='bold')

    ax.imshow(overlay, origin='lower')
    est.aplicar_a_ax(ax, titulo=f'{titulo} — {n_obj} objetos')
    ax.set_xlabel('X [px]')
    ax.set_ylabel('Y [px]')
    ax.grid(False)
    plt.tight_layout()
    if mostrar:
        plt.show()
    return fig, ax


def mapa_propiedades(
    img: np.ndarray,
    mascara_etiquetada: np.ndarray,
    valores_por_objeto: np.ndarray,
    nombre_propiedad: str = 'Intensidad media',
    cmap: Optional[str] = None,
    percentil_clip: Tuple[float, float] = (2.0, 98.0),
    titulo: str = 'Mapa de Propiedades por Objeto',
    estetica: Optional[EsteticaGrafico] = None,
    mostrar: bool = True,
) -> Tuple[Figure, np.ndarray]:
    """
    Mapa 2D donde cada objeto segmentado está coloreado por una propiedad cuantificada.

    Crea un panel de dos subplots:
        Izquierda: imagen de fluorescencia en escala de grises.
        Derecha:   misma imagen con cada objeto coloreado según su valor de propiedad
                   (p.ej. intensidad media, área, excentricidad, cluster).

    Útil para visualizar la distribución espacial de una propiedad a lo largo
    del campo de visión: permite detectar gradientes, patrones y heterogeneidad
    espacial que no serían visibles en gráficos estadísticos.

    Args:
        img: Array 2D de intensidades (imagen de referencia).
        mascara_etiquetada: Array 2D etiquetado (0=fondo, 1..N=objetos).
        valores_por_objeto: Array 1D con un valor de propiedad por objeto.
                           El valor i corresponde al objeto con ID i+1.
        nombre_propiedad: Nombre de la propiedad para la barra de color.
        cmap: Paleta de color. None → usa estetica.paleta_continua.
        percentil_clip: (p_low, p_high) para recortar la escala de color.
                       Evita que outliers extremos dominen la escala.
        titulo: Título del panel.
        estetica: Objeto EsteticaGrafico.
        mostrar: Si True, llama plt.show().

    Returns:
        Tupla (fig, axes) — array (1, 2) de Axes.
    """
    est   = estetica or _EST_DEFAULT
    est.aplicar()
    cmap_ = cmap or est.paleta_continua

    H, W  = img.shape
    n_obj = int(mascara_etiquetada.max())

    if len(valores_por_objeto) < n_obj:
        raise ValueError(
            f"valores_por_objeto tiene {len(valores_por_objeto)} elementos "
            f"pero la máscara contiene {n_obj} objetos."
        )

    # Construir imagen de propiedad
    p_low  = np.percentile(valores_por_objeto, percentil_clip[0])
    p_high = np.percentile(valores_por_objeto, percentil_clip[1])

    img_prop = np.full((H, W), np.nan)
    for obj_id in range(1, n_obj + 1):
        mask = mascara_etiquetada == obj_id
        if np.any(mask):
            img_prop[mask] = valores_por_objeto[obj_id - 1]

    # Imagen de fondo normalizada
    img_float = img.astype(np.float64)
    vmax_bg   = np.percentile(img_float, 99.5)
    img_norm  = np.clip(img_float, 0, vmax_bg) / vmax_bg

    fig, axes = plt.subplots(1, 2, figsize=(est.figsize[0] * 2, est.figsize[1]))

    axes[0].imshow(img_norm, cmap='gray', origin='lower')
    axes[0].set_title('Imagen original', fontsize=est.fuente_ejes,
                       fontfamily=est.fuente_familia)
    axes[0].set_xlabel('X [px]')
    axes[0].set_ylabel('Y [px]')

    axes[1].imshow(img_norm, cmap='gray', origin='lower', alpha=0.4)
    im_prop = axes[1].imshow(
        img_prop, cmap=cmap_, origin='lower', alpha=0.85,
        vmin=p_low, vmax=p_high,
    )
    cbar = plt.colorbar(im_prop, ax=axes[1], shrink=0.85)
    cbar.set_label(nombre_propiedad, fontsize=est.fuente_ejes)
    axes[1].set_title(f'Mapa: {nombre_propiedad}', fontsize=est.fuente_ejes,
                       fontfamily=est.fuente_familia)
    axes[1].set_xlabel('X [px]')
    axes[1].set_ylabel('Y [px]')

    fig.suptitle(titulo, fontsize=est.fuente_titulo,
                 fontweight='bold' if est.negrita_titulo else 'normal',
                 fontfamily=est.fuente_familia)
    plt.tight_layout()
    if mostrar:
        plt.show()
    return fig, axes