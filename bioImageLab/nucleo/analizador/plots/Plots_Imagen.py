# === analizador/plots/Plots_Imagen.py ===
"""
Plots específicos para visualización de datos de imagen (BioImagenData).
Todos reciben BioImagenData y devuelven matplotlib.figure.Figure.
"""
from __future__ import annotations

from typing import Optional, Tuple, List, Any, Dict
from pathlib import Path

import numpy as np

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


def _guardar_o_mostrar(fig, estetica: Estetica, ruta: Optional[Path] = None):
    if estetica.layout.tight_layout:
        fig.tight_layout(pad=estetica.layout.padding)
    if ruta:
        fig.savefig(ruta, dpi=estetica.layout.dpi, bbox_inches="tight")
    return fig


def _normalizar_slice(slice_2d: np.ndarray) -> np.ndarray:
    """Normaliza a [0, 1] para visualización."""
    s = slice_2d.astype(np.float64)
    vmin, vmax = s.min(), s.max()
    if vmax > vmin:
        return (s - vmin) / (vmax - vmin)
    return np.zeros_like(s)


def _mascara_a_rgba(mask: np.ndarray, estetica: Estetica, alpha: float = 0.35) -> np.ndarray:
    """Convierte máscara de etiquetas a RGBA."""
    etiquetas = np.unique(mask)
    etiquetas = etiquetas[etiquetas > 0]
    rgba = np.zeros((*mask.shape, 4), dtype=np.float32)
    for i, etiqueta in enumerate(etiquetas):
        color = estetica.paleta.color_etiqueta(i)
        # Convertir hex a RGB
        r = int(color[1:3], 16) / 255.0
        g = int(color[3:5], 16) / 255.0
        b = int(color[5:7], 16) / 255.0
        mascara_bool = mask == etiqueta
        rgba[:, :, 0][mascara_bool] = r
        rgba[:, :, 1][mascara_bool] = g
        rgba[:, :, 2][mascara_bool] = b
        rgba[:, :, 3][mascara_bool] = alpha
    return rgba


# =========================================================
# CLASES PLOT IMAGEN
# =========================================================

class MIPProyeccion:
    """Maximum Intensity Projection sobre un eje."""
    nombre = "mip_proyeccion"

    def __init__(self, eje: str = "Z", canal: int = 0, t: int = 0,
                 cmap: str = "gray", titulo: Optional[str] = None):
        self.eje = eje.upper()
        self.canal = canal
        self.t = t
        self.cmap = cmap
        self.titulo = titulo

    def __call__(self, data, estetica: Optional[Estetica] = None) -> Any:
        import matplotlib.pyplot as plt
        est = estetica or Estetica()
        fig, ax = _setup_fig(est, "default")

        # data es BioImagenData (import lazy para evitar circular)
        arr = data.datos[self.t, :, self.canal, :, :] if self.eje == "Z" else               data.datos[:, :, self.canal, :, :].max(axis=0) if self.eje == "T" else               data.datos[self.t, :, self.canal, :, :].max(axis=2) if self.eje == "Y" else               data.datos[self.t, :, self.canal, :, :].max(axis=3)

        if self.eje == "Z":
            mip = arr.max(axis=0)
        elif self.eje == "T":
            mip = arr.max(axis=0)
        elif self.eje == "Y":
            mip = arr.max(axis=1)
        elif self.eje == "X":
            mip = arr.max(axis=2)
        else:
            raise ValueError(f"Eje '{self.eje}' no válido. Use Z, T, Y o X.")

        im = ax.imshow(_normalizar_slice(mip), cmap=self.cmap or est.cmap("continuo"))
        _titulo(ax, self.titulo or f"MIP Eje {self.eje}", est)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        return _guardar_o_mostrar(fig, est)


class OrthoView:
    """Vista ortogonal (XY, XZ, YZ) de un punto de referencia."""
    nombre = "ortho_view"

    def __init__(self, z_ref: Optional[int] = None, t_ref: int = 0,
                 canal: int = 0, punto_ref: Optional[Tuple[int, int]] = None,
                 cmap: str = "gray"):
        self.z_ref = z_ref
        self.t_ref = t_ref
        self.canal = canal
        self.punto_ref = punto_ref
        self.cmap = cmap

    def __call__(self, data, estetica: Optional[Estetica] = None) -> Any:
        import matplotlib.pyplot as plt
        est = estetica or Estetica()
        fig, axes = _setup_fig_multi(1, 3, est, "ancho")
        ax_xy, ax_xz, ax_yz = axes[0], axes[1], axes[2]

        dims = data.dims
        z = self.z_ref if self.z_ref is not None else dims.Z // 2
        y = self.punto_ref[1] if self.punto_ref else dims.Y // 2
        x = self.punto_ref[0] if self.punto_ref else dims.X // 2

        # XY
        slice_xy = data.datos[self.t_ref, z, self.canal, :, :]
        ax_xy.imshow(_normalizar_slice(slice_xy), cmap=self.cmap)
        ax_xy.axhline(y, color=est.paleta.primario, lw=1)
        ax_xy.axvline(x, color=est.paleta.primario, lw=1)
        _titulo(ax_xy, f"XY (z={z})", est)
        ax_xy.axis("off")

        # XZ
        slice_xz = data.datos[self.t_ref, :, self.canal, y, :]
        ax_xz.imshow(_normalizar_slice(slice_xz), cmap=self.cmap, aspect="auto")
        ax_xz.axhline(z, color=est.paleta.primario, lw=1)
        ax_xz.axvline(x, color=est.paleta.primario, lw=1)
        _titulo(ax_xz, f"XZ (y={y})", est)
        ax_xz.axis("off")

        # YZ
        slice_yz = data.datos[self.t_ref, :, self.canal, :, x]
        ax_yz.imshow(_normalizar_slice(slice_yz), cmap=self.cmap, aspect="auto")
        ax_yz.axhline(z, color=est.paleta.primario, lw=1)
        ax_yz.axvline(y, color=est.paleta.primario, lw=1)
        _titulo(ax_yz, f"YZ (x={x})", est)
        ax_yz.axis("off")

        fig.suptitle("Vista Ortogonal", fontsize=est.fuentes.tamano_titulo + 2,
                     fontweight=est.fuentes.peso_titulo, y=1.02)
        return _guardar_o_mostrar(fig, est)


class StackViewer:
    """Grid de slices Z para un timepoint y canal."""
    nombre = "stack_viewer"

    def __init__(self, canal: int = 0, t: int = 0, ncols: int = 4,
                 z_range: Optional[Tuple[int, int]] = None,
                 cmap: str = "gray"):
        self.canal = canal
        self.t = t
        self.ncols = ncols
        self.z_range = z_range
        self.cmap = cmap

    def __call__(self, data, estetica: Optional[Estetica] = None) -> Any:
        import matplotlib.pyplot as plt
        est = estetica or Estetica()

        dims = data.dims
        z_start, z_end = self.z_range if self.z_range else (0, dims.Z)
        z_indices = list(range(z_start, min(z_end, dims.Z)))
        n = len(z_indices)
        nrows = (n + self.ncols - 1) // self.ncols

        fig, axes = plt.subplots(nrows, self.ncols,
                                 figsize=(3 * self.ncols, 3 * nrows),
                                 dpi=est.layout.dpi)
        est.aplicar_tema()
        if nrows == 1:
            axes = np.array([axes]) if self.ncols == 1 else axes
        axes = np.array(axes).flatten()

        for i, z in enumerate(z_indices):
            ax = axes[i]
            slice_2d = data.datos[self.t, z, self.canal, :, :]
            ax.imshow(_normalizar_slice(slice_2d), cmap=self.cmap)
            vmin, vmax = slice_2d.min(), slice_2d.max()
            ax.set_title(f"Z={z} [{vmin:.0f},{vmax:.0f}]",
                         fontsize=est.fuentes.tamano_tick)
            ax.axis("off")

        for j in range(i + 1, len(axes)):
            axes[j].axis("off")

        fig.suptitle(f"Stack Z — Canal {self.canal}, T={self.t}",
                     fontsize=est.fuentes.tamano_titulo,
                     fontweight=est.fuentes.peso_titulo)
        return _guardar_o_mostrar(fig, est)


class OverlayMascara:
    """Overlay de máscara sobre imagen base con colores por etiqueta."""
    nombre = "overlay_mascara"

    def __init__(self, mascara_datos: Optional[np.ndarray] = None,
                 canal: int = 0, t: int = 0, z: int = 0,
                 alpha: float = 0.35, color_mascara: Optional[str] = None,
                 titulo: Optional[str] = None):
        self.mascara_datos = mascara_datos
        self.canal = canal
        self.t = t
        self.z = z
        self.alpha = alpha
        self.color_mascara = color_mascara
        self.titulo = titulo

    def __call__(self, data, estetica: Optional[Estetica] = None) -> Any:
        import matplotlib.pyplot as plt
        est = estetica or Estetica()
        fig, ax = _setup_fig(est, "default")

        slice_2d = data.datos[self.t, self.z, self.canal, :, :]
        ax.imshow(_normalizar_slice(slice_2d), cmap="gray")

        # Buscar máscara en metadata si no se pasó explícitamente
        mascara = self.mascara_datos
        if mascara is None:
            mascara = data.metadata.get("mascara_datos")

        if mascara is not None:
            if mascara.ndim == 5:
                mask_2d = mascara[self.t, self.z, self.canal, :, :]
            elif mascara.ndim == 4:
                mask_2d = mascara[self.t, self.z, :, :]
            elif mascara.ndim == 2:
                mask_2d = mascara
            else:
                raise ValueError(f"Dimensión de máscara no soportada: {mascara.ndim}")

            if mask_2d.max() > 0:
                if self.color_mascara:
                    # Máscara binaria con color único
                    rgba = np.zeros((*mask_2d.shape, 4), dtype=np.float32)
                    r = int(self.color_mascara[1:3], 16) / 255.0
                    g = int(self.color_mascara[3:5], 16) / 255.0
                    b = int(self.color_mascara[5:7], 16) / 255.0
                    rgba[:, :, 0][mask_2d > 0] = r
                    rgba[:, :, 1][mask_2d > 0] = g
                    rgba[:, :, 2][mask_2d > 0] = b
                    rgba[:, :, 3][mask_2d > 0] = self.alpha
                    ax.imshow(rgba, interpolation="nearest")
                else:
                    # Máscara de etiquetas multicolor
                    rgba = _mascara_a_rgba(mask_2d, est, self.alpha)
                    ax.imshow(rgba, interpolation="nearest")

        _titulo(ax, self.titulo or "Overlay Máscara", est)
        ax.axis("off")
        return _guardar_o_mostrar(fig, est)


class PerfilIntensidad:
    """Perfil de intensidad a lo largo de una línea."""
    nombre = "perfil_intensidad"

    def __init__(self, punto_inicio: Tuple[int, int] = (0, 0),
                 punto_fin: Optional[Tuple[int, int]] = None,
                 canal: int = 0, t: int = 0, z: int = 0,
                 n_puntos: Optional[int] = None):
        self.punto_inicio = punto_inicio
        self.punto_fin = punto_fin
        self.canal = canal
        self.t = t
        self.z = z
        self.n_puntos = n_puntos

    def __call__(self, data, estetica: Optional[Estetica] = None) -> Any:
        from scipy.ndimage import map_coordinates
        import matplotlib.pyplot as plt
        est = estetica or Estetica()
        fig, axes = _setup_fig_multi(1, 2, est, "ancho")
        ax_img, ax_plot = axes[0], axes[1]

        slice_2d = data.datos[self.t, self.z, self.canal, :, :]
        dims = slice_2d.shape

        p0 = self.punto_inicio
        p1 = self.punto_fin if self.punto_fin else (dims[1] - 1, dims[0] - 1)
        n = self.n_puntos or max(dims)

        # Perfil
        x = np.linspace(p0[0], p1[0], n)
        y = np.linspace(p0[1], p1[1], n)
        perfil = map_coordinates(slice_2d, [y, x], order=1)

        # Imagen con línea
        ax_img.imshow(_normalizar_slice(slice_2d), cmap="gray")
        ax_img.plot([p0[0], p1[0]], [p0[1], p1[1]],
                    color=est.paleta.primario, lw=2)
        ax_img.scatter([p0[0], p1[0]], [p0[1], p1[1]],
                       color=est.paleta.cuaternario, s=50, zorder=5)
        _titulo(ax_img, f"Slice Z={self.z}", est)
        ax_img.axis("off")

        # Plot perfil
        distancia = np.sqrt((p1[0] - p0[0])**2 + (p1[1] - p0[1])**2)
        x_plot = np.linspace(0, distancia, n)
        ax_plot.plot(x_plot, perfil, color=est.paleta.primario,
                     lw=est.linea.ancho, marker=est.linea.marcador,
                     markersize=est.linea.tamano_marcador)
        ax_plot.fill_between(x_plot, perfil, alpha=0.2, color=est.paleta.primario)
        _titulo(ax_plot, "Perfil de Intensidad", est)
        ax_plot.set_xlabel("Distancia (px)", fontsize=est.fuentes.tamano_etiqueta)
        ax_plot.set_ylabel("Intensidad", fontsize=est.fuentes.tamano_etiqueta)
        ax_plot.grid(True, alpha=0.3)

        return _guardar_o_mostrar(fig, est)


class ComparacionCanales:
    """Comparación lado a lado de todos los canales de un slice."""
    nombre = "comparacion_canales"

    def __init__(self, t: int = 0, z: int = 0, canales: Optional[List[int]] = None,
                 cmaps: Optional[List[str]] = None, titulo: Optional[str] = None):
        self.t = t
        self.z = z
        self.canales = canales
        self.cmaps = cmaps or ["gray", "Reds", "Greens", "Blues", "plasma", "viridis"]
        self.titulo = titulo

    def __call__(self, data, estetica: Optional[Estetica] = None) -> Any:
        import matplotlib.pyplot as plt
        est = estetica or Estetica()

        canales = self.canales if self.canales else list(range(data.dims.C))
        n = len(canales)
        ncols = min(n, 4)
        nrows = (n + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(3.5 * ncols, 3.5 * nrows),
                                 dpi=est.layout.dpi)
        est.aplicar_tema()
        if nrows == 1 and ncols == 1:
            axes = np.array([[axes]])
        elif nrows == 1:
            axes = np.array([axes])
        axes = np.array(axes).flatten()

        for i, c in enumerate(canales):
            ax = axes[i]
            slice_2d = data.datos[self.t, self.z, c, :, :]
            cmap = self.cmaps[i % len(self.cmaps)]
            im = ax.imshow(_normalizar_slice(slice_2d), cmap=cmap)
            nombre = data.canales[c] if c < len(data.canales) else f"Canal {c}"
            vmin, vmax = slice_2d.min(), slice_2d.max()
            ax.set_title(f"{nombre}\n[{vmin:.0f},{vmax:.0f}]",
                         fontsize=est.fuentes.tamano_tick)
            ax.axis("off")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        for j in range(i + 1, len(axes)):
            axes[j].axis("off")

        fig.suptitle(self.titulo or f"Comparación Canales — T={self.t}, Z={self.z}",
                     fontsize=est.fuentes.tamano_titulo,
                     fontweight=est.fuentes.peso_titulo)
        return _guardar_o_mostrar(fig, est)
