# === gestorLab/QC_Visualizacion.py ===
"""
Visualización de control de calidad del pipeline.

Muestra las transformaciones imagen paso a paso después de ejecutar.
Cada nodo con BioImagenData se convierte en un subplot.
"""
from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Optional

from .Flujo_Trabajo import GrafoPipeline
from ..controlador.Controlador_BioImagen import BioImagenData


def visualizar_pasos(
    grafo:    GrafoPipeline,
    canal:    int = 0,
    t:        int = 0,
    z:        int = 0,
    ruta:     Optional[Path] = None,
    titulo:   str = "Control de Calidad — Pipeline",
):
    """
    Crea un plot con un subplot por cada nodo que contenga BioImagenData.
    Los nodos se muestran en orden topológico (izquierda → derecha).

    Args:
        grafo:  GrafoPipeline después de ejecutar (nodos con .data poblado)
        canal:  Canal a visualizar
        t:      Timepoint a mostrar
        z:      Plano Z a mostrar
        ruta:   Si se especifica, guarda el plot en esa ruta
        titulo: Título de la figura
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        raise ImportError("matplotlib es requerido para QC. pip install matplotlib")

    orden = grafo.orden_topologico()

    # Recolectar nodos con imagen en orden topológico
    entradas = []
    for nodo_id in orden:
        nodo = grafo.nodos[nodo_id]
        if not nodo.data:
            continue
        ultimo = nodo.data[-1]
        if not isinstance(ultimo, BioImagenData):
            continue
        entrantes = grafo.entrantes(nodo_id)
        op_nombre = (
            entrantes[-1].operacion.nombre if entrantes
            else "input"
        )
        entradas.append((nodo_id, op_nombre, ultimo))

    if not entradas:
        print("[QC] No hay imágenes intermedias para visualizar")
        return None

    n   = len(entradas)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 5), squeeze=False)
    axes = axes[0]

    fig.suptitle(titulo, fontsize=13, fontweight="bold", y=1.02)

    for ax, (nodo_id, op_nombre, data) in zip(axes, entradas):
        _dibujar_slice(ax, data, canal, t, z, op_nombre)

    plt.tight_layout()

    if ruta:
        ruta = Path(ruta)
        ruta.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(ruta, dpi=150, bbox_inches="tight")
        print(f"[QC] Plot guardado en: {ruta}")

    plt.show()
    return fig


def _dibujar_slice(
    ax,
    data:     BioImagenData,
    canal:    int,
    t:        int,
    z:        int,
    op_nombre: str,
):
    """Dibuja un slice 2D en el eje dado, con overlay de máscara si existe."""
    try:
        t_idx = min(t, data.dims.T - 1)
        z_idx = min(z, data.dims.Z - 1)
        c_idx = min(canal, data.dims.C - 1)

        slice_2d = data.datos[t_idx, z_idx, c_idx, :, :]

        # Mostrar imagen base
        ax.imshow(slice_2d, cmap="gray", interpolation="nearest")

        # Overlay de máscara si viene en metadata (post-merge)
        mascara_datos = data.metadata.get("mascara_datos")
        if mascara_datos is not None:
            try:
                if mascara_datos.ndim == 5:
                    mask_2d = mascara_datos[t_idx, z_idx, c_idx, :, :]
                else:
                    mask_2d = mascara_datos[t_idx, z_idx, :, :]

                if mask_2d.max() > 0:
                    mask_rgba = _mascara_a_rgba(mask_2d)
                    ax.imshow(mask_rgba, interpolation="nearest")
            except Exception:
                pass  # overlay opcional — no romper QC si falla

        # Estadísticas del slice
        vmin, vmax = float(slice_2d.min()), float(slice_2d.max())
        stats = f"[{vmin:.1f}, {vmax:.1f}]  dtype={slice_2d.dtype}"

        # Título limpio
        partes    = op_nombre.split("_")
        nombre_corto = "_".join(partes[-2:]) if len(partes) > 2 else op_nombre
        ax.set_title(f"{nombre_corto}\n{stats}", fontsize=7, pad=4)
        ax.axis("off")

    except Exception as e:
        ax.text(
            0.5, 0.5, f"Error:\n{e}",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=8, color="red"
        )
        ax.axis("off")


def _mascara_a_rgba(mask: np.ndarray) -> np.ndarray:
    """
    Convierte máscara de etiquetas a RGBA para overlay semitransparente.
    Cada etiqueta recibe un color distinto.
    """
    etiquetas = np.unique(mask)
    etiquetas = etiquetas[etiquetas > 0]  # ignorar fondo

    rgba = np.zeros((*mask.shape, 4), dtype=np.float32)

    colores = [
        [1.0, 0.2, 0.2, 0.45],   # rojo
        [0.2, 1.0, 0.2, 0.45],   # verde
        [0.2, 0.4, 1.0, 0.45],   # azul
        [1.0, 0.8, 0.0, 0.45],   # amarillo
        [0.8, 0.2, 1.0, 0.45],   # violeta
        [0.0, 0.9, 0.9, 0.45],   # cian
    ]

    for i, etiqueta in enumerate(etiquetas):
        color = colores[i % len(colores)]
        mascara_bool = mask == etiqueta
        for c_idx in range(4):
            rgba[:, :, c_idx][mascara_bool] = color[c_idx]

    return rgba