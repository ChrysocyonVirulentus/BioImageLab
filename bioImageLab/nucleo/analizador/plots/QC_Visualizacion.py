# === analizador/qc/QC_Visualizacion.py ===
"""
Control de Calidad (QC) para pipelines de bioimagen.

Visualización de transformaciones paso a paso usando los plots del analizador.
Cada nodo con BioImagenData se convierte en un subplot.

Integra con:
  - Plots_Imagen (OverlayMascara, ComparacionCanales, StackViewer, etc.)
  - Plots_Estadisticos (HistogramaIntensidad, BoxplotCanales)
  - Estetica (configuración visual global)
"""
from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any, Union
from dataclasses import dataclass

import pandas as pd

# Importamos los plots del analizador
from ..plots.Plots_Imagen import (
    OverlayMascara, ComparacionCanales, StackViewer,
    OrthoView, MIPProyeccion, PerfilIntensidad,
)
from ..plots.Plots_Estadisticos import (
    HistogramaIntensidad, BoxplotCanales,
)
from ..plots.Estetica import Estetica, estetica_publicacion

# Tipos del sistema
from ...controlador.Resultado_Either import Resultado, Ok, Err
from ...controlador.Controlador_BioImagen import BioImagenData


# ==================== ERRORES QC ====================

@dataclass(frozen=True)
class ErrorQC:
    etapa:   str
    mensaje: str
    causa:   Optional[Exception] = None

    def con_contexto(self, nueva_etapa: str) -> "ErrorQC":
        from dataclasses import replace
        return replace(self, etapa=f"{nueva_etapa} -> {self.etapa}")


# ==================== CONFIGURACIÓN QC ====================

@dataclass(frozen=True)
class ConfigQC:
    """Configuración para visualización QC."""
    canal: int = 0
    t: int = 0
    z: int = 0
    mostrar_mascara: bool = True
    mostrar_histograma: bool = True
    mostrar_stats: bool = True
    estetica: Estetica = field(default_factory=estetica_publicacion)
    figsize_por_nodo: Tuple[float, float] = (4, 4)
    dpi: int = 150

    def con_canal(self, canal: int) -> "ConfigQC":
        from dataclasses import replace
        return replace(self, canal=canal)

    def con_t(self, t: int) -> "ConfigQC":
        from dataclasses import replace
        return replace(self, t=t)

    def con_z(self, z: int) -> "ConfigQC":
        from dataclasses import replace
        return replace(self, z=z)


# ==================== EXTRACTOR DE PASOS ====================

class ExtractorPasosQC:
    """
    Extrae pasos ejecutables del pipeline para QC.
    Cada paso produce una figura QC.
    """

    def __init__(self, config: ConfigQC):
        self.config = config

    def extraer_pasos_imagen(
        self,
        secuencia: List[Tuple[str, BioImagenData]],
    ) -> List[Tuple[str, Any]]:
        """
        Extrae figuras QC de una secuencia de (nombre_operacion, BioImagenData).

        Retorna lista de (nombre_operacion, figura).
        """
        pasos = []
        for nombre_op, data in secuencia:
            fig = self._figura_paso(nombre_op, data)
            if fig is not None:
                pasos.append((nombre_op, fig))
        return pasos

    def _figura_paso(self, nombre_op: str, data: BioImagenData) -> Optional[Any]:
        """Genera figura para un paso individual."""
        est = self.config.estetica
        c, t, z = self.config.canal, self.config.t, self.config.z

        try:
            # Validar índices
            t = min(t, data.dims.T - 1)
            z = min(z, data.dims.Z - 1)
            c = min(c, data.dims.C - 1)

            # Overlay con máscara si existe
            if self.config.mostrar_mascara and "mascara_datos" in data.metadata:
                plot = OverlayMascara(
                    canal=c, t=t, z=z,
                    titulo=f"{nombre_op}",
                )
                return plot(data, estetica=est)

            # Comparación de canales si hay múltiples
            if data.dims.C > 1 and nombre_op == "input":
                plot = ComparacionCanales(t=t, z=z, titulo=f"{nombre_op}")
                return plot(data, estetica=est)

            # Stack viewer para Z > 1
            if data.dims.Z > 1 and data.dims.Z <= 8:
                plot = StackViewer(canal=c, t=t, ncols=4, cmap="gray")
                return plot(data, estetica=est)

            # Vista ortogonal para volúmenes
            if data.dims.Z > 1 and data.dims.Z > 8:
                plot = OrthoView(canal=c, t=t, cmap="gray")
                return plot(data, estetica=est)

            # Default: overlay simple (imagen base)
            plot = OverlayMascara(canal=c, t=t, z=z, titulo=f"{nombre_op}")
            return plot(data, estetica=est)

        except Exception as e:
            # Fallback: figura con mensaje de error
            return self._figura_error(nombre_op, str(e), est)

    def _figura_error(self, nombre_op: str, mensaje: str, estetica: Estetica) -> Any:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=estetica.figsize("default"), dpi=estetica.layout.dpi)
        estetica.aplicar_tema()
        ax.text(0.5, 0.5, f"QC Error\n{nombre_op}\n{mensaje}",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=estetica.fuentes.tamano_etiqueta,
                color=estetica.paleta.cuaternario, wrap=True)
        ax.set_title(nombre_op, fontsize=estetica.fuentes.tamano_titulo,
                     fontweight=estetica.fuentes.peso_titulo)
        ax.axis("off")
        fig.tight_layout(pad=estetica.layout.padding)
        return fig


# ==================== VISUALIZADOR QC PRINCIPAL ====================

def visualizar_pasos(
    secuencia: List[Tuple[str, BioImagenData]],
    config: Optional[ConfigQC] = None,
    ruta: Optional[Path] = None,
    titulo: str = "Control de Calidad — Pipeline",
) -> Resultado[Any, ErrorQC]:
    """
    Crea un plot QC con un subplot por cada paso que contenga BioImagenData.

    Args:
        secuencia: Lista de (nombre_operacion, BioImagenData)
        config: Configuración QC (usa default si None)
        ruta: Si se especifica, guarda el plot
        titulo: Título de la figura

    Retorna:
        Resultado[Figure, ErrorQC]
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return Err(ErrorQC("import", "matplotlib es requerido para QC. pip install matplotlib"))

    cfg = config or ConfigQC()
    extractor = ExtractorPasosQC(cfg)

    try:
        pasos = extractor.extraer_pasos_imagen(secuencia)
        if not pasos:
            return Err(ErrorQC("qc", "No hay imágenes intermedias para visualizar"))

        n = len(pasos)
        figsize = (cfg.figsize_por_nodo[0] * n, cfg.figsize_por_nodo[1])
        fig, axes = plt.subplots(1, n, figsize=figsize, dpi=cfg.dpi, squeeze=False)
        axes = axes[0]

        fig.suptitle(titulo, fontsize=cfg.estetica.fuentes.tamano_titulo + 2,
                     fontweight=cfg.estetica.fuentes.peso_titulo, y=1.02)

        for ax, (nombre_op, fig_paso) in zip(axes, pasos):
            # Transferir contenido de fig_paso a ax del grid
            _transferir_a_eje(ax, fig_paso, nombre_op, cfg)

        plt.tight_layout()

        if ruta:
            ruta = Path(ruta)
            ruta.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(ruta, dpi=cfg.dpi, bbox_inches="tight")

        return Ok(fig)

    except Exception as e:
        return Err(ErrorQC("visualizacion", f"Error generando QC: {e}", causa=e))


def _transferir_a_eje(ax_destino, fig_origen, nombre_op: str, config: ConfigQC):
    """Transfiere contenido de una figura individual a un eje del grid QC."""
    est = config.estetica

    # Extraer el primer axes de la figura origen
    ax_origen = fig_origen.axes[0] if fig_origen.axes else None
    if ax_origen is None:
        ax_destino.text(0.5, 0.5, "Sin datos", transform=ax_destino.transAxes,
                        ha="center", va="center", fontsize=est.fuentes.tamano_tick)
        ax_destino.axis("off")
        return

    # Copiar imágenes
    for img in ax_origen.images:
        extent = img.get_extent()
        ax_destino.imshow(img.get_array(), cmap=img.get_cmap(),
                         vmin=img.norm.vmin, vmax=img.norm.vmax,
                         extent=extent, interpolation=img.get_interpolation())

    # Copiar líneas
    for line in ax_origen.lines:
        ax_destino.plot(line.get_xdata(), line.get_ydata(),
                       color=line.get_color(), lw=line.get_linewidth(),
                       ls=line.get_linestyle(), marker=line.get_marker(),
                       ms=line.get_markersize(), alpha=line.get_alpha())

    # Copiar scatter
    for coll in ax_origen.collections:
        if hasattr(coll, 'get_offsets'):
            offsets = coll.get_offsets()
            if len(offsets) > 0:
                ax_destino.scatter(offsets[:, 0], offsets[:, 1],
                                  c=coll.get_facecolors(), s=20, alpha=0.6)

    # Copiar textos
    for text in ax_origen.texts:
        ax_destino.text(text.get_position()[0], text.get_position()[1],
                       text.get_text(), fontsize=text.get_fontsize(),
                       color=text.get_color(), ha=text.get_ha(), va=text.get_va())

    # Título y stats
    ax_destino.set_title(nombre_op, fontsize=est.fuentes.tamano_subtitulo,
                         fontweight=est.fuentes.peso_subtitulo, pad=4)
    ax_destino.axis("off")

    # Cerrar figura origen para liberar memoria
    import matplotlib.pyplot as plt
    plt.close(fig_origen)


# ==================== VISUALIZACIÓN COMPARATIVA (Side-by-Side) ====================

def comparar_antes_despues(
    data_antes: BioImagenData,
    data_despues: BioImagenData,
    nombre_op: str = "Operación",
    config: Optional[ConfigQC] = None,
    ruta: Optional[Path] = None,
) -> Resultado[Any, ErrorQC]:
    """
    Comparación lado a lado de imagen antes y después de una operación.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return Err(ErrorQC("import", "matplotlib es requerido"))

    cfg = config or ConfigQC()
    est = cfg.estetica

    try:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5), dpi=cfg.dpi)
        est.aplicar_tema()

        for ax, data, titulo in zip(axes, [data_antes, data_despues], ["Antes", "Después"]):
            t = min(cfg.t, data.dims.T - 1)
            z = min(cfg.z, data.dims.Z - 1)
            c = min(cfg.canal, data.dims.C - 1)
            slice_2d = data.datos[t, z, c, :, :]

            vmin, vmax = float(slice_2d.min()), float(slice_2d.max())
            ax.imshow(slice_2d, cmap="gray", vmin=vmin, vmax=vmax)
            ax.set_title(f"{titulo}\n[{vmin:.1f}, {vmax:.1f}] dtype={slice_2d.dtype}",
                        fontsize=est.fuentes.tamano_subtitulo)
            ax.axis("off")

        fig.suptitle(f"QC: {nombre_op}", fontsize=est.fuentes.tamano_titulo + 2,
                     fontweight=est.fuentes.peso_titulo)
        plt.tight_layout()

        if ruta:
            ruta = Path(ruta)
            ruta.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(ruta, dpi=cfg.dpi, bbox_inches="tight")

        return Ok(fig)

    except Exception as e:
        return Err(ErrorQC("comparacion", f"Error en comparación: {e}", causa=e))


# ==================== REPORTE QC COMPLETO ====================

def reporte_qc_completo(
    secuencia: List[Tuple[str, BioImagenData]],
    config: Optional[ConfigQC] = None,
    ruta_salida: Optional[Path] = None,
    incluir_histogramas: bool = True,
) -> Resultado[Dict[str, Any], ErrorQC]:
    """
    Genera un reporte QC completo con:
      - Visualización paso a paso
      - Histogramas de intensidad por paso
      - Estadísticas resumen

    Retorna diccionario con figuras y DataFrame de stats.
    """
    cfg = config or ConfigQC()
    resultado = {"figuras": {}, "stats": None}

    # Figura principal de pasos
    res_pasos = visualizar_pasos(secuencia, config=cfg,
                                  titulo="QC — Pasos del Pipeline")
    if res_pasos.es_err():
        return res_pasos.map_err(lambda e: e)
    resultado["figuras"]["pasos"] = res_pasos.unwrap()

    # Histogramas si se piden
    if incluir_histogramas:
        try:
            stats_rows = []
            for nombre_op, data in secuencia:
                t = min(cfg.t, data.dims.T - 1)
                z = min(cfg.z, data.dims.Z - 1)
                c = min(cfg.canal, data.dims.C - 1)
                slice_2d = data.datos[t, z, c, :, :]
                stats_rows.append({
                    "operacion": nombre_op,
                    "min": float(slice_2d.min()),
                    "max": float(slice_2d.max()),
                    "mean": float(slice_2d.mean()),
                    "std": float(slice_2d.std()),
                    "median": float(np.median(slice_2d)),
                    "dtype": str(slice_2d.dtype),
                })
            resultado["stats"] = pd.DataFrame(stats_rows)
        except Exception as e:
            return Err(ErrorQC("stats", f"Error calculando estadísticas: {e}", causa=e))

    # Guardar si se especificó ruta
    if ruta_salida:
        try:
            ruta_salida = Path(ruta_salida)
            ruta_salida.mkdir(parents=True, exist_ok=True)
            for nombre, fig in resultado["figuras"].items():
                fig.savefig(ruta_salida / f"qc_{nombre}.png", dpi=cfg.dpi, bbox_inches="tight")
            if resultado["stats"] is not None:
                resultado["stats"].to_csv(ruta_salida / "qc_stats.csv", index=False)
        except Exception as e:
            return Err(ErrorQC("guardado", f"Error guardando reporte: {e}", causa=e))

    return Ok(resultado)
