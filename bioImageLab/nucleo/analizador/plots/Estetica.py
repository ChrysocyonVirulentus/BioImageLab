# === analizador/plots/Estetica.py ===
"""
Configuración estética centralizada para todos los plots del analizador.
Permite personalizar títulos, fuentes, paletas de colores, tamaños, etc.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any, Callable
import matplotlib
from matplotlib import font_manager as fm


@dataclass(frozen=True)
class PaletaColores:
    """Conjunto de colores para plots categóricos y continuos."""
    primario:   str = "#2E86AB"      # azul principal
    secundario: str = "#A23B72"      # magenta
    terciario:  str = "#F18F01"      # naranja
    cuaternario:str = "#C73E1D"      # rojo
    quinario:   str = "#3B1F2B"      # violeta oscuro

    # Continuos
    continuo:   str = "viridis"
    divergente: str = "RdBu_r"
    secuencial: str = "plasma"

    # Categórico para máscaras/etiquetas
    categorico: Tuple[str, ...] = field(default_factory=lambda: (
        "#E41A1C", "#377EB8", "#4DAF4A", "#984EA3",
        "#FF7F00", "#FFFF33", "#A65628", "#F781BF",
        "#999999", "#66C2A5", "#FC8D62", "#8DA0CB",
    ))

    def color_etiqueta(self, idx: int) -> str:
        return self.categorico[idx % len(self.categorico)]


@dataclass(frozen=True)
class Fuentes:
    """Configuración tipográfica."""
    familia:        str = "sans-serif"
    familia_titulo: Optional[str] = None
    tamano_titulo:  int = 14
    tamano_subtitulo:int = 12
    tamano_etiqueta: int = 10
    tamano_tick:    int = 9
    tamano_leyenda: int = 9
    tamano_anotacion:int = 8
    peso_titulo:    str = "bold"
    peso_subtitulo: str = "normal"


@dataclass(frozen=True)
class Layout:
    """Dimensiones y espaciado de figuras."""
    figsize_default: Tuple[float, float] = (8, 6)
    figsize_ancho:   Tuple[float, float] = (12, 5)
    figsize_alto:    Tuple[float, float] = (6, 10)
    figsize_cuadrado:Tuple[float, float] = (7, 7)
    dpi:             int = 150
    padding:         float = 0.3
    tight_layout:    bool = True


@dataclass(frozen=True)
class EstiloLinea:
    """Configuración de líneas para plots."""
    ancho:      float = 1.5
    estilo:     str = "-"
    marcador:   Optional[str] = None
    tamano_marcador: float = 6.0
    alpha:      float = 0.9


@dataclass(frozen=True)
class Estetica:
    """
    Configuración estética global para plots.
    Inmutable — para cambiar, crear nueva instancia.
    """
    paleta:     PaletaColores = field(default_factory=PaletaColores)
    fuentes:    Fuentes       = field(default_factory=Fuentes)
    layout:     Layout        = field(default_factory=Layout)
    linea:      EstiloLinea   = field(default_factory=EstiloLinea)

    # Overrides por tipo de plot
    estilo_mascara_alpha: float = 0.35
    estilo_overlay_linewidth: float = 1.0

    # Tema matplotlib
    tema: str = "default"  # "default", "seaborn-v0_8-darkgrid", "ggplot", etc.

    def aplicar_tema(self) -> None:
        """Aplica tema matplotlib si está disponible."""
        if self.tema != "default":
            try:
                matplotlib.style.use(self.tema)
            except (OSError, ValueError):
                pass  # tema no disponible, seguir con default

        # Configuración global de fuentes
        matplotlib.rcParams["font.family"] = self.fuentes.familia
        matplotlib.rcParams["axes.titlesize"] = self.fuentes.tamano_titulo
        matplotlib.rcParams["axes.labelsize"] = self.fuentes.tamano_etiqueta
        matplotlib.rcParams["xtick.labelsize"] = self.fuentes.tamano_tick
        matplotlib.rcParams["ytick.labelsize"] = self.fuentes.tamano_tick
        matplotlib.rcParams["legend.fontsize"] = self.fuentes.tamano_leyenda

    def figsize(self, tipo: str = "default") -> Tuple[float, float]:
        mapping = {
            "default":  self.layout.figsize_default,
            "ancho":    self.layout.figsize_ancho,
            "alto":     self.layout.figsize_alto,
            "cuadrado": self.layout.figsize_cuadrado,
        }
        return mapping.get(tipo, self.layout.figsize_default)

    def color(self, idx: int = 0, nombre: Optional[str] = None) -> str:
        if nombre:
            return getattr(self.paleta, nombre, self.paleta.primario)
        return self.paleta.color_etiqueta(idx)

    def cmap(self, tipo: str = "continuo") -> str:
        mapping = {
            "continuo":   self.paleta.continuo,
            "divergente": self.paleta.divergente,
            "secuencial": self.paleta.secuencial,
        }
        return mapping.get(tipo, self.paleta.continuo)

    def con_tema(self, tema: str) -> "Estetica":
        from dataclasses import replace
        return replace(self, tema=tema)

    def con_paleta(self, paleta: PaletaColores) -> "Estetica":
        from dataclasses import replace
        return replace(self, paleta=paleta)

    def con_fuentes(self, fuentes: Fuentes) -> "Estetica":
        from dataclasses import replace
        return replace(self, fuentes=fuentes)

    def con_layout(self, layout: Layout) -> "Estetica":
        from dataclasses import replace
        return replace(self, layout=layout)


def estetica_publicacion() -> Estetica:
    """Estética lista para publicación científica."""
    return Estetica(
        paleta=PaletaColores(
            primario="#000000", secundario="#E69F00",
            terciario="#56B4E9", cuaternario="#009E73",
            continuo="cividis", divergente="PuOr_r",
        ),
        fuentes=Fuentes(
            familia="serif", tamano_titulo=12, tamano_etiqueta=10,
            tamano_tick=8, peso_titulo="bold",
        ),
        layout=Layout(dpi=300, figsize_default=(6, 4.5)),
    )


def estetica_oscuro() -> Estetica:
    """Estética con fondo oscuro para presentaciones."""
    return Estetica(
        paleta=PaletaColores(
            primario="#00D9FF", secundario="#FF6B9D",
            terciario="#FFE66D", cuaternario="#95E1D3",
            continuo="magma", divergente="coolwarm",
        ),
        fuentes=Fuentes(
            familia="sans-serif", tamano_titulo=16,
            tamano_etiqueta=12, tamano_tick=10,
        ),
        layout=Layout(dpi=150, figsize_default=(10, 7)),
        tema="dark_background",
    )
