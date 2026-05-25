# === analizador/exportacion/figures.py ===
"""
Exportador de figuras matplotlib a diversos formatos de imagen.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Union, Dict, Any

import matplotlib
from matplotlib.figure import Figure


class ExportadorFiguras:
    """Exporta Figure matplotlib a archivo de imagen (PNG, SVG, PDF, etc.)."""
    nombre = "exportador_figuras"

    def __init__(self, formato: str = "png", dpi: Optional[int] = None,
                 bbox_inches: str = "tight", pad_inches: float = 0.1,
                 transparente: bool = False, facecolor: Optional[str] = None,
                 edgecolor: Optional[str] = None,
                 kwargs_adicionales: Optional[Dict[str, Any]] = None):
        self.formato = formato.lower().lstrip(".")
        self.dpi = dpi
        self.bbox_inches = bbox_inches
        self.pad_inches = pad_inches
        self.transparente = transparente
        self.facecolor = facecolor
        self.edgecolor = edgecolor
        self.kwargs_adicionales = kwargs_adicionales or {}

    def __call__(self, figura: Figure, ruta_salida: Path) -> Path:
        ruta_salida = Path(ruta_salida)
        ruta_salida.parent.mkdir(parents=True, exist_ok=True)

        # Asegurar extensión correcta
        if ruta_salida.suffix.lstrip(".").lower() != self.formato:
            ruta_salida = ruta_salida.with_suffix(f".{self.formato}")

        kwargs = {
            "format": self.formato,
            "bbox_inches": self.bbox_inches,
            "pad_inches": self.pad_inches,
            "transparent": self.transparente,
            **self.kwargs_adicionales,
        }
        if self.dpi is not None:
            kwargs["dpi"] = self.dpi
        if self.facecolor is not None:
            kwargs["facecolor"] = self.facecolor
        if self.edgecolor is not None:
            kwargs["edgecolor"] = self.edgecolor

        figura.savefig(ruta_salida, **kwargs)
        return ruta_salida
