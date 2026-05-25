# === analizador/exportacion/csv.py ===
"""
Exportador de DataFrames a formato CSV.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd


class ExportadorCSV:
    """Exporta DataFrame a archivo CSV."""
    nombre = "exportador_csv"

    def __init__(self, separador: str = ",", encoding: str = "utf-8",
                 index: bool = False, float_format: Optional[str] = None,
                 na_rep: str = "", line_terminator: str = "\n"):
        self.separador = separador
        self.encoding = encoding
        self.index = index
        self.float_format = float_format
        self.na_rep = na_rep
        self.line_terminator = line_terminator

    def __call__(self, data: pd.DataFrame, ruta_salida: Path) -> Path:
        ruta_salida = Path(ruta_salida)
        ruta_salida.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(
            ruta_salida,
            sep=self.separador,
            encoding=self.encoding,
            index=self.index,
            float_format=self.float_format,
            na_rep=self.na_rep,
            lineterminator=self.line_terminator,
        )
        return ruta_salida
