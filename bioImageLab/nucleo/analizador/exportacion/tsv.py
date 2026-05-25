# === analizador/exportacion/tsv.py ===
"""
Exportador de DataFrames a formato TSV (Tab-Separated Values).
Convenience wrapper sobre ExportadorCSV con separador tab.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from .csv import ExportadorCSV


class ExportadorTSV(ExportadorCSV):
    """Exporta DataFrame a archivo TSV (tab-separated)."""
    nombre = "exportador_tsv"

    def __init__(self, encoding: str = "utf-8", index: bool = False,
                 float_format: Optional[str] = None, na_rep: str = ""):
        super().__init__(
            separador="\t",
            encoding=encoding,
            index=index,
            float_format=float_format,
            na_rep=na_rep,
            line_terminator="\n",
        )
