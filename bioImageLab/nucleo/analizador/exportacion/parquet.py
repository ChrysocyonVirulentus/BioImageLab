# === analizador/exportacion/parquet.py ===
"""
Exportador de DataFrames a formato Parquet.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Any

import pandas as pd


class ExportadorParquet:
    """Exporta DataFrame a archivo Parquet con compresión opcional."""
    nombre = "exportador_parquet"

    def __init__(self, compresion: str = "snappy", engine: str = "pyarrow",
                 index: bool = False, partition_cols: Optional[list] = None,
                 kwargs_adicionales: Optional[Dict[str, Any]] = None):
        self.compresion = compresion
        self.engine = engine
        self.index = index
        self.partition_cols = partition_cols
        self.kwargs_adicionales = kwargs_adicionales or {}

    def __call__(self, data: pd.DataFrame, ruta_salida: Path) -> Path:
        ruta_salida = Path(ruta_salida)
        ruta_salida.parent.mkdir(parents=True, exist_ok=True)

        kwargs = {
            "compression": self.compresion,
            "engine": self.engine,
            "index": self.index,
            **self.kwargs_adicionales,
        }
        if self.partition_cols:
            kwargs["partition_cols"] = self.partition_cols

        data.to_parquet(ruta_salida, **kwargs)
        return ruta_salida
