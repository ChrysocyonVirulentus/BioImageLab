"""
Exportación e importación de DataFrames en formato Apache Parquet.

Cuándo preferir Parquet sobre CSV:
    ┌─────────────────────────────┬────────────┬────────────┐
    │ Criterio                    │ CSV/TSV    │ Parquet    │
    ├─────────────────────────────┼────────────┼────────────┤
    │ Legibilidad humana          │ ✓          │ ✗ (binario)│
    │ Compatibilidad universal    │ ✓          │ Necesita   │
    │ (Excel, R, Prism)           │            │ pyarrow    │
    │ Preserva dtype exacto       │ ✗          │ ✓          │
    │ (int vs float vs category)  │            │            │
    │ Velocidad de escritura/     │ Lento para │ Muy rápido │
    │ lectura para N > 100k filas │ N grande   │            │
    │ Tamaño en disco             │ Grande     │ 3-10x menor│
    │ Lectura parcial (columnas)  │ ✗          │ ✓          │
    │ Soporte de metadatos        │ Solo '#'   │ Schema     │
    │ nativos del schema          │ como texto │ tipado     │
    └─────────────────────────────┴────────────┴────────────┘

Uso recomendado en el pipeline:
    - Guardar DataFrames grandes de métricas (> 50k imágenes).
    - Preservar columnas categóricas (grupos experimentales, genotipo).
    - Cachear resultados intermedios de modelos costosos (PCA, UMAP).
    - Compartir datos entre etapas del pipeline sin conversión de tipos.

Requiere: pip install pyarrow  (o fastparquet como alternativa)

IMPORTANTE — Separación de responsabilidades:
    Este módulo NO transforma ni filtra datos.
    Recibe DataFrames ya construidos por etapas anteriores del pipeline.
"""

import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Literal, Optional, Union

import pandas as pd


def guardar(
    df: pd.DataFrame,
    ruta: Union[str, Path],
    motor: Literal['pyarrow', 'fastparquet'] = 'pyarrow',
    compresion: Literal['snappy', 'gzip', 'brotli', 'zstd', 'none'] = 'snappy',
    columnas: Optional[List[str]] = None,
    incluir_indice: bool = False,
    sobreescribir: bool = True,
    agregar_timestamp: bool = False,
    metadatos: Optional[Dict[str, str]] = None,
    verbose: bool = True,
) -> Path:
    """
    Guarda un DataFrame en formato Parquet con compresión configurable.

    Preserva exactamente los dtypes del DataFrame (int64, float32, category,
    bool, datetime, etc.) sin necesidad de inferencia posterior.

    Args:
        df: DataFrame a exportar. Todos los dtypes se preservan.

        ruta: Ruta de destino. La extensión '.parquet' se añade si no está.
              Ejemplo: 'cache/umap_embeddings'

        motor: Motor de serialización Parquet.
              'pyarrow'    : recomendado — más rápido, mejor compatibilidad.
              'fastparquet': alternativa más ligera, menos funciones.

        compresion: Algoritmo de compresión.
                   'snappy': velocidad/compresión balanceada (por defecto).
                   'gzip':   mayor compresión, más lento.
                   'brotli': muy alta compresión, más lento aún.
                   'zstd':   muy alta compresión, velocidad media.
                   'none':   sin compresión, máxima velocidad de lectura.

        columnas: Subconjunto de columnas a exportar. None → todas.

        incluir_indice: Si True, guarda el índice del DataFrame.
                       Necesario si el índice contiene IDs significativos.

        sobreescribir: False → lanza FileExistsError si el archivo ya existe.

        agregar_timestamp: Si True, añade '_YYYYMMDD_HHMMSS' al nombre de archivo.

        metadatos: Dict {clave: valor} guardado en los metadatos del schema Parquet.
                  A diferencia de CSV, los metadatos Parquet son nativos del formato
                  y se recuperan con cargar() sin necesidad de parsear comentarios.
                  Ejemplo: {'experimento': 'wt_vs_lon2', 'etapa': 'post_pca'}

        verbose: Si True, imprime ruta, dimensiones y tamaño en disco.

    Returns:
        Path del archivo guardado.

    Raises:
        ImportError: Si pyarrow (o fastparquet) no está instalado.
        FileExistsError: Si sobreescribir=False y el archivo ya existe.
        ValueError: Si el DataFrame está vacío.

    Ejemplo:
        from parquet import guardar
        guardar(
            df=df_pca_scores,
            ruta='cache/pca_scores',
            compresion='snappy',
            metadatos={'n_componentes': '10', 'estandarizacion': 'robusto'},
        )
    """
    _verificar_motor(motor)

    if df.empty:
        raise ValueError("El DataFrame está vacío. No se generó ningún archivo.")

    ruta = Path(ruta).with_suffix('.parquet')

    if agregar_timestamp:
        ts   = datetime.now().strftime('%Y%m%d_%H%M%S')
        ruta = ruta.with_name(f'{ruta.stem}_{ts}.parquet')

    if ruta.exists() and not sobreescribir:
        raise FileExistsError(
            f"El archivo ya existe: {ruta}. "
            "Usar sobreescribir=True o agregar_timestamp=True."
        )
    if ruta.exists() and sobreescribir and verbose:
        warnings.warn(f"Sobreescribiendo: {ruta}", UserWarning, stacklevel=2)

    ruta.parent.mkdir(parents=True, exist_ok=True)

    df_exp = df[columnas] if columnas else df

    kwargs_to_parquet: Dict = dict(
        path=str(ruta),
        engine=motor,
        compression=None if compresion == 'none' else compresion,
        index=incluir_indice,
    )

    # Inyectar metadatos en el schema (solo pyarrow soporta esto nativamente)
    if metadatos and motor == 'pyarrow':
        import pyarrow as pa
        import pyarrow.parquet as pq

        meta_bytes = {
            k.encode(): v.encode() for k, v in metadatos.items()
        }
        meta_bytes[b'generado'] = datetime.now().isoformat().encode()

        tabla = pa.Table.from_pandas(df_exp, preserve_index=incluir_indice)
        schema_con_meta = tabla.schema.with_metadata(
            {**tabla.schema.metadata, **meta_bytes}
        ) if tabla.schema.metadata else tabla.schema.with_metadata(meta_bytes)
        tabla = tabla.cast(schema_con_meta)
        pq.write_table(
            tabla, str(ruta),
            compression=None if compresion == 'none' else compresion,
        )
    else:
        df_exp.to_parquet(**kwargs_to_parquet)

    if verbose:
        tam_mb = ruta.stat().st_size / 1_048_576
        print(
            f"[parquet] {ruta}  "
            f"({len(df_exp)} filas × {len(df_exp.columns)} cols | "
            f"{tam_mb:.2f} MB | compresión: {compresion})"
        )

    return ruta


def cargar(
    ruta: Union[str, Path],
    columnas: Optional[List[str]] = None,
    motor: Literal['pyarrow', 'fastparquet'] = 'pyarrow',
    recuperar_metadatos: bool = False,
    verbose: bool = True,
) -> Union[pd.DataFrame, tuple]:
    """
    Carga un archivo Parquet como DataFrame, con lectura parcial por columnas.

    La lectura parcial por columnas es una ventaja clave de Parquet:
    si el DataFrame tiene 200 columnas pero solo se necesitan 5, solo se
    leen esas 5 del disco, reduciendo I/O y uso de memoria.

    Args:
        ruta: Ruta del archivo .parquet.

        columnas: Lista de columnas a cargar. None → todas.
                 Ejemplo: ['intensidad_mediana', 'area_iqr', 'Genotipo']

        motor: Motor de lectura. Debe coincidir con el usado en guardar().

        recuperar_metadatos: Si True, devuelve una tupla (DataFrame, dict_metadatos)
                            en lugar de solo el DataFrame.
                            Útil para recuperar el contexto del experimento.

        verbose: Si True, imprime ruta y dimensiones.

    Returns:
        DataFrame si recuperar_metadatos=False.
        Tupla (DataFrame, Dict[str, str]) si recuperar_metadatos=True.

    Raises:
        FileNotFoundError: Si el archivo no existe.
        ImportError: Si pyarrow no está instalado.
    """
    _verificar_motor(motor)

    ruta = Path(ruta)
    if not ruta.exists():
        raise FileNotFoundError(f"Archivo no encontrado: {ruta}")

    df = pd.read_parquet(str(ruta), engine=motor, columns=columnas)

    if verbose:
        tam_mb = ruta.stat().st_size / 1_048_576
        print(
            f"[parquet] {ruta}  "
            f"({len(df)} filas × {len(df.columns)} cols | {tam_mb:.2f} MB)"
        )

    if recuperar_metadatos and motor == 'pyarrow':
        import pyarrow.parquet as pq
        schema   = pq.read_schema(str(ruta))
        meta_raw = schema.metadata or {}
        metadatos = {
            k.decode(): v.decode()
            for k, v in meta_raw.items()
            if not k.startswith(b'pandas')
        }
        return df, metadatos

    return df


def guardar_multiples(
    dataframes: Dict[str, pd.DataFrame],
    directorio: Union[str, Path],
    metadatos_comunes: Optional[Dict[str, str]] = None,
    **kwargs,
) -> Dict[str, Path]:
    """
    Guarda múltiples DataFrames en un directorio, uno por archivo Parquet.

    Equivalente a csv.guardar_multiples() pero en formato binario eficiente.
    Recomendado para cachear resultados costosos (UMAP, modelos RF, etc.).

    Args:
        dataframes: Dict {nombre: DataFrame}.
        directorio: Directorio de destino.
        metadatos_comunes: Metadatos escritos en todos los archivos.
        **kwargs: Argumentos adicionales pasados a guardar().

    Returns:
        Dict {nombre: Path}.

    Ejemplo:
        guardar_multiples(
            dataframes={
                'umap_2d':     df_umap,
                'cluster_hdb': df_clusters,
                'pca_loadings': df_loadings,
            },
            directorio='cache/run_20250601',
            compresion='snappy',
        )
    """
    directorio = Path(directorio)
    directorio.mkdir(parents=True, exist_ok=True)

    rutas: Dict[str, Path] = {}
    for nombre, df in dataframes.items():
        rutas[nombre] = guardar(
            df=df,
            ruta=directorio / nombre,
            metadatos=metadatos_comunes,
            **kwargs,
        )
    return rutas


def inspeccionar(ruta: Union[str, Path]) -> Dict:
    """
    Inspecciona un archivo Parquet sin cargarlo completamente en memoria.

    Útil para verificar el schema, dtypes, número de filas y metadatos
    antes de decidir si cargarlo completo o por columnas seleccionadas.

    Args:
        ruta: Ruta del archivo .parquet.

    Returns:
        Dict con: 'n_filas', 'n_columnas', 'columnas', 'dtypes', 'metadatos',
                  'tam_mb'.
    """
    _verificar_motor('pyarrow')
    import pyarrow.parquet as pq

    ruta = Path(ruta)
    if not ruta.exists():
        raise FileNotFoundError(f"Archivo no encontrado: {ruta}")

    pf       = pq.ParquetFile(str(ruta))
    schema   = pf.schema_arrow
    meta_raw = schema.metadata or {}

    metadatos = {
        k.decode(): v.decode()
        for k, v in meta_raw.items()
        if not k.startswith(b'pandas')
    }

    info = {
        'n_filas':    pf.metadata.num_rows,
        'n_columnas': len(schema.names),
        'columnas':   list(schema.names),
        'dtypes':     {name: str(schema.field(name).type)
                       for name in schema.names},
        'metadatos':  metadatos,
        'tam_mb':     ruta.stat().st_size / 1_048_576,
    }

    print(f"[parquet] {ruta}")
    print(f"  Filas:    {info['n_filas']}")
    print(f"  Columnas: {info['n_columnas']}  →  {info['columnas'][:8]}"
          f"{'...' if info['n_columnas'] > 8 else ''}")
    print(f"  Tamaño:   {info['tam_mb']:.2f} MB")
    if metadatos:
        print(f"  Metadatos: {metadatos}")

    return info


# ─────────────────────────────────────────────────────────────
# Helper interno
# ─────────────────────────────────────────────────────────────

def _verificar_motor(motor: str) -> None:
    """Verifica que el motor solicitado esté instalado."""
    if motor == 'pyarrow':
        try:
            import pyarrow  # noqa: F401
        except ImportError:
            raise ImportError(
                "pyarrow no está instalado. "
                "Instalar con: pip install pyarrow"
            )
    elif motor == 'fastparquet':
        try:
            import fastparquet  # noqa: F401
        except ImportError:
            raise ImportError(
                "fastparquet no está instalado. "
                "Instalar con: pip install fastparquet"
            )