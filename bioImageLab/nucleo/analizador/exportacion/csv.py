"""
Exportación e importación de DataFrames en formato CSV / TSV.

En el pipeline los DataFrames circulan entre etapas como objetos de memoria.
Este módulo provee la persistencia tabular liviana:

    Cuantificación  → DataFrame de métricas por imagen
    Modelado        → DataFrame con clusters / clases / componentes asignados
    Estadísticos    → DataFrame de resúmenes por grupo experimental

Cuándo usar CSV vs. Parquet:
    CSV/TSV  : intercambio con otros programas (Excel, R, ImageJ, Prism),
            archivos pequeños (< 1 millón de filas), legibilidad humana.
    Parquet  : almacenamiento eficiente de DataFrames grandes, columnas de
            tipos mixtos, carga parcial por columnas. Ver parquet.py.

IMPORTANTE — Separación de responsabilidades:
    Este módulo NO transforma ni filtra datos.
    Recibe el DataFrame ya construido por etapas anteriores y lo escribe a disco.
"""

import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Literal, Optional, Union

import pandas as pd


def guardar(
    df: pd.DataFrame,
    ruta: Union[str, Path],
    formato: Literal['csv', 'tsv'] = 'csv',
    separador: Optional[str] = None,
    encoding: str = 'utf-8-sig',
    incluir_indice: bool = False,
    columnas: Optional[List[str]] = None,
    decimales: int = 6,
    na_rep: str = '',
    sobreescribir: bool = True,
    agregar_timestamp: bool = False,
    metadatos: Optional[Dict[str, str]] = None,
    verbose: bool = True,
) -> Path:
    """
    Guarda un DataFrame en formato CSV o TSV.

    Maneja automáticamente:
        - Creación de directorios intermedios si no existen.
        - Timestamp opcional al nombre de archivo para versionar.
        - Metadatos como líneas '#' al inicio del archivo.
        - Protección contra sobreescritura accidental.

    Args:
        df: DataFrame a exportar.

        ruta: Ruta de destino incluyendo nombre base del archivo.
            La extensión se resuelve automáticamente según `formato`.
            Ejemplo: 'resultados/metricas_experimento_A'

        formato: Formato de salida.
                'csv': separado por comas  — compatible con Excel y mayoría de tools.
                'tsv': separado por tabulación — preferido cuando los datos
                    contienen texto con comas (nombres, etiquetas largas).

        separador: Carácter separador explícito. None → usa el del formato.
                Ejemplo: ';' para Excel en locales europeos con CSV.

        encoding: Codificación del archivo.
                'utf-8-sig': UTF-8 con BOM — abre correctamente en Excel (Windows).
                'utf-8':     Sin BOM — estándar en Linux/Mac y para uso en Python.

        incluir_indice: Si True, escribe el índice como primera columna.
                    Útil cuando el índice contiene IDs de imagen significativos.

        columnas: Subconjunto de columnas a exportar. None → todas.

        decimales: Decimales para valores flotantes (formato f'%.{decimales}f').
                Aumentar para preservar precisión numérica entre etapas.
                Reducir para legibilidad humana o compatibilidad con Prism.

        na_rep: Cadena que representa NaN en el archivo.
            ''    → celda vacía (compatible con la mayoría de lectores).
            'NaN' → explícito, más seguro para parsing posterior en Python.
            'NA'  → convención de R.

        sobreescribir: False → lanza FileExistsError si el archivo ya existe.
                    True  → sobreescribe (emite UserWarning si verbose=True).

        agregar_timestamp: Si True, añade '_YYYYMMDD_HHMMSS' antes de la extensión.
                        Permite versionar resultados de múltiples ejecuciones.

        metadatos: Dict {clave: valor} escrito como comentarios '#' al inicio.
                  No interfieren con pd.read_csv() al cargar (se omiten con comment='#').
                  Ejemplo: {'experimento': 'wt_vs_lon2', 'pipeline_version': '2.1'}

        verbose: Si True, imprime ruta, filas y columnas al guardar.

    Returns:
        Path del archivo guardado (útil si se usó agregar_timestamp).

    Raises:
        FileExistsError: Si sobreescribir=False y el archivo ya existe.
        ValueError: Si el DataFrame está vacío.

    Ejemplo:
        from csv import guardar
        ruta = guardar(
            df=df_metricas,
            ruta='resultados/metricas_ctrl',
            formato='csv',
            metadatos={'experimento': 'ctrl', 'n_imagenes': str(len(df_metricas))},
            agregar_timestamp=True,
        )
    """
    if df.empty:
        raise ValueError("El DataFrame está vacío. No se generó ningún archivo.")

    ruta     = Path(ruta)
    sep_real = separador if separador is not None else (',' if formato == 'csv' else '\t')
    ext      = f'.{formato}'
    ruta     = ruta.with_suffix(ext)

    if agregar_timestamp:
        ts   = datetime.now().strftime('%Y%m%d_%H%M%S')
        ruta = ruta.with_name(f'{ruta.stem}_{ts}{ext}')

    if ruta.exists() and not sobreescribir:
        raise FileExistsError(
            f"El archivo ya existe: {ruta}. "
            "Usar sobreescribir=True o agregar_timestamp=True."
        )
    if ruta.exists() and sobreescribir and verbose:
        warnings.warn(f"Sobreescribiendo: {ruta}", UserWarning, stacklevel=2)

    ruta.parent.mkdir(parents=True, exist_ok=True)

    df_exp = df[columnas] if columnas else df

    if metadatos:
        with open(ruta, 'w', encoding=encoding, newline='') as f:
            f.write(f'# generado: {datetime.now().isoformat()}\n')
            for k, v in metadatos.items():
                f.write(f'# {k}: {v}\n')
            df_exp.to_csv(
                f,
                sep=sep_real,
                index=incluir_indice,
                float_format=f'%.{decimales}f',
                na_rep=na_rep,
                lineterminator='\n',
            )
    else:
        df_exp.to_csv(
            ruta,
            sep=sep_real,
            index=incluir_indice,
            float_format=f'%.{decimales}f',
            na_rep=na_rep,
            encoding=encoding,
        )

    if verbose:
        print(f"[csv] {ruta}  ({len(df_exp)} filas × {len(df_exp.columns)} cols)")

    return ruta


def cargar(
    ruta: Union[str, Path],
    separador: Optional[str] = None,
    encoding: str = 'utf-8-sig',
    omitir_comentarios: bool = True,
    forzar_numericos: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Carga un CSV o TSV (con o sin líneas de metadatos '#') como DataFrame.

    Complemento directo de guardar(). Detecta automáticamente el separador
    por extensión del archivo y omite los metadatos escritos con '#'.

    Args:
        ruta: Ruta del archivo a cargar.
        separador: Separador explícito. None → detecta por extensión (.tsv → '\\t').
        encoding: Codificación del archivo.
        omitir_comentarios: Si True, ignora líneas que comienzan con '#'.
        forzar_numericos: Si True, convierte columnas a numérico donde sea posible.
        verbose: Si True, imprime ruta y dimensiones.

    Returns:
        DataFrame con los datos del archivo.

    Raises:
        FileNotFoundError: Si el archivo no existe.
    """
    ruta = Path(ruta)
    if not ruta.exists():
        raise FileNotFoundError(f"Archivo no encontrado: {ruta}")

    if separador is None:
        separador = '\t' if ruta.suffix.lower() == '.tsv' else ','

    df = pd.read_csv(
        ruta,
        sep=separador,
        encoding=encoding,
        comment='#' if omitir_comentarios else None,
    )

    if forzar_numericos:
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='ignore')

    if verbose:
        print(f"[csv] {ruta}  ({len(df)} filas × {len(df.columns)} cols)")
    return df


def guardar_multiples(
    dataframes: Dict[str, pd.DataFrame],
    directorio: Union[str, Path],
    formato: Literal['csv', 'tsv'] = 'csv',
    metadatos_comunes: Optional[Dict[str, str]] = None,
    **kwargs,
) -> Dict[str, Path]:
    """
    Guarda múltiples DataFrames en un directorio, uno por archivo.

    Útil para exportar al final de un pipeline:
        metricas crudas, clusters asignados, resúmenes por grupo, loadings PCA.

    Args:
        dataframes: Dict {nombre_archivo: DataFrame}.
        directorio: Directorio de destino (se crea si no existe).
        formato: 'csv' o 'tsv'.
        metadatos_comunes: Metadatos escritos en todos los archivos.
        **kwargs: Argumentos adicionales pasados a guardar().

    Returns:
        Dict {nombre: Path} con las rutas guardadas.

    Ejemplo:
        guardar_multiples(
            dataframes={
                'metricas':   df_metricas,
                'clusters':   df_clusters,
                'pca_scores': df_pca,
            },
            directorio='resultados/experimento_A',
            formato='csv',
            metadatos_comunes={'fecha': '2025-06-01'},
        )
    """
    directorio = Path(directorio)
    directorio.mkdir(parents=True, exist_ok=True)

    rutas: Dict[str, Path] = {}
    for nombre, df in dataframes.items():
        rutas[nombre] = guardar(
            df=df,
            ruta=directorio / nombre,
            formato=formato,
            metadatos=metadatos_comunes,
            **kwargs,
        )
    return rutas