"""
Guardado de figuras de matplotlib generadas por los módulos de plots del pipeline.

Por qué existe este módulo separado de los módulos de plots:
    En Google Colab o Jupyter Notebook, llamar a cualquier función de
    Plots_Estadisticos / Plots_Modelos / Plots_Imagen muestra la figura
    inline automáticamente. Fuera de Colab (scripts, ejecución en batch,
    servidores sin pantalla), la figura NO se muestra pero SÍ puede
    guardarse a disco.

    Este módulo desacopla la GENERACIÓN del gráfico de su PERSISTENCIA:

        ┌───────────────────────────────────────────────────────┐
        │  Módulos de plots                                     │
        │  Plots_Estadisticos / Plots_Modelos / Plots_Imagen /  │
        │  VisualizadorDimensionalidad                          │
        │                                                       │
        │   → devuelven (fig, ax) con mostrar=False             │
        │   → son agnósticos al entorno de ejecución            │
        └───────────────────────────────┬───────────────────────┘
                                        │ (fig, ax)
                                        ▼
        ┌──────────────────────────────────────────────────────────┐
        │  figures.py  (este módulo)                               │
        │                                                          │
        │   guardar_figura(fig, ruta, dpi, formato)               │
        │   guardar_desde_funcion(fn, ruta, dpi, ...)             │
        │   guardar_lote([(fn1, ruta1), ...])                     │
        └──────────────────────────────────────────────────────────┘

Uso en el pipeline:
    El pipeline pasa funciones ya parametrizada con functools.partial
    o lambdas; este módulo simplemente las ejecuta y guarda el resultado.

IMPORTANTE — Separación de responsabilidades:
    Este módulo NO decide qué graficar ni qué datos usar.
    Solo ejecuta la función de plot provista y persiste su salida.
    La parametrización del gráfico (qué columnas, qué modelo, qué estética)
    es responsabilidad del módulo de plot correspondiente.
"""

import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


# Formatos soportados y sus extensiones canónicas
_FORMATOS_VALIDOS = {
    'png':  '.png',
    'pdf':  '.pdf',
    'svg':  '.svg',
    'tiff': '.tiff',
    'eps':  '.eps',
    'jpg':  '.jpg',
    'webp': '.webp',
}


def guardar_figura(
    fig: Figure,
    ruta: Union[str, Path],
    dpi: int = 150,
    formato: Literal['png', 'pdf', 'svg', 'tiff', 'eps', 'jpg', 'webp'] = 'png',
    bbox_inches: str = 'tight',
    facecolor: str = 'white',
    transparent: bool = False,
    cerrar_tras_guardar: bool = True,
    sobreescribir: bool = True,
    agregar_timestamp: bool = False,
    verbose: bool = True,
) -> Path:
    """
    Guarda una figura de matplotlib ya generada a disco.

    Uso directo cuando el código ya tiene el objeto Figure (fig):

        fig, ax = Plots_Estadisticos.histograma(df, 'intensidad', mostrar=False)
        ruta = guardar_figura(fig, 'figuras/histograma_intensidad', dpi=300)

    Args:
        fig: Figura de matplotlib a guardar (resultado de cualquier función
             de Plots_Estadisticos / Plots_Modelos / Plots_Imagen).

        ruta: Ruta de destino incluyendo nombre base.
              La extensión se resuelve según `formato`.
              Ejemplo: 'figuras/experimento_A/pca_biplot'

        dpi: Resolución en puntos por pulgada.
            72–96  : pantalla / web.
            150    : calidad intermedia (por defecto).
            300    : publicación estándar.
            600    : publicación de alta calidad (microscopía).

        formato: Formato de imagen.
                'png'  : raster con transparencia, sin compresión con pérdida.
                'pdf'  : vectorial, editable, ideal para figuras de publicación.
                'svg'  : vectorial, editable en Inkscape/Illustrator.
                'tiff' : raster sin pérdida, estándar en microscopía y revistas.
                'eps'  : vectorial, compatible con LaTeX.
                'jpg'  : raster con pérdida, menor tamaño (no recomendado para ciencia).

        bbox_inches: 'tight' recorta márgenes vacíos automáticamente.
                    None    mantiene el tamaño exacto de la figura.

        facecolor: Color de fondo de la figura.
                  'white' : fondo blanco (por defecto, compatible con la mayoría de usos).
                  'none'  : fondo transparente (usar con transparent=True y formato 'png'/'svg').

        transparent: Si True, el fondo es transparente (solo efectivo en 'png' y 'svg').

        cerrar_tras_guardar: Si True, llama plt.close(fig) para liberar memoria.
                            Siempre True en ejecución batch para evitar acumulación.
                            Poner False solo si se necesita seguir modificando la figura.

        sobreescribir: False → lanza FileExistsError si el archivo ya existe.

        agregar_timestamp: Si True, añade '_YYYYMMDD_HHMMSS' al nombre.

        verbose: Si True, imprime ruta y tamaño en disco.

    Returns:
        Path del archivo guardado.

    Raises:
        ValueError: Si el formato no está en _FORMATOS_VALIDOS.
        FileExistsError: Si sobreescribir=False y el archivo ya existe.
    """
    if formato not in _FORMATOS_VALIDOS:
        raise ValueError(
            f"Formato '{formato}' no reconocido. "
            f"Opciones: {list(_FORMATOS_VALIDOS)}"
        )

    ruta = Path(ruta).with_suffix(_FORMATOS_VALIDOS[formato])

    if agregar_timestamp:
        ts   = datetime.now().strftime('%Y%m%d_%H%M%S')
        ruta = ruta.with_name(f'{ruta.stem}_{ts}{_FORMATOS_VALIDOS[formato]}')

    if ruta.exists() and not sobreescribir:
        raise FileExistsError(
            f"El archivo ya existe: {ruta}. "
            "Usar sobreescribir=True o agregar_timestamp=True."
        )

    ruta.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(
        str(ruta),
        dpi=dpi,
        format=formato,
        bbox_inches=bbox_inches,
        facecolor=facecolor,
        transparent=transparent,
    )

    if cerrar_tras_guardar:
        plt.close(fig)

    if verbose:
        tam_kb = ruta.stat().st_size / 1024
        print(
            f"[figures] {ruta}  "
            f"({formato.upper()} | {dpi} dpi | {tam_kb:.1f} KB)"
        )

    return ruta


def guardar_desde_funcion(
    funcion_plot: Callable[..., Tuple[Figure, Any]],
    ruta: Union[str, Path],
    dpi: int = 150,
    formato: Literal['png', 'pdf', 'svg', 'tiff', 'eps', 'jpg', 'webp'] = 'png',
    kwargs_figura: Optional[Dict] = None,
    **kwargs_guardar,
) -> Path:
    """
    Ejecuta una función de plot (con mostrar=False) y guarda su figura.

    Este es el método principal usado por el pipeline en ejecución batch:
    la función ya viene parametrizada con functools.partial o como lambda,
    y este método solo la ejecuta y persiste el resultado.

    La función debe:
        - Aceptar el argumento mostrar=False para no intentar renderizar en pantalla.
        - Devolver una tupla cuyo primer elemento sea un objeto Figure de matplotlib.

    Uso típico en el pipeline:

        from functools import partial
        import Plots_Estadisticos as pe
        import figures

        fn = partial(pe.histograma,
                     df=df_metricas,
                     columnas='intensidad_mediana',
                     mostrar_kde=True,
                     mostrar=False)

        figures.guardar_desde_funcion(fn, 'figuras/histograma_intensidad', dpi=300)

    Args:
        funcion_plot: Función ya parametrizada que devuelve (fig, ax[es]).
                     Debe aceptar mostrar=False para suprimir plt.show().
                     Funciones compatibles: todas las de Plots_Estadisticos,
                     Plots_Modelos, Plots_Imagen y VisualizadorDimensionalidad.

        ruta: Ruta de destino (sin extensión).

        dpi: Resolución de guardado.

        formato: Formato de imagen.

        kwargs_figura: Argumentos adicionales pasados a la funcion_plot.
                      Permiten sobreescribir parámetros sin recrear el partial.
                      Ejemplo: {'titulo': 'Figura final con título actualizado'}

        **kwargs_guardar: Argumentos adicionales pasados a guardar_figura().
                         Ejemplo: transparent=True, facecolor='none'

    Returns:
        Path del archivo guardado.

    Raises:
        TypeError: Si funcion_plot no devuelve un objeto Figure como primer elemento.
    """
    # Asegurar backend no-interactivo si no hay display
    backend_original = matplotlib.get_backend()
    try:
        if not _hay_display():
            matplotlib.use('Agg')
    except Exception:
        pass

    kw = kwargs_figura or {}

    try:
        resultado = funcion_plot(mostrar=False, **kw)
    except TypeError:
        # Algunas funciones (ej. pairplot) devuelven solo Figure, sin ax
        resultado = funcion_plot(**kw)

    # Extraer Figure del resultado (puede ser fig, (fig, ax), (fig, axes), etc.)
    if isinstance(resultado, Figure):
        fig = resultado
    elif isinstance(resultado, (tuple, list)) and isinstance(resultado[0], Figure):
        fig = resultado[0]
    else:
        raise TypeError(
            f"La función de plot debe devolver un objeto Figure como primer elemento. "
            f"Recibido: {type(resultado)}"
        )

    return guardar_figura(fig, ruta, dpi=dpi, formato=formato, **kwargs_guardar)


def guardar_lote(
    tareas: List[Dict],
    directorio_base: Union[str, Path],
    dpi: int = 150,
    formato: Literal['png', 'pdf', 'svg', 'tiff', 'eps', 'jpg', 'webp'] = 'png',
    continuar_si_error: bool = True,
    verbose: bool = True,
) -> Dict[str, Union[Path, Exception]]:
    """
    Guarda múltiples figuras en lote desde funciones de plot parametrizadas.

    Diseñado para el final del pipeline, donde se generan y guardan todas
    las figuras de diagnóstico y análisis en una sola llamada.

    Args:
        tareas: Lista de dicts, cada uno describe una figura a guardar.
               Cada dict puede contener:
                 'funcion':  Callable ya parametrizado (requerido si no hay 'fig').
                 'fig':      Figure ya generada (alternativa a 'funcion').
                 'nombre':   Nombre base del archivo (requerido).
                 'dpi':      DPI específico de esta figura (sobreescribe el global).
                 'formato':  Formato específico (sobreescribe el global).
                 'kwargs_figura': Dict de kwargs adicionales para la función.

        directorio_base: Directorio raíz donde se guardan todos los archivos.

        dpi: DPI global para todas las figuras (sobreescribible por tarea).

        formato: Formato global (sobreescribible por tarea).

        continuar_si_error: Si True, registra el error y continúa con las demás
                           figuras en lugar de lanzar excepción.

        verbose: Si True, imprime progreso y resumen final.

    Returns:
        Dict {nombre: Path o Exception}.
        Los éxitos contienen el Path guardado.
        Los errores contienen la excepción capturada.

    Ejemplo:
        from functools import partial
        import Plots_Estadisticos as pe
        import Plots_Modelos     as pm

        tareas = [
            {
                'nombre':  'histograma_intensidad',
                'funcion': partial(pe.histograma, df=df, columnas='intensidad_mediana',
                                   mostrar_kde=True),
            },
            {
                'nombre':  'elbow_kmeans',
                'funcion': partial(pm.elbow_kmeans, inercias=inercias,
                                   rango_k=list(range(2, 11)), k_optimo=4),
                'dpi': 200,
            },
            {
                'nombre': 'pca_ya_generado',
                'fig':    fig_pca,
            },
        ]

        resultados = guardar_lote(
            tareas=tareas,
            directorio_base='figuras/experimento_A',
            dpi=300,
            formato='png',
        )
    """
    directorio_base = Path(directorio_base)
    directorio_base.mkdir(parents=True, exist_ok=True)

    resultados: Dict[str, Union[Path, Exception]] = {}
    n_ok  = 0
    n_err = 0

    for i, tarea in enumerate(tareas):
        nombre = tarea.get('nombre', f'figura_{i:03d}')
        dpi_t  = tarea.get('dpi', dpi)
        fmt_t  = tarea.get('formato', formato)
        ruta_t = directorio_base / nombre

        try:
            if 'fig' in tarea:
                ruta_guardada = guardar_figura(
                    tarea['fig'], ruta_t, dpi=dpi_t, formato=fmt_t,
                    verbose=verbose,
                )
            elif 'funcion' in tarea:
                ruta_guardada = guardar_desde_funcion(
                    tarea['funcion'], ruta_t,
                    dpi=dpi_t, formato=fmt_t,
                    kwargs_figura=tarea.get('kwargs_figura'),
                    verbose=verbose,
                )
            else:
                raise ValueError(
                    f"La tarea '{nombre}' debe tener 'funcion' o 'fig'."
                )
            resultados[nombre] = ruta_guardada
            n_ok += 1

        except Exception as e:
            resultados[nombre] = e
            n_err += 1
            if continuar_si_error:
                warnings.warn(
                    f"Error en tarea '{nombre}': {e}",
                    UserWarning, stacklevel=2,
                )
            else:
                raise

    if verbose:
        print(
            f"\n[figures] Lote completo — "
            f"{n_ok} guardadas, {n_err} errores | directorio: {directorio_base}"
        )

    return resultados


# ─────────────────────────────────────────────────────────────
# Helpers internos
# ─────────────────────────────────────────────────────────────

def _hay_display() -> bool:
    """Detecta si hay un display disponible (False en servidores headless)."""
    import os
    import sys

    if sys.platform == 'win32':
        return True  # Windows siempre tiene GDI disponible

    # En Linux/Mac, verificar la variable de entorno DISPLAY
    if os.environ.get('DISPLAY'):
        return True

    # Colab y Jupyter setean variables específicas
    if 'google.colab' in sys.modules or 'ipykernel' in sys.modules:
        return True

    return False