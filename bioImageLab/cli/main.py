import click

from pathlib import Path
from typing import Optional

import numpy as np

try:
    import pandas as pd
except ImportError:
    pd = None

from bioImageLab.nucleo.gestorLab.Gestor_Lab import GestorLab
from bioImageLab.nucleo.controlador.Controlador_BioImagen import BioImagenData


@click.group()
def cli():
    """
    BioImageLab - procesamiento y análisis de bioimágenes.
    """
    pass


# ==========================================
# HELPERS: resumen / guardado de resultados
# ==========================================

def _resumir_dato(nombre_nodo: str, dato) -> str:
    """
    Genera una descripción legible de qué hay en un nodo final del pipeline:
      - BioImagenData -> shape, dtype, rango de valores
      - DataFrame     -> shape, columnas
      - Otro          -> repr básico
    """
    if isinstance(dato, BioImagenData):
        arr = dato.datos
        return (
            f"  • {nombre_nodo}\n"
            f"      tipo:   imagen (BioImagenData)\n"
            f"      shape:  {dato.dims.shape}  (T,Z,C,Y,X)\n"
            f"      dtype:  {arr.dtype}\n"
            f"      rango:  [{arr.min():.4g}, {arr.max():.4g}]"
        )
    if pd is not None and isinstance(dato, pd.DataFrame):
        return (
            f"  • {nombre_nodo}\n"
            f"      tipo:   tabla (DataFrame)\n"
            f"      shape:  {dato.shape}\n"
            f"      columnas: {list(dato.columns)}"
        )
    return f"  • {nombre_nodo}\n      tipo:   {type(dato).__name__}  valor: {dato!r}"


def _guardar_dato(nombre_nodo: str, dato, dir_salida: Path) -> Optional[Path]:
    """
    Guarda el dato de un nodo final a disco.
    BioImagenData -> .npy (array crudo)
    DataFrame      -> .csv
    Devuelve la ruta guardada, o None si no se sabe cómo persistir ese tipo.
    """
    dir_salida.mkdir(parents=True, exist_ok=True)

    if isinstance(dato, BioImagenData):
        ruta = dir_salida / f"{nombre_nodo}.npy"
        np.save(ruta, dato.datos)
        return ruta

    if pd is not None and isinstance(dato, pd.DataFrame):
        ruta = dir_salida / f"{nombre_nodo}.csv"
        dato.to_csv(ruta, index=False)
        return ruta

    return None


# ==========================================
# COMANDO: run
# ==========================================

@cli.command()
@click.option(
    "--yaml",
    "yaml_path",
    type=click.Path(exists=True, path_type=Path),
    required=True,
)
@click.option(
    "--imagen",
    "imagen_path",
    type=click.Path(exists=True, path_type=Path),
    required=True,
)
@click.option(
    "--debug",
    is_flag=True,
    default=False,
    help="Activa debug: imprime cada paso y genera QC visual.",
)
@click.option(
    "--output",
    "dir_salida",
    type=click.Path(path_type=Path),
    default=None,
    help="Directorio donde guardar los resultados (imagen .npy / tabla .csv).",
)
def run(yaml_path, imagen_path, debug, dir_salida):
    """
    Ejecuta un pipeline sobre una imagen.
    """
    gestor = GestorLab()

    try:
        flujo = gestor.registrar_desde_yaml(yaml_path)
    except (FileNotFoundError, ValueError, KeyError) as e:
        raise click.ClickException(f"No se pudo registrar el pipeline: {e}")

    click.echo(f"Pipeline registrado: {flujo.nombre}")

    resultado = gestor.ejecutar_desde_ruta(flujo.nombre, imagen_path, debug=debug)

    if resultado.es_ok():
        salida, logs = resultado.unwrap()
        n_err  = sum(1 for l in logs if l.nivel.value == "error")
        n_warn = sum(1 for l in logs if l.nivel.value == "warn")

        click.secho(f"✓ Ejecución completada ({len(salida)} nodo(s) final(es))", fg="green")
        if n_err or n_warn:
            click.echo(f"  ({n_err} errores, {n_warn} warnings en el log)")

        click.echo()
        for nombre_nodo, dato in salida.items():
            click.echo(_resumir_dato(nombre_nodo, dato))

            if dir_salida is not None:
                ruta_guardada = _guardar_dato(nombre_nodo, dato, dir_salida)
                if ruta_guardada:
                    click.echo(f"      guardado: {ruta_guardada}")
                else:
                    click.echo(f"      (no se sabe guardar tipo {type(dato).__name__}, se omite)")
    else:
        error = resultado.error
        click.secho(f"✗ Error: {getattr(error, 'mensaje', str(error))}", fg="red", err=True)
        raise SystemExit(1)


# ==========================================
# GRUPO: pipeline
# ==========================================

@cli.group()
def pipeline():
    """
    Operaciones relacionadas con pipelines.
    """
    pass


# ==========================================
# COMANDO: pipeline listar
# ==========================================

@pipeline.command()
@click.option(
    "--dir",
    "dir_path",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=Path("."),
    show_default=True,
    help="Directorio donde buscar archivos YAML de pipelines.",
)
def listar(dir_path):
    """
    Lista los pipelines disponibles a partir de los YAML encontrados
    en el directorio indicado.
    """
    yamls = sorted(dir_path.glob("*.yaml")) + sorted(dir_path.glob("*.yml"))

    if not yamls:
        click.echo(f"No se encontraron archivos YAML en '{dir_path}'.")
        return

    gestor = GestorLab()
    click.echo(f"Pipelines encontrados en '{dir_path}':\n")

    for ruta in yamls:
        try:
            flujo = gestor.registrar_desde_yaml(ruta)
            n_nodos = len(flujo.grafo.nodos)
            click.echo(f"  • {flujo.nombre}  ({ruta.name}, {n_nodos} nodos)")
        except Exception as e:
            click.echo(f"  ✗ {ruta.name}: no se pudo cargar — {e}")


# ==========================================
# COMANDO: pipeline validar
# ==========================================

@pipeline.command()
def validar():
    """
    Valida un pipeline.
    """
    click.echo("Validando pipeline...")


# ==========================================
# COMANDO: pipeline grafo
# ==========================================

@pipeline.command()
def grafo():
    """
    Muestra el grafo de un pipeline.
    """
    click.echo("Mostrando grafo...")


# ==========================================
# COMANDO: pipeline orden
# ==========================================

@pipeline.command()
def orden():
    """
    Muestra el orden de ejecución de un pipeline.
    """
    click.echo("Mostrando orden de ejecución...")


# ==========================================
# ENTRY POINT
# ==========================================
# IMPORTANTE: este bloque va al FINAL del archivo, después de que todos
# los comandos (incluidos los del grupo `pipeline`) ya fueron registrados
# con sus decoradores. Si se ejecuta antes, los subcomandos definidos más
# abajo en el archivo no existen todavía cuando se corre como script.

if __name__ == "__main__":
    cli()