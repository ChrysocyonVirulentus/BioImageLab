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

# CAMBIO 1: formatos de imagen aceptados por BioImageLab para --dir.
FORMATOS_BIOIMAGEN = {".ids", ".ics", ".tif", ".tiff"}


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
    if isinstance(dato, BioImagenData):
        arr = dato.datos
        lineas = [
            f"  • {nombre_nodo}",
            f"      tipo:   imagen (BioImagenData)",
            f"      shape:  {dato.dims.shape}  (T,Z,C,Y,X)",
            f"      dtype:  {arr.dtype}",
        ]
        for c in range(dato.dims.C):
            canal_arr = arr[:, :, c, :, :]
            nombre_canal = dato.canales[c] if c < len(dato.canales) else str(c)
            lineas.append(
                f"      canal {c} ({nombre_canal}): rango [{canal_arr.min():.4g}, {canal_arr.max():.4g}]"
            )
        return "\n".join(lineas)

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

    # CAMBIO 2: proteger la ejecución individual para que una excepción
    # inesperada no termine mostrando un traceback completo al usuario.
    try:
        resultado = gestor.ejecutar_desde_ruta(
            flujo.nombre,
            imagen_path,
            debug=debug,
        )
    except Exception as e:
        raise click.ClickException(f"Error durante la ejecución: {e}")

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
# COMANDO: run batch
# ==========================================

@cli.command(name="batch")  # CAMBIO 3: antes era run-batch
@click.option(
    "--yaml",
    "yaml_path",
    type=click.Path(exists=True, path_type=Path),
    required=True,
)
@click.option(
    "--dir",
    "dir_imagenes",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=None,
    help="Carpeta con las imágenes a procesar.",
)
@click.option(
    "--glob",
    "patron_glob",
    default="*",
    show_default=True,
    help="Patrón para filtrar archivos dentro de --dir (ej: '*.tif').",
)
@click.option(
    "--lista",
    "ruta_lista",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Archivo .txt con un nombre de imagen por línea.",
)
@click.option(
    "--base-dir",
    "dir_base",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=None,
    help="Directorio base para las rutas indicadas en --lista.",
)
@click.option(
    "--debug",
    is_flag=True,
    default=False,
    help="Activa debug: imprime cada paso por imagen.",
)
@click.option(
    "--output",
    "dir_salida",
    type=click.Path(path_type=Path),
    default=None,
    help="Directorio donde guardar los resultados de cada imagen.",
)
@click.option(
    "--log-batch",
    "ruta_log_batch",
    type=click.Path(path_type=Path),
    default=None,
    help="Ruta donde guardar el resumen TSV del batch.",
)
def batch(yaml_path, dir_imagenes, patron_glob, ruta_lista, dir_base, debug, dir_salida, ruta_log_batch):
    """
    Ejecuta un pipeline sobre un batch de imágenes (carpeta+glob o lista .txt).
    """
    # CAMBIO 4: validación explícita de las dos fuentes posibles del batch.
    if dir_imagenes is None and ruta_lista is None:
        raise click.ClickException("Tenés que pasar --dir o --lista (uno de los dos).")

    if dir_imagenes is not None and ruta_lista is not None:
        raise click.ClickException("--dir y --lista son mutuamente excluyentes, pasá solo uno.")

    gestor = GestorLab()

    try:
        flujo = gestor.registrar_desde_yaml(yaml_path)
    except (FileNotFoundError, ValueError, KeyError) as e:
        raise click.ClickException(f"No se pudo registrar el pipeline: {e}")

    click.echo(f"Pipeline registrado: {flujo.nombre}")

    # --- armar y disparar el batch según la opción elegida ---
    if dir_imagenes is not None:
        # CAMBIO 5: filtrar directorios y formatos no soportados.
        rutas = sorted(
            ruta
            for ruta in dir_imagenes.glob(patron_glob)
            if ruta.is_file() and ruta.suffix.lower() in FORMATOS_BIOIMAGEN
        )
        if not rutas:
            raise click.ClickException(
                f"No se encontraron imágenes válidas con patrón '{patron_glob}' en '{dir_imagenes}'."
            )
        click.echo(f"  {len(rutas)} imagen(es) encontradas en '{dir_imagenes}'")
        resultados = gestor.ejecutar_batch(
            nombre=flujo.nombre,
            rutas_imagenes=rutas,
            debug=debug,
            ruta_log_batch=ruta_log_batch,
        )
    else:
        resultados = gestor.ejecutar_batch_desde_archivo(
            nombre=flujo.nombre,
            ruta_lista=ruta_lista,
            directorio=dir_base,  # CAMBIO 6: base-dir para rutas relativas del TXT.
            debug=debug,
            ruta_log_batch=ruta_log_batch,
        )

    # --- reportar / guardar cada resultado ---
    n_ok = n_err = 0
    for ruta_str, resultado in resultados.items():
        click.echo(f"\n── {ruta_str}")
        if resultado.es_ok():
            n_ok += 1
            salida, logs = resultado.unwrap()
            n_e = sum(1 for l in logs if l.nivel.value == "error")
            n_w = sum(1 for l in logs if l.nivel.value == "warn")
            click.secho(f"  ✓ OK ({len(salida)} nodo(s) final(es))", fg="green")
            if n_e or n_w:
                click.echo(f"    ({n_e} errores, {n_w} warnings en el log)")

            for nombre_nodo, dato in salida.items():
                click.echo(_resumir_dato(nombre_nodo, dato))
                if dir_salida is not None:
                    # subcarpeta por imagen para no pisar resultados entre sí
                    subdir = dir_salida / Path(ruta_str).stem
                    ruta_guardada = _guardar_dato(nombre_nodo, dato, subdir)
                    if ruta_guardada:
                        click.echo(f"      guardado: {ruta_guardada}")
        else:
            n_err += 1
            error = resultado.error
            click.secho(f"  ✗ Error: {getattr(error, 'mensaje', str(error))}", fg="red")

    click.echo()
    click.secho(f"Batch terminado: {n_ok} ok, {n_err} con error(es)", fg=("green" if n_err == 0 else "yellow"))
    if ruta_log_batch:
        click.echo(f"Resumen TSV: {ruta_log_batch}")



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

@pipeline.command(name="validar_pipeline")
@click.option(
    "--yaml",
    "yaml_path",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Archivo YAML del pipeline a validar.",
)
def validar(yaml_path):
    """
    Valida un pipeline sin ejecutar imágenes.
    """
    # CAMBIO 7: validar ahora hace trabajo real usando el validador del núcleo.
    from bioImageLab.nucleo.gestorLab.Validar_Flujo_Trabajo import validar_pipeline

    gestor = GestorLab()
    try:
        flujo = gestor.registrar_desde_yaml(yaml_path)
    except (FileNotFoundError, ValueError, KeyError) as e:
        raise click.ClickException(f"No se pudo cargar el pipeline: {e}")

    diagnostico = validar_pipeline(flujo.grafo).unwrap()
    click.echo(f"Pipeline: {flujo.nombre}")
    click.echo(diagnostico.resumen())

    for evento in diagnostico.eventos:
        nivel = evento.nivel.value.upper()
        click.echo(f"  [{nivel}] {evento.mensaje}")

    if not diagnostico.es_valido:
        raise click.exceptions.Exit(1)


# ==========================================
# COMANDO: pipeline grafo
# ==========================================

@pipeline.command()
@click.option(
    "--yaml",
    "yaml_path",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Archivo YAML del pipeline.",
)
def grafo(yaml_path):
    """
    Muestra el grafo de un pipeline.
    """
    # CAMBIO 8: usa la visualización que ya existe en GestorLab.
    gestor = GestorLab()
    try:
        flujo = gestor.registrar_desde_yaml(yaml_path)
    except (FileNotFoundError, ValueError, KeyError) as e:
        raise click.ClickException(f"No se pudo cargar el pipeline: {e}")

    gestor.mostrar_grafo(flujo.nombre)


# ==========================================
# COMANDO: pipeline orden
# ==========================================

@pipeline.command()
@click.option(
    "--yaml",
    "yaml_path",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Archivo YAML del pipeline.",
)
def orden(yaml_path):
    """
    Muestra el orden topológico de ejecución de un pipeline.
    """
    # CAMBIO 9: usa la visualización que ya existe en GestorLab.
    gestor = GestorLab()
    try:
        flujo = gestor.registrar_desde_yaml(yaml_path)
    except (FileNotFoundError, ValueError, KeyError) as e:
        raise click.ClickException(f"No se pudo cargar el pipeline: {e}")

    gestor.mostrar_orden_ejecucion(flujo.nombre)


# ==========================================
# ENTRY POINT
# ==========================================
# IMPORTANTE: este bloque va al FINAL del archivo, después de que todos
# los comandos (incluidos los del grupo `pipeline`) ya fueron registrados
# con sus decoradores. Si se ejecuta antes, los subcomandos definidos más
# abajo en el archivo no existen todavía cuando se corre como script.

if __name__ == "__main__":
    cli()