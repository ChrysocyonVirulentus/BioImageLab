import click

from pathlib import Path

from bioImageLab.nucleo.gestorLab.Gestor_Lab import GestorLab


@click.group()
def cli():
    """
    BioImageLab - procesamiento y análisis de bioimágenes.
    """
    pass


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
def run(yaml_path, imagen_path, debug):
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
        click.secho(f"✓ Ejecución completada. Nodos finales: {list(salida.keys())}", fg="green")
        if n_err or n_warn:
            click.echo(f"  ({n_err} errores, {n_warn} warnings en el log)")
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
# COMANDO: pipeline 
# ==========================================

@pipeline.command()
def listar():
    """
    Lista los pipelines disponibles.
    """
    click.echo("Lista de pipelines")


if __name__ == "__main__":
    cli()


@pipeline.command()
def validar():
    """
    Valida un pipeline.
    """
    click.echo("Validando pipeline...")

@pipeline.command()
def grafo():
    click.echo("Mostrando grafo...")


@pipeline.command()
def orden():
    click.echo("Mostrando orden de ejecución...")