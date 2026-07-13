#!/usr/bin/env python3
# === main.py ===
"""
Punto de entrada de BioImageLab.

Modos de uso
────────────
  # Ejecución individual (interactiva)
  python main.py

  # Batch desde archivo de lista
  python main.py --batch lista_imagenes.txt --dir /ruta/imagenes

  # Con debug QC visual
  python main.py --debug

  # Especificar YAML distinto
  python main.py --yaml mi_experimento.yaml

  # Modo estricto: aborta si la validación detecta errores duros
  python main.py --estricto
"""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

# ── Ajustar sys.path para importaciones relativas si se corre como script ──
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _construir_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="BioImageLab — pipeline de bioimagen"
    )
    p.add_argument(
        "--yaml", type=Path,
        default=Path("test.yaml"),
        help="Ruta al archivo YAML de configuración del pipeline",
    )
    p.add_argument(
        "--imagen", type=Path, default=None,
        help="Ruta de imagen a procesar (sobreescribe ruta_imagen del YAML)",
    )
    p.add_argument(
        "--batch", type=Path, default=None,
        help="Archivo .txt con lista de nombres de imagen (modo batch)",
    )
    p.add_argument(
        "--dir", type=Path, default=None,
        help="Directorio base de imágenes (modo batch)",
    )
    p.add_argument(
        "--debug", action="store_true",
        help="Activa debug: imprime cada paso y genera QC visual",
    )
    p.add_argument(
        "--estricto", action="store_true",
        help="Aborta si la validación detecta errores duros",
    )
    p.add_argument(
        "--log-batch", type=Path, default=None,
        help="Ruta del TSV de resumen batch (por defecto: junto al YAML)",
    )
    return p


# ──────────────────────────────────────────────────────────────────────────────
# HELPERS DE SALIDA
# ──────────────────────────────────────────────────────────────────────────────

def _imprimir_salida(salida: dict, logs: list, debug: bool = False) -> None:
    from nucleo.controlador.Resultado_Either import NivelLog

    print("\n" + "═" * 60)
    print("  NODOS FINALES")
    print("═" * 60)

    for nodo_id, valor in salida.items():
        print(f"\n▸ {nodo_id}")
        tipo = type(valor).__name__
        print(f"  tipo   : {tipo}")

        if valor is None:
            print("  ⚠ nodo sin datos")
            continue

        if hasattr(valor, "datos"):          # BioImagenData
            print(f"  shape  : {valor.datos.shape}")
            print(f"  dtype  : {valor.datos.dtype}")
            print(f"  canales: {valor.canales}")
            if valor.metadata.get("mascara_datos") is not None:
                print("  ✓ tiene máscara en metadata")

        elif hasattr(valor, "columns"):      # pd.DataFrame
            import pandas as pd
            print(f"  columnas: {list(valor.columns)}")
            print(f"  filas   : {len(valor)}")
            if debug:
                print(valor.head().to_string(index=False))

        elif hasattr(valor, "shape"):        # np.ndarray
            print(f"  shape  : {valor.shape}")
            print(f"  dtype  : {valor.dtype}")

        else:
            print(f"  repr   : {repr(valor)[:100]}")

    print(f"\n{'─'*60}")
    print(f"  LOG  ({len(logs)} eventos)")
    print(f"{'─'*60}")

    conteo = {k: 0 for k in ("info", "warn", "error")}
    for ev in logs:
        conteo[ev.nivel.value] = conteo.get(ev.nivel.value, 0) + 1

    print(f"  ✓ INFO   : {conteo['info']}")
    print(f"  ⚠ WARN   : {conteo['warn']}")
    print(f"  ✗ ERROR  : {conteo['error']}")

    if debug:
        print()
        for ev in logs:
            icono = {"info": "✓", "warn": "⚠", "error": "✗"}.get(ev.nivel.value, "·")
            print(f"  {icono} [{ev.nivel.value.upper()}] {ev.etapa}")
            print(f"      {ev.mensaje}")


def _imprimir_resumen_batch(resultados: dict, debug: bool = False) -> None:
    from nucleo.controlador.Resultado_Either import NivelLog

    ok_count  = sum(1 for r in resultados.values() if r.es_ok())
    err_count = len(resultados) - ok_count

    print(f"\n{'═'*60}")
    print(f"  RESUMEN BATCH  ({len(resultados)} imágenes)")
    print(f"{'═'*60}")
    print(f"  ✓ Exitosas : {ok_count}")
    print(f"  ✗ Fallidas : {err_count}")

    for ruta, res in resultados.items():
        nombre_corto = Path(ruta).name
        if res.es_ok():
            salida, logs = res.unwrap()
            n_err = sum(1 for l in logs if l.nivel == NivelLog.ERROR)
            tag   = f"[OK]  nodos={list(salida.keys())} err_log={n_err}"
        else:
            err = res.error
            tag = f"[ERR] {getattr(err, 'mensaje', str(err))[:60]}"
        print(f"  {nombre_corto:40} {tag}")


# ──────────────────────────────────────────────────────────────────────────────
# FLUJO PRINCIPAL
# ──────────────────────────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> int:
    """Punto de entrada real. Retorna código de salida."""
    import yaml as _yaml
    from nucleo.gestorLab.Gestor_Lab import GestorLab
    from nucleo.controlador.Controlador_BioImagen import ModoImagen

    ruta_yaml = args.yaml
    if not ruta_yaml.exists():
        print(f"❌  YAML no encontrado: {ruta_yaml}")
        return 1

    # ── Leer config raw para extraer ruta_imagen ──────────────
    with open(ruta_yaml, "r", encoding="utf-8") as f:
        config_raw = _yaml.safe_load(f)

    # ── Instanciar gestor y registrar pipeline ─────────────────
    gestor = GestorLab()

    print(f"\n[main] Cargando pipeline desde: {ruta_yaml}")
    try:
        pipeline = gestor.registrar_desde_yaml(ruta_yaml)
    except Exception as e:
        print(f"❌  Error cargando YAML: {e}")
        if args.debug:
            traceback.print_exc()
        return 1

    print(f"[main] Pipeline '{pipeline.nombre}' registrado")

    # ── Estructura del grafo ───────────────────────────────────
    gestor.mostrar_grafo(pipeline.nombre)
    gestor.mostrar_orden_ejecucion(pipeline.nombre)

    # ── Diagnóstico de validación (sin bloquear) ───────────────
    print(f"\n{'─'*60}")
    print("  DIAGNÓSTICO DE VALIDACIÓN")
    print(f"{'─'*60}")
    gestor.mostrar_diagnostico(pipeline.nombre)

    # ── Modo estricto: abortar si hay errores duros ────────────
    if args.estricto or config_raw.get("modo_estricto", False):
        from nucleo.gestorLab.Validar_Flujo_Trabajo import validar_pipeline
        diag = validar_pipeline(pipeline.grafo).unwrap()
        if diag.tiene_errores_duros:
            print(f"\n❌  Modo estricto: {len(diag.errores)} errores duros. Abortando.")
            return 1

    # ── MODO BATCH ─────────────────────────────────────────────
    if args.batch:
        print(f"\n[main] Modo BATCH — lista: {args.batch}")
        ruta_log_batch = args.log_batch or ruta_yaml.parent / f"{pipeline.nombre}_batch.tsv"
        resultados = gestor.ejecutar_batch_desde_archivo(
            nombre        = pipeline.nombre,
            ruta_lista    = args.batch,
            directorio    = args.dir,
            debug         = args.debug,
            ruta_log_batch= ruta_log_batch,
        )
        _imprimir_resumen_batch(resultados, debug=args.debug)
        print(f"\n📄  Resumen batch: {ruta_log_batch}")
        n_err = sum(1 for r in resultados.values() if r.es_err())
        return 0 if n_err == 0 else 2   # 2 = batch con algunos fallos

    # ── MODO INDIVIDUAL ────────────────────────────────────────
    ruta_imagen = args.imagen or config_raw.get("ruta_imagen")
    if not ruta_imagen:
        print("❌  No se especificó ruta de imagen (--imagen o ruta_imagen en YAML)")
        return 1

    ruta_imagen = Path(ruta_imagen)
    if not ruta_imagen.exists():
        print(f"❌  Imagen no encontrada: {ruta_imagen}")
        return 1

    print(f"\n[main] Ejecutando sobre: {ruta_imagen.name}")
    resultado = gestor.ejecutar_desde_ruta(
        nombre      = pipeline.nombre,
        ruta_imagen = ruta_imagen,
        debug       = args.debug,
    )

    if resultado.es_ok():
        salida, logs = resultado.unwrap()
        print("\n✅  Ejecución completada")
        _imprimir_salida(salida, logs, debug=args.debug)

        # Generar QC si se activa por argumento o por YAML
        modo_qc  = args.debug or config_raw.get("modo_qc", False)
        ruta_qc  = config_raw.get("ruta_qc")
        ruta_log = config_raw.get("ruta_log")

        if ruta_log:
            print(f"\n📄  Log: {ruta_log}")
        if modo_qc and ruta_qc:
            print(f"🖼   QC guardado en: {ruta_qc}")
        elif modo_qc:
            print("🖼   QC generado (sin ruta_qc en YAML, solo pantalla)")

        return 0

    else:
        err = resultado.error
        print(f"\n❌  Ejecución fallida")
        print(f"   etapa  : {getattr(err, 'etapa', '?')}")
        print(f"   mensaje: {getattr(err, 'mensaje', str(err))}")
        if args.debug and hasattr(err, "causa") and err.causa:
            traceback.print_exception(type(err.causa), err.causa, err.causa.__traceback__)
        return 1


def main() -> None:
    parser = _construir_parser()
    args   = parser.parse_args()
    sys.exit(run(args))


if __name__ == "__main__":
    main()
