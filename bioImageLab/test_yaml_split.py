# test_yaml.py
from pathlib import Path
from nucleo.gestorLab.Gestor_Lab import GestorLab

DEBUG         = True
DEBUG_DETALLE = True

RUTA_YAML = Path(
    "/home/nyarlathotep/Documentos/Programacion/ProjectosBioinformaticos"
    "/BioImageLab/bioImageLab/test_split_merge.yaml"
)


def debug_print(*args):
    if DEBUG:
        print(*args)


def mostrar_salida(salida: dict, logs: list):
    print("\n=== NODOS FINALES ===")
    for k, v in salida.items():
        print(f"\n🔹 {k}")
        print(f"   Tipo  : {type(v).__name__}")
        if hasattr(v, "datos"):
            print(f"   Shape : {v.datos.shape}")
            print(f"   Dtype : {v.datos.dtype}")
            if v.metadata.get("mascara_datos") is not None:
                print(f"   [Tiene máscara en metadata]")
        elif hasattr(v, "shape"):
            print(f"   Shape : {v.shape}")
        elif hasattr(v, "columns"):   # DataFrame
            print(f"   Cols  : {list(v.columns)}")
            print(f"   Filas : {len(v)}")
            if DEBUG_DETALLE:
                print(v.head())
        if DEBUG_DETALLE and hasattr(v, "datos"):
            print(f"   Repr  : {repr(v)[:120]}...")

    print(f"\n=== LOG ({len(logs)} eventos) ===")
    conteo = {"info": 0, "warn": 0, "error": 0}
    for ev in logs:
        conteo[ev.nivel.value] = conteo.get(ev.nivel.value, 0) + 1

    print(f"  ✓ INFO  : {conteo['info']}")
    print(f"  ⚠ WARN  : {conteo['warn']}")
    print(f"  ✗ ERROR : {conteo['error']}")

    if DEBUG_DETALLE:
        print()
        for ev in logs:
            icono = {"info": "✓", "warn": "⚠", "error": "✗"}.get(ev.nivel.value, "·")
            print(f"  {icono} [{ev.nivel.value.upper()}] {ev.etapa}")
            print(f"      {ev.mensaje}")


def main():
    if not RUTA_YAML.exists():
        print(f"❌ YAML no encontrado: {RUTA_YAML}")
        return

    gestor = GestorLab()

    # ── Registro ──────────────────────────────────────────────
    debug_print(f"\n[DEBUG] Cargando YAML: {RUTA_YAML}")
    try:
        pipeline = gestor.registrar_desde_yaml(RUTA_YAML)
    except Exception as e:
        print(f"❌ Error al cargar YAML: {e}")
        import traceback; traceback.print_exc()
        return

    debug_print(f"[DEBUG] Pipeline: {pipeline}")

    # ── Estructura del grafo ───────────────────────────────────
    print("\n=== GRAFO ===")
    gestor.mostrar_grafo(pipeline.nombre)
    gestor.mostrar_orden_ejecucion(pipeline.nombre)

    if DEBUG_DETALLE:
        debug_print("\n[DEBUG] CHECKPOINTS del constructor:")
        # (acceso directo para debug — no parte de la API pública)
        debug_print("\n[DEBUG] NODOS:")
        for n in pipeline.grafo.nodos.values():
            debug_print(f"  {n}")
        debug_print("\n[DEBUG] ARISTAS:")
        for a in pipeline.grafo.aristas:
            debug_print(f"  {a}")

    # ── Ruta de imagen ─────────────────────────────────────────
    import yaml
    with open(RUTA_YAML, "r", encoding="utf-8") as f:
        config_raw = yaml.safe_load(f)

    ruta_imagen = config_raw.get("ruta_imagen")
    if not ruta_imagen:
        print("❌ El YAML no contiene 'ruta_imagen'")
        return

    ruta_imagen = Path(ruta_imagen)
    if not ruta_imagen.exists():
        print(f"❌ Imagen no encontrada: {ruta_imagen}")
        return

    debug_print(f"\n[DEBUG] Imagen: {ruta_imagen}")

    # ── Ejecución ─────────────────────────────────────────────
    print("\n=== EJECUTANDO PIPELINE (split + merge) ===")
    resultado = gestor.ejecutar_desde_ruta(
        pipeline.nombre,
        ruta_imagen,
        validar=True,
        debug=DEBUG,
    )

    debug_print(f"\n[DEBUG] Resultado raw: {resultado}")

    if resultado.es_err():
        print("❌ Error en ejecución:")
        error = resultado.error
        print(f"   mensaje : {getattr(error, 'mensaje', str(error))}")
        print(f"   etapa   : {getattr(error, 'etapa', '?')}")
        if hasattr(error, "causa") and error.causa:
            print(f"   causa   : {error.causa}")
        import traceback
        if hasattr(error, "causa") and error.causa:
            traceback.print_exception(type(error.causa), error.causa, error.causa.__traceback__)
        return

    print("✅ Ejecución exitosa")
    salida, logs = resultado.unwrap()
    mostrar_salida(salida, logs)

    ruta_log = config_raw.get("ruta_log")
    ruta_qc  = config_raw.get("ruta_qc")
    modo_qc  = config_raw.get("modo_qc", False)

    if ruta_log:
        print(f"\n📄 Log guardado en: {ruta_log}")
    if modo_qc and ruta_qc:
        print(f"🖼  QC plot guardado en: {ruta_qc}")
    elif modo_qc:
        print("🖼  QC plot mostrado (sin ruta_qc especificada en YAML)")


if __name__ == "__main__":
    main()