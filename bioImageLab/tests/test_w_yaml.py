# === test_yaml.py ===

from pathlib import Path
from nucleo.gestorLab.Gestor_Lab import GestorLab

DEBUG         = True
DEBUG_DETALLE = True

RUTA_YAML = Path(
    "/home/nyarlathotep/Documentos/Programacion/ProjectosBioinformaticos"
    "/BioImageLab/bioImageLab/test.yaml"
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
        elif hasattr(v, "shape"):
            print(f"   Shape : {v.shape}")
        if DEBUG_DETALLE:
            print(f"   Repr  : {repr(v)[:120]}...")

    print(f"\n=== LOG DE EJECUCIÓN ({len(logs)} eventos) ===")
    infos    = [e for e in logs if e.nivel.value == "info"]
    warnings = [e for e in logs if e.nivel.value == "warn"]
    errores  = [e for e in logs if e.nivel.value == "error"]

    print(f"   ✓ INFO    : {len(infos)}")
    print(f"   ⚠ WARN    : {len(warnings)}")
    print(f"   ✗ ERROR   : {len(errores)}")

    if DEBUG_DETALLE:
        print()
        for ev in logs:
            icono = {"info": "✓", "warn": "⚠", "error": "✗"}.get(ev.nivel.value, "·")
            print(f"  {icono} [{ev.nivel.value.upper()}] {ev.etapa} — {ev.mensaje}")
            if ev.metadata:
                for mk, mv in ev.metadata.items():
                    print(f"      {mk}: {mv}")


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
        return

    debug_print("[DEBUG] Pipeline registrado:", pipeline)
    debug_print("[DEBUG] Nombre:", pipeline.nombre)

    # ── Grafo ─────────────────────────────────────────────────
    print("\n=== GRAFO ===")
    gestor.mostrar_grafo(pipeline.nombre)
    gestor.mostrar_orden_ejecucion(pipeline.nombre)

    if DEBUG_DETALLE:
        debug_print("\n[DEBUG] NODOS:")
        for n in pipeline.grafo.nodos.values():
            debug_print(" -", n)
        debug_print("\n[DEBUG] ARISTAS:")
        for a in pipeline.grafo.aristas:
            debug_print(" -", a)

    # ── Ruta de imagen desde el YAML ──────────────────────────
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

    debug_print(f"[DEBUG] Imagen: {ruta_imagen}")

    # ── Ejecución ─────────────────────────────────────────────
    print("\n=== EJECUTANDO PIPELINE ===")
    resultado = gestor.ejecutar_desde_ruta(
        pipeline.nombre,
        ruta_imagen,
        validar=True,
        debug=DEBUG,
    )

    debug_print("\n[DEBUG] Resultado raw:", resultado)

    if resultado.es_err():
        print("❌ Error en ejecución:")
        error = resultado.error
        print(f"   mensaje : {getattr(error, 'mensaje', str(error))}")
        print(f"   etapa   : {getattr(error, 'etapa', '?')}")
        if hasattr(error, "causa") and error.causa:
            print(f"   causa   : {error.causa}")
        return

    # ── Salida ────────────────────────────────────────────────
    print("✅ Ejecución exitosa")
    salida, logs = resultado.unwrap()
    mostrar_salida(salida, logs)

    ruta_log = config_raw.get("ruta_log")
    if ruta_log:
        print(f"\n📄 Log guardado en: {ruta_log}")
    else:
        print("\n💡 Tip: añade 'ruta_log' al YAML para guardar el log automáticamente")


if __name__ == "__main__":
    main()