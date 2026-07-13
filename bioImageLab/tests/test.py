# === test.py ===

from pathlib import Path
from nucleo.gestorLab.Gestor_Lab import GestorLab
from nucleo.gestorLab.Log import RecolectorLog

DEBUG        = True
DEBUG_DETALLE = True


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
    for ev in logs:
        icono = {"info": "✓", "warn": "⚠", "error": "✗"}.get(ev.nivel.value, "·")
        print(f"  [{ev.timestamp}] {icono} [{ev.nivel.value.upper()}] {ev.etapa}")
        print(f"    {ev.mensaje}")
        if ev.metadata:
            for k, v in ev.metadata.items():
                print(f"    {k}: {v}")


def main():
    ruta_imagen = Path(
        "/home/nyarlathotep/Documentos/Programacion/ProjectosBioinformaticos"
        "/BioImageLab/bioImageLab/test_data/glp1_1.ids"
    )
    ruta_log = Path(
        "/home/nyarlathotep/Documentos/Programacion/ProjectosBioinformaticos"
        "/BioImageLab/bioImageLab/test_data/mi_pipeline.log"
    )

    config = {
        "nombre_pipeline": "pipeline_test_simple",
        "ruta_log": str(ruta_log),           # ← el gestor guarda el log automáticamente
        "etapas": [
            {
                "preprocesamiento": [
                    {
                        "metodo":          "max_norm",
                        "dominio":         "normalizacion",
                        "canal":           0,
                        "tipo_aplicacion": "global",
                    }
                ]
            },
            {
                "filtracion": [
                    {
                        "metodo":          "fft_pasabajo",
                        "dominio":         "filtracion",
                        "canal":           0,
                        "tipo_aplicacion": "por_corte_espaciotemporal",
                        "params":          {"radio": 5},
                    }
                ]
            },
            {
                "preprocesamiento": [
                    {
                        "metodo":          "to_uint8",
                        "dominio":         "normalizacion",
                        "canal":           0,
                        "tipo_aplicacion": "global",
                    }
                ]
            },
            {
                "segmentacion": [
                    {
                        "metodo":          "otsu",
                        "dominio":         "segmentacion",
                        "canal":           0,
                        "tipo_aplicacion": "por_corte_espaciotemporal",
                    }
                ]
            },
        ],
    }

    gestor = GestorLab()

    # ── Registro ──────────────────────────────────────────────
    debug_print("\n[DEBUG] Registrando pipeline desde config...")
    pipeline = gestor.registrar_desde_config(config)
    debug_print("[DEBUG] Pipeline registrado:", pipeline)

    # ── Grafo ─────────────────────────────────────────────────
    print("\n=== GRAFO ===")
    gestor.mostrar_grafo("pipeline_test_simple")
    gestor.mostrar_orden_ejecucion("pipeline_test_simple")

    if DEBUG_DETALLE:
        debug_print("\n[DEBUG] NODOS:")
        for n in pipeline.grafo.nodos.values():
            debug_print(" -", n)
        debug_print("\n[DEBUG] ARISTAS:")
        for a in pipeline.grafo.aristas:
            debug_print(" -", a)

    # ── Validación ────────────────────────────────────────────
    debug_print("\n[DEBUG] Validando pipeline...")
    recolector = RecolectorLog(pipeline.nombre)
    resultado_val = gestor._validar(pipeline, debug=DEBUG, recolector=recolector)
    if resultado_val.es_err():
        print("❌ Pipeline inválido:")
        print(resultado_val.error.mensaje)
        return
    debug_print("✅ Pipeline válido")

    # ── Ejecución ─────────────────────────────────────────────
    print("\n=== EJECUTANDO PIPELINE ===")
    resultado = gestor.ejecutar_desde_ruta(
        "pipeline_test_simple",
        ruta_imagen,
        validar=True,
        debug=DEBUG,
    )

    debug_print("\n[DEBUG] Resultado raw:", resultado)

    if resultado.es_err():
        print("❌ Error en ejecución:")
        print(resultado.error.mensaje)
        print(f"   etapa : {resultado.error.etapa}")
        if resultado.error.causa:
            print(f"   causa : {resultado.error.causa}")
        return

    # ── Salida ────────────────────────────────────────────────
    print("✅ Ejecución exitosa")
    salida, logs = resultado.unwrap()
    mostrar_salida(salida, logs)

    print(f"\n📄 Log guardado en: {ruta_log}")


if __name__ == "__main__":
    main()