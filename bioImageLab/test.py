# === test.py ===

from nucleo.gestorLab.Gestor_Lab import GestorLab

DEBUG = True
DEBUG_DETALLE = True


def debug_print(*args):
    if DEBUG:
        print(*args)


def main():

    config = {
        "input": {
            "ruta": "/home/nyarlathotep/Documentos/Programacion/ProjectosBioinformaticos/BioImageLab/bioImageLab/test_data/glp1_1.ids"
        },
        "nombre_pipeline": "pipeline_test_simple",
        "etapas": [
            {
                "preprocesamiento": [
                    {"metodo": "max_norm", "dominio": "normalizacion"}
                ]
            },
            {
                "filtracion": [
                    {"metodo": "fft_pasabajo", "params": {"radio": 5.0}}
                ]
            },
            {
                "segmentacion": [
                    {"metodo": "otsu"}
                ]
            }
        ]
    }

    gestor = GestorLab()

    debug_print("\n[DEBUG] Registrando pipeline...")
    pipeline = gestor.registrar_desde_config(config)
    
    debug_print("\n[DEBUG] Pipeline:", pipeline)

    print("\n=== DEBUG GRAFO ===")
    gestor.mostrar_grafo("pipeline_test_simple")

    # 🔍 DEBUG ESTRUCTURAL
    if DEBUG_DETALLE:
        debug_print("\n[DEBUG] NODOS:")
        for n in pipeline.grafo.nodos.values():
            debug_print(" -", n)

        debug_print("\n[DEBUG] ARISTAS:")
        for a in pipeline.grafo.aristas:
            debug_print(" -", a)

    # VALIDACIÓN
    debug_print("\n[DEBUG] Validando pipeline...")
    ok, errores = pipeline.validar_pipeline()

    if not ok:
        print("❌ Pipeline inválido:")
        for e in errores:
            print(" -", e)
        return
    else:
        debug_print("✅ Pipeline válido")

    # EJECUCIÓN
    print("\n=== EJECUTANDO PIPELINE ===")

    ruta = config["input"]["ruta"]

    resultado = gestor.ejecutar_desde_ruta(
        "pipeline_test_simple",
        ruta,
        debug=DEBUG
    )

    # 🔥 FORZAR VISIBILIDAD DEL RESULTADO
    debug_print("\n[DEBUG] Resultado raw:", resultado)

    if resultado is None:
        print("❌ Resultado es None (ERROR SILENCIOSO)")
        return

    if resultado.es_err():
        print("❌ Error en ejecución:")
        print(resultado.error)
    else:
        print("✅ Ejecución exitosa")

        salida = resultado.unwrap()

        debug_print("[DEBUG] Salida completa:", salida)

        if isinstance(salida, dict):
            debug_print("[DEBUG] Claves salida:", list(salida.keys()))

        # Mostrar contenido real
        for k, v in salida.items():
            print(f"\n🔹 Nodo final: {k}")
            print("Tipo:", type(v))

            if hasattr(v, "datos"):
                print("Shape:", v.datos.shape)

            if DEBUG_DETALLE:
                print("Contenido:", repr(v))


if __name__ == "__main__":
    main()