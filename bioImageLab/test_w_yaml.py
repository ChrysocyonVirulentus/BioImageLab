from pathlib import Path
from nucleo.gestorLab.Gestor_Lab import GestorLab

DEBUG = True
DEBUG_DETALLE = True

def debug_print(*args):
    if DEBUG:
        print(*args)

def main():

    ruta_yaml = Path("/home/nyarlathotep/Documentos/Programacion/ProjectosBioinformaticos/BioImageLab/bioImageLab/test.yaml")

    gestor = GestorLab()

    debug_print("\n[DEBUG] Registrando pipeline desde YAML...")
    pipeline = gestor.registrar_desde_yaml(ruta_yaml)

    debug_print("\n[DEBUG] Pipeline:", pipeline)

    print("\n=== DEBUG GRAFO ===")
    gestor.mostrar_grafo(pipeline.nombre)

    # 🔍 DEBUG ESTRUCTURAL
    if DEBUG_DETALLE:
        debug_print("\n[DEBUG] NODOS:")
        for n in pipeline.grafo.nodos.values():
            debug_print(" -", n)

        debug_print("\n[DEBUG] ARISTAS:")
        for a in pipeline.grafo.aristas:
            debug_print(" -", a)

    # ✅ VALIDACIÓN
    debug_print("\n[DEBUG] Validando pipeline...")
    resultado_val = gestor._validar(pipeline, debug=DEBUG)

    if resultado_val.es_err():
        print("❌ Pipeline inválido:")
        print(resultado_val.error)
        return
    else:
        debug_print("✅ Pipeline válido")

    # 🚀 EJECUCIÓN
    print("\n=== EJECUTANDO PIPELINE ===")
    resultado = gestor.ejecutar_desde_ruta(
        pipeline.nombre,
        pipeline.etapas[0].get("ruta_imagen", None),  # tu YAML debe tener ruta_imagen
        debug=DEBUG
    )

    # 🔥 DEBUG RESULTADO
    debug_print("\n[DEBUG] Resultado raw:", resultado)

    if resultado.es_err():
        print("❌ Error en ejecución:")
        print(resultado.error)
    else:
        print("✅ Ejecución exitosa")
        salida = resultado.unwrap()

        debug_print("[DEBUG] Salida completa:", salida)
        for k, v in salida.items():
            print(f"\n🔹 Nodo final: {k}")
            print("Tipo:", type(v))
            if hasattr(v, "datos"):
                print("Shape:", v.datos.shape)
            if DEBUG_DETALLE:
                print("Contenido:", repr(v))

if __name__ == "__main__":
    main()