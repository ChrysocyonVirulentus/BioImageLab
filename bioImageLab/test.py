# === test.py ===

from nucleo.gestorLab.Gestor_Flujo_Trabajo import GestorFlujoTrabajo


def main():

    # =========================================================
    # CONFIG (simula YAML)
    # =========================================================

    config = {
        "input": {
            "ruta": "/home/nyarlathotep/Documentos/Programacion/ProjectosBioinformaticos/BioImageLab/bioImageLab/test_pipeline_resultado.png"
        },

        "nombre_pipeline": "pipeline_test_simple",

        "etapas": [

            {
                "preprocesamiento": [
                    {
                        "metodo": "max_norm",
                        "dominio": "normalizacion"
                    }
                ]
            },

            {
                "filtracion": [
                    {
                        "metodo": "fft_pasabajo",
                        "params": {
                            "radio": 5.0
                        }
                    }
                ]
            },

            {
                "segmentacion": [
                    {
                        "metodo": "otsu"
                    }
                ]
            }

        ]
    }

    # =========================================================
    # GESTOR (CONSTRUCCIÓN + REGISTRO)
    # =========================================================

    gestor = GestorFlujoTrabajo()

    pipeline = gestor.registrar_desde_config(config)

    print("\n=== PIPELINE CONSTRUIDO ===")
    print(pipeline)

    # =========================================================
    # VALIDACIÓN
    # =========================================================

    print("\n=== VALIDANDO PIPELINE ===")
    ok, errores = pipeline.validar_pipeline()

    if not ok:
        print("❌ Pipeline inválido:")
        for e in errores:
            print(" -", e)
        return
    else:
        print("✅ Pipeline válido")

    # =========================================================
    # EJECUCIÓN (NUEVO MODELO)
    # =========================================================

    print("\n=== EJECUTANDO PIPELINE ===")

    ruta = config["input"]["ruta"]

    resultado = gestor.ejecutar_desde_ruta(
        "pipeline_test_simple",
        ruta
    )

    if resultado.es_err():
        print("❌ Error en ejecución:")
        print(resultado.error)
    else:
        print("✅ Ejecución exitosa")

        salida = resultado.unwrap()

        print("Tipo de salida:", type(salida))

        # Debug útil
        if hasattr(salida, "datos"):
            print("Shape:", salida.datos.shape)

        # Opcional: ver canal
        try:
            print("Canales:", salida.canales)
        except:
            pass


if __name__ == "__main__":
    main()