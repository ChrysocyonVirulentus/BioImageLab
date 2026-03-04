#!/usr/bin/env python3
"""
Test de pipeline funcional con bioimagen real.
Ejecuta: python test_pipeline_bioimagen.py
"""

from pathlib import Path
import numpy as np

# === IMPORTS DEL SISTEMA ===
from ..controlador.Controlador_BioImagen import ControladorBioImagen, ModoImagen
from ..controlador.Resultado_Either import Resultado, Ok, Err

from .Categoria_Operacion import CategoriaOperacion
from .Operacion import Operacion, TipoSalida
from .Pipeline import Pipeline
from .Pipeline_Builder import PipelineBuilder
from .Flujo_Procesamiento import FlujoProcesamiento

# === IMPORTS DE TUS MÓDULOS DE PROCESAMIENTO ===
from ..preprocesador.normalizador.Metodos_Normalizacion import MaxNorm
from ..controlador.Controlador_Normalizador import Normalizador

from ..filtrador.espectrales.Filtros_Ffts import FFTPasaAlto
from ..controlador.Controlador_Filtrador import Controlador_Filtrador

from ..realzador.morfologicos.Realzadores_Morfologicos import Apertura
from ..controlador.Controlador_Realzador import Controlador_Realzador

from ..segmentador.binarizacion.Segmentadores_Binarizacion import Otsu
from ..controlador.Controlador_Segmentador import Controlador_Segmentador

# === VISUALIZACIÓN (asumiendo que bioimagenes.py está en analizador/plots/) ===
try:
    from ..analizador.plots.bioimagenes import (
        panel_transformaciones,
        panel_antes_despues,
        panel_segmentacion
    )
    VISUALIZACION_DISPONIBLE = True
except ImportError:
    VISUALIZACION_DISPONIBLE = False
    print("⚠️  Módulo de visualización no encontrado, se omitirán los plots")


def crear_operaciones_puras() -> list[Operacion]:
    """
    Crea las operaciones del pipeline con instancias callables de tus módulos.
    Cada operación envuelve tu clase callable específica.
    """
    operaciones = []
    
    # 1. NORMALIZACIÓN (PREPROCESAMIENTO)
    # Crear instancia del método de normalización
    metodo_norm = MaxNorm()  # Tu clase callable de Metodos_Normalizacion
    
    # El Normalizador es el wrapper que aplica el método
    normalizador = Normalizador(
        tipo=...,  # Tu tipo de normalización (ej: Norm_Global())
        metodo=metodo_norm
    )
    
    op_norm = Operacion(
        nombre="max_norm_global",
        categoria=CategoriaOperacion.PREPROCESAMIENTO,
        instancia_callable=normalizador,  # El wrapper es callable: img → img_norm
        canal_objetivo=None,  # Aplica a todos los canales
        parametros_originales={"metodo": "max", "tipo": "global"},
        tipo_salida=TipoSalida.IMAGEN
    )
    operaciones.append(op_norm)
    
    # 2. FILTRACIÓN ESPECTRAL (FILTRACION)
    # Instancia directa de tu filtro FFT
    filtro_alto = FFTPasaalto(radio=30)  # Tu clase de Filtros_Ffts
    
    # O via Controlador_Filtrador si prefieres el wrapper
    # controlador_fft = Controlador_Filtrador()
    # filtro_alto = controlador_fft.crear_pasaalto(radio=30)
    
    op_fft = Operacion(
        nombre="fft_pasaalto",
        categoria=CategoriaOperacion.FILTRACION,
        instancia_callable=filtro_alto,  # FFTPasaalto.__call__ recibe img → img_filtrada
        canal_objetivo=None,
        parametros_originales={"radio": 30, "tipo": "pasaalto"},
        tipo_salida=TipoSalida.IMAGEN
    )
    operaciones.append(op_fft)
    
    # 3. REALCE MORFOLÓGICO (REALZADOR)
    # Apertura: erosión seguida de dilatación (elimina ruido pequeño, preserva forma)
    apertura = Apertura(
        elemento_estructurante=np.ones((3, 3)),  # Kernel 3x3
        iteraciones=1
    )
    
    op_apertura = Operacion(
        nombre="apertura_morfologica",
        categoria=CategoriaOperacion.REALZADOR,
        instancia_callable=apertura,
        canal_objetivo=None,
        parametros_originales={"kernel": "3x3", "iteraciones": 1},
        tipo_salida=TipoSalida.IMAGEN
    )
    operaciones.append(op_apertura)
    
    # 4. SEGMENTACIÓN (SEGMENTADOR)
    # Otsu: umbralización automática
    otsu = Otsu()  # Tu clase de Segmentadores_Binarizacion
    
    op_otsu = Operacion(
        nombre="otsu_automatico",
        categoria=CategoriaOperacion.SEGMENTADOR,
        instancia_callable=otsu,  # Retorna máscara binaria
        canal_objetivo=0,  # Segmentar canal 0 (ej: canal de fluorescencia)
        parametros_originales={"metodo": "otsu", "automatico": True},
        tipo_salida=TipoSalida.MASCARA,  # Importante: marca que es máscara
        es_operacion_split=True  # Marca punto de bifurcación potencial
    )
    operaciones.append(op_otsu)
    
    return operaciones


def ejecutar_pipeline_con_visualizacion(
    ruta_imagen: Path,
    canal_visualizacion: int = 0
) -> Resultado[dict, Exception]:
    """
    Ejecuta el pipeline completo con visualizaciones en cada etapa.
    """
    print(f"\n{'='*60}")
    print(f"TEST PIPELINE FUNCIONAL - BioImagen")
    print(f"{'='*60}")
    print(f"Archivo: {ruta_imagen}")
    print(f"Canal para visualización: {canal_visualizacion}")
    
    resultados_intermedios = {}
    
    # === 1. CARGAR IMAGEN ===
    print(f"\n[1/5] CARGANDO IMAGEN...")
    controlador = ControladorBioImagen(ruta_imagen)
    
    resultado_carga = controlador.cargar_ImagenResultado(ModoImagen.AUTO)
    if resultado_carga.es_err():
        return Err(resultado_carga.error)
    
    data_original = resultado_carga.unwrap()
    print(f"  ✓ Cargada: {data_original.dims}")
    print(f"  ✓ Canales: {data_original.canales}")
    print(f"  ✓ Rango intensidad: [{data_original.datos.min():.2f}, {data_original.datos.max():.2f}]")
    
    # Guardar para visualización
    resultados_intermedios["original"] = data_original
    
    # === 2. CONSTRUIR PIPELINE ===
    print(f"\n[2/5] CONSTRUYENDO PIPELINE...")
    
    operaciones = crear_operaciones_puras()
    for i, op in enumerate(operaciones):
        print(f"  {i+1}. {op}")
    
    try:
        builder = PipelineBuilder(modo_estricto=True)
        for op in operaciones:
            builder.agregar(op)
        
        pipeline = builder.construir()
        print(f"  ✓ Pipeline válido: {len(pipeline)} operaciones")
        print(f"  ✓ Puntos de split detectados: {pipeline.puntos_split}")
        
    except ValueError as e:
        return Err(Exception(f"Error construyendo pipeline: {e}"))
    
    # === 3. EJECUTAR CON SNAPSHOTS ===
    print(f"\n[3/5] EJECUTANDO PIPELINE...")
    
    # Ejecutar con snapshots para visualización
    resultado_final, snapshots = pipeline.ejecutar_con_snapshots(data_original)
    
    # Organizar snapshots por etapa
    etapas = {}
    for idx, nombre, data_snap in snapshots:
        if "despues_" in nombre:
            etapa = nombre.replace("despues_", "")
            etapas[etapa] = data_snap
            resultados_intermedios[etapa] = data_snap
    
    if resultado_final.es_err():
        print(f"  ✗ Fallo en pipeline: {resultado_final.error.mensaje}")
        return Err(Exception(resultado_final.error.mensaje))
    
    data_final = resultado_final.unwrap()
    print(f"  ✓ Pipeline completado")
    print(f"  ✓ Resultado final: {data_final.dims}")
    
    # === 4. VISUALIZACIÓN ===
    if VISUALIZACION_DISPONIBLE:
        print(f"\n[4/5] GENERANDO VISUALIZACIONES...")
        
        # Extraer arrays 2D del canal específico para visualización
        def extraer_corte_2d(data: BioImagenData, t=0, z=0, c=canal_visualizacion):
            return data.datos[t, z, c, :, :]
        
        # Panel de transformaciones: todas las etapas en fila
        imagenes_etapas = [
            extraer_corte_2d(resultados_intermedios["original"]),
            extraer_corte_2d(etapas.get("max_norm_global", resultados_intermedios["original"])),
            extraer_corte_2d(etapas.get("fft_pasaalto", resultados_intermedios["original"])),
            extraer_corte_2d(etapas.get("apertura_morfologica", resultados_intermedios["original"])),
            extraer_corte_2d(etapas.get("otsu_automatico", resultados_intermedios["original"])),
        ]
        
        titulos = ["Original", "Normalizada", "FFT Pasa-alto", "Apertura", "Otsu"]
        
        fig_transformaciones = panel_transformaciones(
            imagenes_etapas,
            titulos=titulos,
            titulo_general=f"Pipeline: {ruta_imagen.name}",
            figsize=(20, 4)
        )
        fig_transformaciones.savefig("test_pipeline_transformaciones.png", dpi=150, bbox_inches="tight")
        print(f"  ✓ Guardado: test_pipeline_transformaciones.png")
        
        # Panel antes/después: original vs final
        fig_antes_despues = panel_antes_despues(
            extraer_corte_2d(resultados_intermedios["original"]),
            extraer_corte_2d(data_final),
            titulo_antes="Original",
            titulo_despues="Procesada (Post-Otsu)",
            mostrar_diferencia=True
        )
        fig_antes_despues.savefig("test_pipeline_antes_despues.png", dpi=150, bbox_inches="tight")
        print(f"  ✓ Guardado: test_pipeline_antes_despues.png")
        
        # Panel de segmentación: imagen + máscara + overlay
        if "otsu_automatico" in etapas:
            fig_segmentacion = panel_segmentacion(
                imagen_original=extraer_corte_2d(resultados_intermedios["original"]),
                mascara_segmentada=extraer_corte_2d(etapas["otsu_automatico"]),
                titulo="Segmentación Otsu"
            )
            fig_segmentacion.savefig("test_pipeline_segmentacion.png", dpi=150, bbox_inches="tight")
            print(f"  ✓ Guardado: test_pipeline_segmentacion.png")
        
    else:
        print(f"\n[4/5] VISUALIZACIÓN OMITIDA (módulo no disponible)")
    
    # === 5. RESUMEN ===
    print(f"\n[5/5] RESUMEN EJECUCIÓN")
    print(f"  Etapas ejecutadas: {len(etapas)}")
    print(f"  Datos intermedios guardados: {list(resultados_intermedios.keys())}")
    print(f"  Estado final: {'✓ ÉXITO' if resultado_final.es_ok() else '✗ FALLO'}")
    
    return Ok({
        "data_final": data_final,
        "snapshots": resultados_intermedios,
        "pipeline": pipeline,
        "controlador": controlador
    })


def test_pipeline_simple():
    """
    Test mínimo sin visualización, solo validación funcional.
    """
    print("\n" + "="*60)
    print("TEST SIMPLE - Validación funcional básica")
    print("="*60)
    
    # Usar imagen de test o crear datos sintéticos si no hay archivo
    ruta_test = Path("test_data/celula_sintetica.tiff")
    
    if not ruta_test.exists():
        print(f"Creando datos sintéticos de prueba...")
        # Crear imagen sintética 5D para test
        T, Z, C, Y, X = 1, 5, 2, 256, 256
        
        # Canal 0: células con ruido
        canal0 = np.random.poisson(100, (T, Z, Y, X)).astype(np.float64)
        # Agregar "células" artificiales
        for z in range(Z):
            canal0[0, z, 100:150, 100:150] += 200  # Célula cuadrada
        
        # Canal 1: fondo estructurado
        canal1 = np.random.normal(50, 10, (T, Z, Y, X))
        
        datos_sinteticos = np.stack([canal0, canal1], axis=2)  # [T,Z,C,Y,X]
        
        # Guardar como TIFF para test
        from tifffile import imwrite
        ruta_test.parent.mkdir(exist_ok=True)
        imwrite(ruta_test, datos_sinteticos.astype(np.uint16))
        print(f"  ✓ Creada imagen sintética: {ruta_test}")
        print(f"    Shape: {datos_sinteticos.shape}")
    
    # Ejecutar pipeline
    resultado = ejecutar_pipeline_con_visualizacion(ruta_test, canal_visualizacion=0)
    
    if resultado.es_ok():
        datos = resultado.unwrap()
        print(f"\n{'='*60}")
        print("TEST COMPLETADO EXITOSAMENTE")
        print(f"{'='*60}")
        print(f"Data final shape: {datos['data_final'].dims}")
        print(f"Snapshots disponibles: {list(datos['snapshots'].keys())}")
        return True
    else:
        print(f"\n{'='*60}")
        print("TEST FALLÓ")
        print(f"{'='*60}")
        print(f"Error: {resultado.error}")
        return False


def test_con_imagen_real(ruta: str):
    """
    Test con imagen real proporcionada por el usuario.
    """
    ruta_path = Path(ruta)
    if not ruta_path.exists():
        print(f"Error: No se encuentra {ruta}")
        return False
    
    resultado = ejecutar_pipeline_con_visualizacion(ruta_path)
    return resultado.es_ok()


if __name__ == "__main__":
    import sys
    
    # Test con datos sintéticos por defecto
    if len(sys.argv) == 1:
        exito = test_pipeline_simple()
    else:
        # Test con imagen real proporcionada
        exito = test_con_imagen_real(sys.argv[1])
    
    sys.exit(0 if exito else 1)