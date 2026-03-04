#!/usr/bin/env python3
"""
Test de pipeline funcional con bioimagen real.
Ejecuta: python test_pipeline_bioimagen.py
"""

from pathlib import Path
import numpy as np
import traceback
from dataclasses import replace

# === IMPORTS DEL SISTEMA ===
from nucleo.controlador.Controlador_BioImagen import ControladorBioImagen, ModoImagen, BioImagenData, ErrorBioImagen
from nucleo.controlador.Resultado_Either import Resultado, Ok, Err
from .Categoria_Operacion import CategoriaOperacion
from .Operacion import Operacion, TipoSalida
from .Pipeline import Pipeline
from .Pipeline_Builder import PipelineBuilder
from ..analizador.exportacion.bioimagenes import panel_transformaciones, EsteticaGrafico

# === IMPORTS DE CONTROLADORES - FACTORIES ===
from ..controlador.Controlador_Normalizador import (
    operacion_normalizacion,
    Norm_Global,
    MaxNorm,
)

from ..controlador.Controlador_Filtrador import (
    operacion_filtro,
    Filtro_Global,
    FFTPasaAlto,
)

from ..controlador.Controlador_Realzador import (
    operacion_realce,
    Realce_PorCorteEspaciotemporal,
    Apertura,
)

from ..controlador.Controlador_Segmentador import (
    operacion_segmentacion,
    Segmentacion_PorCorteEspaciotemporal,
    Otsu,
)


def crear_operaciones_pipeline() -> list[Operacion]:
    """
    Crea operaciones usando las factories de los controladores.
    Estas factories ya retornan objetos Operacion con callables configurados.
    """
    operaciones = []
    
    # 1. NORMALIZACIÓN (PREPROCESAMIENTO)
    op_norm = operacion_normalizacion(
        tipo=Norm_Global(),
        metodo=MaxNorm(),
        canal=0,
        nombre="max_norm_global"
    )
    operaciones.append(op_norm)
    
    # 2. FILTRACIÓN ESPECTRAL (FILTRACION)
    metodo_fft = FFTPasaAlto(radio=30)
    
    op_fft = operacion_filtro(
        metodo=metodo_fft,
        tipo=Filtro_Global(),
        canal=0,
        nombre="fft_pasaalto_30"
    )
    operaciones.append(op_fft)
    
    # 3. REALCE MORFOLÓGICO (REALZADOR)
    metodo_apertura = Apertura(
        tamanio=(3, 3),
        forma='rect',
        iteraciones=1
    )
    
    op_apertura = operacion_realce(
        metodo=metodo_apertura,
        tipo=Realce_PorCorteEspaciotemporal(),
        canal=0,
        nombre="apertura_3x3"
    )
    operaciones.append(op_apertura)
    
    # 4. SEGMENTACIÓN (SEGMENTADOR)
    metodo_otsu = Otsu()
    
    op_otsu = operacion_segmentacion(
        metodo=metodo_otsu,
        tipo=Segmentacion_PorCorteEspaciotemporal(),
        canal=0,
        nombre="otsu_auto"
    )
    operaciones.append(op_otsu)
    
    return operaciones


def ejecutar_pipeline_test(
    ruta_imagen: Path,
    canal_visualizacion: int = 0
) -> Resultado[dict, Exception]:
    """
    Ejecuta pipeline de 4 etapas y genera visualización obligatoria.
    """
    print(f"\n{'='*70}")
    print(f"TEST PIPELINE FUNCIONAL - BioImagen")
    print(f"{'='*70}")
    print(f"Archivo: {ruta_imagen}")
    
    resultados_intermedios = {}
    
    # === 1. CARGAR IMAGEN ===
    print(f"\n[1/5] CARGANDO IMAGEN...")
    ctrl_bio = ControladorBioImagen(ruta_imagen)
    
    resultado_carga = ctrl_bio.cargar_ImagenResultado(ModoImagen.AUTO)
    if resultado_carga.es_err():
        return Err(Exception(f"Error carga: {resultado_carga.error.mensaje}"))
    
    data = resultado_carga.unwrap()
    print(f"  ✓ Cargada: {data.dims}")
    print(f"  ✓ Canales: {data.canales}")
    print(f"  ✓ Rango: [{data.datos.min():.2f}, {data.datos.max():.2f}]")
    
    resultados_intermedios["original"] = data
    
    # === 2. CREAR OPERACIONES ===
    print(f"\n[2/5] CREANDO OPERACIONES...")
    try:
        operaciones = crear_operaciones_pipeline()
        
        for i, op in enumerate(operaciones):
            print(f"  {i+1}. {op}")
            print(f"     Categoría: {op.categoria}")
            print(f"     Callable: {type(op.instancia_callable)}")
            print(f"     Es callable: {callable(op.instancia_callable)}")
            
    except Exception as e:
        print(f"  ✗ Error creando operaciones: {e}")
        traceback.print_exc()
        return Err(Exception(f"Error creando operaciones: {e}"))
    
    # === 3. CONSTRUIR PIPELINE ===
    print(f"\n[3/5] CONSTRUYENDO PIPELINE...")
    
    try:
        builder = PipelineBuilder(modo_estricto=False)
        
        for op in operaciones:
            builder.agregar(op)
        
        pipeline = builder.construir(forzar=True)
        print(f"  ✓ Pipeline construido: {len(pipeline)} etapas")
        
    except Exception as e:
        print(f"\n  ✗ Error: {e}")
        traceback.print_exc()
        return Err(Exception(f"Error construcción: {e}"))
    
    # === 4. EJECUTAR ===
    print(f"\n[4/5] EJECUTANDO PIPELINE...")
    
    resultado_actual: Resultado[BioImagenData, ErrorBioImagen] = Ok(data)
    nombres_etapas = ["original"]
    
    for i, op in enumerate(pipeline.operaciones):
        print(f"\n  Paso {i+1}: {op.nombre}")
        
        if resultado_actual.es_err():
            print(f"    ⊘ Saltado por error previo")
            continue
        
        try:
            resultado_nuevo = op.ejecutar(resultado_actual.unwrap())
        except Exception as e:
            print(f"    ✗ Excepción: {e}")
            traceback.print_exc()
            resultado_actual = Err(ErrorBioImagen(f"Excepción en {op.nombre}: {e}", "EJECUCION"))
            break
        
        if resultado_nuevo.es_err():
            print(f"    ✗ Falló: {resultado_nuevo.error.mensaje}")
            resultado_actual = resultado_nuevo
            break
        
        data_nueva = resultado_nuevo.unwrap()
        print(f"    ✓ OK: shape={data_nueva.dims}, rango=[{data_nueva.datos.min():.2f}, {data_nueva.datos.max():.2f}]")
        
        resultados_intermedios[op.nombre] = data_nueva
        nombres_etapas.append(op.nombre)
        resultado_actual = resultado_nuevo
    
    if resultado_actual.es_err():
        return Err(Exception(f"Pipeline falló: {resultado_actual.error.mensaje}"))
    
    data_final = resultado_actual.unwrap()
    print(f"\n  ✓ Pipeline completado")
    
    # === 5. VISUALIZACIÓN ===
    print(f"\n[5/5] GENERANDO VISUALIZACIÓN...")
    
    def extraer_corte(data: BioImagenData, t=0, z=0, c=canal_visualizacion):
        """Extrae corte 2D para visualización."""
        if t >= data.dims.T:
            t = 0
        if z >= data.dims.Z:
            z = data.dims.Z // 2
        return data.datos[t, z, c, :, :]
    
    imagenes = [extraer_corte(resultados_intermedios[nom]) for nom in nombres_etapas]
    etiquetas = [nom.replace("_", " ").title() for nom in nombres_etapas]
    
    estetica = EsteticaGrafico(
        tema='oscuro',
        fuente='sans-serif',
        tamano_fuente=10
    )
    
    try:
        fig = panel_transformaciones(
            imagenes=imagenes,
            etiquetas=etiquetas,
            titulo_general=f"Pipeline: {ruta_imagen.name}",
            cmap='gray',
            normalizar='percentil',
            mostrar_histograma=True,
            mostrar_stats=True,
            colorbar=True,
            ancho_por_imagen=3.5,
            alto=4.0,
            fuente_titulo_general=12,
            fuente_subtitulo=9,
            estetica=estetica,
            mostrar=False
        )
        
        output_path = Path("test_pipeline_resultado.png")
        fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='black')
        print(f"  ✓ Guardado: {output_path.absolute()}")
    except Exception as e:
        print(f"  ⚠ Error visualización: {e}")
        traceback.print_exc()
        output_path = None
        fig = None
    
    print(f"\n{'='*70}")
    print("RESUMEN")
    print(f"{'='*70}")
    print(f"Etapas: {len(nombres_etapas)} ({', '.join(nombres_etapas)})")
    print(f"Resultado: {data_final.dims}")
    if output_path:
        print(f"Visualización: {output_path}")
    print(f"Estado: ✓ ÉXITO")
    
    return Ok({
        "data_final": data_final,
        "snapshots": resultados_intermedios,
        "pipeline": pipeline,
        "figura": fig
    })


def test_con_imagen_real(ruta: str):
    """Test con imagen real."""
    ruta_path = Path(ruta)
    if not ruta_path.exists():
        print(f"Error: No se encuentra {ruta}")
        return False
    
    resultado = ejecutar_pipeline_test(ruta_path, canal_visualizacion=0)
    
    if resultado.es_ok():
        print(f"\n{'='*70}")
        print("✓ TEST COMPLETADO EXITOSAMENTE")
        print(f"{'='*70}")
        return True
    else:
        print(f"\n{'='*70}")
        print("✗ TEST FALLÓ")
        print(f"{'='*70}")
        print(f"Error: {resultado.error}")
        return False


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Uso: python test_pipeline_bioimagen.py <ruta_imagen>")
        sys.exit(1)
    
    exito = test_con_imagen_real(sys.argv[1])
    sys.exit(0 if exito else 1)