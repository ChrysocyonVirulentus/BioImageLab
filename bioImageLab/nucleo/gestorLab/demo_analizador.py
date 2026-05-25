"""
Demo / Ejemplo de uso del módulo analizador completo.

Este archivo muestra cómo usar:
  - Plots estadísticos, de imagen y de modelos
  - Exportación a CSV, TSV, Parquet y figuras
  - QC visual con pasos del pipeline
  - Configuración estética personalizada
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# IMPORTS DEL SISTEMA
# ---------------------------------------------------------------------------

from analizador.plots.Estetica import (
    Estetica, PaletaColores, Fuentes, Layout,
    estetica_publicacion, estetica_oscuro,
)
from analizador.plots.Plots_Estadisticos import (
    HistogramaIntensidad, BoxplotCanales, ScatterFeatures,
    HeatmapCorrelacion, CurvaROC, MatrizConfusion,
)
from analizador.plots.Plots_Imagen import (
    OverlayMascara, ComparacionCanales, StackViewer,
    OrthoView, MIPProyeccion, PerfilIntensidad,
)
from analizador.plots.Plots_Modelos import (
    BiplotPCA, ImportanciaFeatures, SeparabilidadClases,
)
from analizador.exportacion.csv import ExportadorCSV
from analizador.exportacion.tsv import ExportadorTSV
from analizador.exportacion.parquet import ExportadorParquet
from analizador.exportacion.figures import ExportadorFiguras
from analizador.qc.QC_Visualizacion import (
    ConfigQC, visualizar_pasos, comparar_antes_despues,
    reporte_qc_completo,
)
from analizador.Controlador_Analizador import Controlador_Analizador

# ---------------------------------------------------------------------------
# EJEMPLO 1: PLOT ESTADÍSTICO PERSONALIZADO
# ---------------------------------------------------------------------------

def demo_plot_estadistico():
    """Histograma con estética personalizada."""
    # Crear datos de ejemplo
    df = pd.DataFrame({
        "canal_1": np.random.normal(100, 20, 1000),
        "canal_2": np.random.normal(150, 30, 1000),
        "canal_3": np.random.exponential(50, 1000),
    })

    # Estética personalizada
    mi_estetica = Estetica(
        paleta=PaletaColores(
            primario="#FF6B6B",
            secundario="#4ECDC4",
            terciario="#45B7D1",
        ),
        fuentes=Fuentes(
            familia="sans-serif",
            tamano_titulo=16,
            tamano_etiqueta=12,
            peso_titulo="bold",
        ),
        layout=Layout(dpi=150, figsize_default=(10, 6)),
    )

    # Crear plot
    plot = HistogramaIntensidad(columna="canal_1", bins=50, kde=True)
    fig = plot(df, estetica=mi_estetica)
    # fig.savefig("histograma.png", dpi=150)
    print("✓ Histograma generado")
    return fig


# ---------------------------------------------------------------------------
# EJEMPLO 2: PLOT DE IMAGEN (BioImagenData)
# ---------------------------------------------------------------------------

def demo_plot_imagen():
    """Overlay de máscara sobre imagen."""
    # Simular BioImagenData (en producción viene de ControladorBioImagen)
    from controlador.Controlador_BioImagen import BioImagenData, Dimensiones

    np.random.seed(42)
    datos = np.random.randint(0, 65535, (1, 5, 3, 256, 256), dtype=np.uint16)
    # Crear máscara de ejemplo
    mascara = np.zeros((1, 5, 3, 256, 256), dtype=np.uint16)
    mascara[0, 2, 1, 50:150, 50:150] = 1
    mascara[0, 2, 1, 100:200, 120:220] = 2

    data = BioImagenData(
        datos=datos,
        dims=Dimensiones(T=1, Z=5, C=3, Y=256, X=256),
        canais=("Rojo", "Verde", "Azul"),
        ruta_origen=Path("demo.tif"),
        metadata={"mascara_datos": mascara},
    )

    # Plot con overlay
    est = Estetica(
        paleta=PaletaColores(categorico=("#FF0000", "#00FF00", "#0000FF")),
    )
    plot = OverlayMascara(canal=1, t=0, z=2, alpha=0.4)
    fig = plot(data, estetica=est)
    print("✓ Overlay de máscara generado")
    return fig


# ---------------------------------------------------------------------------
# EJEMPLO 3: EXPORTACIÓN
# ---------------------------------------------------------------------------

def demo_exportacion():
    """Exportar DataFrame a múltiples formatos."""
    df = pd.DataFrame({
        "celula_id": range(100),
        "area": np.random.lognormal(5, 0.5, 100),
        "intensidad_media": np.random.normal(2000, 500, 100),
        "clase": np.random.choice(["tipo_a", "tipo_b", "tipo_c"], 100),
    })

    # CSV
    exp_csv = ExportadorCSV(separador=";", index=False)
    ruta_csv = exp_csv(df, Path("/tmp/salida.csv"))
    print(f"✓ CSV exportado: {ruta_csv}")

    # TSV
    exp_tsv = ExportadorTSV(index=False)
    ruta_tsv = exp_tsv(df, Path("/tmp/salida.tsv"))
    print(f"✓ TSV exportado: {ruta_tsv}")

    # Parquet
    exp_parquet = ExportadorParquet(compresion="snappy")
    ruta_parquet = exp_parquet(df, Path("/tmp/salida.parquet"))
    print(f"✓ Parquet exportado: {ruta_parquet}")

    return ruta_csv, ruta_tsv, ruta_parquet


# ---------------------------------------------------------------------------
# EJEMPLO 4: QC — VISUALIZACIÓN DE PASOS DEL PIPELINE
# ---------------------------------------------------------------------------

def demo_qc_pasos():
    """Visualizar pasos de un pipeline de procesamiento."""
    from controlador.Controlador_BioImagen import BioImagenData, Dimensiones

    # Simular secuencia de pasos
    np.random.seed(123)
    pasos = []

    # Paso 0: Input original
    datos0 = np.random.randint(0, 65535, (1, 3, 2, 128, 128), dtype=np.uint16)
    data0 = BioImagenData(
        datos=datos0,
        dims=Dimensiones(1, 3, 2, 128, 128),
        canais=("Canal_A", "Canal_B"),
        ruta_origen=Path("input.tif"),
    )
    pasos.append(("input", data0))

    # Paso 1: Después de filtrado gaussiano
    datos1 = datos0.astype(np.float32)
    from scipy.ndimage import gaussian_filter
    for t in range(datos1.shape[0]):
        for z in range(datos1.shape[1]):
            for c in range(datos1.shape[2]):
                datos1[t, z, c] = gaussian_filter(datos1[t, z, c], sigma=1.5)
    datos1 = datos1.astype(np.uint16)
    data1 = BioImagenData(
        datos=datos1,
        dims=Dimensiones(1, 3, 2, 128, 128),
        canais=("Canal_A", "Canal_B"),
        ruta_origen=Path("input.tif"),
    )
    pasos.append(("gaussian_filter", data1))

    # Paso 2: Después de segmentación (máscara en metadata)
    mascara = np.zeros_like(datos1)
    mascara[0, 1, 0, 30:90, 30:90] = 1
    mascara[0, 1, 0, 60:110, 70:120] = 2
    data2 = BioImagenData(
        datos=datos1,
        dims=Dimensiones(1, 3, 2, 128, 128),
        canais=("Canal_A", "Canal_B"),
        ruta_origen=Path("input.tif"),
        metadata={"mascara_datos": mascara},
    )
    pasos.append(("segmentacion", data2))

    # Configuración QC
    config = ConfigQC(
        canal=0, t=0, z=1,
        mostrar_mascara=True,
        estetica=Estetica(),
    )

    # Visualizar pasos
    res = visualizar_pasos(pasos, config=config, titulo="QC — Pipeline Demo")
    if res.es_ok():
        fig = res.unwrap()
        print("✓ QC pasos generado")
        return fig
    else:
        print(f"✗ Error QC: {res.error}")
        return None


# ---------------------------------------------------------------------------
# EJEMPLO 5: CONTROLADOR ANALIZADOR COMPLETO
# ---------------------------------------------------------------------------

def demo_controlador_completo():
    """Uso completo del Controlador_Analizador."""
    ctrl = Controlador_Analizador()

    # Cambiar estética
    ctrl.cambiar_estetica(estetica_publicacion())

    # Crear operaciones para pipeline
    op_hist = ctrl.crear_operacion_histograma_intensidad(
        columna="intensidad", bins=50, nombre="hist_intensidad"
    )
    op_box = ctrl.crear_operacion_boxplot_canales(
        columnas=["canal_1", "canal_2"], nombre="box_canales"
    )
    op_export = ctrl.crear_operacion_exportar_csv(
        ruta_salida=Path("/tmp/resultados.csv"), separador=";"
    )

    print(f"✓ Operaciones creadas: {op_hist.nombre}, {op_box.nombre}, {op_export.nombre}")
    print(f"  Categorías: {op_hist.categoria}, {op_box.categoria}, {op_export.categoria}")

    return ctrl


# ---------------------------------------------------------------------------
# EJECUCIÓN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("DEMO — Módulo Analizador Completo")
    print("=" * 60)

    # Nota: Los plots requieren matplotlib instalado
    # demo_plot_estadistico()
    # demo_plot_imagen()
    demo_exportacion()
    # demo_qc_pasos()
    demo_controlador_completo()

    print("\n✅ Demo completado")
