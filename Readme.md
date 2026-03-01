## Descripción

BioPictureTools es un conjunto de herramientas escritas en Python para procesar y analizar imágenes de microscopía. Permite, entre otras cosas:

    -Detección y rastreo de núcleos
    -Análisis de fluorescencia
    -Manejo de sets de imágenes
    -Versiones compatibles con ambientes locales y Google Colab

## Características

    -Interfaz de línea de comando / scripts fáciles de usar
    -Utiliza bioio y bioio_bioformats para leer formatos comunes de bioimagen
    -Modular: varios scripts para distintos tipos de análisis

## Requisitos

Para poder usar BioPictureTools, necesitas tener:

    -Python 3.x
    -Java (OpenJDK u otra distribución compatible)
    -Dependencias de Python que puedan estar en requirements.txt (o las que los scripts usen)

## Instalación

Aquí los pasos para instalar/configurar el entorno en una PC con Linux (u otro sistema *nix).

```bash
# Ver qué versión de Java está siendo usada o dónde está instalado:
readlink -f $(which java)

# Si usás bash:
nano ~/.bashrc

# O si usás zsh:
nano ~/.zshrc

# Agregar/editar estas líneas en el archivo correspondiente:
export JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64
export PATH=$JAVA_HOME/bin:$PATH

# Luego recargar el archivo de configuración:

source ~/.bashrc
# o si usás zsh
source ~/.zshrc

```

Instala las dependencias 

```bash
pip3 install -r requirements.txt
```

## Estructura del proyecto:

```bash
┌─────────────────────────────────────────────────────────────┐
│                    PIPELINE DE BIOIMAGEN                     │
├─────────────────────────────────────────────────────────────┤
│  1. PREPROCESAMIENTO  │ Normalización, corrección iluminación │
│     (Preparar datos)  │ Flat-field, dark-current, white balance│
├─────────────────────────────────────────────────────────────┤
│  2. FILTRACIÓN        │ Reducción de ruido, suavizado         │
│     (Limpiar señal)   │ Gaussiano, mediana, Wiener, NLMeans   │
├─────────────────────────────────────────────────────────────┤
│  3. REALZADORES       │ Mejora de características visibles    │
│     (Resaltar info)   │ Contraste adaptativo, sharpening,     │
│                       │ unsharp mask, gradientes (Sobel, etc)  │
├─────────────────────────────────────────────────────────────┤
│  4. TRANSFORMADORES   │ Cambios estructurales de la imagen    │
│     (Reestructurar)   │ Rotación, registro, warping,          │
│                       │ esqueletización, distancia transform  │
├─────────────────────────────────────────────────────────────┤
│  5. SEGMENTADORES     │ Separación de objetos de interés      │
│     (Identificar ROI) │ Umbralización, watershed, region      │
│                       │ growing, modelos de contorno (snakes)  │
├─────────────────────────────────────────────────────────────┤
│  6. CUANTIFICADORES   │ Medición de propiedades de objetos    │
│     (Extraer números) │ Área, volumen, intensidad, textura,    │
│                       │ forma, conteo de células, colocalización│
├─────────────────────────────────────────────────────────────┤
│  7. MODELADORES       │ Análisis estadístico/machine learning │
│     (Inferir patrones)│ PCA, clustering, clasificación,       │
│                       │ regresión, redes neuronales            │
├─────────────────────────────────────────────────────────────┤
│  8. ANALIZADORES      │ Visualización y exportación           │
│     (Comunicar)       │ Plots, histogramas, heatmaps,         │
│                       │ export a CSV/Excel, reportes PDF       │
└─────────────────────────────────────────────────────────────┘
```
```bash
nucleo/
│
├── controlador/
│   ├── ControladorPreprocesador.py
│   ├── ControladorFiltrador.py
│   ├── ControladorRealzador.py
│   ├── ControladorTransformador.py
│   ├── ControladorModelador.py
│   ├── ControladorSegmentador.py
│   ├── ControladorExtractor.py
│   ├── ControladorAnalizador.py
│   └── Controlador_BioImagen.py # I/O, metadatos, e iteracion
│
├── preprocesador/
│   ├── normalizador/
│   │   ├── normalizador.py               # Handler de la normalizacion : por el metodo y por el corte confocal.
│   │   └── metodosNormalizacion.py       # Metodos zscore, max, mim_max, y por percentil.
│   │
│   └── corrector/                        # Cada uno con dos metodos : el real con imagenes de referencia, y el estimado (usando filtros, realzadores o modelado).
│          ├── iluminacion/
│          │    ├── flat_field.py 
│          │    ├── correccion_fondo.py 
│          │    ├── rolling_ball.py 
│          │    └── sombreado.py 
│          │ 
│          ├── artefactos/
│          │    ├── dead_pixels.py 
│          │    ├── hot_pixels.py         # Esto es mas una correccion de artefactos sencilla sin modelado.
│          │    └── striping.py 
│          │ 
│          └── deformaciones/             # Correciones mas complejas usando transformaciones y modelado rigido, afin y elastico. Utilizan modelado.
│              ├── afin.py
│              ├── rigida.py
│              ├── elastica.py
│              ├── demons.py
│              └── b_splines.py
│
│
├── filtrador/                            # Su misión es la REDUCCIÓN (Ruido/Fondo)
│   ├── locales/                          # Dominio espacial
│   │   └── Metodos_Locales.py            # CajaBlur, Gaussiano, Bilateral, Mediana, DifusionAnisotropica
│   │
│   ├── espectrales/                      # Dominio frecuencial
│   │   └── Metodos_Ffts.py               # Fast Fourier Transformations : PasaBajo, PasaAlto, PasaBanda, BandStop, FiltradoNotch
│   │
│   ├── multiescala/                      # Dominio Multiescala
│   │   └── Metodos_Multiescala.py        # Diferencia Laplaciana, Diferencia Gaussiana, Wavelets, PiramideLaplaciana
│   │
│   ├── variacionales/                    
│   │   └── total_variacion.py
│   │
│   └── noLocales/                      # Dominio No local
│        └── Metodos_NoLocales.py       # Non-local medians, Block-Matching 3D
│
│
├── realzador/                          # Su misión es la EXPLICITACIÓN (Bordes/Detalle)
│   │
│   ├── contraste/
│   │   └── Metodos_Constraste.py       # CLAHE, Gamma, Logaritmico, Retinex, EcuacionHistograma
│   │
│   ├── convolucion/
│   │   ├── kernel_personalizado.py
│   │   └── psf_simulacion.py
│   │
│   ├── deconvolucion/
│   │   ├── wiener.py
│   │   ├── richardson_lucy.py
│   │   ├── blind_deconvolucion.py
│   │   └── tikhonov.py
│   │
│   ├── morfologicos/
│   │   └── Metodos_Morfologicos.py     # Apertura, Cierre, Top-Hat, Bottom-Hat, Gradiente, Reconstruccion 
│   │
│   ├── afilacion/
│   │   └── Metodos_Afilacion.py        # AfilacionLaplaciana, FiltroHighBoost, MascaraEnfoque, AfilacionGradiente, AfilacionWavelet, DeconvolucionLucy
│   │
│   ├── estructura/                     # Vesselness filters : Son realzadores que no buscan bordes, sino "tubos" (neuritas, vasos, filamentos de actina).
│   │   └── Metodos_Estructurales.py    # Hessiano, Frangi, Sato, TensorEstructural
│   │ 
│   └── gradientes/
│       ├── laplaciano.py
│       ├── canny.py
│       ├── sobel.py
│       └── scharr.py
│
├── segmentador/
│   ├── binarizacion/
│   │   ├── metodosBinarizacion.py      # otsu, global, adaptativo, percentil
│   │   └── binarizador.py
│   │
│   ├── instancial/
│   │   ├── watershed.py
│   │   ├── marcado.py                  # watershedMarcado
│   │   ├── distancia_watersher.py
│   │   └── splitDistancial.py
│   │
│   ├── regional/
│   │   ├── region_growing.py
│   │   ├── random_walk.py
│   │   ├── corte_grafico.py
│   │   └── superpixel.py
│   │
│   ├── contornos_activos/
│   │   ├── serpientes.py
│   │   └── conjuntos_nivel.py
│   │
│   └── etiquetado/
│       ├── connected_components.py
│       └── reetiquetado.py
│
├── transformador/                  # Rotaciones, escalado, Warp manual  
│   ├── geometrico/
│   │   ├── deformar.py             # Warp
│   │   ├── redimensionar.py        # Resize
│   │   ├── rotacion.py            
│   │   └── remuestreo.py  
│   │
│   └── matematico/        
│        ├── transformacion_distancia.py
│        ├── esqueletizacion.py
│        ├── eje_medial.py            
│        ├── radon.py
│        ├── fourier.py            
│        └── tranformacion_wavelet.py      
│
├── modelador/                 
│   ├── dimensionalidad/                 
│   │   ├── pca.py
│   │   ├── umap.py
│   │   └── tsne.py
│   │
│   ├── clustering/              
│   │   ├── kmeans.py
│   │   └── dbscan.py
│   │
│   ├── clasificacion/  
│   │   ├── svm.py
│   │   └── random_forest.py
│   │
│   ├── tracking/  
│   │   └── multi_objeto.py
│   │
│   └── ajuste/  
│       ├── ajuste_superficie.py
│       └── ajuste_psf.py
│
├── extractor/
│   ├── contornos/
│   │   ├── encontrar_contornos.py
│   │   └── hull.py
│   │
│   ├── geometria/
│   │   ├── centroides.py
│   │   ├── caja_frontera.py
│   │   ├── area.py
│   │   └── diametro.py
│   │
│   ├── textura/
│   │   ├── glcm.py
│   │   ├── haralick.py
│   │   └── lbp.py
│   │
│   ├── topologia/
│   │   ├── metricas_esqueleticas.py
│   │   └── branching.py
│   │
│   └── relaciones_espaciales/
│       ├── vecinos.py
│       └── colocalizacion.py
│
├── cuantificador/
│   ├── intensidad/
│   │   ├── media.py
│   │   ├── integrada.py
│   │   ├── maximo.py
│   │   └── perfil_lineal.py
│   │
│   ├── morfometria/
│   │   ├── area.py
│   │   ├── perimetro.py
│   │   ├── dimension_fractal.py
│   │   └── circularidad.py
│   │
│   ├── dinamica_temporal/
│   │   ├── serie_temporales.py
│   │   ├── tracking.py
│   │   └── curva_bleaching.py
│   │
│   └── estadisticas_objetos/
│       ├── distribuciones.py
│       └── correlaciones.py
│
├── analizador/
│   ├── plots/
│   │   ├── histogramas.py
│   │   ├── scatter.py
│   │   └── boxplot.py
│   │
│   ├── qc/
│   │   ├── overlays.py
│   │   └── sanity_checks.py
│   │
│   └── exportacion/
│       ├── csv.py
│       ├── parquet.py
│       └── figures.py
│
└── gestorLab/
    ├── pipelinesClasicos/
    │   ├── nuclei_fluorescence.yaml
    │   └── spots_detection.yaml
    │
    ├── personalizados/
    └── validacion/
```

```bash
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Archivo       │────▶│  BioImage        │────▶│ BioImagenData   │
│   (.ics/.ids)   │     │  (librería bioio)│     │ (tu estructura) │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                                                              │
┌─────────────────┐     ┌──────────────────┐                  │
│   Archivo       │────▶│  np.ndarray      │──────────────────┘
│   (.png/.jpg)   │     │  (OpenCV)        │
└─────────────────┘     └──────────────────┘
```

## Ejemplo de uso

```bash
# Ejecutar script principal (supuesto nombre, reemplazá según tu estructura)
python main.py --input ruta/a/las/imagenes --output carpeta_de_salida

# Otro ejemplo para detección de núcleos
python RastreadorNucleos.py --input example.png --threshold 0.2
```

Aquí un ejemplo del resultado generado por BioPictureTools:

![Ejemplo de salida](output/output_example.png)

## Licencia

Este proyecto está bajo la licencia MIT. Para más detalles, revisá el archivo LICENSE.
