## Descripción

BioPictureTools es un conjunto de herramientas escritas en Python para procesar y analizar imágenes de microscopía. Permite, entre otras cosas:

    -Detección y rastreo de núcleos
    -Análisis de fluorescencia
    -Manejo de sets de imágenes
    -Versiones compatibles con ambientes locales y Google Colab

```bash
¡AVISO! AUN EN DESARROLLO!
```

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

Controlador       _requiere_estrategia  _preprocesar  _validar_salida   _postprocesar
─────────────────────────────────────────────────────────────────────────────────────
Filtrador         base  (True)          base (2D)     base (2D==2D)     base
Realzador         base  (True)          base (2D)     base (2D==2D)     base
Normalizador      base  (True)          4D override   4D==4D override   base
Segmentador       base  (True)          base (2D)     multi-shape       squeeze+uint16
Transformador     base  (True)          base (2D)     shape-libre       base
Cuantificador     False override        4D override   DataFrame check   base (pasa-directo)
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
│   └── corrector/                        # ATENCION : NO IMPLEMENTADO TODAVIA
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
│   │   └── Filtros_Locales.py            # CajaBlur, Gaussiano, Bilateral, Mediana, DifusionAnisotropica
│   │
│   ├── espectrales/                      # Dominio frecuencial
│   │   └── Filtros_Ffts.py               # Fast Fourier Transformations : PasaBajo, PasaAlto, PasaBanda, BandStop, FiltradoNotch
│   │
│   ├── multiescala/                      # Dominio Multiescala
│   │   └── Filtros_Multiescala.py        # Diferencia Laplaciana, Diferencia Gaussiana, Wavelets, PiramideLaplaciana
│   │
│   ├── variacionales/                    # ATENCION : NO IMPLEMENTADO TODAVIA
│   │   └── total_variacion.py
│   │
│   └── noLocales/                      # Dominio No local
│        └── Filtros_NoLocales.py       # Non-local medians, Block-Matching 3D
│
│
├── realzador/                          # Su misión es la EXPLICITACIÓN (Bordes/Detalle)
│   │
│   ├── contraste/
│   │   └── Realzadores_Constraste.py       # CLAHE, Gamma, Logaritmico, Retinex, EcuacionHistograma
│   │
│   ├── convolucion/
│   │   └── Realzadores_Convolucion.py      # KernelPersonalizado, PSFSimulacion, KernelSeparable, ConvolucionFrecuencia, CorreccionBordes
│   │
│   ├── deconvolucion/
│   │   └── Realzadores_Deconvolucion.py    # Wiener, RichardsonLucy, BlindDeconvolucion, Tikhonov
│   │
│   ├── morfologicos/
│   │   └── Realzadores_Morfologicos.py     # Apertura, Cierre, Top-Hat, Bottom-Hat, Gradiente, Reconstruccion 
│   │
│   ├── afilacion/
│   │   └── Realzadores_Afilacion.py        # AfilacionLaplaciana, FiltroHighBoost, MascaraEnfoque, AfilacionGradiente, AfilacionWavelet
│   │
│   ├── estructura/                         # Vesselness filters : Son realzadores que no buscan bordes, sino "tubos" (neuritas, vasos, filamentos de actina).
│   │   └── Realzadores_Estructurales.py    # Hessiano, Frangi, Sato, TensorEstructural
│   │ 
│   └── gradientes/
│       └── Realzadores_Gradientes.py       # Laplaciano, LaplacianoCero, Canny, Sobel, Scharr, Prewitt, Roberts 
│
├── segmentador/
│   ├── binarizacion/  
│   │   └── Segmentadores_Binarizacion.py     # otsu, global, adaptativo, percentil, triangle, yen, li, isodata, minimum, mean
│   │
│   ├── instancial/
│   │   └── Segmentadores_Instanciales.py     # Watershed, WatershedMarcado, DistanciaWatershed, SplitDistancial, WatershedHibrido, SplitWatershed
│   │
│   ├── regional/
│   │   └── Segmentaodores_Regioneales.py     # RegionGrowing, RandomWalk, CorteGrafico, SuperpixelSLIC, SuperpixelFelzenszwalb, WatershedRegiones, MeanShiftSegmentacion
│   │
│   ├── contornos_activos/  # ATENCION : NO IMPLEMENTADO TODAVIA
│   │   ├── serpientes.py
│   │   └── conjuntos_nivel.py
│   │
│   └── etiquetado/         # ATENCION : NO IMPLEMENTADO TODAVIA
│       ├── connected_components.py
│       └── reetiquetado.py
│
├── transformador/                          # Rotaciones, escalado, Warp manual  
│   ├── geometricos/ 
│   │   └── Transfromadores_Geometricos.py  # TransformacionDistancia,  Esqueletizacion, EjeMedial, Deformar, Redimensionar, Rotacion, Remuestreo
│   │
│   ├── espectrales/ 
│   │   └── Transfromadores_Espectrales.py  # Fourier, Wavelet, Gabor
│   │ 
│   └── integrales/             
│        └── Transfromadores_Proyectivos.py # Radon, IntegralDeLinea, Hough, TransformadaDistanciaGeodesica, TransformadaHilbert, Abel
│
│
├── cuantificador/
│   ├── intensidad/
│   │   └── Cuantificadores_Intensidad.py # MediaIntensidad, IntensidadIntegrada, MaximoIntensidad, MinimoIntensidad, MedianaIntensidad, DesviacionEstandar, CoeficienteVariacion, PercentilIntensidad, RelacionSenialRuido, AsimetriaIntensidad, CurtosisIntensidad, PerfilLineal
│   │
│   ├── morfometria/
│   │   └── Cuantificadores_Morfometria.py # Geométricas básicas, Forma, Escala, Orientación, Intensidad de forma
│   │
│   ├── topologicos/
│   │   └── Cuantificadores_Topologicos.py # Esqueléticas, Ramificación, Contornos, Conectividad, índice de betti, grafo de adyacencia, distancia geodésica
│   │
│   ├── textura/
│   │   └── Cuantificadores_Textura.py  # GLCM, CaracteristicasHaralick, LBP, FiltrosGabor, GLRLM, EnergiaLaws
│   │
│   └── estadisticos/
│       └── Estadisticos.py             # Estadisticos , distribuciones, correlaciones
│
├── modelador/                 
│   ├── dimensionalidad/                 
│   │   └── Modelador_Dimensionalidad.py  # PCA, UMPA, tSNE
│   │
│   ├── clustering/              
│   │   └── Modelador_Clustering.py # KMneas, DBSCAN, HDBSCAN
│   │
│   ├── clasificacion/  
│   │   └── Modelador_Clasificacion.py # SVM, RegresionLogistica, RandomForest
│   │
│   ├── dinamica_temporal/              # ATENCION : NO IMPLEMENTADO TODAVIA
│   │   ├── serie_temporales.py
│   │   ├── tracking.py
│   │   └── curva_bleaching.py
│   ├── tracking/                       # ATENCION : NO IMPLEMENTADO TODAVIA
│   │   └── multi_objeto.py
│   │
│   └── ajuste/                 # ATENCION : NO IMPLEMENTADO TODAVIA
│       ├── ajuste_superficie.py
│       └── ajuste_psf.py
│
│
│
├── analizador/
│   ├── plots/
│   │   ├── Estetica.py
│   │   ├── Plots_Estadisticos.py
│   │   ├── Plots_Imagen.py
│   │   └── Plots_Modelos.py
│   │
│   ├── qc/                     # ATENCION : NO IMPLEMENTADO TODAVIA
│   │   ├── overlays.py
│   │   └── sanity_checks.py
│   │
│   └── exportacion/
│       ├── csv.py
│       ├── parquet.py
│       └── figures.py
│
└── gestorLab/
    ├── Categoria_Operacion.py
    ├── Operacion.py
    ├── Flujo_PRocesameinto.py
    ├── Flujo_Trabajo.py
    ├── Log.py
    ├── Pipeline_Builder.py
    └── Gestor_Lab.py
```

```bash
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Archivo       │────▶│  BioImage        │────▶│ BioImagenData   │
│   (.ics/.ids)   │     │  (librería bioio)│     │ (estructura) │
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
