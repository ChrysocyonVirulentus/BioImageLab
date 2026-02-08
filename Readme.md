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
│   └── ControladorBioImagen.py # I/O, metadatos, e iteracion
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
│   │   ├── gaussiano.py
│   │   ├── mediana.py
│   │   ├── difusionAnisotropica.py
│   │   ├── cajaBlur.py
│   │   └── bilateral.py
│   │
│   ├── espectrales/                      # Dominio frecuencial
│   │   ├── metodos_ffts.py
│   │   └── filtradoNotch.py
│   │
│   ├── multiescala/                      # Dominio Multiescala
│   │   ├── diferenciaGaussiana.py
│   │   ├── diferenciaLaplaciana.py
│   │   ├── piramedesLaplacianas.py
│   │   └── wavelets.py
│   │
│   ├── variacionales/                    
│   │   └── total_variacion.py
│   │
│   └── noLocales/                        # Dominio No local
│        ├── bm3d.py
│        └── nlm.py                       # Non-local medians
│
│
├── realzador/                            # Su misión es la EXPLICITACIÓN (Bordes/Detalle)
│   │
│   ├── contraste/
│   │   ├── clahe.py
│   │   ├── gamma.py
│   │   ├── log.py
│   │   ├── retinex.py
│   │   └── hist_eq.py
│   │
│   ├── convolucion/
│   │   ├── kernel_personalizado.py
│   │   └── psf_simulacion.py
│   │
│   │
│   ├── deconvolucion/
│   │   ├── wiener.py
│   │   ├── richardson_lucy.py
│   │   ├── blind_deconvolucion.py
│   │   └── tikhonov.py
│   │
│   ├── morfologicos/
│   │   ├── apertura.py
│   │   ├── cierre.py
│   │   ├── top_hat.py
│   │   ├── bottom_hat.py
│   │   ├── reconstruccion_morfologica.py
│   │   └── gradienteMorfologico.py
│   │
│   ├── afilacion/
│   │   ├── afilacionLaplaciana.py
│   │   ├── filtroHighBoost.py
│   │   └── mascaraEnforque.py
│   │
│   ├── estructura/                     # Vesselness filters : Son realzadores que no buscan bordes, sino "tubos" (neuritas, vasos, filamentos de actina).
│   │   ├── hessiano.py
│   │   ├── frangi.py
│   │   ├── sato.py
│   │   └── tensor_estructura.py
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
