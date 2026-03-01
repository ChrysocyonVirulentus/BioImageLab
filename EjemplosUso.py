## Uso Filtro Locales

from filtros_locales import Gaussiano, Mediana, Bilateral, DifusionAnisotropica

# Filtro gaussiano
filtro1 = Gaussiano(sigma=2.0, mascara=(5, 5))
img_suave = filtro1(img)

# Filtro de mediana para ruido sal y pimienta
filtro2 = Mediana(mascara=5)
img_sin_ruido = filtro2(img)

# Bilateral para preservar bordes
filtro3 = Bilateral(diam=9, sigma_color=75, sigma_espacio=75)
img_bordes_nitidos = filtro3(img)

# Difusión anisotrópica para microscopía
filtro4 = DifusionAnisotropica(n_iter=15, kappa=50, gamma=0.15, opcion=1)
img_mejorada = filtro4(img)

## Filtro espectral

from filtros_locales import Gaussiano, Bilateral
from filtros_espectrales import FFTPasabajo, FiltradoNotch

# 1. Eliminar ruido periódico específico (dominio frecuencial)
notch = FiltradoNotch(puntos_ruido=[(25, 30), (40, 15)], radio=5)
img_sin_periodico = notch(img)

# 2. Suavizado general (dominio frecuencial)
pasabajo = FFTPasabajo(radio=30)
img_suave_fft = pasabajo(img_sin_periodico)

# 3. Preservar bordes (dominio espacial)
bilateral = Bilateral(diam=9, sigma_color=75, sigma_espacio=75)
img_final = bilateral(img_suave_fft)

## Filtro multiescala

from filtros_multiescala import (
    DiferenciaGaussiana, 
    DiferenciaLaplaciana,
    PiramideLaplaciana, 
    Wavelet
)

# --- 1. Detección de núcleos con DoG ---
dog = DiferenciaGaussiana(sigma1=2.0, k=1.6)  # k=1.6 es óptimo en Scale-Space
img_nucleos = dog(img)

# --- 2. Detección precisa con DoL ---
dol = DiferenciaLaplaciana(sigma1=1.5, sigma2=3.0, ksize=3)
img_spots = dol(img)

# --- 3. Análisis multiescala con Pirámide Laplaciana ---
piramide = PiramideLaplaciana(niveles=4)
niveles = piramide(img)  # [L0, L1, L2, L3, Residuo]

# Visualizar nivel específico
import matplotlib.pyplot as plt
plt.imshow(niveles[0], cmap='gray')  # Detalles más finos
plt.imshow(niveles[2], cmap='gray')  # Detalles medios

# Reconstruir imagen original
img_reconstruida = piramide.reconstruir()

# --- 4. Análisis Wavelet ---
wavelet = Wavelet(wavelet='db4', nivel=3, modo='symmetric')
cA, detalles = wavelet(img)

# cA = aproximación (bajas frecuencias)
# detalles = [(cH1, cV1, cD1), (cH2, cV2, cD2), (cH3, cV3, cD3)]

# Denoising con wavelets
img_limpia = wavelet.denoising(img, umbral=30.0)

# Modificar coeficientes y reconstruir
cA_modificada = cA * 1.5  # Realzar bajas frecuencias
coefs_modificados = [cA_modificada] + detalles
img_realzada = wavelet.reconstruir(coefs_modificados)

## Filtros no locales

from filtros_no_locales import NonLocalMeans, BM3D, NonLocalMeansMultiescala

# --- 1. Non-Local Means básico ---
nlm = NonLocalMeans(
    h=15.0,                      # Fuerza del denoising
    template_window_size=7,       # Tamaño del parche
    search_window_size=21         # Ventana de búsqueda
)
img_limpia = nlm(img_confocal)

# --- 2. BM3D para máxima calidad ---
bm3d = BM3D(
    sigma_psd=30.0,              # Nivel de ruido estimado
    stage_arg='all'              # Algoritmo completo
)

# Estimar sigma automáticamente
sigma_estimado = bm3d.estimar_sigma(img_ruidosa, metodo='mad')
print(f"Ruido estimado: {sigma_estimado:.2f}")

# Aplicar con sigma estimado
bm3d_auto = BM3D(sigma_psd=sigma_estimado, stage_arg='all')
img_super_limpia = bm3d_auto(img_ruidosa)

# --- 3. NLM Multiescala para imágenes complejas ---
nlm_multi = NonLocalMeansMultiescala(
    escalas=3,
    h_base=10.0,
    template_window_size=7,
    search_window_size=21
)
img_multiescala = nlm_multi(img_compleja)

from filtros_locales import Bilateral
from filtros_multiescala import Wavelet
from filtros_no_locales import BM3D, NonLocalMeans

# Pipeline agresivo para máxima calidad
def pipeline_denoising_premium(img):
    # 1. BM3D para eliminar ruido principal
    bm3d = BM3D(sigma_psd=30, stage_arg='all')
    img_bm3d = bm3d(img)
    
    # 2. NLM para refinar texturas
    nlm = NonLocalMeans(h=8, template_window_size=5, search_window_size=15)
    img_nlm = nlm(img_bm3d)
    
    # 3. Bilateral suave para suavizado final
    bilateral = Bilateral(diam=5, sigma_color=50, sigma_espacio=50)
    img_final = bilateral(img_nlm)
    
    return img_final

# Pipeline rápido para tiempo real
def pipeline_denoising_rapido(img):
    # Solo NLM con parámetros optimizados
    nlm = NonLocalMeans(h=10, template_window_size=5, search_window_size=15)
    return nlm(img)

# Aplicar
img_premium = pipeline_denoising_premium(img_ruidosa)
img_rapido = pipeline_denoising_rapido(img_ruidosa)

# Realzador Morfologico

from operadores_morfologicos import (
    Apertura, Cierre, TopHat, BottomHat,
    GradienteMorfologico, ReconstruccionMorfologica
)

# --- 1. Limpieza de segmentación de núcleos ---
apertura = Apertura(tamaño=(5, 5), forma='elipse', iteraciones=1)
nucleos_limpios = apertura(nucleos_binarios)

# --- 2. Rellenar huecos en células ---
cierre = Cierre(tamaño=(7, 7), forma='elipse', iteraciones=2)
celulas_completas = cierre(celulas_fragmentadas)

# --- 3. Detección de puncta de fluorescencia ---
tophat = TopHat(tamaño=(11, 11), forma='elipse')
puncta = tophat(img_fluorescencia)

# --- 4. Detección de vacuolas ---
bottomhat = BottomHat(tamaño=(15, 15), forma='elipse')
vacuolas = bottomhat(img_campo_claro)

# --- 5. Extracción de bordes celulares ---
gradiente = GradienteMorfologico(
    tamaño=(3, 3), 
    forma='elipse', 
    tipo='basico'
)
bordes = gradiente(celulas_binarias)

# --- 6. Selección de células específicas ---
reconstruccion = ReconstruccionMorfologica(conectividad=8)

# Marcadores: solo células de interés (ej: después de filtrado por área)
marcador = seleccionar_celulas_grandes(celulas_binarias)
# Máscara: todas las células
mascara = celulas_binarias
# Reconstruir solo las marcadas
celulas_seleccionadas = reconstruccion(marcador, mascara, tipo='dilatacion')


import matplotlib.pyplot as plt

# Cargar imagen de prueba
img = cargar_imagen_nucleos()

# Aplicar diferentes operadores
apertura = Apertura(tamaño=(7, 7))
cierre = Cierre(tamaño=(7, 7))
tophat = TopHat(tamaño=(21, 21))
gradiente = GradienteMorfologico(tamaño=(5, 5))

# Visualizar
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes[0, 0].imshow(img, cmap='gray')
axes[0, 0].set_title('Original')
axes[0, 1].imshow(apertura(img), cmap='gray')
axes[0, 1].set_title('Apertura')
axes[0, 2].imshow(cierre(img), cmap='gray')
axes[0, 2].set_title('Cierre')
axes[1, 0].imshow(tophat(img), cmap='gray')
axes[1, 0].set_title('Top-Hat')
axes[1, 1].imshow(gradiente(img), cmap='gray')
axes[1, 1].set_title('Gradiente')
plt.tight_layout()
plt.show()



# --- Pipeline 1: Segmentación de núcleos ---
def segmentar_nucleos(img_dapi):
    from operadores_morfologicos import TopHat, Apertura, Cierre
    from segmentador import Otsu
    
    # 1. Realzar núcleos con Top-Hat
    tophat = TopHat(tamaño=(21, 21), forma='elipse')
    nucleos_realzados = tophat(img_dapi)
    
    # 2. Binarizar
    otsu = Otsu()
    binaria = otsu(nucleos_realzados)
    
    # 3. Apertura para limpiar
    apertura = Apertura(tamaño=(3, 3), forma='elipse')
    binaria_limpia = apertura(binaria)
    
    # 4. Cierre para rellenar huecos
    cierre = Cierre(tamaño=(5, 5), forma='elipse')
    nucleos_finales = cierre(binaria_limpia)
    
    return nucleos_finales

# --- Pipeline 2: Detección de membranas ---
def detectar_membranas(img_membrana):
    from operadores_morfologicos import GradienteMorfologico, Apertura
    
    # 1. Gradiente morfológico para bordes
    gradiente = GradienteMorfologico(tamaño=(3, 3), forma='elipse')
    bordes = gradiente(img_membrana)
    
    # 2. Binarizar bordes
    _, bordes_binarios = cv2.threshold(bordes, 0, 255, cv2.THRESH_OTSU)
    
    # 3. Limpiar con apertura pequeña
    apertura = Apertura(tamaño=(2, 2), forma='cruz')
    membranas = apertura(bordes_binarios)
    
    return membranas

# --- Pipeline 3: Eliminar objetos tocando bordes ---
def eliminar_objetos_borde(img_binaria):
    from operadores_morfologicos import ReconstruccionMorfologica
    import numpy as np
    
    # Crear marcador: solo píxeles en los bordes
    h, w = img_binaria.shape
    marcador = np.zeros_like(img_binaria)
    marcador[0, :] = img_binaria[0, :]      # Borde superior
    marcador[-1, :] = img_binaria[-1, :]    # Borde inferior
    marcador[:, 0] = img_binaria[:, 0]      # Borde izquierdo
    marcador[:, -1] = img_binaria[:, -1]    # Borde derecho
    
    # Reconstruir objetos conectados a bordes
    reconstruccion = ReconstruccionMorfologica(conectividad=8)
    objetos_borde = reconstruccion(marcador, img_binaria)
    
    # Restar para obtener solo objetos internos
    objetos_internos = cv2.subtract(img_binaria, objetos_borde)
    
    return objetos_internos

# --- Pipeline 4: H-maxima (máximos locales robustos) ---
def h_maxima(img, h=10):
    """
    Detecta máximos locales con altura mínima h.
    Útil para detección robusta de núcleos/puncta.
    """
    from operadores_morfologicos import ReconstruccionMorfologica
    
    # Marcador: imagen - h
    marcador = cv2.subtract(img, h)
    
    # Reconstruir por dilatación
    reconstruccion = ReconstruccionMorfologica(conectividad=8)
    reconstruida = reconstruccion(marcador, img, tipo='dilatacion')
    
    # H-maxima = Original - Reconstruida
    h_max = cv2.subtract(img, reconstruida)
    
    return h_max

# --- Pipeline 5: Separación de objetos tocantes ---
def separar_objetos_tocantes(img_binaria):
    from operadores_morfologicos import Apertura, ReconstruccionMorfologica
    from scipy import ndimage
    
    # 1. Apertura para crear marcadores
    apertura = Apertura(tamaño=(5, 5), forma='elipse', iteraciones=2)
    marcadores = apertura(img_binaria)
    
    # 2. Etiquetar marcadores
    marcadores_etiquetados, num_objetos = ndimage.label(marcadores)
    
    # 3. Reconstruir cada objeto por separado
    reconstruccion = ReconstruccionMorfologica(conectividad=8)
    resultado = np.zeros_like(img_binaria)
    
    for i in range(1, num_objetos + 1):
        marcador_i = (marcadores_etiquetados == i).astype(np.uint8) * 255
        objeto_i = reconstruccion(marcador_i, img_binaria)
        resultado = cv2.bitwise_or(resultado, objeto_i)
    
    return resultado


## Realce de constraste

from metodos_contraste import (
    CLAHE, Gamma, Logaritmico, Retinex, EcualizacionHistograma
)

# --- 1. CLAHE para campo claro con iluminación desigual ---
clahe = CLAHE(clip_limit=2.0, tile_grid_size=(8, 8))
img_realzada = clahe(img_brightfield)

# --- 2. Gamma para ajuste rápido de brillo ---
# Aclarar señales débiles
gamma_aclara = Gamma(gamma=0.7)
img_clara = gamma_aclara(img_debil)

# Oscurecer sobre-exposición
gamma_oscurece = Gamma(gamma=1.5)
img_oscura = gamma_oscurece(img_saturada)

# --- 3. Logarítmico para alto rango dinámico ---
log_transform = Logaritmico(c=None)  # c automático
img_comprimida = log_transform(img_hdr)

# --- 4. Retinex para corrección de iluminación ---
# Single Scale Retinex
ssr = Retinex(sigma=250, multi_escala=False)
img_corregida = ssr(img_vignetting)

# Multi Scale Retinex (más robusto)
msr = Retinex(
    sigma=250,  # Ignorado en multi-escala
    multi_escala=True,
    sigmas=(15, 80, 250)  # Pequeño, medio, grande
)
img_msr = msr(img_brightfield)

# --- 5. Ecualización simple ---
eq_hist = EcualizacionHistograma()
img_ecualizada = eq_hist(img_bajo_contraste)

import matplotlib.pyplot as plt

# Cargar imagen de prueba
img = cargar_imagen_bajo_contraste()

# Aplicar todos los métodos
clahe = CLAHE(clip_limit=2.0, tile_grid_size=(8, 8))
gamma = Gamma(gamma=0.7)
logaritmico = Logaritmico()
retinex = Retinex(sigma=250)
eq_hist = EcualizacionHistograma()

# Visualizar
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

axes[0, 0].imshow(img, cmap='gray')
axes[0, 0].set_title('Original')
axes[0, 0].hist(img.ravel(), bins=256, alpha=0.3)

axes[0, 1].imshow(clahe(img), cmap='gray')
axes[0, 1].set_title('CLAHE')

axes[0, 2].imshow(gamma(img), cmap='gray')
axes[0, 2].set_title('Gamma (γ=0.7)')

axes[1, 0].imshow(logaritmico(img), cmap='gray')
axes[1, 0].set_title('Logarítmico')

axes[1, 1].imshow(retinex(img), cmap='gray')
axes[1, 1].set_title('Retinex (MSR)')

axes[1, 2].imshow(eq_hist(img), cmap='gray')
axes[1, 2].set_title('Ecualización Histograma')

plt.tight_layout()
plt.show()


# --- Pipeline 1: Brightfield con iluminación desigual ---
def mejorar_brightfield(img):
    # 1. Corrección de iluminación con Retinex
    retinex = Retinex(sigma=300, multi_escala=True)
    img_corregida = retinex(img)
    
    # 2. Realce de contraste con CLAHE
    clahe = CLAHE(clip_limit=2.0, tile_grid_size=(8, 8))
    img_final = clahe(img_corregida)
    
    return img_final

# --- Pipeline 2: Fluorescencia débil ---
def realzar_fluorescencia_debil(img):
    # 1. Corrección gamma para aclarar
    gamma = Gamma(gamma=0.6)
    img_clara = gamma(img)
    
    # 2. CLAHE moderado
    clahe = CLAHE(clip_limit=1.5, tile_grid_size=(16, 16))
    img_realzada = clahe(img_clara)
    
    return img_realzada

# --- Pipeline 3: Time-lapse con bleaching ---
def corregir_bleaching_timelapse(frames):
    """
    Corrige pérdida de intensidad por bleaching en time-lapse.
    """
    gamma = Gamma(gamma=0.8)
    clahe = CLAHE(clip_limit=2.0, tile_grid_size=(8, 8))
    
    frames_corregidos = []
    for i, frame in enumerate(frames):
        # Corrección progresiva: más gamma conforme avanza el tiempo
        gamma_frame = Gamma(gamma=0.8 - 0.01 * i)  # Aclara más con el tiempo
        frame_corregido = gamma_frame(frame)
        frame_corregido = clahe(frame_corregido)
        frames_corregidos.append(frame_corregido)
    
    return frames_corregidos

# --- Pipeline 4: Visualización óptima ---
def preparar_para_visualizacion(img, tipo='auto'):
    """
    Prepara imagen para visualización óptima.
    """
    if tipo == 'auto':
        # Detectar si es bajo contraste
        std = np.std(img)
        rango = img.max() - img.min()
        
        if std < 30 and rango < 100:  # Muy bajo contraste
            clahe = CLAHE(clip_limit=3.0, tile_grid_size=(8, 8))
            return clahe(img)
        else:
            gamma = Gamma(gamma=0.85)  # Ligero realce
            return gamma(img)
    
    elif tipo == 'brightfield':
        return mejorar_brightfield(img)
    
    elif tipo == 'fluorescencia':
        return realzar_fluorescencia_debil(img)

# --- Pipeline 5: Análisis cuantitativo (mínimo procesamiento) ---
def preparar_para_analisis(img):
    """
    Realce mínimo para análisis cuantitativo (preservar valores).
    """
    # Solo CLAHE moderado (preserva mejor intensidades relativas)
    clahe = CLAHE(clip_limit=1.5, tile_grid_size=(16, 16))
    return clahe(img)

from normalizador import Normalizador, Norm_Global
from metodosNormalizacion import PercentilNorm
from metodos_contraste import CLAHE, Gamma

# 1. NORMALIZAR PRIMERO (si es necesario)
norm = Normalizador(tipo=Norm_Global(), metodo=PercentilNorm(2, 98))
img_normalizada = norm(img_raw, canal=0)  # → float64 [0, 1]

# 2. Convertir a uint8 para realzadores que lo requieren
img_uint8 = (img_normalizada[0, 0, 0, :, :] * 255).astype(np.uint8)

# 3. REALZAR (trabaja con valores tal como vienen)
clahe = CLAHE(clip_limit=2.0, tile_grid_size=(8, 8))
img_realzada = clahe(img_uint8)  # uint8 → uint8

# O para Gamma (no requiere uint8)
gamma = Gamma(gamma=0.7)
img_gamma = gamma(img_uint8)  # uint8 → uint8