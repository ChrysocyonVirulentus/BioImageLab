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

from metodos_binarizacion import (
    Otsu, Global, Adaptativo, Percentil,
    Triangle, Mean, Isodata, Minimum
)

# --- 1. Otsu (automático, óptimo) ---
otsu = Otsu()
umbral, img_binaria = otsu(img_nucleos)
print(f"Umbral Otsu: {umbral:.2f}")

# --- 2. Global (manual, control total) ---
global_bin = Global(umbral=128, invertir=False)
umbral, img_binaria = global_bin(img)

# --- 3. Adaptativo (iluminación desigual) ---
adaptativo = Adaptativo(
    tamaño_ventana=15,
    metodo='gaussian',  # o 'mean'
    C=2.0
)
_, img_binaria = adaptativo(img_brightfield)

# --- 4. Percentil (control por porcentaje) ---
percentil = Percentil(percentil=95)  # Top 5% más brillante
umbral, img_binaria = percentil(img_spots)

# --- 5. Triangle (histograma asimétrico) ---
triangle = Triangle()
umbral, img_binaria = triangle(img_asimetrica)

# --- 6. Mean (rápido y simple) ---
mean = Mean()
umbral, img_binaria = mean(img)

# --- 7. Isodata (iterativo robusto) ---
isodata = Isodata(max_iter=100, tol=0.5)
umbral, img_binaria = isodata(img_celulas)

# --- 8. Minimum (mínimo entre picos) ---
minimum = Minimum(suavizado=5)
umbral, img_binaria = minimum(img_bimodal)

import matplotlib.pyplot as plt

img = cargar_imagen_nucleos()

metodos = {
    'Otsu': Otsu(),
    'Percentil 90': Percentil(90),
    'Triangle': Triangle(),
    'Adaptativo': Adaptativo(15, 'gaussian', 2),
    'Isodata': Isodata(),
    'Mean': Mean()
}

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.ravel()

for i, (nombre, metodo) in enumerate(metodos.items()):
    umbral, img_bin = metodo(img)
    axes[i].imshow(img_bin, cmap='gray')
    if umbral is not None:
        axes[i].set_title(f'{nombre}\nUmbral: {umbral:.1f}')
    else:
        axes[i].set_title(f'{nombre}\n(Umbral adaptativo)')
    axes[i].axis('off')

plt.tight_layout()
plt.show()

def visualizar_umbrales(img):
    """Visualiza imagen, histograma y diferentes umbrales."""
    metodos = {
        'Otsu': Otsu(),
        'Triangle': Triangle(),
        'Mean': Mean(),
        'Percentil 90': Percentil(90)
    }
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Imagen original
    axes[0].imshow(img, cmap='gray')
    axes[0].set_title('Imagen Original')
    axes[0].axis('off')
    
    # Histograma con umbrales marcados
    axes[1].hist(img.ravel(), bins=256, alpha=0.7, color='blue')
    axes[1].set_title('Histograma con Umbrales')
    axes[1].set_xlabel('Intensidad')
    axes[1].set_ylabel('Frecuencia')
    
    colores = ['red', 'green', 'orange', 'purple']
    for (nombre, metodo), color in zip(metodos.items(), colores):
        umbral, _ = metodo(img)
        if umbral is not None:
            axes[1].axvline(umbral, color=color, linestyle='--', 
                          linewidth=2, label=f'{nombre}: {umbral:.1f}')
    
    axes[1].legend()
    plt.tight_layout()
    plt.show()

visualizar_umbrales(img_test)

# --- Pipeline 1: Segmentación de núcleos DAPI ---
def segmentar_nucleos_dapi(img):
    from operadores_morfologicos import Apertura, Cierre
    
    # 1. Binarizar con Otsu
    otsu = Otsu()
    _, img_bin = otsu(img)
    
    # 2. Limpiar con morfología
    apertura = Apertura(tamaño=(3, 3), forma='elipse')
    img_limpia = apertura(img_bin)
    
    cierre = Cierre(tamaño=(5, 5), forma='elipse')
    img_final = cierre(img_limpia)
    
    return img_final

# --- Pipeline 2: Brightfield con iluminación desigual ---
def segmentar_brightfield(img):
    # Adaptativo es mejor para iluminación variable
    adaptativo = Adaptativo(
        tamaño_ventana=21,
        metodo='gaussian',
        C=5.0
    )
    _, img_bin = adaptativo(img)
    return img_bin

# --- Pipeline 3: Detección de spots ---
def detectar_spots(img, sensibilidad='medio'):
    """
    Detecta spots usando percentil.
    
    Args:
        sensibilidad: 'alto'=98, 'medio'=95, 'bajo'=90
    """
    percentiles = {'alto': 98, 'medio': 95, 'bajo': 90}
    p = percentiles.get(sensibilidad, 95)
    
    percentil = Percentil(percentil=p)
    umbral, img_spots = percentil(img)
    
    print(f"Umbral (percentil {p}): {umbral:.2f}")
    return img_spots

# --- Pipeline 4: Comparación de métodos automáticos ---
def comparar_metodos_automaticos(img):
    """
    Compara diferentes métodos automáticos y retorna el mejor.
    """
    metodos = {
        'Otsu': Otsu(),
        'Triangle': Triangle(),
        'Isodata': Isodata(),
    }
    
    resultados = {}
    for nombre, metodo in metodos.items():
        umbral, img_bin = metodo(img)
        # Calcular métricas (ej: % de píxeles blancos)
        porcentaje_objetos = (img_bin == 255).sum() / img_bin.size * 100
        resultados[nombre] = {
            'umbral': umbral,
            'binaria': img_bin,
            'porcentaje': porcentaje_objetos
        }
    
    return resultados

# --- Pipeline 5: Binarización robusta con validación ---
def binarizar_con_validacion(img, metodo_preferido='otsu'):
    """
    Binariza con validación de calidad.
    """
    metodos_backup = [
        ('otsu', Otsu()),
        ('triangle', Triangle()),
        ('isodata', Isodata()),
    ]
    
    for nombre, metodo in metodos_backup:
        umbral, img_bin = metodo(img)
        
        # Validar resultado
        porcentaje = (img_bin == 255).sum() / img_bin.size * 100
        
        # Criterio: objetos deben ser 5-70% de la imagen
        if 5 <= porcentaje <= 70:
            print(f"Método '{nombre}' exitoso: {porcentaje:.1f}% objetos")
            return umbral, img_bin
        else:
            print(f"Método '{nombre}' descartado: {porcentaje:.1f}% objetos")
    
    # Si todos fallan, usar percentil 90 como fallback
    print("Usando Percentil 90 como fallback")
    return Percentil(90)(img)

def analizar_binarizacion(img, metodo):
    """Analiza la calidad de la binarización."""
    umbral, img_bin = metodo(img)
    
    # Métricas
    total_pixeles = img.size
    pixeles_objeto = (img_bin == 255).sum()
    pixeles_fondo = (img_bin == 0).sum()
    
    porcentaje_objeto = pixeles_objeto / total_pixeles * 100
    porcentaje_fondo = pixeles_fondo / total_pixeles * 100
    
    # Intensidades promedio
    if pixeles_objeto > 0:
        intensidad_objeto = img[img_bin == 255].mean()
    else:
        intensidad_objeto = 0
    
    if pixeles_fondo > 0:
        intensidad_fondo = img[img_bin == 0].mean()
    else:
        intensidad_fondo = 0
    
    separacion = abs(intensidad_objeto - intensidad_fondo)
    
    print(f"Umbral: {umbral}")
    print(f"Objetos: {porcentaje_objeto:.1f}%")
    print(f"Fondo: {porcentaje_fondo:.1f}%")
    print(f"Intensidad objetos: {intensidad_objeto:.1f}")
    print(f"Intensidad fondo: {intensidad_fondo:.1f}")
    print(f"Separación: {separacion:.1f}")
    
    return {
        'umbral': umbral,
        'porcentaje_objeto': porcentaje_objeto,
        'separacion': separacion
    }

    # =============================================================================
# EJEMPLO DE USO - ESTADISTICOS
# =============================================================================

def ejemplo_uso():
    """
    Ejemplo de uso del módulo de estadísticas.
    """
    # Simular datos de 5 imágenes con métricas morfométricas
    np.random.seed(42)
    
    datos_ejemplo = {}
    for i in range(5):
        n_objetos = np.random.randint(50, 150)
        datos_ejemplo[f"imagen_{i+1}"] = {
            'area': np.random.lognormal(5, 0.5, n_objetos).tolist(),
            'perimetro': np.random.lognormal(4, 0.4, n_objetos).tolist(),
            'circularidad': np.random.beta(2, 5, n_objetos).tolist(),
            'excentricidad': np.random.beta(2, 2, n_objetos).tolist(),
            'convexidad': np.random.beta(8, 2, n_objetos).tolist(),
        }
    
    # Ejecutar pipeline completo
    resultados = pipeline_estadistico_completo(datos_ejemplo)
    
    print("\n=== ESTADÍSTICOS DESCRIPTIVOS ===")
    print(resultados['descriptivos'].head())
    
    print("\n=== DISTRIBUCIONES ===")
    print(resultados['distribuciones'][['mejor_distribucion', 'es_normal_95']].head())
    
    print("\n=== CORRELACIONES ===")
    print(resultados['correlaciones'].round(2))
    
    return resultados


if __name__ == "__main__":
    resultado = ejemplo_uso()

    # =============================================================================
# 5. EJEMPLO DE USO - CLUSTERING
# =============================================================================

def ejemplo_uso():
    """
    Ejemplo completo de uso del módulo de clustering.
    """
    from sklearn.datasets import make_blobs
    
    # Generar datos sintéticos (simulando métricas de células)
    np.random.seed(42)
    X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=1.0, random_state=42)
    
    # Crear DataFrame de ejemplo
    df = pd.DataFrame({
        'imagen': [f'celula_{i:03d}' for i in range(len(X))],
        'grupo_exp': np.random.choice(['control', 'tratamiento'], len(X)),
        'feature_1': X[:, 0],
        'feature_2': X[:, 1],
        'area': np.random.lognormal(5, 0.5, len(X)),
    })
    
    print("=" * 60)
    print("EJEMPLO: Pipeline de Clustering")
    print("=" * 60)
    
    # 1. K-Means con selección automática de k
    print("\n--- K-Means con selección automática de k ---")
    resultado_kmeans = pipeline_clustering_completo(
        df,
        columnas_features=['feature_1', 'feature_2'],
        metodo='kmeans',
        params_clustering={'seleccionar_k_auto': True, 'selector_k': SelectorK(k_max=6)},
        columna_id='imagen'
    )
    
    print(f"\nMétricas K-Means:")
    for k, v in resultado_kmeans['metricas'].items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    
    # 2. DBSCAN con eps automático
    print("\n--- DBSCAN con eps automático ---")
    resultado_dbscan = pipeline_clustering_completo(
        df,
        columnas_features=['feature_1', 'feature_2'],
        metodo='dbscan',
        params_clustering={'calcular_eps_auto': True, 'k_vecinos': 4}
    )
    
    print(f"\nMétricas DBSCAN:")
    for k, v in resultado_dbscan['metricas'].items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    
    # 3. Comparación de métodos
    print("\n--- Comparación de métodos ---")
    from sklearn.preprocessing import StandardScaler
    X_std = StandardScaler().fit_transform(df[['feature_1', 'feature_2']].values)
    
    comparacion = comparar_metodos_clustering(X_std, k_min=2, k_max=5)
    print(comparacion[['metodo', 'n_clusters', 'silhouette_score', 'parametros']])
    
    # Retornar DataFrames de ejemplo
    return {
        'df_kmeans': resultado_kmeans['dataframe'],
        'df_dbscan': resultado_dbscan['dataframe'],
        'comparacion': comparacion
    }


if __name__ == "__main__":
    resultados = ejemplo_uso()