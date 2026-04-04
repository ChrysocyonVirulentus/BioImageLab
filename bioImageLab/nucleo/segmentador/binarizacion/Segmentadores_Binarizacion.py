"""
Métodos de binarización para conversión de imágenes en escala de grises a binarias.

La binarización convierte una imagen en escala de grises a una imagen binaria
(típicamente 0 y 255) mediante la aplicación de un umbral. Los píxeles por
encima del umbral se convierten a 255 (blanco), los demás a 0 (negro).

Ecuación básica:
    I_bin(x,y) = 255 si I(x,y) > T, else 0
    donde T es el umbral

IMPORTANTE - Separación de responsabilidades:
- Estos métodos NO normalizan imágenes (ese rol es de normalizador.py)
- Trabajan con los valores de la imagen tal como vienen
- La normalización previa debe hacerse con Normalizador si es necesaria

Tipos de umbralización:
- Global: Un solo umbral para toda la imagen
- Adaptativo: Umbral variable según contexto local
- Automático: Umbral calculado del histograma
- Manual: Umbral especificado por el usuario

Métodos disponibles:
- Otsu: Umbral óptimo minimizando varianza intra-clase
- Global: Umbral fijo manual
- Adaptativo: Umbral local (media o gaussiano)
- Percentil: Umbral basado en percentil del histograma
- Triangle: Para histogramas unimodales con cola larga
- Yen: Para imágenes de bajo contraste
- Li: Basado en entropía cruzada
- Isodata: Iterativo basado en medias de clases
- Minimum: Para histogramas bimodales claros
- Mean: Umbral en la media de intensidades
"""

import numpy as np
import cv2
from typing import Optional, Tuple, Literal
import warnings


class MetodoBinarizacion:
    """
        Clase base para métodos de binarización.
        
        Los métodos de binarización convierten imágenes en escala de grises
        a imágenes binarias mediante la aplicación de un umbral.
        
        Conceptos clave:
            - Umbral (threshold): Valor de corte para separar clases
            - Objeto/Fondo: Dos clases en imagen binaria
            - Histograma: Distribución de intensidades
    """
    nombre = "metodo_binarizacion_base"
    
    def __call__(self, img: np.ndarray) -> Tuple[float, np.ndarray]:
        """
            Aplica binarización a la imagen.
            
            Args:
                img: Array 2D (Y, X) en escala de grises
                
            Returns:
                Tupla (umbral_usado, imagen_binaria)
                - umbral_usado: Valor de umbral calculado/usado
                - imagen_binaria: Imagen binaria (0 o 255)
        """
        raise NotImplementedError("Subclases deben implementar __call__")
    
    def _validar_imagen(self, img: np.ndarray):
        """Valida que la imagen sea 2D."""
        if img.ndim != 2:
            raise ValueError(f"La imagen debe ser 2D, tiene {img.ndim} dimensiones")

@registrar_en("segmentacion")
class Otsu(MetodoBinarizacion):
    """
        Método de Otsu para umbralización automática óptima.
        
        Otsu (1979) calcula el umbral que minimiza la varianza intra-clase
        (o equivalentemente, maximiza la varianza inter-clase) de las dos clases
        resultantes (objeto y fondo).
        
        Algoritmo:
            1. Calcular histograma de la imagen
            2. Para cada posible umbral t:
            - Calcular probabilidades de cada clase
            - Calcular medias de cada clase
            - Calcular varianza inter-clase
            3. Seleccionar t que maximiza varianza inter-clase
        
        Ecuación (varianza inter-clase):
            σ²_b(t) = w₀(t) * w₁(t) * [μ₀(t) - μ₁(t)]²
            donde w₀, w₁ son probabilidades de clases, μ₀, μ₁ sus medias
        
        Ventajas:
            - Completamente automático (no requiere parámetros)
            - Óptimo según criterio de varianza
            - Rápido y robusto
            - Funciona bien para histogramas bimodales
            - Ampliamente usado y validado
        
        Desventajas:
            - Asume histograma bimodal (dos picos claros)
            - Puede fallar con ruido alto
            - Sensible a desbalance severo de clases
            - No funciona bien si una clase es muy pequeña
        
        Usos típicos en microscopía:
            - Segmentación de núcleos en DAPI
            - Separación células/fondo en contraste de fase
            - Segmentación de objetos en fluorescencia uniforme
            - Preprocesamiento para análisis cuantitativo
            - Cuando histograma tiene dos picos claros
    """
    nombre = "otsu"
    
    def __init__(self):
        """
            Inicializa el método de Otsu.
            
            No requiere parámetros (completamente automático).
        """
        pass
    
    def __call__(self, img: np.ndarray) -> np.ndarray:
        """
            Aplica umbralización de Otsu.
            
            Args:
                img: Imagen 2D en escala de grises (uint8 o uint16)
                
            Returns:
                Tupla (umbral, imagen_binaria)
            
            Nota:
                OpenCV calcula Otsu solo para uint8. Para uint16 se escala temporalmente.
        """
        self._validar_imagen(img)
        
        if img.dtype == np.uint8:
            umbral, img_binaria = cv2.threshold(
                img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            return img_binaria
        
        elif img.dtype == np.uint16:
            # Escalar a uint8 para Otsu
            img_uint8 = (img / 256).astype(np.uint8)
            umbral_uint8, img_binaria_uint8 = cv2.threshold(
                img_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            # Umbral real en escala uint16
            umbral_real = umbral_uint8 * 256
            # Aplicar umbral a imagen original
            img_binaria = np.where(img > umbral_real, 255, 0).astype(np.uint8)
            return img_binaria
        
        else:
            raise TypeError(
                f"Otsu requiere uint8 o uint16, recibió {img.dtype}. "
                "Use normalizador.py para convertir primero."
            )

@registrar_en("segmentacion")
class Global(MetodoBinarizacion):
    """
        Umbralización global con valor fijo manual.
        
        Aplica un umbral constante especificado por el usuario a toda la imagen.
        Es el método más simple y directo cuando se conoce el valor apropiado.
        
        Ecuación:
            I_bin(x,y) = max_val si I(x,y) > umbral, else 0
        
        Ventajas:
            - Extremadamente simple y rápido
            - Control total sobre el umbral
            - Reproducible (mismo umbral → mismo resultado)
            - Útil cuando se conoce el valor óptimo
            - No depende del contenido de la imagen
        
        Desventajas:
            - Requiere conocimiento previo del umbral
            - No se adapta a variaciones de iluminación
            - Puede necesitar ajuste manual
            - No óptimo para todas las regiones
        
        Usos típicos en microscopía:
            - Cuando se conoce el umbral óptimo por calibración
            - Procesamiento por lotes con condiciones consistentes
            - Análisis cuantitativo con umbral estandarizado
            - Validación de métodos automáticos
            - Control de calidad con criterios fijos
    """
    nombre = "global"
    
    def __init__(self, umbral: float = 127.0, invertir: bool = False):
        """
            Args:
                umbral: Valor de umbral fijo
                    Para uint8: típicamente 0-255
                    Para uint16: típicamente 0-65535
                invertir: Si True, invierte la binarización
                        (píxeles < umbral → 255, resto → 0)
        """
        if umbral < 0:
            raise ValueError("umbral debe ser >= 0")
        
        self.umbral = umbral
        self.invertir = invertir
    
    def __call__(self, img: np.ndarray) -> np.ndarray:
        """
            Aplica umbralización global.
            
            Args:
                img: Imagen 2D en escala de grises
                
            Returns:
                Tupla (umbral, imagen_binaria)
        """
        self._validar_imagen(img)
        
        if self.invertir:
            img_binaria = np.where(img < self.umbral, 255, 0).astype(np.uint8)
        else:
            img_binaria = np.where(img > self.umbral, 255, 0).astype(np.uint8)
        
        return img_binaria

@registrar_en("segmentacion")
class Adaptativo(MetodoBinarizacion):
    """
        Umbralización adaptativa con umbral local variable.
        
        Calcula un umbral diferente para cada región de la imagen basado en
        estadísticas locales. Ideal para imágenes con iluminación no uniforme.
        
        Algoritmo:
            1. Para cada píxel, considerar ventana de vecindad
            2. Calcular estadística local (media o media gaussiana ponderada)
            3. Umbral_local = estadística_local - C
            4. Binarizar comparando con umbral local
        
        Tipos:
            - Mean: Umbral = media local - C
            - Gaussian: Umbral = media gaussiana ponderada - C
        
        Ventajas:
            - Se adapta a variaciones locales de iluminación
            - Robusto ante fondos no uniformes
            - No requiere preprocesamiento de iluminación
            - Efectivo en imágenes con gradientes de luz
        
        Desventajas:
            - Más lento que métodos globales
            - Sensible al tamaño de ventana
            - Parámetro C requiere ajuste
            - Puede crear artefactos en regiones uniformes
        
        Usos típicos en microscopía:
            - Imágenes de campo claro con vignetting
            - Segmentación con iluminación desigual
            - Fluorescencia con gradientes de fondo
            - Cuando corrección de iluminación no es posible
            - Imágenes con variaciones lentas de intensidad
    """
    nombre = "adaptativo"
    
    def __init__(self,
                tamaño_ventana: int = 11,
                metodo: Literal['mean', 'gaussian'] = 'gaussian',
                C: float = 2.0):
        """
            Args:
                tamaño_ventana: Tamaño de la vecindad (debe ser impar)
                            Valores típicos:
                                - 5-11: detalles finos, rápido
                                - 11-21: balance (recomendado)
                                - 21-51: variaciones amplias, más lento
                            Mayor tamaño = más suavizado del umbral
                
                metodo: Método de cálculo del umbral local
                    'mean': Media aritmética simple
                    'gaussian': Media ponderada gaussiana (más suave)
                
                C: Constante sustraída del umbral calculado
                Valores típicos: 0-10
                Positivo: umbral más bajo (más píxeles → blancos)
                Negativo: umbral más alto (menos píxeles → blancos)
                Permite ajuste fino del resultado
        """
        if tamaño_ventana < 3 or tamaño_ventana % 2 == 0:
            raise ValueError("tamaño_ventana debe ser impar y >= 3")
        if metodo not in ['mean', 'gaussian']:
            raise ValueError("metodo debe ser 'mean' o 'gaussian'")
        
        self.tamaño_ventana = tamaño_ventana
        self.metodo = metodo
        self.C = C
    
    def __call__(self, img: np.ndarray) -> np.ndarray:
        """
            Aplica umbralización adaptativa.
            
            Args:
                img: Imagen 2D en escala de grises (uint8 o uint16)
                
            Returns:
                Tupla (None, imagen_binaria)
                - None porque no hay un solo umbral global
                - imagen_binaria: Resultado binario
            
            Nota:
                OpenCV adaptiveThreshold solo funciona con uint8.
        """
        self._validar_imagen(img)
        
        if img.dtype == np.uint8:
            img_procesada = img
        elif img.dtype == np.uint16:
            img_procesada = (img / 256).astype(np.uint8)
        else:
            raise TypeError(
                f"Adaptativo requiere uint8 o uint16, recibió {img.dtype}. "
                "Use normalizador.py para convertir primero."
            )
        
        # Seleccionar método adaptativo
        if self.metodo == 'mean':
            metodo_cv = cv2.ADAPTIVE_THRESH_MEAN_C
        else:  # gaussian
            metodo_cv = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
        
        img_binaria = cv2.adaptiveThreshold(
            img_procesada,
            255,
            metodo_cv,
            cv2.THRESH_BINARY,
            self.tamaño_ventana,
            self.C
        )
        
        return img_binaria

@registrar_en("segmentacion")
class Percentil(MetodoBinarizacion):
    """
        Umbralización basada en percentil del histograma.
        
        Calcula el umbral como un percentil específico de la distribución de
        intensidades. Útil cuando se desea segmentar un porcentaje conocido
        de la imagen.
        
        Ejemplo:
            percentil=90 → umbral que deja el 10% más brillante como objetos
        
        Ventajas:
            - Control intuitivo (porcentaje de imagen a segmentar)
            - Robusto ante outliers (usa percentil, no valores extremos)
            - Útil cuando se conoce la proporción objeto/fondo
            - No asume forma particular del histograma
        
        Desventajas:
            - Requiere conocimiento previo de la proporción
            - No considera forma del histograma (bimodalidad)
            - Puede no ser óptimo si proporción varía entre imágenes
            - Sensible a desbalance de clases
        
        Usos típicos en microscopía:
            - Cuando se conoce % aproximado de área de objetos
            - Segmentación de spots (ej: 2-5% más brillante)
            - Extracción de núcleos (~30-40% del área)
            - Control de sensibilidad en detección
            - Análisis exploratorio para determinar umbral
    """
    nombre = "percentil"
    
    def __init__(self, percentil: float = 90.0):
        """
            Args:
                percentil: Percentil del histograma a usar como umbral
                        Valores: 0-100
                        Valores típicos:
                            - 50: Mediana (split 50-50)
                            - 70-80: Segmentar ~20-30% más brillante
                            - 90-95: Segmentar ~5-10% más brillante (spots)
                            - 95-99: Solo píxeles muy brillantes
        """
        if not (0 <= percentil <= 100):
            raise ValueError("percentil debe estar en [0, 100]")
        
        self.percentil = percentil
    
    def __call__(self, img: np.ndarray) -> np.ndarray:
        """
            Aplica umbralización por percentil.
            
            Args:
                img: Imagen 2D en escala de grises
                
            Returns:
                Tupla (umbral, imagen_binaria)
        """
        self._validar_imagen(img)
        
        # Calcular percentil
        umbral = np.percentile(img, self.percentil)
        
        # Binarizar
        img_binaria = np.where(img > umbral, 255, 0).astype(np.uint8)
        
        return img_binaria

@registrar_en("segmentacion")
class Triangle(MetodoBinarizacion):
    """
        Método Triangle para histogramas unimodales con cola larga.
        
        Triangle (Zack et al. 1977) encuentra el umbral trazando una línea desde
        el pico del histograma hasta el extremo de la cola, y seleccionando el
        punto de máxima distancia perpendicular.
        
        Algoritmo:
            1. Encontrar pico del histograma
            2. Encontrar extremo de la cola (lado opuesto)
            3. Trazar línea entre pico y extremo
            4. Seleccionar punto del histograma con mayor distancia a la línea
        
        Ventajas:
            - Funciona bien con histogramas unimodales asimétricos
            - Robusto ante ruido en la cola
            - No requiere bimodalidad como Otsu
            - Automático (sin parámetros)
        
        Desventajas:
            - Requiere histograma unimodal claro
            - No funciona con histogramas bimodales equilibrados
            - Puede fallar si pico no está en extremo
        
        Usos típicos en microscopía:
            - Segmentación con fondo dominante (histograma asimétrico)
            - Objetos brillantes en fondo oscuro (o viceversa)
            - Imágenes con distribución exponencial
            - Cuando Otsu falla por falta de bimodalidad
    """
    nombre = "triangle"
    
    def __init__(self):
        """Inicializa el método Triangle (sin parámetros)."""
        pass
    
    def __call__(self, img: np.ndarray) -> np.ndarray:
        """
            Aplica umbralización Triangle.
            
            Args:
                img: Imagen 2D en escala de grises (uint8 o uint16)
                
            Returns:
                Tupla (umbral, imagen_binaria)
        """
        self._validar_imagen(img)
        
        if img.dtype == np.uint8:
            umbral, img_binaria = cv2.threshold(
                img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE
            )
            return img_binaria
        
        elif img.dtype == np.uint16:
            img_uint8 = (img / 256).astype(np.uint8)
            umbral_uint8, _ = cv2.threshold(
                img_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE
            )
            umbral_real = umbral_uint8 * 256
            img_binaria = np.where(img > umbral_real, 255, 0).astype(np.uint8)
            return img_binaria
        
        else:
            raise TypeError(
                f"Triangle requiere uint8 o uint16, recibió {img.dtype}."
            )

@registrar_en("segmentacion")
class Mean(MetodoBinarizacion):
    """
        Umbralización usando la media de intensidades.
        
        Método simple que usa la media de todos los píxeles como umbral.
        Equivalente a percentil=50 para distribuciones simétricas.
        
        Ventajas:
            - Extremadamente simple
            - Muy rápido
            - No requiere parámetros
            - Intuitivo
        
        Desventajas:
            - No considera forma del histograma
            - Sensible a outliers
            - Subóptimo para distribuciones asimétricas
            - No adaptativo
        
        Usos típicos:
            - Pruebas rápidas y exploratorias
            - Baseline para comparar otros métodos
            - Imágenes con distribución aproximadamente uniforme
    """
    nombre = "mean"
    
    def __init__(self):
        """Inicializa el método Mean (sin parámetros)."""
        pass
    
    def __call__(self, img: np.ndarray) -> np.ndarray:
        """
        Aplica umbralización por media.
        
        Args:
            img: Imagen 2D en escala de grises
            
        Returns:
            Tupla (umbral, imagen_binaria)
        """
        self._validar_imagen(img)
        
        umbral = np.mean(img)
        img_binaria = np.where(img > umbral, 255, 0).astype(np.uint8)
        
        return img_binaria

@registrar_en("segmentacion")
class Isodata(MetodoBinarizacion):
    """
        Método Isodata (Ridler-Calvard) iterativo.
        
        Algoritmo iterativo que converge al umbral óptimo basándose en las medias
        de las dos clases (objeto y fondo).
        
        Algoritmo:
            1. Inicializar umbral (media de la imagen)
            2. Separar píxeles en dos clases según umbral actual
            3. Calcular media de cada clase
            4. Nuevo umbral = (media_clase1 + media_clase2) / 2
            5. Repetir hasta convergencia
        
        Ventajas:
            - Generalmente converge rápido (pocas iteraciones)
            - Robusto y estable
            - Similar a k-means con k=2
            - Funciona bien para histogramas bimodales
        
        Desventajas:
            - Más lento que Otsu (iterativo)
            - Puede dar resultados similares a Otsu
            - Sensible a inicialización (aunque usualmente buena)
        
        Usos típicos:
            - Alternativa a Otsu para histogramas bimodales
            - Cuando se requiere interpretabilidad del proceso
            - Segmentación robusta de objetos/fondo
    """
    nombre = "isodata"
    
    def __init__(self, max_iter: int = 100, tol: float = 0.5):
        """
        Args:
            max_iter: Número máximo de iteraciones
            tol: Tolerancia para convergencia (cambio mínimo de umbral)
        """
        if max_iter < 1:
            raise ValueError("max_iter debe ser >= 1")
        if tol < 0:
            raise ValueError("tol debe ser >= 0")
        
        self.max_iter = max_iter
        self.tol = tol
    
    def __call__(self, img: np.ndarray) -> np.ndarray:
        """
            Aplica umbralización Isodata.
            
            Args:
                img: Imagen 2D en escala de grises
                
            Returns:
                Tupla (umbral, imagen_binaria)
        """
        self._validar_imagen(img)
        
        # Inicializar umbral con la media
        umbral = np.mean(img)
        
        for _ in range(self.max_iter):
            # Separar en dos clases
            clase_fondo = img[img <= umbral]
            clase_objeto = img[img > umbral]
            
            # Evitar división por cero
            if len(clase_fondo) == 0 or len(clase_objeto) == 0:
                break
            
            # Calcular nuevas medias
            media_fondo = np.mean(clase_fondo)
            media_objeto = np.mean(clase_objeto)
            
            # Nuevo umbral
            nuevo_umbral = (media_fondo + media_objeto) / 2.0
            
            # Verificar convergencia
            if abs(nuevo_umbral - umbral) < self.tol:
                umbral = nuevo_umbral
                break
            
            umbral = nuevo_umbral
        
        # Binarizar
        img_binaria = np.where(img > umbral, 255, 0).astype(np.uint8)
        
        return img_binaria

@registrar_en("segmentacion")
class Minimum(MetodoBinarizacion):
    """
        Método Minimum para histogramas bimodales.
        
        Encuentra el mínimo local entre dos picos en un histograma bimodal.
        Asume que el histograma tiene dos modas claras con un valle entre ellas.
        
        Algoritmo:
            1. Suavizar histograma
            2. Encontrar dos picos principales
            3. Buscar mínimo local entre los picos
            4. Usar ese mínimo como umbral
        
        Ventajas:
            - Intuitivo para histogramas bimodales claros
            - Funciona bien cuando las clases están bien separadas
            - Robusto si hay separación clara
        
        Desventajas:
            - Requiere histograma bimodal claro
            - Sensible a ruido en el histograma
            - Puede fallar si picos no son evidentes
            - Requiere suavizado del histograma
        
        Usos típicos:
            - Segmentación cuando histograma muestra dos picos claros
            - Imágenes con buena separación objeto/fondo
            - Cuando inspección visual del histograma muestra bimodalidad
    """
    nombre = "minimum"
    
    def __init__(self, suavizado: int = 3):
        """
            Args:
                suavizado: Tamaño de kernel para suavizar histograma
                        Mayor valor = más suavizado (más robusto a ruido)
        """
        if suavizado < 1:
            raise ValueError("suavizado debe ser >= 1")
        
        self.suavizado = suavizado
    
    def __call__(self, img: np.ndarray) -> np.ndarray:
        """
            Aplica umbralización por mínimo entre picos.
            
            Args:
                img: Imagen 2D en escala de grises
                
            Returns:
                Tupla (umbral, imagen_binaria)
        """
        self._validar_imagen(img)
        
        # Calcular histograma
        if img.dtype == np.uint8:
            hist, bins = np.histogram(img.ravel(), bins=256, range=(0, 256))
        elif img.dtype == np.uint16:
            hist, bins = np.histogram(img.ravel(), bins=256, range=(0, 65536))
        else:
            hist, bins = np.histogram(img.ravel(), bins=256)
        
        # Suavizar histograma
        from scipy.ndimage import uniform_filter1d
        hist_suavizado = uniform_filter1d(hist.astype(np.float64), size=self.suavizado)
        
        # Encontrar mínimo en la región central (ignorar extremos)
        region_central = hist_suavizado[len(hist_suavizado)//4 : 3*len(hist_suavizado)//4]
        idx_minimo_local = np.argmin(region_central) + len(hist_suavizado)//4
        
        # Umbral corresponde al bin del mínimo
        umbral = bins[idx_minimo_local]
        
        # Binarizar
        img_binaria = np.where(img > umbral, 255, 0).astype(np.uint8)
        
        return img_binaria