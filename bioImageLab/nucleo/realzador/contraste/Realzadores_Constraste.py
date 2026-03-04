"""
Métodos de realce de contraste para mejora de visibilidad en imágenes.

Los realzadores de contraste mejoran la visibilidad de estructuras mediante
manipulación de la distribución de intensidades. Hacen más evidentes los
detalles que pueden estar ocultos por bajo contraste o iluminación inadecuada.

Principio fundamental:
Expandir o redistribuir el rango dinámico de la imagen para aprovechar mejor
el espectro de intensidades disponible, mejorando la discriminación visual.

IMPORTANTE - Separación de responsabilidades:
- Estos métodos NO normalizan imágenes (ese rol es de normalizador.py) : POR ENDE NORMALIZAR ANTES DE USAR ESTO!
- Solo realizan conversiones de tipo cuando OpenCV lo requiere estrictamente
- Trabajan con los valores de la imagen tal como vienen

Tipos de realce:
- Global: Aplica la misma transformación a toda la imagen
- Adaptativo: Ajusta la transformación según el contexto local
- No lineal: Enfatiza selectivamente rangos específicos de intensidad

Métodos disponibles:
- CLAHE: Ecualización adaptativa con límite de contraste
- Gamma: Corrección no lineal de brillo
- Logarítmico: Compresión de rango dinámico alto
- Retinex: Separación iluminación-reflectancia (corrección de iluminación)
- Ecualización de histograma: Distribución uniforme de intensidades
"""

import numpy as np
import cv2
from typing import Optional, Tuple, Literal
import warnings


class RealzadorContraste:
    """
        Clase base para métodos de realce de contraste.
        
        Los realzadores de contraste mejoran la visibilidad de estructuras
        manipulando la distribución de intensidades en la imagen.
        
        Conceptos clave:
            - Contraste: Diferencia entre intensidades claras y oscuras
            - Rango dinámico: Span de intensidades presentes
            - Histograma: Distribución de intensidades
    """
    nombre = "realzador_contraste_base"
    
    def __call__(self, img: np.ndarray) -> np.ndarray:
        """
            Aplica el realce de contraste a la imagen.
            
            Args:
                img: Array 2D (Y, X) con la imagen a realzar
                
            Returns:
                Imagen con contraste realzado del mismo tipo y forma
        """
        raise NotImplementedError("Subclases deben implementar __call__")
    
    def _validar_imagen(self, img: np.ndarray):
        """Valida que la imagen sea 2D."""
        if img.ndim != 2:
            raise ValueError(f"La imagen debe ser 2D, tiene {img.ndim} dimensiones")


class CLAHE(RealzadorContraste):
    """
        Contrast Limited Adaptive Histogram Equalization.
        
        CLAHE es una variante mejorada de la ecualización de histograma adaptativa (AHE)
        que previene sobre-amplificación del ruido mediante un límite de contraste.
        
        Algoritmo:
            1. Divide la imagen en tiles (bloques)
            2. Ecualiza el histograma de cada tile
            3. Limita el contraste recortando el histograma
            4. Interpola suavemente entre tiles
        
        Ecuación del límite:
            clip_limit = (1 + α/100) * (tile_size / 256)
            donde α es el parámetro de límite de contraste
        
        Ventajas:
            - Realce adaptativo a condiciones locales
            - Previene sobre-amplificación de ruido
            - Excelente para imágenes con iluminación desigual
            - Preserva detalles locales mejor que ecualización global
            - Resultados naturales
        
        Desventajas:
            - Puede crear artefactos en bordes de tiles (si interpolación insuficiente)
            - Más lento que ecualización global
            - Parámetros requieren ajuste según imagen
        
        Usos típicos en microscopía:
            - Mejora de imágenes de campo claro con iluminación desigual
            - Realce de detalles en imágenes de bajo contraste
            - Preprocesamiento para visualización y análisis
            - Mejora de imágenes de fluorescencia débil
            - Realce de estructuras celulares en brightfield
    """
    nombre = "clahe"
    
    def __init__(self,
                clip_limit: float = 2.0,
                tile_grid_size: Tuple[int, int] = (8, 8)):
        """
            Args:
                clip_limit: Límite de contraste para recorte de histograma
                        Valores típicos: 1.0-4.0
                        1.0: realce suave, menos artefactos
                        2.0: balance recomendado para microscopía
                        4.0: realce agresivo, puede amplificar ruido
                        Mayor valor = más contraste (pero más ruido)
                
                tile_grid_size: Tamaño de la grilla de tiles (filas, columnas)
                            Valores típicos: (8,8), (16,16)
                            Tiles pequeños (16x16): realce muy local, puede crear artefactos
                            Tiles grandes (4x4): más suave, menos adaptativo
                            Recomendado: (8,8) para balance
            
            Nota:
                La imagen se divide en tile_grid_size[0] x tile_grid_size[1] bloques.
                Cada bloque tiene su propio histograma ecualizado.
        """
        if clip_limit <= 0:
            raise ValueError("clip_limit debe ser > 0")
        if tile_grid_size[0] < 1 or tile_grid_size[1] < 1:
            raise ValueError("tile_grid_size debe tener valores >= 1")
        
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        
        # Crear objeto CLAHE de OpenCV
        self.clahe = cv2.createCLAHE(
            clipLimit=clip_limit,
            tileGridSize=tile_grid_size
        )
    
    def __call__(self, img: np.ndarray) -> np.ndarray:
        """
            Aplica CLAHE a la imagen.
            
            Args:
                img: Imagen 2D (preferiblemente uint8, se convierte si es necesario)
                
            Returns:
                Imagen con contraste realzado del mismo tipo
            
            Nota:
                OpenCV CLAHE solo acepta uint8. Si la imagen es otro tipo,
                se convierte temporalmente manteniendo el rango de valores.
        """
        self._validar_imagen(img)
        
        tipo_original = img.dtype
        
        if img.dtype == np.uint8:
            return self.clahe.apply(img)
        
        elif img.dtype == np.uint16:
            # Escalar a uint8 manteniendo proporciones
            # uint16 [0, 65535] → uint8 [0, 255]
            img_uint8 = (img / 256).astype(np.uint8)
            img_clahe = self.clahe.apply(img_uint8)
            # Volver a uint16
            return (img_clahe.astype(np.uint16) * 256)
        
        else:
            raise TypeError(
                f"CLAHE requiere uint8 o uint16, recibió {img.dtype}. "
                "Use normalizador.py para convertir a uint8/uint16 primero."
            )


class Gamma(RealzadorContraste):
    """
        Corrección Gamma para ajuste no lineal de brillo y contraste.
        
        La corrección gamma aplica una transformación no lineal que enfatiza
        selectivamente intensidades bajas (gamma < 1) o altas (gamma > 1).
        
        Ecuación:
            I_out = I_in^γ (normalizado a [0, 1])
            o equivalente: I_out = 255 * (I_in / 255)^γ
        
        Interpretación:
            - γ < 1: Aclara la imagen (power < 1 → valores pequeños crecen más)
            - γ = 1: Sin cambio (identidad)
            - γ > 1: Oscurece la imagen (power > 1 → valores pequeños se reducen más)
        
        Ventajas:
            - Muy simple y rápido
            - Control intuitivo con un solo parámetro
            - Preserva el orden de intensidades
            - No amplifica ruido tanto como ecualización
            - Útil para corrección de visualización
        
        Desventajas:
            - No adaptativo (global)
            - No optimiza uso del rango dinámico
            - Puede saturar o sub-utilizar intensidades
            - Sensible a la elección de gamma
        
        Usos típicos en microscopía:
            - Ajuste rápido de brillo para visualización
            - Corrección de gamma de cámara/monitor
            - Realce de señales débiles (gamma < 1)
            - Reducción de sobre-exposición (gamma > 1)
            - Preprocesamiento simple antes de segmentación
    """
    nombre = "gamma"
    
    def __init__(self, gamma: float = 1.0):
        """
            Args:
                gamma: Factor de corrección gamma
                    Valores típicos:
                        - 0.3-0.7: Aclara mucho (señales muy débiles)
                        - 0.8-0.9: Aclara moderadamente
                        - 1.0: Sin cambio
                        - 1.1-1.5: Oscurece moderadamente
                        - 1.5-3.0: Oscurece mucho (reducir saturación)
            
            Regla práctica:
                gamma = 0.5 → duplica la "intensidad percibida" de sombras
                gamma = 2.0 → reduce a la mitad la "intensidad percibida" de sombras
        """
        if gamma <= 0:
            raise ValueError("gamma debe ser > 0")
        
        self.gamma = gamma
    
    def __call__(self, img: np.ndarray) -> np.ndarray:
        """
            Aplica corrección gamma.
            
            Args:
                img: Imagen 2D en cualquier formato
                
            Returns:
                Imagen con gamma aplicado del mismo tipo
            
            Nota:
                Trabaja directamente con los valores de la imagen.
                Para uint8/uint16, asume rango [0, max_tipo].
        """
        self._validar_imagen(img)
        
        if np.issubdtype(img.dtype, np.integer):
            # Para enteros, trabajar en el rango del tipo
            info = np.iinfo(img.dtype)
            img_float = img.astype(np.float64) / info.max
            img_gamma = np.power(img_float, self.gamma)
            img_gamma = img_gamma * info.max
            return np.clip(img_gamma, info.min, info.max).astype(img.dtype)
        else:
            # Para float, asumir que ya está en rango apropiado
            img_float = img.astype(np.float64)
            return np.power(img_float, self.gamma).astype(img.dtype)


class Logaritmico(RealzadorContraste):
    """
        Transformación logarítmica para compresión de rango dinámico.
        
        La transformación logarítmica comprime el rango dinámico expandiendo
        valores bajos y comprimiendo valores altos. Útil para imágenes con
        alto rango dinámico (HDR).
        
        Ecuación:
            I_out = c * log(1 + I_in)
            donde c es una constante de escalamiento
        
        Interpretación:
            - Valores bajos: cambios pequeños → cambios grandes en salida
            - Valores altos: cambios grandes → cambios pequeños en salida
            - Efecto: "acerca" los valores extremos, mejora visibilidad de sombras
        
        Ventajas:
            - Excelente para alto rango dinámico
            - Realza detalles en regiones oscuras
            - Comprime brillos sin pérdida total de información
            - Útil para visualizar exponenciales (ej: decaimiento)
        
        Desventajas:
            - Puede sobre-comprimir detalles en zonas brillantes
            - Amplifica ruido en regiones oscuras
            - No tan intuitivo como gamma
            - Requiere ajuste de parámetro c
        
        Usos típicos en microscopía:
            - Visualización de imágenes con amplio rango dinámico
            - Realce de señales débiles junto a señales fuertes
            - Análisis de decaimiento exponencial (bleaching)
            - Imágenes con sobre-exposición local
            - Visualización de conteos de fotones
    """
    nombre = "logaritmico"
    
    def __init__(self, c: Optional[float] = None):
        """
            Args:
                c: Constante de escalamiento
                Si None, se calcula automáticamente como:
                c = 255 / log(1 + max_valor)
                para mapear el rango completo a [0, 255]
                
                Valores típicos si se especifica manualmente:
                    - 20-50: Compresión moderada
                    - 50-100: Compresión fuerte
        """
        if c is not None and c <= 0:
            raise ValueError("c debe ser > 0")
        
        self.c = c
    
    def __call__(self, img: np.ndarray) -> np.ndarray:
        """
            Aplica transformación logarítmica.
            
            Args:
                img: Imagen 2D en cualquier formato
                
            Returns:
                Imagen con transformación log aplicada del mismo tipo
            
            Nota:
                Trabaja directamente con los valores de la imagen.
                La constante c se calcula según el rango del tipo de dato.
        """
        self._validar_imagen(img)
        
        img_float = img.astype(np.float64)
        
        # Calcular c según el tipo de dato
        if self.c is None:
            if np.issubdtype(img.dtype, np.integer):
                info = np.iinfo(img.dtype)
                c_auto = info.max / np.log(1.0 + info.max)
            else:
                # Para float, usar el rango actual de la imagen
                c_auto = img_float.max() / np.log(1.0 + img_float.max())
        else:
            c_auto = self.c
        
        # Aplicar transformación logarítmica
        img_log = c_auto * np.log(1.0 + img_float)
        
        # Clipear al rango del tipo
        if np.issubdtype(img.dtype, np.integer):
            info = np.iinfo(img.dtype)
            img_log = np.clip(img_log, info.min, info.max)
        
        return img_log.astype(img.dtype)


class Retinex(RealzadorContraste):
    """
        Algoritmo Retinex para corrección de iluminación desigual.
        
        Retinex (Retina + Cortex) es un modelo de percepción visual que separa
        la iluminación del reflejo (propiedades intrínsecas del objeto).
        
        Algoritmo Single Scale Retinex (SSR):
            1. Estimar iluminación: L(x,y) = I(x,y) * G(x,y)
            donde G es un filtro gaussiano de gran sigma
            2. Separar reflectancia: R(x,y) = log(I(x,y)) - log(L(x,y))
        
        Algoritmo Multi Scale Retinex (MSR):
            Promedio de SSR en múltiples escalas para mejor robustez
        
        Ventajas:
            - Corrige iluminación desigual efectivamente
            - Realza contraste local
            - Invariante a cambios de iluminación global
            - Preserva detalles de la imagen
            - No requiere imagen de referencia
        
        Desventajas:
            - Puede crear halos alrededor de bordes fuertes
            - Computacionalmente más costoso
            - Sensible a la elección de sigma
            - Puede amplificar ruido
        
        Usos típicos en microscopía:
            - Corrección de iluminación desigual en campo claro
            - Realce de contraste en brightfield
            - Preprocesamiento para segmentación con fondo variable
            - Mejora de visibilidad en imágenes con vignetting
            - Normalización de iluminación en time-lapse
    """
    nombre = "retinex"
    
    def __init__(self,
                sigma: float = 300.0,
                multi_escala: bool = False,
                sigmas: Optional[Tuple[float, ...]] = None):
        """
            Args:
                sigma: Desviación estándar del filtro gaussiano para SSR
                    Valores típicos:
                        - 50-150: Detalles locales (estructuras pequeñas)
                        - 150-300: Balance (recomendado para microscopía)
                        - 300-600: Variaciones globales (fondos amplios)
                    Mayor sigma = corrige variaciones más amplias
                
                multi_escala: Si True, usa Multi Scale Retinex (MSR)
                
                sigmas: Tupla de sigmas para MSR (solo si multi_escala=True)
                    Default: (15, 80, 250) - pequeño, medio, grande
                    Usa múltiples escalas para robustez
        """
        if sigma <= 0:
            raise ValueError("sigma debe ser > 0")
        
        self.sigma = sigma
        self.multi_escala = multi_escala
        
        if multi_escala:
            self.sigmas = sigmas if sigmas is not None else (15.0, 80.0, 250.0)
        else:
            self.sigmas = (sigma,)
    
    def __call__(self, img: np.ndarray) -> np.ndarray:
        """
            Aplica Retinex (SSR o MSR).
            
            Args:
                img: Imagen 2D en escala de grises
                
            Returns:
                Imagen con iluminación corregida del mismo tipo
            
            Nota:
                Trabaja en el rango del tipo de dato original.
        """
        self._validar_imagen(img)
        
        # Convertir a float y añadir pequeño offset para evitar log(0)
        img_float = img.astype(np.float64) + 1.0
        
        # Calcular SSR para cada escala
        retinex_outputs = []
        for sigma in self.sigmas:
            # Estimar iluminación con filtro gaussiano
            iluminacion = cv2.GaussianBlur(img_float, (0, 0), sigmaX=sigma)
            
            # Calcular log del cociente
            retinex = np.log10(img_float) - np.log10(iluminacion)
            retinex_outputs.append(retinex)
        
        # Promediar escalas si es multi-escala
        if self.multi_escala and len(retinex_outputs) > 1:
            retinex_final = np.mean(retinex_outputs, axis=0)
        else:
            retinex_final = retinex_outputs[0]
        
        # Escalar de vuelta al rango del tipo
        # Retinex produce valores log, necesitamos mapear de vuelta
        if np.issubdtype(img.dtype, np.integer):
            info = np.iinfo(img.dtype)
            # Mapear rango de retinex a rango del tipo
            retinex_min, retinex_max = retinex_final.min(), retinex_final.max()
            if retinex_max > retinex_min:
                retinex_final = (retinex_final - retinex_min) / (retinex_max - retinex_min)
                retinex_final = retinex_final * info.max
            retinex_final = np.clip(retinex_final, info.min, info.max)
        
        return retinex_final.astype(img.dtype)


class EcualizacionHistograma(RealzadorContraste):
    """
        Ecualización de histograma global para distribución uniforme de intensidades.
        
        La ecualización transforma el histograma de la imagen para que sea
        aproximadamente uniforme, maximizando el uso del rango dinámico disponible.
        
        Algoritmo:
            1. Calcular histograma de la imagen
            2. Calcular función de distribución acumulativa (CDF)
            3. Normalizar CDF a [0, 255]
            4. Mapear cada pixel usando CDF como lookup table
        
        Ecuación:
            I_out(x,y) = (L-1) * CDF(I_in(x,y))
            donde L es el número de niveles (256 para uint8)
        
        Ventajas:
            - Maximiza uso del rango dinámico
            - Muy simple y rápido
            - No requiere parámetros
            - Efectivo para imágenes de bajo contraste uniforme
        
        Desventajas:
            - Puede sobre-amplificar ruido
            - No adaptativo (global)
            - Puede crear artefactos en imágenes con distribución bimodal
            - Modifica el histograma drásticamente (puede no ser natural)
            - Sensible a outliers
        
        Usos típicos en microscopía:
            - Realce rápido para visualización
            - Preprocesamiento cuando contraste es muy bajo
            - Análisis exploratorio de imágenes
            - Cuando no hay tiempo para ajuste manual
            
        Nota:
            Para imágenes de microscopía, CLAHE suele ser preferible por ser
            más robusto y adaptativo.
    """
    nombre = "ecualizacion_histograma"
    
    def __init__(self):
        """
            Inicializa el ecualizador de histograma.
            
            No requiere parámetros (transformación completamente automática).
        """
        pass
    
    def __call__(self, img: np.ndarray) -> np.ndarray:
        """
            Aplica ecualización de histograma global.
            
            Args:
                img: Imagen 2D (preferiblemente uint8, se convierte si es necesario)
                
            Returns:
                Imagen con histograma ecualizado del mismo tipo
            
            Nota:
                OpenCV equalizeHist solo acepta uint8. Si la imagen es otro tipo,
                se convierte temporalmente manteniendo proporciones.
        """
        self._validar_imagen(img)
        
        tipo_original = img.dtype
        
        if img.dtype == np.uint8:
            return cv2.equalizeHist(img)
        
        elif img.dtype == np.uint16:
            # Escalar a uint8 manteniendo proporciones
            img_uint8 = (img / 256).astype(np.uint8)
            img_eq = cv2.equalizeHist(img_uint8)
            # Volver a uint16
            return (img_eq.astype(np.uint16) * 256)
        
        else:
            raise TypeError(
                f"EcualizacionHistograma requiere uint8 o uint16, recibió {img.dtype}. "
                "Use normalizador.py para convertir a uint8/uint16 primero."
            )