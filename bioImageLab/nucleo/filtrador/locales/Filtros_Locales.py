"""
Filtros locales en el dominio espacial para reducción de ruido.

Estos filtros operan directamente sobre los píxeles de la imagen mediante
convolución con kernels o máscaras locales.

Características:
- Operan píxel por píxel en vecindades locales
- Buenos para ruido aleatorio y suavizado general
- Algunos preservan bordes mejor que otros
- Computacionalmente eficientes

Tipos disponibles:
- Gaussiano: Suavizado general con preservación de estructura
- Mediana: Eliminación de ruido sal y pimienta, preserva bordes
- CajaBlur: Promedio uniforme, muy rápido
- Bilateral: Suavizado preservando bordes nítidos
- DifusiónAnisotropica: Suavizado adaptativo que respeta estructuras
"""

import cv2
import numpy as np
import warnings
from typing import Tuple


class FiltroLocal:
    """
    Clase base para filtros locales en el dominio espacial.
    
    Los filtros locales operan sobre vecindades de píxeles para reducir ruido
    o suavizar la imagen.
    """
    nombre = "filtro_local_base"
    
    def __call__(self, img: np.ndarray) -> np.ndarray:
        """
        Aplica el filtro local a la imagen.
        
        Args:
            img: Array 2D (Y, X) con la imagen a filtrar
            
        Returns:
            Imagen filtrada del mismo tipo y forma
        """
        raise NotImplementedError("Subclases deben implementar __call__")
    
    def _validar_imagen(self, img: np.ndarray):
        """Valida que la imagen sea 2D."""
        if img.ndim != 2:
            raise ValueError(f"La imagen debe ser 2D, tiene {img.ndim} dimensiones")

@registrar_en("filtracion")
class Gaussiano(FiltroLocal):
    """
    Filtro gaussiano para suavizado espacial general.
    
    Elimina ruido electrónico de fondo mediante convolución con kernel gaussiano.
    
    Ventajas:
        - Suavizado isotrópico (igual en todas direcciones)
        - Preserva características generales de la imagen
        - Elimina ruido de alta frecuencia
    
    Desventajas:
        - Difumina bordes (como membranas celulares)
        - No distingue entre señal y ruido en los bordes
    
    Usos típicos:
        - Preprocesamiento antes de segmentación
        - Reducción de ruido en imágenes de microscopía
        - Suavizado de mapas de intensidad
    """
    nombre = "gaussiano"
    
    def __init__(self, sigma: float, mascara: Tuple[int, int] = (0, 0)):
        """
            Args:
                sigma: Desviación estándar del kernel gaussiano (controla la fuerza del suavizado)
                    Valores típicos: 0.5-3.0 para microscopía
                mascara: Tamaño del kernel (ancho, alto). Deben ser impares.
                        Si es (0, 0), OpenCV calcula automáticamente según sigma.
        """
        if sigma <= 0:
            raise ValueError("sigma debe ser > 0")
        
        self.sigma = sigma
        self.mascara = self._chequear_mascara(mascara)
    
    def _chequear_mascara(self, mascara: Tuple[int, int]) -> Tuple[int, int]:
        """
            Verifica que los valores de la máscara sean impares.
            Si son pares, les suma 1 para corregirlos automáticamente.
            
            Args:
                mascara: Tupla (ancho, alto)
                
            Returns:
                Tupla corregida con valores impares
        """
        ancho, alto = mascara
        
        # Si se pasa (0, 0), OpenCV calcula automáticamente según sigma
        if ancho == 0 and alto == 0:
            return (0, 0)
        
        # Asegurar que sean impares (OpenCV requiere ksize impar)
        nuevo_ancho = ancho if ancho % 2 != 0 else ancho + 1
        nuevo_alto = alto if alto % 2 != 0 else alto + 1
        
        if nuevo_ancho != ancho or nuevo_alto != alto:
            warnings.warn(
                f"Máscara corregida de ({ancho}, {alto}) "
                f"a ({nuevo_ancho}, {nuevo_alto}) para ser impar.",
                RuntimeWarning
            )
        
        return (nuevo_ancho, nuevo_alto)
    
    def __call__(self, img: np.ndarray) -> np.ndarray:
        """
            Aplica el filtro gaussiano.
            
            Args:
                img: Imagen 2D a filtrar
                
            Returns:
                Imagen suavizada del mismo tipo
        """
        self._validar_imagen(img)
        return cv2.GaussianBlur(img, self.mascara, self.sigma)

@registrar_en("filtracion")
class Mediana(FiltroLocal):
    """
        Filtro de mediana para eliminación de ruido impulsivo.
        
        Reemplaza cada píxel por la mediana de su vecindad, excelente para
        ruido tipo 'sal y pimienta' (píxeles saturados o muertos).
        
        Ventajas:
            - Elimina ruido impulsivo efectivamente
            - Preserva bordes mejor que filtros lineales
            - No introduce nuevos valores de intensidad
        
        Desventajas:
            - Más lento que filtros lineales
            - Puede eliminar detalles pequeños (menor al tamaño de la máscara)
        
        Usos típicos:
            - Corrección de píxeles muertos o calientes
            - Eliminación de artefactos puntuales
            - Preprocesamiento de imágenes con ruido impulsivo
    """
    nombre = "mediana"
    
    def __init__(self, mascara: int = 3):
        """
            Args:
                mascara: Tamaño de la ventana (debe ser impar, típicamente 3, 5, 7)
                        Valores más grandes eliminan más ruido pero pueden difuminar
        """
        if mascara <= 0:
            raise ValueError("mascara debe ser > 0")
        
        self.mascara = self._chequear_mascara(mascara)
    
    def _chequear_mascara(self, mascara: int) -> int:
        """
            Verifica si la máscara es impar. Si es par, la incrementa en 1.
            
            Args:
                mascara: Tamaño de la ventana
                
            Returns:
                Tamaño corregido (impar)
        """
        mascara_corregida = mascara if mascara % 2 != 0 else mascara + 1
        
        if mascara_corregida != mascara:
            warnings.warn(
                f"Máscara de mediana corregida de {mascara} a {mascara_corregida} (debe ser impar).",
                RuntimeWarning
            )
        
        return mascara_corregida
    
    def __call__(self, img: np.ndarray) -> np.ndarray:
        """
            Aplica el filtro de mediana.
            
            Args:
                img: Imagen 2D a filtrar
                
            Returns:
                Imagen filtrada del mismo tipo
        """
        self._validar_imagen(img)
        return cv2.medianBlur(img, self.mascara)

@registrar_en("filtracion")
class CajaBlur(FiltroLocal):
    """
        Filtro de caja (promedio uniforme) para suavizado rápido.
        
        Reemplaza cada píxel por el promedio de su vecindad rectangular.
        Es el filtro más simple y rápido, pero puede producir artefactos.
        
        Ventajas:
            - Extremadamente rápido
            - Implementación simple
            - Bueno para reducción de ruido general
        
        Desventajas:
            - Difumina bordes más que el gaussiano
            - Puede crear artefactos rectangulares
            - No pondera píxeles por distancia
        
        Usos típicos:
            - Suavizado rápido en pipelines de tiempo real
            - Preprocesamiento donde velocidad es crítica
            - Reducción de ruido cuando los bordes no son importantes
    """
    nombre = "caja_blur"
    
    def __init__(self, mascara: Tuple[int, int] = (3, 3)):
        """
        Args:
            mascara: Tupla (ancho, alto) del kernel de promedio.
                    No es obligatorio que sean impares, pero es recomendable.
                    Típicamente (3, 3), (5, 5), (7, 7)
        """
        if mascara[0] <= 0 or mascara[1] <= 0:
            raise ValueError("Ambas dimensiones de mascara deben ser > 0")
        
        self.mascara = mascara
    
    def __call__(self, img: np.ndarray) -> np.ndarray:
        """
            Aplica el filtro de caja (promedio).
            
            Args:
                img: Imagen 2D a filtrar
                
            Returns:
                Imagen suavizada del mismo tipo
        """
        self._validar_imagen(img)
        # cv2.blur es el alias para el filtro de caja normalizado
        return cv2.blur(img, self.mascara)

@registrar_en("filtracion")
class Bilateral(FiltroLocal):
    """
        Filtro bilateral para suavizado preservando bordes.
        
        Combina cercanía espacial con similitud de intensidad para suavizar
        áreas homogéneas mientras mantiene bordes nítidos.
        
        Ventajas:
            - Excelente preservación de bordes
            - Suaviza texturas internas sin difuminar estructuras
            - Ideal para microscopía celular
        
        Desventajas:
            - Más lento que filtros lineales
            - Parámetros sensibles, requieren ajuste
            - Puede sobre-suavizar en algunas regiones
        
        Usos típicos:
            - Reducción de ruido en células manteniendo membranas nítidas
            - Preprocesamiento antes de segmentación
            - Mejora de contraste local sin difuminar bordes
        
        Nota:
            Requiere imagen en uint8 o float32. Se convierte automáticamente si es uint16.
    """
    nombre = "bilateral"
    
    def __init__(self, 
                diam: int = 5,
                sigma_color: float = 75.0, 
                sigma_espacio: float = 75.0):
        """
            Args:
                diam: Diámetro de la vecindad de píxeles
                    5 para filtrado rápido en tiempo real
                    9 para procesamiento offline de mayor calidad
                sigma_color: Filtro en el espacio de color (rango de intensidad)
                            Mayor valor: colores más distantes se mezclan
                            Típico: 50-150
                sigma_espacio: Filtro en el espacio de coordenadas (distancia espacial)
                            Mayor valor: píxeles más lejanos se influencian
                            Típico: 50-150
        """
        if diam <= 0:
            raise ValueError("diam debe ser > 0")
        if sigma_color <= 0 or sigma_espacio <= 0:
            raise ValueError("sigma_color y sigma_espacio deben ser > 0")
        
        self.diam = diam
        self.sigma_color = sigma_color
        self.sigma_espacio = sigma_espacio
    
    def __call__(self, img: np.ndarray) -> np.ndarray:
        """
            Aplica el filtro bilateral.
            
            Nota: Si la imagen es uint16, se convierte temporalmente a float32
            para el filtrado, luego se convierte de vuelta.
            
            Args:
                img: Imagen 2D a filtrar
                
            Returns:
                Imagen filtrada del mismo tipo que la entrada
        """
        self._validar_imagen(img)
        
        es_uint16 = img.dtype == np.uint16
        
        if es_uint16:
            # Convertir a float32, filtrar, y volver a uint16
            img_float = img.astype(np.float32)
            img_filtrada = cv2.bilateralFilter(
                img_float,
                self.diam,
                self.sigma_color,
                self.sigma_espacio
            )
            return img_filtrada.astype(np.uint16)
        else:
            return cv2.bilateralFilter(
                img,
                self.diam,
                self.sigma_color,
                self.sigma_espacio
            )

@registrar_en("filtracion")
class DifusionAnisotropica(FiltroLocal):
    """
        Filtro de difusión anisotrópica para suavizado adaptativo.
        
        También conocido como filtro de Perona-Malik, reduce el ruido mediante
        un proceso de difusión que se detiene en los bordes.
        
        Ventajas:
            - Preservación superior de bordes y estructuras
            - Suavizado adaptativo según contenido local
            - Excelente para imágenes de microscopía
        
        Desventajas:
            - Más lento que otros filtros
            - Requiere múltiples iteraciones
            - Parámetros sensibles
        
        Usos típicos:
            - Reducción de ruido en microscopía confocal
            - Preprocesamiento para segmentación de células
            - Mejora de calidad manteniendo detalles finos
        
        Nota:
            No está implementado nativamente en OpenCV, requiere implementación manual
            o uso de bibliotecas especializadas como scikit-image.
    """
    nombre = "difusion_anisotropica"
    
    def __init__(self, 
                n_iter: int = 10,
                kappa: float = 50.0,
                gamma: float = 0.1,
                opcion: int = 1):
        """
            Args:
                n_iter: Número de iteraciones de difusión (típico: 5-20)
                kappa: Constante de conductividad (controla sensibilidad a bordes)
                    Valores más altos preservan más bordes (típico: 20-100)
                gamma: Tasa de difusión, debe cumplir 0 < gamma <= 0.25 para estabilidad
                opcion: Función de conductividad (1: privilegia bordes altos, 2: bordes anchos)
        """
        if n_iter <= 0:
            raise ValueError("n_iter debe ser > 0")
        if kappa <= 0:
            raise ValueError("kappa debe ser > 0")
        if not (0 < gamma <= 0.25):
            raise ValueError("gamma debe estar en (0, 0.25] para estabilidad")
        if opcion not in [1, 2]:
            raise ValueError("opcion debe ser 1 o 2")
        
        self.n_iter = n_iter
        self.kappa = kappa
        self.gamma = gamma
        self.opcion = opcion
    
    def __call__(self, img: np.ndarray) -> np.ndarray:
        """
            Aplica difusión anisotrópica.
            
            Implementación del algoritmo de Perona-Malik:
            I^(t+1) = I^t + gamma * div(c(∇I) * ∇I)
            
            Args:
                img: Imagen 2D a filtrar
                
            Returns:
                Imagen filtrada del mismo tipo
        """
        self._validar_imagen(img)
        
        # Convertir a float para cálculos
        img_float = img.astype(np.float64)
        
        # Iterar proceso de difusión
        for _ in range(self.n_iter):
            # Calcular gradientes en las 4 direcciones (N, S, E, W)
            delta_N = np.roll(img_float, 1, axis=0) - img_float
            delta_S = np.roll(img_float, -1, axis=0) - img_float
            delta_E = np.roll(img_float, -1, axis=1) - img_float
            delta_W = np.roll(img_float, 1, axis=1) - img_float
            
            # Calcular coeficientes de conductividad
            if self.opcion == 1:
                # Función exponencial (privilegia bordes altos)
                c_N = np.exp(-(delta_N / self.kappa) ** 2)
                c_S = np.exp(-(delta_S / self.kappa) ** 2)
                c_E = np.exp(-(delta_E / self.kappa) ** 2)
                c_W = np.exp(-(delta_W / self.kappa) ** 2)
            else:
                # Función racional (privilegia bordes anchos)
                c_N = 1.0 / (1.0 + (delta_N / self.kappa) ** 2)
                c_S = 1.0 / (1.0 + (delta_S / self.kappa) ** 2)
                c_E = 1.0 / (1.0 + (delta_E / self.kappa) ** 2)
                c_W = 1.0 / (1.0 + (delta_W / self.kappa) ** 2)
            
            # Actualizar imagen mediante difusión
            img_float += self.gamma * (c_N * delta_N + c_S * delta_S + 
                                       c_E * delta_E + c_W * delta_W)
        
        # Clipear y convertir de vuelta al tipo original
        if np.issubdtype(img.dtype, np.integer):
            info = np.iinfo(img.dtype)
            img_float = np.clip(img_float, info.min, info.max)
        
        return img_float.astype(img.dtype)