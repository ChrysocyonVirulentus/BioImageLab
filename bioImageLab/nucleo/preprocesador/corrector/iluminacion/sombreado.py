"""
    Corrección de iluminación sin flat-field, basada en la propia imagen.

    Modelo: I(x,y) = S(x,y) · L(x,y)
    Donde:
        I(x,y) = Imagen observada
        S(x,y) = Señal verdadera (specimen)
        L(x,y) = Iluminación suave estimada

    Métodos:
        - División por mapa real
        - Ajuste polinomial (usa ajuste_superficie.py)
        - Mediana temporal/z (usa ajuste_superficie.py)
        - Low-pass filter + división

    Corrige:
        - Gradientes suaves
        - Iluminación desigual cuando no hay flat-field
"""

import numpy as np
from typing import Optional
# Importar desde el módulo de ajuste de superficies
from ajuste_superficie import (
    AjusteSuperficie,
    AjustePolinomial,
    AjustePlano,
    AjusteMediana
)


class Sombreado:
    """Clase base abstracta para métodos de corrección de luz por sombreado (shading)."""
    nombre = "sombreado_base"
    
    def __call__(self, img: np.ndarray) -> np.ndarray:
        """
        Aplica corrección de sombreado a una imagen 2D.
        
        Args:
            img: Array 2D (Y, X) con la imagen a corregir
            
        Returns:
            Array 2D corregido
        """
        raise NotImplementedError("Este método debe ser implementado por subclases")
    
    def _corregir_division(self, img: np.ndarray, mapa_iluminacion: np.ndarray) -> np.ndarray:
        """
        Método auxiliar para corregir dividiendo por el mapa de iluminación.
        
        Args:
            img: Imagen a corregir
            mapa_iluminacion: Mapa estimado de iluminación
            
        Returns:
            Imagen corregida
        """
        if img.shape != mapa_iluminacion.shape:
            raise ValueError(f"Forma de imagen {img.shape} no coincide con mapa {mapa_iluminacion.shape}")
        
        # Convertir a float
        img_float = img.astype(np.float64)
        mapa_float = mapa_iluminacion.astype(np.float64)
        
        # Normalizar el mapa al promedio de la imagen para preservar intensidades
        mapa_normalizado = mapa_float / mapa_float.mean()
        
        # Evitar división por cero
        mapa_normalizado[mapa_normalizado < 1e-10] = 1e-10
        
        # Corregir
        img_corregida = img_float / mapa_normalizado
        
        # Clipear para evitar valores fuera de rango
        if np.issubdtype(img.dtype, np.integer):
            info = np.iinfo(img.dtype)
            img_corregida = np.clip(img_corregida, info.min, info.max)
        
        return img_corregida.astype(img.dtype)


class SombreadoReal(Sombreado):
    """
    Corrección usando un mapa de sombreado real (medido o calculado externamente).
    
    Operación: I_corregida(x,y) = I(x,y) / (L(x,y) / mean(L))
    
    Se asume que el shading_map provisto viene procesado/normalizado
    si así se requiere externamente.
    """
    nombre = "sombreado_real"
    
    def __init__(self, shading_map: np.ndarray):
        """
        Args:
            shading_map: Array 2D con el mapa de iluminación estimado.
                        Debe tener la misma forma que las imágenes a corregir.
        """
        if shading_map.ndim != 2:
            raise ValueError(f"shading_map debe ser 2D, tiene {shading_map.ndim} dimensiones")
        
        self.map = shading_map.astype(np.float64)
    
    def __call__(self, img: np.ndarray) -> np.ndarray:
        """
        Aplica la corrección dividiendo por el mapa de sombreado.
        
        Args:
            img: Imagen 2D a corregir
            
        Returns:
            Imagen corregida del mismo tipo que la entrada
        """
        return self._corregir_division(img, self.map)


class SombreadoAjuste(Sombreado):
    """
        Corrección estimando el sombreado mediante un ajuste de superficie.
        
        Usa el módulo ajuste_superficie.py para ajustar diferentes tipos de superficies
        (polinomial, plano, gaussiano, etc.) que modelan la iluminación desigual.
    """
    nombre = "sombreado_ajuste"
    
    def __init__(self, metodo_ajuste: AjusteSuperficie):
        """
        Args:
            metodo_ajuste: Instancia de AjusteSuperficie (AjustePolinomial, AjustePlano, etc.)
        """
        self.metodo_ajuste = metodo_ajuste
    
    def __call__(self, img: np.ndarray) -> np.ndarray:
        """
        Estima el sombreado mediante ajuste y corrige la imagen.
        
        Args:
            img: Imagen 2D a corregir
            
        Returns:
            Imagen corregida del mismo tipo que la entrada
        """
        if img.ndim != 2:
            raise ValueError(f"img debe ser 2D, tiene {img.ndim} dimensiones")
        
        # Ajustar superficie a la imagen (esto estima el mapa de iluminación)
        mapa_iluminacion = self.metodo_ajuste(img)
        
        # Corregir dividiendo
        return self._corregir_division(img, mapa_iluminacion)


class SombreadoMedianaTemporal(Sombreado):
    """
        Corrección usando la mediana temporal/z-stack como estimación del sombreado.
        
        Útil cuando tienes múltiples imágenes (time-lapse o z-stack) y asumes que
        la mediana representa el fondo/iluminación sin la señal de interés.
        
        Usa AjusteMediana del módulo ajuste_superficie.py.
    """
    nombre = "sombreado_mediana_temporal"
    
    def __init__(self, stack_referencia: np.ndarray, axis: int = 0):
        """
        Args:
            stack_referencia: Array 3D [N, Y, X] donde N es tiempo o z.
            axis: Eje a lo largo del cual calcular la mediana (default: 0)
        """
        # Usar el módulo de ajuste para calcular la mediana
        self.ajuste_mediana = AjusteMediana(stack_referencia, axis=axis)
    
    def __call__(self, img: np.ndarray) -> np.ndarray:
        """
        Aplica la corrección dividiendo por la mediana temporal.
        
        Args:
            img: Imagen 2D a corregir
            
        Returns:
            Imagen corregida del mismo tipo que la entrada
        """
        if img.ndim != 2:
            raise ValueError(f"img debe ser 2D, tiene {img.ndim} dimensiones")
        
        # Obtener mapa de mediana
        mapa_mediana = self.ajuste_mediana(img)
        
        # Corregir dividiendo
        return self._corregir_division(img, mapa_mediana)


class SombreadoLowPass(Sombreado):
    """
    Corrección usando filtro pasa-bajos para estimar iluminación de fondo.
    
    Aplica un filtro gaussiano de gran sigma para obtener una versión suave
    de la imagen que representa la iluminación, luego divide.
    """
    nombre = "sombreado_lowpass"
    
    def __init__(self, sigma: float = 50):
        """
        Args:
            sigma: Desviación estándar del filtro gaussiano.
                  Valores grandes (30-100) para capturar gradientes suaves.
        """
        if sigma <= 0:
            raise ValueError("sigma debe ser > 0")
        
        self.sigma = sigma
    
    def __call__(self, img: np.ndarray) -> np.ndarray:
        """
        Estima el sombreado mediante filtrado pasa-bajos y corrige.
        
        Args:
            img: Imagen 2D a corregir
            
        Returns:
            Imagen corregida del mismo tipo que la entrada
        """
        if img.ndim != 2:
            raise ValueError(f"img debe ser 2D, tiene {img.ndim} dimensiones")
        
        # Importar scipy para filtrado gaussiano
        try:
            from scipy.ndimage import gaussian_filter
        except ImportError:
            raise ImportError("Se requiere scipy para SombreadoLowPass")
        
        # Aplicar filtro gaussiano para estimar fondo
        img_float = img.astype(np.float64)
        mapa_iluminacion = gaussian_filter(img_float, sigma=self.sigma)
        
        # Corregir dividiendo
        return self._corregir_division(img, mapa_iluminacion)