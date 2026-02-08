"""
    Módulo para ajuste de superficies matemáticas a datos 2D.

    Proporciona diferentes métodos de ajuste que pueden ser usados por otros módulos
    como corrección de iluminación, modelado de PSF, etc.
"""

import numpy as np
from typing import Tuple, Optional


class AjusteSuperficie:
    """Clase base abstracta para métodos de ajuste de superficies."""
    nombre = "ajuste_base"
    
    def ajustar(self, img: np.ndarray) -> np.ndarray:
        """
        Ajusta una superficie a los datos de la imagen.
        
        Argumentos:
            img: Array 2D (Y, X) con los datos a ajustar
            
        Retorna:
            Array 2D con la superficie ajustada
        """
        raise NotImplementedError("Este método debe ser implementado por subclases")
    
    def __call__(self, img: np.ndarray) -> np.ndarray:
        """Alias para ajustar()."""
        return self.ajustar(img)


class AjustePolinomial(AjusteSuperficie):
    """
        Ajusta una superficie polinomial 2D mediante mínimos cuadrados.
        
        Para grado n, genera términos: x^i * y^j donde i+j <= n
        Ejemplo grado 2: [1, x, y, x^2, xy, y^2]
    """
    nombre = "ajuste_polinomial"
    
    def __init__(self, grado: int = 2, normalizar: bool = True):
        """
        Argumento:
            grado: Grado del polinomio 2D (1=plano, 2=cuadrático, 3=cúbico)
            normalizar: Si True, normaliza la superficie al rango [0, 1]
        """
        if grado < 1:
            raise ValueError("El grado debe ser >= 1")
        
        self.grado = grado
        self.normalizar = normalizar
        self.coeficientes: Optional[np.ndarray] = None
    
    def _construir_matriz_diseno(self, h: int, w: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Construye la matriz de diseño para el ajuste polinomial.
        
        Argumentos:
            h: Altura de la imagen
            w: Ancho de la imagen
            
        Retorna:
            Tupla (A, x_norm, y_norm) donde A es la matriz de diseño

        Coste:
            O(n^2)
        """
        # Crear grilla de coordenadas
        y, x = np.mgrid[0:h, 0:w]
        
        # Normalizar coordenadas a [0, 1] para estabilidad numérica
        x_norm = x / w
        y_norm = y / h
        
        # Construir términos del polinomio
        terms = []
        for i in range(self.grado + 1):
            for j in range(self.grado + 1 - i):
                terms.append((x_norm ** i) * (y_norm ** j))
        
        # Matriz de diseño: cada columna es un término del polinomio
        A = np.column_stack([term.ravel() for term in terms])
        
        return A, x_norm, y_norm
    
    def ajustar(self, img: np.ndarray) -> np.ndarray:
        """
        Ajusta una superficie polinomial a la imagen.
        
        Args:
            img: Imagen 2D a ajustar
            
        Returns:
            Superficie polinomial ajustada
        """
        if img.ndim != 2:
            raise ValueError(f"img debe ser 2D, tiene {img.ndim} dimensiones")
        
        h, w = img.shape
        img_float = img.astype(np.float64)
        
        # Construir matriz de diseño
        A, x_norm, y_norm = self._construir_matriz_diseno(h, w)
        b = img_float.ravel()
        
        # Resolver sistema de mínimos cuadrados: A·coef = b
        self.coeficientes, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
        
        # Evaluar el polinomio ajustado
        superficie = A @ self.coeficientes
        superficie = superficie.reshape(h, w)
        
        # Normalizar si se solicita
        if self.normalizar:
            superficie = superficie - superficie.min()
            if superficie.max() > 0:
                superficie = superficie / superficie.max()
        
        return superficie
    
    def evaluar(self, h: int, w: int) -> np.ndarray:
        """
        Evalúa el polinomio ajustado en una grilla de tamaño (h, w).
        
        Args:
            h: Altura de la grilla
            w: Ancho de la grilla
            
        Returns:
            Superficie evaluada
            
        Raises:
            RuntimeError: Si no se ha ajustado ningún polinomio
        """
        if self.coeficientes is None:
            raise RuntimeError("Primero debes ajustar el polinomio con ajustar()")
        
        A, _, _ = self._construir_matriz_diseno(h, w)
        superficie = A @ self.coeficientes
        superficie = superficie.reshape(h, w)
        
        if self.normalizar:
            superficie = superficie - superficie.min()
            if superficie.max() > 0:
                superficie = superficie / superficie.max()
        
        return superficie


class AjusteGaussiano(AjusteSuperficie):
    """
    Ajusta una superficie Gaussiana 2D a los datos.
    
    Forma: A * exp(-((x-x0)²/2σx² + (y-y0)²/2σy²))
    
    Útil para modelar PSF, spots, o iluminación con perfil gaussiano.
    """
    nombre = "ajuste_gaussiano"
    
    def __init__(self):
        """Inicializa el ajuste gaussiano."""
        self.parametros: Optional[dict] = None
    
    def ajustar(self, img: np.ndarray) -> np.ndarray:
        """
        Ajusta una gaussiana 2D a la imagen.
        
        Args:
            img: Imagen 2D a ajustar
            
        Returns:
            Gaussiana 2D ajustada
        """
        if img.ndim != 2:
            raise ValueError(f"img debe ser 2D, tiene {img.ndim} dimensiones")
        
        h, w = img.shape
        img_float = img.astype(np.float64)
        
        # Estimación inicial de parámetros
        y, x = np.mgrid[0:h, 0:w]
        
        # Centro de masa como estimación inicial del centroide
        total = img_float.sum()
        if total == 0:
            x0, y0 = w / 2, h / 2
        else:
            x0 = (x * img_float).sum() / total
            y0 = (y * img_float).sum() / total
        
        # Amplitud como el valor máximo
        A = img_float.max()
        
        # Sigma como 1/6 del ancho (regla empírica)
        sigma_x = w / 6
        sigma_y = h / 6
        
        # Construir gaussiana con parámetros estimados
        # (Para ajuste más preciso, usar scipy.optimize.curve_fit)
        gauss = A * np.exp(-((x - x0)**2 / (2 * sigma_x**2) + 
                             (y - y0)**2 / (2 * sigma_y**2)))
        
        # Guardar parámetros
        self.parametros = {
            'amplitud': A,
            'x0': x0,
            'y0': y0,
            'sigma_x': sigma_x,
            'sigma_y': sigma_y
        }
        
        return gauss


class AjustePlano(AjusteSuperficie):
    """
    Ajusta un plano a los datos (polinomio grado 1).
    
    Forma: a*x + b*y + c
    
    Útil para corregir gradientes lineales de iluminación.
    """
    nombre = "ajuste_plano"
    
    def __init__(self):
        """Inicializa el ajuste de plano."""
        self.coeficientes: Optional[np.ndarray] = None
    
    def ajustar(self, img: np.ndarray) -> np.ndarray:
        """
        Ajusta un plano a la imagen.
        
        Args:
            img: Imagen 2D a ajustar
            
        Returns:
            Plano ajustado
        """
        if img.ndim != 2:
            raise ValueError(f"img debe ser 2D, tiene {img.ndim} dimensiones")
        
        h, w = img.shape
        img_float = img.astype(np.float64)
        
        # Crear grilla de coordenadas normalizadas
        y, x = np.mgrid[0:h, 0:w]
        x_norm = x / w
        y_norm = y / h
        
        # Matriz de diseño para plano: [1, x, y]
        ones = np.ones((h, w))
        A = np.column_stack([ones.ravel(), x_norm.ravel(), y_norm.ravel()])
        b = img_float.ravel()
        
        # Resolver mínimos cuadrados
        self.coeficientes, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        
        # Evaluar plano
        plano = A @ self.coeficientes
        plano = plano.reshape(h, w)
        
        return plano


class AjusteMediana(AjusteSuperficie):
    """
    "Ajuste" usando la mediana de un stack 3D.
    
    No es un ajuste matemático per se, sino una estimación estadística
    útil para time-lapse o z-stacks.
    """
    nombre = "ajuste_mediana"
    
    def __init__(self, stack: np.ndarray, axis: int = 0):
        """
        Args:
            stack: Array 3D [N, Y, X] o [Y, X, N]
            axis: Eje a lo largo del cual calcular la mediana (default: 0)
        """
        if stack.ndim != 3:
            raise ValueError(f"stack debe ser 3D, tiene {stack.ndim} dimensiones")
        
        self.mediana = np.median(stack, axis=axis).astype(np.float64)
    
    def ajustar(self, img: np.ndarray) -> np.ndarray:
        """
        Retorna la mediana calculada (ignora img).
        
        Args:
            img: Ignorado (solo para consistencia de interfaz)
            
        Returns:
            Mediana del stack
        """
        return self.mediana