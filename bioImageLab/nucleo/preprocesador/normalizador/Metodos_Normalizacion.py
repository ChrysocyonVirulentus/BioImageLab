import numpy as np 

# Funciones para normalizar
# Se aceptan que esten en float32 o float64

class MetodoNormalizacion:
    nombre = "base_norm"

    """
        Clase base abstracta para métodos de normalización.
    """
    
    def __call__(self, data: np.ndarray) -> np.ndarray:
        raise NotImplementedError

# Normalizar solo por el pixel maximo.
class MaxNorm(MetodoNormalizacion):
    nombre = "max_norm"

    """
        Normaliza dividiendo por el valor máximo.
        
        Argumentos:
            img: Array de cualquier forma
            
        Retorna:
            Array normalizado a [0, 1]
    """
    def __call__(self, img: np.ndarray) -> np.ndarray:
        return img / img.max() if img.max() > 0 else img.astype(np.float64)

# Normalizar por pixel maximo y minimo.abs
class MinMaxNorm(MetodoNormalizacion):
    nombre = "min_max_norm"

    """
        Normaliza al rango [0, 1] usando min-max scaling.
        
        Argumento:
            img: Array de cualquier forma
            
        Retorna:
            Array normalizado a [0, 1]
    """

    def __call__(self, img: np.ndarray) -> np.ndarray:
        return (img - img.min()) / (img.max() - img.min()) if img.max() != img.min() else img.astype(np.float64)

# Normalizar por percentil:
class PercentilNorm(MetodoNormalizacion):
    nombre = "percentil_norm"
    def __init__(self, p_bajo: int = 2, p_alto: int = 98):
        self.p_bajo, self.p_alto = p_bajo, p_alto
    
    def __call__(self, img: np.ndarray) -> np.ndarray:

        """
            Normaliza usando percentiles y clipea a [0, 1].
            
            Argumento:
                img: Array de cualquier forma
                
            Retorna:
                Array normalizado y clipeado a [0, 1]
        """

        # Variable local por pura eficiencia de CPU (O(N log N))
        limites = np.percentile(img, [self.p_bajo, self.p_alto])
        return np.clip((img - limites[0]) / (limites[1] - limites[0]), 0, 1) if limites[1] > limites[0] else img.astype(np.float64)

# Normalizar por ZScore
class ZScoreNorm(MetodoNormalizacion):
    nombre = "zscore_norm"

    """
        Normaliza restando media y dividiendo por desviación estándar.
        Nota: El resultado NO está en [0, 1], sino centrado en 0.
        
        Argumento:
            img: Array de cualquier forma
            
        Retorna:
            Array estandarizado (media=0, sigma=1)
    """
    
    def __call__(self, img: np.ndarray) -> np.ndarray:
        return (img - img.mean()) / img.std() if img.std() > 0 else img.astype(np.float64)