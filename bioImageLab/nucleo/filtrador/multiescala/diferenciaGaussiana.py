import numpy as np
from ..locales.gaussiano import Gaussiano

# Diferencia de Gaussianas (DoG)
class DiferenciaGaussiana:
    nombre = "dog"

    """
        Funcion atomica de filtrado gaussiano que permite un suavizado espacial general de los pixeles,
        eliminando ruido electronico de fondo. 
        Problema : Produce difuminado de los bordes, como bordes de celulas o estructuras.
    """

    def __init__(self, sigma_pequeno: float = 1.0, sigma_grande: float = 2.0):
        self.g1 = Gaussiano(sigma=sigma_pequeno)
        self.g2 = Gaussiano(sigma=sigma_grande)

    def __call__(self, img: np.ndarray) -> np.ndarray:
        # DoG = G1 - G2
        img_f = img.astype(np.float64)
        return (self.g1(img_f) - self.g2(img_f)).astype(img.dtype)