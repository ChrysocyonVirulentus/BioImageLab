import numpy as np
from typing import List, Tuple

class FiltroEspectralBase:
    """
        Clase base para filtros espectrales basados en FFT.
        Maneja la infraestructura de FFT; subclases implementan 'generar_mascara'.
    """
    nombre = "base_fft"

    def generar_mascara(self, forma: Tuple[int, int], centro: Tuple[int, int]) -> np.ndarray:
        raise NotImplementedError("Subclases deben implementar generar_mascara")

    def __call__(self, img: np.ndarray) -> np.ndarray:
        if img.ndim != 2:
            raise ValueError("La imagen debe ser 2D")
        
        # 1. Transformada y centrado
        transformada_f = np.fft.fft2(img.astype(np.float64))
        f_shift = np.fft.fftshift(transformada_f)
        
        # 2. Generar máscara
        filas, columnas = img.shape
        cfilas, ccolumnas = filas // 2, columnas // 2
        mascara = self.generar_mascara((filas, columnas), (cfilas, ccolumnas))
        
        # 3. Aplicar máscara
        f_shift_filtrado = f_shift * mascara
        
        # 4. Transformada inversa
        f_ishift = np.fft.ifftshift(f_shift_filtrado)
        img_retornada = np.fft.ifft2(f_ishift)
        
        # Retornar magnitud real, manteniendo dtype original
        return np.abs(img_retornada).astype(img.dtype)

class FFTPasabajo(FiltroEspectralBase):
    nombre = "fft_pasabajo"
    def __init__(self, radio: int = 30):
        if radio <= 0:
            raise ValueError("Radio debe ser positivo")
        self.radio = radio

    def generar_mascara(self, forma: Tuple[int, int], centro: Tuple[int, int]) -> np.ndarray:
        y, x = np.ogrid[-centro[0]:forma[0]-centro[0], -centro[1]:forma[1]-centro[1]]
        return np.exp(-(x**2 + y**2) / (2 * self.radio**2))

class FFTPasaalto(FiltroEspectralBase):
    nombre = "fft_pasaalto"
    def __init__(self, radio: int = 10):
        if radio <= 0:
            raise ValueError("Radio debe ser positivo")
        self.radio = radio

    def generar_mascara(self, forma: Tuple[int, int], centro: Tuple[int, int]) -> np.ndarray:
        y, x = np.ogrid[-centro[0]:forma[0]-centro[0], -centro[1]:forma[1]-centro[1]]
        return 1 - np.exp(-(x**2 + y**2) / (2 * self.radio**2))

class FFTPasabanda(FiltroEspectralBase):
    nombre = "fft_pasabanda"
    def __init__(self, r_bajo: int = 5, r_alto: int = 50):
        if r_bajo >= r_alto or r_bajo <= 0 or r_alto <= 0:
            raise ValueError("r_bajo debe ser < r_alto y ambos positivos")
        self.r_bajo = r_bajo
        self.r_alto = r_alto

    def generar_mascara(self, forma: Tuple[int, int], centro: Tuple[int, int]) -> np.ndarray:
        y, x = np.ogrid[-centro[0]:forma[0]-centro[0], -centro[1]:forma[1]-centro[1]]
        dist_sq = x**2 + y**2
        m1 = np.exp(-dist_sq / (2 * self.r_alto**2))
        m2 = np.exp(-dist_sq / (2 * self.r_bajo**2))
        return m1 - m2

class FFTBandstop(FiltroEspectralBase):
    nombre = "fft_bandstop"
    def __init__(self, r_centro: int = 30, ancho: int = 5):
        if r_centro <= 0 or ancho <= 0:
            raise ValueError("r_centro y ancho deben ser positivos")
        self.r_centro = r_centro
        self.ancho = ancho

    def generar_mascara(self, forma: Tuple[int, int], centro: Tuple[int, int]) -> np.ndarray:
        y, x = np.ogrid[-centro[0]:forma[0]-centro[0], -centro[1]:forma[1]-centro[1]]
        dist = np.sqrt(x**2 + y**2)
        # Evita división por cero; usa un valor alto si dist == r_centro
        denom = np.where(dist == self.r_centro, 1e-6, dist**2 - self.r_centro**2)
        return 1 / (1 + ((dist * self.ancho) / denom)**(2 * 2))

class FiltradoNotch(FiltroEspectralBase):
    nombre = "filtrado_notch"
    def __init__(self, puntos_ruido: List[Tuple[int, int]], radio: int = 5):
        if not puntos_ruido or radio <= 0:
            raise ValueError("puntos_ruido no puede estar vacío y radio debe ser positivo")
        self.puntos = puntos_ruido
        self.radio = radio

    def generar_mascara(self, forma: Tuple[int, int], centro: Tuple[int, int]) -> np.ndarray:
        y, x = np.ogrid[-centro[0]:forma[0]-centro[0], -centro[1]:forma[1]-centro[1]]
        mascara = np.ones(forma, dtype=np.float64)
        
        for u, v in self.puntos:
            dist_p1 = (x - u)**2 + (y - v)**2
            dist_p2 = (x + u)**2 + (y + v)**2  # Punto simétrico
            notcha = 1 - np.exp(-dist_p1 / (2 * self.radio**2))
            notchb = 1 - np.exp(-dist_p2 / (2 * self.radio**2))
            mascara *= (notcha * notchb)
        
        return mascara