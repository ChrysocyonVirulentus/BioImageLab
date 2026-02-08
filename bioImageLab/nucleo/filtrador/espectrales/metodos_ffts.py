import numpy as np

class FiltroEspectralBase:
    """
        Clase base que gestiona la infraestructura de la Transformada de Fourier discreta.

        F(u,v) = Sumatario(x = 0, M-1) Sumatorio (y = 0, N - 1) f(x,y) e^((-1)2*pi*(ux/M + vy/N))

        donde f(x,y) es imagen de NxM.

        Las subclases solo deben implementar el método 'generar_mascara'. 

        - FFT Pasabajo (Low-pass) : Deja pasar solo el centro del espectro, bloquea la periferia (Elimina las altas frecuencias). Suavizado Extremo (Elimina bordes y ruido fino)
        Se puede usar para estimar el fondo o eliminar el ruido de "grano" muy fino.

        - FFT Pasaalto (High-pass) : Bloquea las frecuencias por debajo de la frecuencia de corte y permite pasar las altas, esencial 
        para detectar border o cambios rápidos. O sea, bloquea el centro y deja la periferia. Realzar membranas o estructuras muy finas ignorando la masa
        de la célula.

        - FFT Pasabanda (Band-pass) : Deja pasar un "anillo" de frecuencias, bloquea el centro y los bordes externos. Sirve
        para aislamiento de estructuras de un tamaño específico. "filtro deoro" para detectar núcleos o spots de un diámetro conocido.

        - FFT Bandstop (Notch) : Bloquea un rango especifico omnidireccional (un anillo o valores aislados). Elimina patrones quitando interferencias
        periódicas. Elimina el ruido de red eléctrica (granulosidad) o patrones de striping del microscopio.

        - FiltradoNotch : Es un filtro puntual o direccional. No bloquea un anillo completo, sino puntos específicos (o pares de puntos simétricos) del
        espectro. Sirve para eliminar puntos o lineas horizontales perfectas (Como un bisturí).
    """
    nombre = "base_fft"

    def generar_mascara(self, forma: tuple, centro: tuple) -> np.ndarray:
        raise NotImplementedError

    def __call__(self, img: np.ndarray) -> np.ndarray:
        # 1. Transformada y centrado
        transformada_f = np.fft.fft2(img.astype(np.float64))
        f_shift = np.fft.fftshift(transformada_f)
        
        # 2. Generar y aplicar máscara
        filas, columnas = img.shape
        cfilas, ccolumnas = rows // 2, cols // 2
        mascara = self.generar_mascara((filas, columnas), (cfilas, ccolumnas))
        
        f_shift_filtrado = f_shift * mascara
        
        # 3. Transformada inversa
        f_ishift = np.fft.ifftshift(f_shift_filtrado)
        img_retornada = np.fft.ifft2(f_ishift)
        
        # Retornamos la magnitud real, manteniendo el tipo de dato original
        return np.abs(img_retornada).astype(img.dtype)

# Modos de filtrado segun los "modos normales" de la imagen

class FFTPasabajo(FiltroEspectralBase):
    nombre = "fft_pasabajo"
    def __init__(self, radio: int = 30):
        self.radio = radio

    def generar_mascara(self, forma, centro):
        y, x = np.ogrid[-centro[0]:forma[0]-centro[0], -centro[1]:forma[1]-centro[1]]
        # Filtro Gaussiano (suave, evita ringing)
        return np.exp(-(x**2 + y**2) / (2 * self.radio**2))

class FFTPasaalto(FiltroEspectralBase):
    nombre = "fft_pasaalto"
    def __init__(self, radio: int = 10):
        self.radio = radio

    def generar_mascara(self, forma, centro):
        y, x = np.ogrid[-centro[0]:forma[0]-centro[0], -centro[1]:forma[1]-centro[1]]
        # 1 - Gaussiano = Pasa-alto suave
        return 1 - np.exp(-(x**2 + y**2) / (2 * self.radio**2))

class FFTPasabanda(FiltroEspectralBase):
    nombre = "fft_pasabanda"
    def __init__(self, r_bajo: int = 5, r_alto: int = 50):
        self.r_bajo = r_bajo
        self.r_alto = r_alto

    def generar_mascara(self, forma, centro):
        y, x = np.ogrid[-centro[0]:forma[0]-centro[0], -centro[1]:forma[1]-centro[1]]
        dist_sq = x**2 + y**2
        # Combinación de dos Gaussianos para crear el "anillo"
        m1 = np.exp(-dist_sq / (2 * self.r_alto**2))
        m2 = np.exp(-dist_sq / (2 * self.r_bajo**2))
        return m1 - m2

class FFTBandstop(FiltroEspectralBase):
    nombre = "fft_bandstop"
    def __init__(self, r_centro: int = 30, ancho: int = 5):
        self.r_centro = r_centro
        self.ancho = ancho

    def generar_mascara(self, forma, centro):
        y, x = np.ogrid[-centro[0]:forma[0]-centro[0], -centro[1]:forma[1]-centro[1]]
        dist = np.sqrt(x**2 + y**2)
        # Filtro Butterworth para crear un "bloqueo" suave en un radio específico
        return 1 / (1 + ( (dist * self.ancho) / (dist**2 - self.r_centro**2) )**(2 * 2))

class FiltradoNotch(FiltroEspectralBase):
    nombre = "filtrado_notch"
    
    def __init__(self, puntos_ruido: list[tuple[int, int]], radio: int = 5):
        """
            Argumentos:
                -puntos_ruido: Lista de coordenadas (u, v) relativas al centro 
                            donde se detectaron picos de interferencia.
                -radio: Qué tan "ancho" es el "bisturí" para borrar el punto.
        """
        self.puntos = puntos_ruido
        self.radio = radio

    def generar_mascara(self, forma, centro):
        y, x = np.ogrid[-centro[0]:forma[0]-centro[0], -centro[1]:forma[1]-centro[1]]
        mascara = np.ones(forma, dtype=np.float64)
        
        for u, v in self.puntos:
            # Filtro Notch Gaussiano para cada punto de ruido y su simétrico
            dist_p1 = (x - u)**2 + (y - v)**2
            dist_p2 = (x + u)**2 + (y + v)**2 # Punto simétrico
            
            notcha = 1 - np.exp(-dist_p1 / (2 * self.radio**2))
            notchb = 1 - np.exp(-dist_p2 / (2 * self.radio**2))
            mascara *= (notcha * notchb)
            
        return mascara