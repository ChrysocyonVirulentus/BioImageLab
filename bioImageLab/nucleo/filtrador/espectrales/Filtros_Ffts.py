"""
Filtros espectrales en el dominio de frecuencias (FFT) para procesamiento de imágenes.

Estos filtros operan sobre la transformada de Fourier de la imagen, permitiendo
manipulación selectiva de frecuencias espaciales.

Características:
- Operan en el dominio de Fourier (frecuencias)
- Permiten manipulación selectiva de componentes frecuenciales
- Excelentes para ruido periódico y patrones estructurados
- Implementación eficiente mediante FFT

Ventajas sobre filtros espaciales:
- Eliminan ruido periódico (ej: interferencia de red eléctrica)
- Pueden aislar rangos específicos de frecuencias
- Más eficientes para kernels grandes

Tipos disponibles:
- Pasabajo: Elimina altas frecuencias (ruido, detalles finos)
- Pasaalto: Elimina bajas frecuencias (iluminación desigual, tendencias)
- Pasabanda: Preserva un rango específico de frecuencias
- Bandstop (Rechaza banda): Elimina un rango específico de frecuencias
- Notch: Elimina frecuencias específicas (ruido periódico)
"""

import numpy as np
from typing import List, Tuple

@registrar_en("filtrado")
class FiltroEspectral:
    """
        Clase base para filtros espectrales basados en FFT.
        
        Maneja la infraestructura común de transformada de Fourier.
        Las subclases solo necesitan implementar la generación de la máscara
        en el dominio de frecuencias.
        
        Pipeline de filtrado:
            1. FFT2D de la imagen
            2. Centrar frecuencias (fftshift)
            3. Aplicar máscara frecuencial
            4. Descentrar (ifftshift)
            5. IFFT2D para volver al dominio espacial
    """
    nombre = "filtro_espectral_base"
    
    def generar_mascara(self, forma: Tuple[int, int], centro: Tuple[int, int]) -> np.ndarray:
        """
            Genera la máscara de filtrado en el dominio de frecuencias.
            
            Args:
                forma: (filas, columnas) de la imagen
                centro: (centro_y, centro_x) de las frecuencias
                
            Returns:
                Array 2D con la máscara (valores entre 0 y 1)
        """
        raise NotImplementedError("Subclases deben implementar generar_mascara")
    
    def __call__(self, img: np.ndarray) -> np.ndarray:
        """
            Aplica el filtro espectral a la imagen.
            
            Args:
                img: Array 2D (Y, X) con la imagen a filtrar
                
            Returns:
                Imagen filtrada del mismo tipo y forma
        """
        if img.ndim != 2:
            raise ValueError(f"La imagen debe ser 2D, tiene {img.ndim} dimensiones")
        
        # 1. Transformada de Fourier 2D y centrado
        transformada_f = np.fft.fft2(img.astype(np.float64))
        f_shift = np.fft.fftshift(transformada_f)
        
        # 2. Generar máscara en el dominio de frecuencias
        filas, columnas = img.shape
        centro_filas, centro_columnas = filas // 2, columnas // 2
        mascara = self.generar_mascara((filas, columnas), (centro_filas, centro_columnas))
        
        # 3. Aplicar máscara (multiplicación en frecuencias)
        f_shift_filtrado = f_shift * mascara
        
        # 4. Transformada inversa (volver al dominio espacial)
        f_ishift = np.fft.ifftshift(f_shift_filtrado)
        img_filtrada = np.fft.ifft2(f_ishift)
        
        # 5. Retornar magnitud real, manteniendo dtype original
        img_resultado = np.abs(img_filtrada)
        
        # Clipear para evitar overflow al convertir de vuelta
        if np.issubdtype(img.dtype, np.integer):
            info = np.iinfo(img.dtype)
            img_resultado = np.clip(img_resultado, info.min, info.max)
        
        return img_resultado.astype(img.dtype)

@registrar_en("filtrado")
class FFTPasaBajo(FiltroEspectral):
    """
        Filtro pasabajo (low-pass) gaussiano en el dominio de frecuencias.
        
        Atenúa las altas frecuencias (detalles finos, ruido) mientras preserva
        las bajas frecuencias (estructuras grandes, tendencias).
        
        Ventajas:
            - Suavizado global sin artefactos de borde
            - Eliminación efectiva de ruido de alta frecuencia
            - Control preciso mediante el radio de corte
        
        Desventajas:
            - Difumina bordes y detalles finos
            - Puede introducir ringing (oscilaciones) si el radio es muy pequeño
        
        Usos típicos:
            - Reducción de ruido electrónico
            - Suavizado de mapas de intensidad
            - Preprocesamiento para análisis de estructuras grandes
            - Eliminación de artefactos de alta frecuencia
    """
    nombre = "fft_pasabajo"
    
    def __init__(self, radio: int = 30):
        """
            Args:
                radio: Radio de corte en píxeles (en el dominio de frecuencias)
                    Valores típicos: 20-50 para microscopía
                    Mayor radio = más suavizado
        """
        if radio <= 0:
            raise ValueError("radio debe ser > 0")
        self.radio = radio
    
    def generar_mascara(self, forma: Tuple[int, int], centro: Tuple[int, int]) -> np.ndarray:
        """
            Genera máscara gaussiana que atenúa altas frecuencias.
            
            Forma: exp(-d²/(2σ²)) donde d es la distancia al centro
        """
        y, x = np.ogrid[-centro[0]:forma[0]-centro[0], -centro[1]:forma[1]-centro[1]]
        return np.exp(-(x**2 + y**2) / (2 * self.radio**2))

@registrar_en("filtrado")
class FFTPasaAlto(FiltroEspectral):
    """
        Filtro pasaalto (high-pass) gaussiano en el dominio de frecuencias.
        
        Atenúa las bajas frecuencias (iluminación desigual, fondos) mientras
        preserva las altas frecuencias (bordes, detalles finos).
        
        Ventajas:
            - Elimina variaciones lentas de iluminación
            - Realza bordes y detalles finos
            - Útil para corrección de fondo
        
        Desventajas:
            - Amplifica ruido de alta frecuencia
            - Reduce contraste global
            - Puede crear halos alrededor de estructuras
        
        Usos típicos:
            - Corrección de iluminación desigual
            - Realce de bordes para segmentación
            - Detección de detalles finos (vesículas, puncta)
            - Eliminación de fondos variables
    """
    nombre = "fft_pasaalto"
    
    def __init__(self, radio: int = 10):
        """
            Args:
                radio: Radio de corte en píxeles (en el dominio de frecuencias)
                    Valores típicos: 5-20 para microscopía
                    Menor radio = más agresivo (elimina más frecuencias bajas)
        """
        if radio <= 0:
            raise ValueError("radio debe ser > 0")
        self.radio = radio
    
    def generar_mascara(self, forma: Tuple[int, int], centro: Tuple[int, int]) -> np.ndarray:
        """
        Genera máscara gaussiana que atenúa bajas frecuencias.
        
        Forma: 1 - exp(-d²/(2σ²)) donde d es la distancia al centro
        """
        y, x = np.ogrid[-centro[0]:forma[0]-centro[0], -centro[1]:forma[1]-centro[1]]
        return 1 - np.exp(-(x**2 + y**2) / (2 * self.radio**2))

@registrar_en("filtrado")
class FFTPasaBanda(FiltroEspectral):
    """
        Filtro pasabanda (band-pass) en el dominio de frecuencias.
        
        Preserva solo un rango específico de frecuencias, atenuando tanto
        las muy bajas como las muy altas.
        
        Ventajas:
            - Aislamiento de estructuras de tamaño específico
            - Eliminación simultánea de fondo y ruido
            - Control preciso del rango de frecuencias
        
        Desventajas:
            - Puede eliminar información útil fuera de la banda
            - Requiere conocimiento previo del tamaño de estructuras de interés
        
        Usos típicos:
            - Detección de estructuras de tamaño específico (ej: células de ~20μm)
            - Análisis de patrones periódicos
            - Eliminación de ruido y fondo simultáneamente
            - Análisis de texturas en rangos específicos de escala
    """
    nombre = "fft_pasabanda"
    
    def __init__(self, r_bajo: int = 5, r_alto: int = 50):
        """
            Args:
                r_bajo: Radio interno (frecuencias bajas a eliminar)
                r_alto: Radio externo (frecuencias altas a eliminar)
                    Valores típicos: r_bajo=5-15, r_alto=30-60
                    La banda preservada está entre r_bajo y r_alto
        """
        if r_bajo >= r_alto:
            raise ValueError("r_bajo debe ser < r_alto")
        if r_bajo <= 0 or r_alto <= 0:
            raise ValueError("r_bajo y r_alto deben ser > 0")
        
        self.r_bajo = r_bajo
        self.r_alto = r_alto
    
    def generar_mascara(self, forma: Tuple[int, int], centro: Tuple[int, int]) -> np.ndarray:
        """
        Genera máscara que preserva banda entre r_bajo y r_alto.
        
        Combina dos gaussianas: una de corte alto menos una de corte bajo
        """
        y, x = np.ogrid[-centro[0]:forma[0]-centro[0], -centro[1]:forma[1]-centro[1]]
        dist_sq = x**2 + y**2
        
        # Gaussiana de corte alto - Gaussiana de corte bajo
        mascara_alta = np.exp(-dist_sq / (2 * self.r_alto**2))
        mascara_baja = np.exp(-dist_sq / (2 * self.r_bajo**2))
        
        return mascara_alta - mascara_baja

@registrar_en("filtrado")
class FFTBandStop(FiltroEspectral):
    """
        Filtro rechaza-banda (band-stop/notch) en el dominio de frecuencias.
        
        Elimina un rango específico de frecuencias mientras preserva el resto.
        También conocido como filtro "band-reject" o "notch filter".
        
        Ventajas:
            - Eliminación selectiva de ruido periódico
            - Preserva frecuencias fuera de la banda rechazada
            - Útil para interferencia de frecuencia conocida
        
        Desventajas:
            - Puede crear artefactos si la banda es muy ancha
            - Requiere conocimiento previo de la frecuencia del ruido
        
        Usos típicos:
            - Eliminación de ruido de red eléctrica (50/60 Hz)
            - Remoción de patrones periódicos de iluminación
            - Corrección de interferencia de barrido (scanning artifacts)
            - Eliminación de franjas periódicas (striping)
    """
    nombre = "fft_bandstop"
    
    def __init__(self, r_centro: int = 30, ancho: int = 5):
        """
            Args:
                r_centro: Radio central de la banda a rechazar (en píxeles)
                ancho: Ancho de la banda de rechazo
                    Valores típicos: r_centro=20-50, ancho=3-10
        """
        if r_centro <= 0 or ancho <= 0:
            raise ValueError("r_centro y ancho deben ser > 0")
        
        self.r_centro = r_centro
        self.ancho = ancho
    
    def generar_mascara(self, forma: Tuple[int, int], centro: Tuple[int, int]) -> np.ndarray:
        """
            Genera máscara Butterworth que rechaza banda alrededor de r_centro.
            
            Forma: 1 / (1 + ((D*W) / (D² - D₀²))^(2n))
            donde D es distancia, D₀ es r_centro, W es ancho, n es orden
        """
        y, x = np.ogrid[-centro[0]:forma[0]-centro[0], -centro[1]:forma[1]-centro[1]]
        dist = np.sqrt(x**2 + y**2)
        
        # Evitar división por cero cuando dist == r_centro
        # Usar valor pequeño para crear un "notch" pronunciado
        denom = np.where(
            np.abs(dist - self.r_centro) < 1e-6,
            1e-6,
            dist**2 - self.r_centro**2
        )
        
        # Filtro Butterworth de orden 2
        return 1.0 / (1.0 + ((dist * self.ancho) / denom)**(2 * 2))

@registrar_en("filtrado")
class FiltradoNotch(FiltroEspectral):
    """
        Filtro notch para eliminación de múltiples frecuencias específicas.
        
        Elimina picos específicos en el espectro de frecuencias, ideal para
        ruido periódico con frecuencias conocidas.
        
        Ventajas:
            - Eliminación quirúrgica de múltiples frecuencias
            - Automáticamente maneja puntos simétricos en el espectro
            - Preserva todas las demás frecuencias
        
        Desventajas:
            - Requiere identificación manual de frecuencias del ruido
            - Puede dejar residuos si el radio es muy pequeño
        
        Usos típicos:
            - Eliminación de ruido de red eléctrica
            - Remoción de patrones Moiré
            - Corrección de interferencia de múltiples fuentes
            - Eliminación de artefactos de escaneo periódicos
        
        Nota:
            Los puntos se especifican en coordenadas relativas al centro del espectro.
            El filtro automáticamente maneja los puntos simétricos.
    """
    nombre = "filtrado_notch"
    
    def __init__(self, puntos_ruido: List[Tuple[int, int]], radio: int = 5):
        """
            Args:
                puntos_ruido: Lista de (u, v) coordenadas en el espectro de Fourier
                            donde se encuentran picos de ruido periódico.
                            Coordenadas relativas al centro del espectro.
                radio: Radio de cada notch gaussiano (en píxeles)
                    Valores típicos: 3-10 píxeles
            
            Ejemplo:
                # Eliminar pico en (20, 30) y su simétrico (-20, -30)
                filtro = FiltradoNotch(puntos_ruido=[(20, 30)], radio=5)
        """
        if not puntos_ruido:
            raise ValueError("puntos_ruido no puede estar vacío")
        if radio <= 0:
            raise ValueError("radio debe ser > 0")
        
        self.puntos = puntos_ruido
        self.radio = radio
    
    def generar_mascara(self, forma: Tuple[int, int], centro: Tuple[int, int]) -> np.ndarray:
        """
            Genera máscara con múltiples notches gaussianos.
            
            Cada punto genera dos notches (el punto y su simétrico) debido a
            la simetría hermitiana de la FFT de señales reales.
        """
        y, x = np.ogrid[-centro[0]:forma[0]-centro[0], -centro[1]:forma[1]-centro[1]]
        mascara = np.ones(forma, dtype=np.float64)
        
        for u, v in self.puntos:
            # Distancia al punto y a su simétrico
            dist_punto = (x - u)**2 + (y - v)**2
            dist_simetrico = (x + u)**2 + (y + v)**2
            
            # Notch gaussiano en ambos puntos
            notch_punto = 1.0 - np.exp(-dist_punto / (2 * self.radio**2))
            notch_simetrico = 1.0 - np.exp(-dist_simetrico / (2 * self.radio**2))
            
            # Multiplicar máscaras (efecto acumulativo)
            mascara *= (notch_punto * notch_simetrico)
        
        return mascara