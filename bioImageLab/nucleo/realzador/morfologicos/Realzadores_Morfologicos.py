"""
Operadores morfológicos para procesamiento de imágenes binarias y de escala de grises.

Los operadores morfológicos analizan y modifican la forma (morfología) de objetos
en la imagen usando elementos estructurantes. Originalmente para imágenes binarias,
se extienden a escala de grises mediante erosión/dilatación.

Operaciones fundamentales:
- Erosión: Reduce/adelgaza objetos, elimina píxeles en bordes
- Dilatación: Expande/engrosa objetos, agrega píxeles en bordes

Operaciones compuestas:
- Apertura = Erosión + Dilatación (elimina pequeños objetos brillantes)
- Cierre = Dilatación + Erosión (rellena huecos pequeños)
- Top-Hat = Original - Apertura (extrae objetos pequeños brillantes)
- Bottom-Hat = Cierre - Original (extrae objetos pequeños oscuros)
- Gradiente = Dilatación - Erosión (detecta bordes)
- Reconstrucción = Propagación geodésica (recupera formas específicas)

Usos típicos en microscopía:
- Separar objetos tocantes
- Eliminar ruido preservando forma
- Detectar bordes y contornos
- Rellenar huecos en segmentación
- Extraer objetos de tamaños específicos
- Conectar regiones fragmentadas
"""

import numpy as np
import cv2
from typing import Tuple, Optional, Literal
import warnings


class RealzadorMorfologico:
    """
        Clase base para operadores morfológicos.
        
        Los operadores morfológicos manipulan la forma de objetos en la imagen
        mediante elementos estructurantes (kernels) que definen la vecindad
        y conectividad.
        
        Elemento estructurante:
            - Define la forma de la operación morfológica
            - Común: rectangular, elíptico, cruz
            - Tamaño determina el alcance del operador
    """
    nombre = "operador_morfologico_base"
    
    def __call__(self, img: np.ndarray) -> np.ndarray:
        """
            Aplica el operador morfológico a la imagen.
            
            Args:
                img: Array 2D (Y, X) con la imagen a procesar
                
            Returns:
                Imagen procesada del mismo tipo y forma
        """
        raise NotImplementedError("Subclases deben implementar __call__")
    
    def _validar_imagen(self, img: np.ndarray):
        """Valida que la imagen sea 2D."""
        if img.ndim != 2:
            raise ValueError(f"La imagen debe ser 2D, tiene {img.ndim} dimensiones")
    
    def _crear_elemento_estructurante(self, 
                                    forma: Literal['rect', 'elipse', 'cruz'],
                                    tamanio: Tuple[int, int]) -> np.ndarray:
        """
            Crea un elemento estructurante de forma y tamaño especificados.
            
            Args:
                forma: Tipo de elemento ('rect', 'elipse', 'cruz')
                tamaño: Tupla (ancho, alto) del elemento
                
            Returns:
                Array 2D con el elemento estructurante
        """
        if tamanio[0] < 1 or tamanio[1] < 1:
            raise ValueError("tamaño debe tener valores >= 1")
        
        if forma == 'rect':
            return cv2.getStructuringElement(cv2.MORPH_RECT, tamanio)
        elif forma == 'elipse':
            return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, tamanio)
        elif forma == 'cruz':
            return cv2.getStructuringElement(cv2.MORPH_CROSS, tamanio)
        else:
            raise ValueError(f"forma '{forma}' no válida. Usar 'rect', 'elipse' o 'cruz'")

@registrar_en("realzado")
class Apertura(RealzadorMorfologico):
    """
        Operador de apertura morfológica (erosión seguida de dilatación).
        
        Secuencia: Imagen → Erosión → Dilatación → Resultado
        
        Efecto:
            - Elimina objetos pequeños brillantes (ruido, puncta)
            - Suaviza contornos de objetos
            - Separa objetos conectados por puentes delgados
            - Preserva área aproximada de objetos grandes
        
        Ecuación (teoría de conjuntos):
            A ∘ B = (A ⊖ B) ⊕ B
            donde ⊖ es erosión, ⊕ es dilatación, B es el elemento estructurante
        
        Ventajas:
            - Elimina ruido sin erosionar mucho los objetos
            - Separa objetos tocantes
            - Suaviza bordes irregulares
            - Elimina protuberancias pequeñas
        
        Desventajas:
            - Reduce tamaño de objetos ligeramente
            - Puede eliminar estructuras delgadas deseadas
            - No recupera forma exacta después de erosión
        
        Usos típicos en microscopía:
            - Limpiar segmentación de núcleos (eliminar puncta)
            - Separar células ligeramente tocantes
            - Eliminar artefactos pequeños en máscara binaria
            - Preprocesamiento antes de watershed
            - Suavizar bordes de objetos segmentados
    """
    nombre = "apertura"
    
    def __init__(self, 
                tamanio: Tuple[int, int] = (3, 3),
                forma: Literal['rect', 'elipse', 'cruz'] = 'elipse',
                iteraciones: int = 1):
        """
            Args:
                tamaño: Tupla (ancho, alto) del elemento estructurante
                    Valores típicos: (3,3), (5,5), (7,7)
                    Mayor tamaño = elimina objetos más grandes
                forma: Tipo de elemento estructurante
                    'elipse': mejor para objetos circulares (células, núcleos)
                    'rect': para objetos rectangulares
                    'cruz': para conectividad 4-vecindad
                iteraciones: Número de veces que aplicar la operación
                            Mayor iteraciones = efecto más agresivo
        """
        if iteraciones < 1:
            raise ValueError("iteraciones debe ser >= 1")
        
        self.tamanio = tamanio
        self.forma = forma
        self.iteraciones = iteraciones
        self.kernel = self._crear_elemento_estructurante(forma, tamanio)
    
    def __call__(self, img: np.ndarray) -> np.ndarray:
        """
            Aplica apertura morfológica.
            
            Args:
                img: Imagen 2D (puede ser binaria o escala de grises)
                
            Returns:
                Imagen con apertura aplicada
        """
        self._validar_imagen(img)
        
        return cv2.morphologyEx(
            img,
            cv2.MORPH_OPEN,
            self.kernel,
            iterations=self.iteraciones
        )

@registrar_en("realzado")
class Cierre(RealzadorMorfologico):
    """
        Operador de cierre morfológico (dilatación seguida de erosión).
        
        Secuencia: Imagen → Dilatación → Erosión → Resultado
        
        Efecto:
            - Rellena huecos pequeños en objetos
            - Conecta objetos cercanos
            - Suaviza contornos (llena concavidades)
            - Elimina objetos pequeños oscuros (huecos, ruido)
        
        Ecuación (teoría de conjuntos):
            A • B = (A ⊕ B) ⊖ B
            donde ⊕ es dilatación, ⊖ es erosión, B es el elemento estructurante
        
        Ventajas:
            - Rellena huecos internos en objetos
            - Conecta regiones fragmentadas
            - Suaviza bordes (rellena concavidades)
            - Preserva tamaño aproximado de objetos
        
        Desventajas:
            - Puede unir objetos que deberían estar separados
            - Expande ligeramente los objetos
            - Puede eliminar detalles finos internos
        
        Usos típicos en microscopía:
            - Rellenar huecos en núcleos segmentados
            - Conectar fragmentos de neuritas
            - Completar membranas celulares fragmentadas
            - Suavizar bordes de células después de binarización
            - Unir regiones de fluorescencia fragmentada
    """
    nombre = "cierre"
    
    def __init__(self,
                tamanio: Tuple[int, int] = (3, 3),
                forma: Literal['rect', 'elipse', 'cruz'] = 'elipse',
                iteraciones: int = 1):
        """
            Args:
                tamaño: Tupla (ancho, alto) del elemento estructurante
                    Valores típicos: (3,3), (5,5), (7,7)
                    Mayor tamaño = rellena huecos más grandes
                forma: Tipo de elemento estructurante
                    'elipse': mejor para objetos circulares
                    'rect': para objetos rectangulares
                    'cruz': para conectividad 4-vecindad
                iteraciones: Número de veces que aplicar la operación
                            Mayor iteraciones = rellena huecos más grandes
        """
        if iteraciones < 1:
            raise ValueError("iteraciones debe ser >= 1")
        
        self.tamanio = tamanio
        self.forma = forma
        self.iteraciones = iteraciones
        self.kernel = self._crear_elemento_estructurante(forma, tamanio)
    
    def __call__(self, img: np.ndarray) -> np.ndarray:
        """
            Aplica cierre morfológico.
            
            Args:
                img: Imagen 2D (puede ser binaria o escala de grises)
                
            Returns:
                Imagen con cierre aplicado
        """
        self._validar_imagen(img)
        
        return cv2.morphologyEx(
            img,
            cv2.MORPH_CLOSE,
            self.kernel,
            iterations=self.iteraciones
        )

@registrar_en("realzado")
class TopHat(RealzadorMorfologico):
    """
        Operador Top-Hat (White Top-Hat) para extracción de objetos brillantes pequeños.
        
        Operación: TopHat(I) = I - Apertura(I)
        
        Efecto:
            - Extrae objetos brillantes más pequeños que el elemento estructurante
            - Realza estructuras pequeñas sobre fondo variable
            - Corrige iluminación desigual (similar a high-pass)
            - Detecta spots, puncta, vesículas
        
        Interpretación:
            La apertura elimina detalles pequeños → restarla deja solo los detalles eliminados
        
        Ventajas:
            - Detecta objetos brillantes de tamaño específico
            - Insensible a variaciones lentas de fondo
            - No requiere umbral fijo
            - Útil para fondos no uniformes
        
        Desventajas:
            - Sensible al tamaño del elemento estructurante
            - Puede crear artefactos en bordes
            - No funciona bien si objetos son más grandes que el kernel
        
        Usos típicos en microscopía:
            - Detección de puncta/spots de fluorescencia
            - Extracción de vesículas intracelulares
            - Detección de cromosomas en mitosis
            - Realce de mitocondrias puntuales
            - Corrección de fondo desigual en fluorescencia
            - Preprocesamiento para detección de blobs
    """
    nombre = "top_hat"
    
    def __init__(self,
                tamanio: Tuple[int, int] = (15, 15),
                forma: Literal['rect', 'elipse', 'cruz'] = 'elipse'):
        """
            Args:
                tamaño: Tupla (ancho, alto) del elemento estructurante
                    IMPORTANTE: Debe ser mayor que los objetos a detectar
                    Típicamente 2-3x el tamaño del objeto de interés
                    Ejemplos:
                        - Puncta pequeños: (5, 5) - (11, 11)
                        - Vesículas: (11, 11) - (21, 21)
                        - Núcleos pequeños: (21, 21) - (31, 31)
                forma: Tipo de elemento estructurante
                    'elipse': mejor para objetos circulares (recomendado)
                    'rect': para objetos rectangulares
        """
        self.tamanio = tamanio
        self.forma = forma
        self.kernel = self._crear_elemento_estructurante(forma, tamanio)
    
    def __call__(self, img: np.ndarray) -> np.ndarray:
        """
            Aplica Top-Hat morfológico.
            
            Args:
                img: Imagen 2D en escala de grises
                
            Returns:
                Imagen con objetos pequeños brillantes realzados
        """
        self._validar_imagen(img)
        
        return cv2.morphologyEx(
            img,
            cv2.MORPH_TOPHAT,
            self.kernel
        )

@registrar_en("realzado")
class BottomHat(RealzadorMorfologico):
    """
        Operador Bottom-Hat (Black Top-Hat) para extracción de objetos oscuros pequeños.
        
        Operación: BottomHat(I) = Cierre(I) - I
        
        Efecto:
            - Extrae objetos oscuros más pequeños que el elemento estructurante
            - Realza valles y depresiones en la imagen
            - Detecta huecos, grietas, estructuras oscuras pequeñas
            - Dual del Top-Hat
        
        Interpretación:
            El cierre rellena valles pequeños → restarlo deja solo los valles rellenados
        
        Ventajas:
            - Detecta objetos oscuros de tamaño específico
            - Complementario a Top-Hat
            - Útil para defectos o huecos
            - Insensible a variaciones de fondo
        
        Desventajas:
            - Menos usado que Top-Hat en microscopía
            - Sensible al tamaño del elemento estructurante
            - Puede amplificar ruido oscuro
        
        Usos típicos en microscopía:
            - Detección de vacuolas
            - Identificación de huecos en tejidos
            - Detección de defectos en imágenes de campo claro
            - Extracción de estructuras oscuras (ej: canales iónicos en contraste de fase)
            - Análisis de porosidad
    """
    nombre = "bottom_hat"
    
    def __init__(self,
                tamanio: Tuple[int, int] = (15, 15),
                forma: Literal['rect', 'elipse', 'cruz'] = 'elipse'):
        """
            Args:
                tamaño: Tupla (ancho, alto) del elemento estructurante
                    Debe ser mayor que los objetos oscuros a detectar
                    Similar a Top-Hat pero para estructuras oscuras
                forma: Tipo de elemento estructurante
                    'elipse': mejor para objetos circulares (recomendado)
        """
        self.tamanio = tamanio
        self.forma = forma
        self.kernel = self._crear_elemento_estructurante(forma, tamanio)
    
    def __call__(self, img: np.ndarray) -> np.ndarray:
        """
            Aplica Bottom-Hat morfológico.
            
            Args:
                img: Imagen 2D en escala de grises
                
            Returns:
                Imagen con objetos pequeños oscuros realzados
        """
        self._validar_imagen(img)
        
        return cv2.morphologyEx(
            img,
            cv2.MORPH_BLACKHAT,
            self.kernel
        )

@registrar_en("realzado")
class GradienteMorfologico(RealzadorMorfologico):
    """
        Operador de gradiente morfológico para detección de bordes.
        
        Tipos:
            - Gradiente básico: Dilatación - Erosión
            - Gradiente interno: Original - Erosión
            - Gradiente externo: Dilatación - Original
        
        Efecto:
            - Detecta bordes/contornos de objetos
            - Más robusto al ruido que gradientes de intensidad (Sobel, etc.)
            - Bordes más gruesos que operadores de gradiente clásicos
            - Funciona bien en imágenes binarias y de escala de grises
        
        Ecuación:
            Gradiente básico: G = δ(I) - ε(I)
            donde δ es dilatación, ε es erosión
        
        Ventajas:
            - Robusto ante ruido
            - Funciona bien en binarias y escala de grises
            - Bordes cerrados (útil para segmentación)
            - Menos sensible a cambios de iluminación
        
        Desventajas:
            - Bordes más gruesos que Sobel/Canny
            - Menos preciso en localización
            - Sensible al tamaño del elemento estructurante
        
        Usos típicos en microscopía:
            - Detección de membranas celulares
            - Extracción de contornos de núcleos
            - Preprocesamiento para watershed
            - Análisis de bordes en imágenes binarias
            - Detección de límites de tejidos
    """
    nombre = "gradiente_morfologico"
    
    def __init__(self,
                tamanio: Tuple[int, int] = (3, 3),
                forma: Literal['rect', 'elipse', 'cruz'] = 'elipse',
                tipo: Literal['basico', 'interno', 'externo'] = 'basico'):
        """
            Args:
                tamaño: Tupla (ancho, alto) del elemento estructurante
                    Valores típicos: (3,3), (5,5)
                    Mayor tamaño = bordes más gruesos
                forma: Tipo de elemento estructurante
                    'elipse': bordes más suaves (recomendado)
                    'rect': bordes más angulados
                    'cruz': bordes más delgados
                tipo: Tipo de gradiente
                    'basico': Dilatación - Erosión (bordes completos)
                    'interno': Original - Erosión (borde interior)
                    'externo': Dilatación - Original (borde exterior)
        """
        if tipo not in ['basico', 'interno', 'externo']:
            raise ValueError("tipo debe ser 'basico', 'interno' o 'externo'")
        
        self.tamanio = tamanio
        self.forma = forma
        self.tipo = tipo
        self.kernel = self._crear_elemento_estructurante(forma, tamanio)
    
    def __call__(self, img: np.ndarray) -> np.ndarray:
        """
            Aplica gradiente morfológico.
            
            Args:
                img: Imagen 2D (binaria o escala de grises)
                
            Returns:
                Imagen con bordes detectados
        """
        self._validar_imagen(img)
        
        if self.tipo == 'basico':
            return cv2.morphologyEx(img, cv2.MORPH_GRADIENT, self.kernel)
        
        elif self.tipo == 'interno':
            # Gradiente interno: Original - Erosión
            erosion = cv2.erode(img, self.kernel)
            return cv2.subtract(img, erosion)
        
        else:  # externo
            # Gradiente externo: Dilatación - Original
            dilation = cv2.dilate(img, self.kernel)
            return cv2.subtract(dilation, img)

@registrar_en("realzado")
class ReconstruccionMorfologica(RealzadorMorfologico):
    """
        Reconstrucción morfológica por dilatación geodésica.
        
        Concepto:
            Propaga una imagen marcador dentro de una imagen máscara,
            reconstruyendo solo las regiones conectadas a los marcadores.
        
        Proceso:
            1. Imagen marcador: puntos/regiones semilla
            2. Imagen máscara: define límites de propagación
            3. Dilatación geodésica iterativa: el marcador crece dentro de la máscara
            4. Resultado: objetos completos conectados a marcadores
        
        Ecuación recursiva:
            R^(n+1) = (R^n ⊕ B) ∧ M
            donde R^n es la reconstrucción en iteración n, B es elemento estructurante,
            M es la máscara, ⊕ es dilatación, ∧ es mínimo (intersección)
        
        Ventajas:
            - Recupera objetos específicos (los marcados)
            - Elimina objetos no marcados
            - Preserva forma exacta de objetos
            - Base para operadores más complejos
        
        Desventajas:
            - Requiere definir marcadores apropiados
            - Computacionalmente costoso (iterativo)
            - Sensible a la elección de marcadores
        
        Usos típicos en microscopía:
            - Selección de células específicas de interés
            - Limpieza de segmentación (eliminar falsos positivos)
            - Extracción de regiones conectadas específicas
            - Implementación de h-maxima y h-minima
            - Base para watershed mejorado
            - Remoción de ruido preservando objetos marcados
    """
    nombre = "reconstruccion_morfologica"
    
    def __init__(self,
                conectividad: Literal[4, 8] = 8):
        """
            Args:
                conectividad: Tipo de conectividad para propagación
                                4: Solo vecinos horizontal/vertical
                                8: Incluye vecinos diagonales (más común)
        """
        if conectividad not in [4, 8]:
            raise ValueError("conectividad debe ser 4 u 8")
        
        self.conectividad = conectividad
    
    def __call__(self, 
                marcador: np.ndarray, 
                mascara: np.ndarray,
                tipo: Literal['dilatacion', 'erosion'] = 'dilatacion') -> np.ndarray:
        """
            Aplica reconstrucción morfológica.
            
            Args:
                marcador: Imagen con semillas/marcadores (valores más bajos que máscara)
                mascara: Imagen que define límites de reconstrucción
                tipo: Tipo de reconstrucción
                        'dilatacion': Propaga marcador hacia arriba (más común)
                        'erosion': Propaga marcador hacia abajo (dual)
            
            Returns:
                Imagen reconstruida
                
            Ejemplo:
                # Eliminar objetos pequeños preservando forma exacta
                # 1. Apertura como marcador (elimina pequeños)
                marcador = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
                # 2. Original como máscara
                mascara = img
                # 3. Reconstruir
                resultado = reconstruccion(marcador, mascara)
        """
        self._validar_imagen(marcador)
        self._validar_imagen(mascara)
        
        if marcador.shape != mascara.shape:
            raise ValueError("marcador y mascara deben tener la misma forma")
        
        # Usar implementación iterativa
        if tipo == 'dilatacion':
            return self._reconstruccion_por_dilatacion(marcador, mascara)
        else:
            return self._reconstruccion_por_erosion(marcador, mascara)
    
    def _reconstruccion_por_dilatacion(self, marcador: np.ndarray, mascara: np.ndarray) -> np.ndarray:
        """
            Reconstrucción por dilatación geodésica iterativa.
        """
        # Elemento estructurante para conectividad
        if self.conectividad == 4:
            kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        else:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        
        # Asegurar que marcador <= máscara
        reconstruccion = np.minimum(marcador, mascara)
        
        # Iterar hasta convergencia
        iteracion = 0
        max_iteraciones = 1000  # Prevenir loops infinitos
        
        while iteracion < max_iteraciones:
            # Dilatación del marcador
            reconstruccion_anterior = reconstruccion.copy()
            reconstruccion = cv2.dilate(reconstruccion, kernel)
            
            # Limitar por la máscara (geodésica)
            reconstruccion = np.minimum(reconstruccion, mascara)
            
            # Verificar convergencia
            if np.array_equal(reconstruccion, reconstruccion_anterior):
                break
            
            iteracion += 1
        
        if iteracion == max_iteraciones:
            warnings.warn("Reconstrucción no convergió en {} iteraciones".format(max_iteraciones))
        
        return reconstruccion
    
    def _reconstruccion_por_erosion(self, marcador: np.ndarray, mascara: np.ndarray) -> np.ndarray:
        """
            Reconstrucción por erosión geodésica (dual de dilatación).
        """
        if self.conectividad == 4:
            kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        else:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        
        # Asegurar que marcador >= máscara
        reconstruccion = np.maximum(marcador, mascara)
        
        iteracion = 0
        max_iteraciones = 1000
        
        while iteracion < max_iteraciones:
            reconstruccion_anterior = reconstruccion.copy()
            reconstruccion = cv2.erode(reconstruccion, kernel)
            reconstruccion = np.maximum(reconstruccion, mascara)
            
            if np.array_equal(reconstruccion, reconstruccion_anterior):
                break
            
            iteracion += 1
        
        if iteracion == max_iteraciones:
            warnings.warn("Reconstrucción no convergió en {} iteraciones".format(max_iteraciones))
        
        return reconstruccion