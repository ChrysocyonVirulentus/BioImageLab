"""
Métodos de segmentación instancial para separación de objetos individuales.

La segmentación instancial identifica y separa objetos individuales (instancias)
dentro de una misma clase, a diferencia de la segmentación semántica que solo
clasifica píxeles por categoría. Esencial para análisis de objetos múltiples
que se tocan o se superponen.

Principio fundamental:
Transformar el problema de segmentación en búsqueda de regiones separadas
mediante análisis de topología (watershed) o partición del espacio métrico
(análisis de distancia).

IMPORTANTE - Separación de responsabilidades:
- Estos métodos NO normalizan imágenes (ese rol es de normalizador.py)
- Asumen que la imagen de entrada ya está preprocesada (filtros, contraste)
- Trabajan con máscaras binarias o imágenes de intensidad según el método
- La binarización previa (si es necesaria) debe hacerse con Segmentadores_Umbral
- Estos métodos separan objetos, NO detectan qué es objeto (eso es rol del detector)

Tipos de segmentación instancial:
- Watershed: Basado en topografía de intensidad (inundación de cuencas)
- Análisis de distancia: Partición basada en métrica euclidiana
- Grafo: División basada en contornos y conectividad
- Aprendizaje: Embeddings instanciales (no incluido aquí, ver ML)

Métodos disponibles:
- Watershed: Clásico, basado en gradiente o distancia
- WatershedMarcado: Con semillas controladas para ev sobre-segmentación
- DistanciaWatershed: Watershed sobre transformada de distancia (para objetos redondos)
- SplitDistancial: División de objetos tocantes por máximos de distancia
- WatershedHibrido: Combinación de gradiente y distancia
- SplitWatershed: División jerárquica por líneas de watershed
"""

import numpy as np
import cv2
from ...gestorLab.Registro_Metodos import registrar_en
from typing import Optional, Tuple, List, Literal, Union
from scipy import ndimage
from scipy.ndimage import distance_transform_edt, label, find_objects
from skimage import morphology, segmentation, measure
from skimage.feature import peak_local_max
import warnings


class SegmentadorInstancial:
    """
        Clase base para segmentadores de instancias individuales.
        
        Los segmentadores instanciales separan objetos conectados en
        instancias individuales mediante análisis topológico o métrico.
        
        Conceptos clave:
            - Instancia: Objeto individual identificable (célula, núcleo, vesícula)
            - Touching objects: Objetos conectados que deben separarse
            - Over-segmentation: Dividir un objeto en múltiples partes (error)
            - Under-segmentation: Fusionar múltiples objetos en uno (error)
            - Marker/Seed: Punto de inicio para crecimiento de región
    """
    nombre = "segmentador_instancial_base"
    
    def __call__(self, img: np.ndarray, 
                mascara: Optional[np.ndarray] = None) -> np.ndarray:
        """
            Aplica segmentación instancial.
            
            Args:
                img: Imagen de entrada (intensidad o binaria según método)
                mascara: Máscara binaria opcional de objetos de interés
                
            Returns:
                Imagen etiquetada (int32) donde cada instancia tiene ID único
        """
        raise NotImplementedError("Subclases deben implementar __call__")
    
    def _validar_imagen(self, img: np.ndarray):
        """Valida que la imagen sea 2D."""
        if img.ndim != 2:
            raise ValueError(f"La imagen debe ser 2D, tiene {img.ndim} dimensiones")
    
    def _etiquetar(self, mascara: np.ndarray) -> np.ndarray:
        """Etiqueta componentes conectados de forma robusta."""
        etiquetas, num = label(mascara, structure=np.ones((3,3)))
        return etiquetas.astype(np.int32)
    
    def _visualizar_etiquetas(self, etiquetas: np.ndarray) -> np.ndarray:
        """Convierte etiquetas a imagen RGB para visualización."""
        from skimage.color import label2rgb
        return (label2rgb(etiquetas, bg_label=0, bg_color=(0,0,0)) * 255).astype(np.uint8)

@registrar_en("segmentacion")
class Watershed(SegmentadorInstancial):
    """
        Segmentación Watershed clásica basada en topografía de intensidad.
        
        El algoritmo watershed trata la imagen como relieve topográfico donde
        los valores altos son "montañas" y los bajos son "valles". Simula
        inundación desde mínimos locales, creando cuencas de captación que
        se encuentran en líneas de divisoria (watershed lines).
        
        Algoritmo (Vincent-Soille):
            1. Ordenar píxeles por intensidad creciente
            2. Inicializar cola con mínimos locales (semillas)
            3. Expandir etiquetas a vecinos no procesados de menor intensidad
            4. Cuando dos cuencas se encuentran, marcar línea de watershed
        
        Ecuación:
            W(f) = ∪{B_i} donde B_i = {x | ∃ camino descendente a mínimo i}
            
            Línea de watershed: frontera donde B_i ∩ B_j = ∅ para i≠j
        
        Interpretación:
            - Mínimos locales: Semillas iniciales de cuencas
            - Gradientes altos: Crestas montañosas (líneas de separación)
            - Valles: Regiones que se inundan (objetos)
            - Sin marcadores: Cada mínimo local genera una cuenca
        
        Ventajas:
            - Fundamento topológico sólido
            - Flexible (puede usar gradiente, distancia o intensidad directa)
            - Produce líneas de separación precisas
            - Extensible a múltiples dimensiones (3D)
            - Determinístico (resultado único para entrada dada)
        
        Desventajas:
            - Grave problema de sobre-segmentación sin pre-procesado
            - Sensible al ruido (cada mínimo local es una semilla potencial)
            - Requiere gradiente de calidad (Canny, Sobel, etc.)
            - No maneja bien objetos con intensidad variable interna
            - Líneas de watershed pueden ser irregulares
        
        Usos típicos en microscopía:
            - Separación de células en monolayers densos
            - Segmentación de núcleos agrupados en DAPI
            - División de colonias bacterianas en placas
            - Separación de vesículas secretoras agrupadas
            - Post-procesamiento de segmentaciones binarias fusionadas
        
        Referencia:
            Vincent, L., & Soille, P. (1991). Watersheds in digital spaces:
            An efficient algorithm based on immersion simulations.
            IEEE TPAMI, 13(6), 583-598.
    """
    nombre = "watershed"
    
    def __init__(self,
                usar_gradiente: bool = True,
                sigma_gradiente: float = 1.0,
                compactness: float = 0.0):
        """
            Args:
                usar_gradiente: Si True, aplica watershed sobre magnitud del gradiente
                            Si False, usa imagen de intensidad directa (inverso)
                
                sigma_gradiente: Suavizado para cálculo de gradiente
                                Mayor = menos mínimos locales, menos sobre-segmentación
                
                compactness: Factor de compactación de cuencas (skimage)
                            0: Formas naturales según topografía
                            >0: Favorece cuencas más compactas/redondas
                            Valores típicos: 0-0.1
        """
        self.usar_gradiente = usar_gradiente
        self.sigma_gradiente = sigma_gradiente
        self.compactness = compactness
    
    def __call__(self, 
                img: np.ndarray, 
                mascara: Optional[np.ndarray] = None) -> np.ndarray:
        """
            Aplica watershed clásico.
            
            Args:
                img: Imagen de intensidad (gradiente calculado internamente si se solicita)
                    o imagen de topografía directa
                mascara: Máscara binaria de objetos de interés (opcional)
                
            Returns:
                Imagen etiquetada (int32) con instancias separadas
        """
        self._validar_imagen(img)
        
        # Preparar imagen de topografía
        if self.usar_gradiente:
            # Calcular magnitud de gradiente suavizado
            from skimage.filters import sobel, gaussian
            img_suave = gaussian(img.astype(np.float64), sigma=self.sigma_gradiente)
            topografia = sobel(img_suave)
            # Invertir: bordes altos (montañas), interiores bajos (valles)
            topografia = -topografia
        else:
            # Usar intensidad directa, invertida (asumimos objetos brillantes)
            topografia = -img.astype(np.float64)
        
        # Si hay máscara, restringir watershed a esa región
        if mascara is not None:
            # Encontrar marcadores como mínimos locales dentro de la máscara
            from skimage.feature import peak_local_max
            # Invertir para encontrar mínimos en topografía negativa
            local_min = peak_local_max(-topografia, 
                                    min_distance=5,
                                    exclude_border=False)
            marcadores = np.zeros_like(img, dtype=int)
            marcadores[local_min[:, 0], local_min[:, 1]] = 1
            marcadores, num = label(marcadores)
        else:
            # Sin máscara: todos los mínimos locales son semillas (sobre-segmentación!)
            marcadores = None
        
        # Aplicar watershed
        from skimage.segmentation import watershed
        
        if mascara is not None:
            etiquetas = watershed(topografia, marcadores, 
                                mask=mascara,
                                compactness=self.compactness)
        else:
            # Watershed sin máscara: todos los mínimos son cuencas
            etiquetas = watershed(topografia, 
                                compactness=self.compactness)
        
        return etiquetas.astype(np.int32)

@registrar_en("segmentacion")
class WatershedMarcado(SegmentadorInstancial):
    """
        Watershed con marcadores controlados para evitar sobre-segmentación.
        
        Versión controlada del watershed donde las semillas (marcadores) se
        definen explícitamente mediante detección de núcleos, centros de masa
        u otro criterio, en lugar de usar todos los mínimos locales.
        
        Algoritmo:
            1. Detectar marcadores de forma controlada (núcleos, máximos, etc.)
            2. Calcular topografía (gradiente o distancia)
            3. Expandir marcadores hasta encontrar otros marcadores o límites
            4. Líneas de watershed forman separación entre instancias
        
        Ecuación:
            M = {m_i | m_i = detector(img)}  # Conjunto de marcadores
            W(f, M) = watershed(f) restringido a ∪B_i donde semilla(m_i) ∈ B_i
        
        Estrategias de marcado:
            - Máximos locales: Picos de intensidad (núcleos en DAPI)
            - Centroides: Centros de masa de objetos binarios
            - Detección de blobs: Difference of Gaussians (DoG)
            - Detección de puntos: Harris, SIFT, etc.
            - Manual: Puntos definidos por usuario
        
        Ventajas:
            - Control explícito del número de instancias
            - Elimina sobre-segmentación inherente al watershed clásico
            - Marcadores pueden venir de canal diferente (ej: DAPI para células en actina)
            - Robusto a variaciones de intensidad interna del objeto
            - Flexible: cualquier detector puede generar marcadores
        
        Desventajas:
            - Requiere detector de marcadores confiable
            - Marcadores faltantes → objetos no segmentados (fusión)
            - Marcadores extra → sobre-segmentación
            - Sensibilidad a precisión de localización de marcadores
            - No maneja objetos sin marcador (invisible al detector)
        
        Usos típicos en microscopía:
            - Segmentación de células usando núcleos como marcadores
            - Separación de colonias bacterianas con centroides
            - Análisis de tejido: núcleos (DAPI) → citoplasma (marcador secundario)
            - Seguimiento de células en time-lapse (marcadores previos)
            - Segmentación de organelos con marcadores específicos
        
        Referencia:
            Beucher, S., & Meyer, F. (1993). The morphological approach to
            segmentation: the watershed transformation. Mathematical morphology
            in image processing, 433-481.
    """
    nombre = "watershed_marcado"
    
    def __init__(self,
                metodo_marcador: Literal['maximos', 'distancia', 'blob', 'manual'] = 'maximos',
                min_distance: int = 10,
                umbral_relativo: float = 0.5,
                usar_distancia: bool = True,
                sigma_distancia: float = 2.0):
        """
            Args:
                metodo_marcador: Estrategia para generar marcadores
                            'maximos': Máximos locales de intensidad
                            'distancia': Máximos de transformada de distancia
                            'blob': Detector de blobs (DoG)
                            'manual': Marcadores proporcionados externamente
                
                min_distance: Distancia mínima entre marcadores (píxeles)
                            Evita múltiples marcadores en mismo objeto
                
                umbral_relativo: Umbral relativo para detección de máximos
                            (0-1, fracción del máximo global)
                
                usar_distancia: Si True, usa transformada de distancia como topografía
                            Si False, usa gradiente de intensidad
                
                sigma_distancia: Suavizado de la transformada de distancia
                                Mayor = marcadores más centrados, menos sesgados
        """
        self.metodo_marcador = metodo_marcador
        self.min_distance = min_distance
        self.umbral_relativo = umbral_relativo
        self.usar_distancia = usar_distancia
        self.sigma_distancia = sigma_distancia
    
    def __call__(self,
                img: np.ndarray,
                mascara: np.ndarray,
                marcadores_manual: Optional[np.ndarray] = None) -> np.ndarray:
        """
            Aplica watershed con marcadores controlados.
            
            Args:
                img: Imagen de intensidad (para gradiente) o binaria (para distancia)
                mascara: Máscara binaria de objetos a segmentar
                marcadores_manual: Si metodo='manual', array binario con marcadores
                
            Returns:
                Imagen etiquetada (int32)
        """
        self._validar_imagen(img)
        
        # Generar marcadores según método
        if self.metodo_marcador == 'manual':
            if marcadores_manual is None:
                raise ValueError("Se requieren marcadores_manual cuando metodo='manual'")
            marcadores = marcadores_manual.astype(bool)
        elif self.metodo_marcador == 'maximos':
            marcadores = self._detectar_maximos(img, mascara)
        elif self.metodo_marcador == 'distancia':
            marcadores = self._detectar_maximos_distancia(mascara)
        elif self.metodo_marcador == 'blob':
            marcadores = self._detectar_blobs(img, mascara)
        else:
            raise ValueError(f"Método de marcador '{self.metodo_marcador}' no reconocido")
        
        # Etiquetar marcadores conectados
        marcadores_etiquetados, num_marcadores = label(marcadores)
        
        if num_marcadores == 0:
            warnings.warn("No se detectaron marcadores. Devolviendo máscara sin segmentar.")
            return self._etiquetar(mascara)
        
        # Preparar topografía
        if self.usar_distancia:
            # Transformada de distancia (mejor para objetos redondos/convexos)
            distancia = distance_transform_edt(mascara)
            distancia = ndimage.gaussian_filter(distancia, sigma=self.sigma_distancia)
            topografia = -distancia  # Mínimos en centros, máximos en bordes
        else:
            # Gradiente de intensidad
            from skimage.filters import sobel
            topografia = sobel(img.astype(np.float64))
            topografia = -topografia
        
        # Aplicar watershed
        from skimage.segmentation import watershed
        etiquetas = watershed(topografia, 
                            marcadores_etiquetados, 
                            mask=mascara)
        
        return etiquetas.astype(np.int32)
    
    def _detectar_maximos(self, img: np.ndarray, mascara: np.ndarray) -> np.ndarray:
        """Detecta máximos locales de intensidad dentro de la máscara."""
        from skimage.feature import peak_local_max
        
        img_float = img.astype(np.float64)
        umbral = img_float.max() * self.umbral_relativo
        
        # Aplicar máscara (excluir fuera)
        img_mascara = np.where(mascara, img_float, 0)
        
        maximos = peak_local_max(img_mascara,
                                min_distance=self.min_distance,
                                threshold_abs=umbral,
                                exclude_border=False)
        
        marcadores = np.zeros_like(img, dtype=bool)
        marcadores[maximos[:, 0], maximos[:, 1]] = True
        
        return marcadores
    
    def _detectar_maximos_distancia(self, mascara: np.ndarray) -> np.ndarray:
        """Detecta máximos de transformada de distancia (centros de objetos)."""
        from skimage.feature import peak_local_max
        
        distancia = distance_transform_edt(mascara)
        distancia = ndimage.gaussian_filter(distancia, sigma=self.sigma_distancia)
        
        maximos = peak_local_max(distancia,
                                min_distance=self.min_distance,
                                exclude_border=False)
        
        marcadores = np.zeros_like(mascara, dtype=bool)
        marcadores[maximos[:, 0], maximos[:, 1]] = True
        
        return marcadores
    
    def _detectar_blobs(self, img: np.ndarray, mascara: np.ndarray) -> np.ndarray:
        """Detecta blobs usando Difference of Gaussians."""
        from skimage.feature import blob_dog
        
        blobs = blob_dog(img.astype(np.float64), 
                        min_sigma=1, 
                        max_sigma=5, 
                        threshold=self.umbral_relativo)
        
        marcadores = np.zeros_like(img, dtype=bool)
        for blob in blobs:
            y, x, r = blob
            y, x = int(y), int(x)
            if 0 <= y < img.shape[0] and 0 <= x < img.shape[1] and mascara[y, x]:
                marcadores[y, x] = True
        
        return marcadores

@registrar_en("segmentacion")
class DistanciaWatershed(SegmentadorInstancial):
    """
        Watershed sobre transformada de distancia para objetos redondos/convexos.
        
        Especialización del watershed marcado donde la topografía es la
        transformada de distancia euclidiana (distancia al fondo). Ideal para
        objetos aproximadamente circulares/esféricos como núcleos, células
        redondas, vesículas, gotas, etc.
        
        Algoritmo:
            1. Calcular transformada de distancia de la máscara binaria
            2. Suavizar para reducir ruido en máximos locales
            3. Detectar máximos locales como marcadores (centros de objetos)
            4. Aplicar watershed sobre -distancia (mínimos en centros)
        
        Ecuación:
            D(x) = min{||x - y|| | y ∈ fondo}  para x en objeto
            
            Marcadores: argmax_local{D(x) | x ∈ objeto}
            
            Watershed sobre: -D(x) (mínimos en centros, crestas en contactos)
        
        Interpretación geométrica:
            - Máximos de D(x): Centros geométricos de objetos (más alejados del borde)
            - Crestas de D(x): Líneas de contacto entre objetos (silla de montar)
            - Watershed lines: Separación equidistante a bordes de objetos tocantes
        
        Ventajas:
            - Óptimo para objetos convexos/redondos
            - Marcadores geométricamente centrados (robustos)
            - Separación en punto más estrecho del contacto
            - Invariante a rotación (isotrópico)
            - Funciona con objetos de diferentes tamaños
            - No requiere imagen de intensidad (solo máscara)
        
        Desventajas:
            - Solo para objetos convexos (falla en objetos alargados/tubulares)
            - Sensibilidad a irregularidades de borde (rueda de carro)
            - Puede fallar en contactos muy largos (células muy pegadas)
            - No considera intensidad interna (solo geometría)
            - Suavizado necesario para evitar múltiples máximos por objeto
        
        Usos típicos en microscopía:
            - Segmentación de núcleos en DAPI/Hoechst (redondos)
            - Separación de células en suspensión (esféricas)
            - Conteo de colonias bacterianas en placas
            - Segmentación de vesículas lipídicas
            - Análisis de gotas en microfluidica
            - Cuantificación de foci/puncta
        
        Referencia:
            Malpica, N., et al. (1997). Applying watershed algorithms to the
            segmentation of clustered nuclei. Cytometry, 28(4), 289-297.
    """
    nombre = "distancia_watershed"
    
    def __init__(self,
                min_distance: int = 5,
                sigma_suavizado: float = 1.0,
                umbral_distancia_relativo: float = 0.5,
                compactness: float = 0.0):
        """
            Args:
                min_distance: Distancia mínima entre centros de objetos (píxeles)
                            Previene múltiples marcadores en objetos grandes
                
                sigma_suavizado: Suavizado gaussiano de la transformada de distancia
                            Mayor = menos marcadores por objeto, más centrados
                            Valores típicos: 0.5-2.0
                
                umbral_distancia_relativo: Umbral relativo para máximos de distancia
                                        (0-1, fracción del máximo global)
                                        Filtra máximos débiles en bordes irregulares
                
                compactness: Favorece segmentaciones más compactas (skimage)
                            0: Formas naturales, >0: más redondas
        """
        self.min_distance = min_distance
        self.sigma_suavizado = sigma_suavizado
        self.umbral_distancia_relativo = umbral_distancia_relativo
        self.compactness = compactness
    
    def __call__(self, mascara: np.ndarray) -> np.ndarray:
        """
        Aplica watershed sobre transformada de distancia.
        
        Args:
            mascara: Máscara binaria de objetos a segmentar
            
        Returns:
            Imagen etiquetada (int32) con instancias separadas
        """
        self._validar_imagen(mascara)
        
        if not mascara.dtype == bool:
            mascara_bin = mascara > 0
        else:
            mascara_bin = mascara
        
        # Transformada de distancia
        distancia = distance_transform_edt(mascara_bin)
        
        # Suavizar para reducir ruido
        if self.sigma_suavizado > 0:
            distancia = ndimage.gaussian_filter(distancia, 
                                            sigma=self.sigma_suavizado)
        
        # Detectar máximos locales (centros de objetos)
        from skimage.feature import peak_local_max
        
        distancia_max = distancia.max()
        if distancia_max == 0:
            return self._etiquetar(mascara_bin)
        
        umbral_abs = distancia_max * self.umbral_distancia_relativo
        
        maximos = peak_local_max(distancia,
                                min_distance=self.min_distance,
                                threshold_abs=umbral_abs,
                                exclude_border=False)
        
        # Crear marcadores
        marcadores = np.zeros_like(mascara, dtype=int)
        marcadores[maximos[:, 0], maximos[:, 1]] = 1
        marcadores, num_marcadores = label(marcadores)
        
        if num_marcadores == 0:
            return self._etiquetar(mascara_bin)
        
        # Watershed sobre distancia negativa
        from skimage.segmentation import watershed
        etiquetas = watershed(-distancia, 
                            marcadores, 
                            mask=mascara_bin,
                            compactness=self.compactness)
        
        return etiquetas.astype(np.int32)

@registrar_en("segmentacion")
class SplitDistancial(SegmentadorInstancial):
    """
        División de objetos tocantes por análisis de máximos de distancia.
        
        Método específico para separar objetos binarios que se tocan,
        identificando puntos de división por análisis de la transformada
        de distancia sin watershed completo (más rápido para casos simples).
        
        Algoritmo:
            1. Calcular transformada de distancia
            2. Detectar máximos locales (centros potenciales)
            3. Para cada máximo, expandir región hasta tocar otra región o borde
            4. Líneas de contacto definen separación
        
        Variantes:
            - Voronoi: Partición del espacio por proximidad a centros
            - Esqueleto: Línea media del objeto, ramificaciones en contactos
            - Watershed local: Solo en regiones de contacto detectadas
        
        Ecuación:
            Partición: R_i = {x | D_i(x) > D_j(x) ∀ j≠i}
            
            donde D_i es distancia al i-ésimo centro (máximo local)
        
        Ventajas:
            - Más rápido que watershed completo para objetos simples
            - Separación geométrica pura (no requiere intensidad)
            - Produce líneas de separación rectas (Voronoi) o curvas (watershed)
            - Eficiente para muchos objetos pequeños (núcleos, vesículas)
            - Fácil de paralelizar
        
        Desventajas:
            - Solo funciona para contactos simples (no superposiciones)
            - Puede producir separaciones no naturales (líneas rectas en Voronoi)
            - No maneja objetos con concavidades complejas
            - Sensibilidad a número de máximos detectados
            - Menos preciso que watershed en contactos irregulares
        
        Usos típicos en microscopía:
            - Separación rápida de núcleos en imágenes de alta densidad
            - División de células en culturas confluentes simples
            - Post-procesamiento de segmentaciones umbralizadas
            - Análisis de packing de células en tejidos
            - Conteo rápido de objetos redondos agrupados
        
        Comparación con DistanciaWatershed:
            - SplitDistancial: Más rápido, menos preciso, para casos simples
            - DistanciaWatershed: Más robusto, mejor para contactos complejos
    """
    nombre = "split_distancial"
    
    def __init__(self,
                metodo: Literal['voronoi', 'esqueleto', 'watershed_local'] = 'voronoi',
                min_distance: int = 5,
                sigma_suavizado: float = 1.0,
                umbral_area: int = 50):
        """
            Args:
                metodo: Estrategia de partición
                    'voronoi': Partición por diagrama de Voronoi (más rápido)
                    'esqueleto': Usa esqueleto morfológico para detectar contactos
                    'watershed_local': Watershed solo en zonas de contacto
                
                min_distance: Distancia mínima entre centros
                
                sigma_suavizado: Suavizado de transformada de distancia
                
                umbral_area: Área mínima de objeto válido (filtra ruido)
        """
        self.metodo = metodo
        self.min_distance = min_distance
        self.sigma_suavizado = sigma_suavizado
        self.umbral_area = umbral_area
    
    def __call__(self, mascara: np.ndarray) -> np.ndarray:
        """
        Separa objetos tocantes por análisis de distancia.
        
        Args:
            mascara: Máscara binaria con objetos conectados
            
        Returns:
            Imagen etiquetada (int32) con objetos separados
        """
        self._validar_imagen(mascara)
        
        mascara_bin = mascara > 0 if not mascara.dtype == bool else mascara
        
        # Calcular distancia y detectar centros
        distancia = distance_transform_edt(mascara_bin)
        
        if self.sigma_suavizado > 0:
            distancia = ndimage.gaussian_filter(distancia, 
                                            sigma=self.sigma_suavizado)
        
        # Detectar máximos
        from skimage.feature import peak_local_max
        
        maximos = peak_local_max(distancia,
                                min_distance=self.min_distance,
                                exclude_border=False)
        
        if len(maximos) == 0:
            return self._etiquetar(mascara_bin)
        
        # Crear marcadores
        marcadores = np.zeros_like(mascara, dtype=int)
        for i, (y, x) in enumerate(maximos, 1):
            marcadores[y, x] = i
        
        if self.metodo == 'voronoi':
            return self._split_voronoi(mascara_bin, marcadores)
        elif self.metodo == 'esqueleto':
            return self._split_esqueleto(mascara_bin, marcadores)
        else:  # watershed_local
            return self._split_watershed_local(mascara_bin, marcadores, distancia)
    
    def _split_voronoi(self, mascara: np.ndarray, marcadores: np.ndarray) -> np.ndarray:
        """División por diagrama de Voronoi."""
        from scipy.ndimage import distance_transform_edt
        
        # Calcular transformada de distancia a cada marcador
        # Usar watershed como aproximación a Voronoi en grid discreto
        from skimage.segmentation import watershed
        
        # Invertir marcadores para watershed (mínimos)
        marcadores_inv = np.where(marcadores > 0, -marcadores, 0)
        
        # Distancia a marcadores más cercano
        distancia_a_centros = distance_transform_edt(marcadores == 0)
        
        etiquetas = watershed(distancia_a_centros, 
                            marcadores, 
                            mask=mascara)
        
        return etiquetas.astype(np.int32)
    
    def _split_esqueleto(self, mascara: np.ndarray, marcadores: np.ndarray) -> np.ndarray:
        """División usando esqueleto morfológico."""
        from skimage.morphology import skeletonize
        
        # Esqueleto de la máscara
        esqueleto = skeletonize(mascara)
        
        # Encontrar puntos de ramificación (contactos)
        # Kernel 3x3 para contar vecinos
        vecinos = ndimage.convolve(esqueleto.astype(int), 
                                np.ones((3,3)), 
                                mode='constant')
        puntos_contacto = (esqueleto & (vecinos > 3))
        
        # Separar en puntos de contacto
        mascara_separada = mascara & ~puntos_contacto
        
        # Re-etiquetar
        return self._etiquetar(mascara_separada)
    
    def _split_watershed_local(self, 
                            mascara: np.ndarray, 
                            marcadores: np.ndarray,
                            distancia: np.ndarray) -> np.ndarray:
        """Watershed solo en regiones de contacto."""
        from skimage.segmentation import watershed
        
        # Identificar zonas de contacto (bajos en distancia entre máximos)
        umbral_contacto = distancia.max() * 0.3
        zona_contacto = (distancia < umbral_contacto) & mascara
        
        # Dilatar zonas de contacto para asegurar separación
        zona_contacto = ndimage.binary_dilation(zona_contacto, 
                                                iterations=2)
        
        # Aplicar watershed solo en zona de contacto
        etiquetas = watershed(-distancia, 
                            marcadores, 
                            mask=mascara)
        
        return etiquetas.astype(np.int32)

@registrar_en("segmentacion")
class WatershedHibrido(SegmentadorInstancial):
    """
        Watershed híbrido combinando gradiente de intensidad y distancia geométrica.
        
        Combina información de intensidad (bordes reales) con información
        geométrica (centros de objetos) para segmentación robusta en casos
        donde ni el gradiente ni la distancia solos son suficientes.
        
        Algoritmo:
            1. Calcular gradiente de intensidad (bordes reales)
            2. Calcular transformada de distancia (geometría interna)
            3. Combinar topografías: T = α·G + (1-α)·(-D)
            4. Detectar marcadores en máximos de distancia suavizada
            5. Aplicar watershed sobre topografía combinada
        
        Ecuación de topografía híbrida:
            T(x) = α · ||∇I(x)|| - (1-α) · D(x)
            
            donde:
                α ∈ [0,1]: Peso del gradiente vs distancia
                ||∇I||: Magnitud del gradiente de intensidad
                D: Transformada de distancia
        
        Interpretación:
            - α ≈ 1: Casi watershed clásico (sigue bordes de intensidad)
            - α ≈ 0: Casi distancia watershed (sigue geometría)
            - α = 0.5: Balance entre evidencia de borde y centro
        
        Ventajas:
            - Combina fortalezas de ambos enfoques
            - Robusto a bordes débiles (apoyo geométrico)
            - Robusto a objetos irregulares (apoyo de intensidad)
            - Flexible mediante parámetro α
            - Mejor que cualquiera solo en casos mixtos
        
        Desventajas:
            - Más parámetros que ajustar (α, sigmas de ambos)
            - Más costoso computacionalmente (dos transformadas)
            - Puede ser overkill para casos simples
            - Balance α depende de cada imagen
        
        Usos típicos en microscopía:
            - Segmentación de células con citoplasma irregular (intensidad) 
            y núcleos redondos (geometría)
            - Tejidos epiteliales con bordes celulares variables
            - Células en diferentes fases del ciclo (formas variables)
            - Imágenes con contraste de fase (bordes complejos)
            - Segmentación de organelos con morfología mixta
        
        Referencia:
            Jones, T. R., et al. (2005). CellProfiler: Novel software for
            high-throughput cell analysis. Cytometry A, 63(1), 231-236.
    """
    nombre = "watershed_hibrido"
    
    def __init__(self,
                alpha: float = 0.5,
                sigma_gradiente: float = 1.0,
                sigma_distancia: float = 2.0,
                min_distance: int = 5,
                compactness: float = 0.0):
        """
            Args:
                alpha: Peso del gradiente (0-1). 1=puro gradiente, 0=pura distancia
                
                sigma_gradiente: Suavizado para cálculo de gradiente
                
                sigma_distancia: Suavizado para transformada de distancia
                
                min_distance: Distancia mínima entre marcadores
                
                compactness: Favorece segmentaciones compactas
        """
        if not 0 <= alpha <= 1:
            raise ValueError("alpha debe estar en [0, 1]")
        
        self.alpha = alpha
        self.sigma_gradiente = sigma_gradiente
        self.sigma_distancia = sigma_distancia
        self.min_distance = min_distance
        self.compactness = compactness
    
    def __call__(self, 
                img: np.ndarray, 
                mascara: np.ndarray) -> np.ndarray:
        """
            Aplica watershed híbrido.
            
            Args:
                img: Imagen de intensidad
                mascara: Máscara binaria de objetos
                
            Returns:
                Imagen etiquetada (int32)
        """
        self._validar_imagen(img)
        
        # Gradientes de intensidad
        from skimage.filters import sobel, gaussian
        img_suave = gaussian(img.astype(np.float64), sigma=self.sigma_gradiente)
        gradiente = sobel(img_suave)
        gradiente = gradiente / (gradiente.max() + 1e-10)  # Normalizar
        
        # Transformada de distancia (normalizada)
        distancia = distance_transform_edt(mascara > 0)
        distancia = ndimage.gaussian_filter(distancia, sigma=self.sigma_distancia)
        distancia = distancia / (distancia.max() + 1e-10)
        
        # Topografía híbrida
        topografia = self.alpha * gradiente - (1 - self.alpha) * distancia
        
        # Marcadores en máximos de distancia
        from skimage.feature import peak_local_max
        
        maximos = peak_local_max(distancia,
                                min_distance=self.min_distance,
                                exclude_border=False)
        
        marcadores = np.zeros_like(img, dtype=int)
        marcadores[maximos[:, 0], maximos[:, 1]] = 1
        marcadores, _ = label(marcadores)
        
        # Watershed
        from skimage.segmentation import watershed
        etiquetas = watershed(topografia, 
                            marcadores, 
                            mask=mascara > 0,
                            compactness=self.compactness)
        
        return etiquetas.astype(np.int32)

@registrar_en("segmentacion")
class SplitWatershed(SegmentadorInstancial):
    """
        División jerárquica por watershed para objetos anidados o múltiples escalas.
        
        Aplica watershed de forma jerárquica: primero separa grandes grupos,
        luego subdivide si los objetos resultantes aún son compuestos (por
        análisis de concavidad o múltiples máximos de distancia).
        
        Algoritmo:
            1. Watershed inicial con parámetros conservadores (pocos marcadores)
            2. Para cada objeto resultante, analizar si es unitario o compuesto
            - Múltiples máximos de distancia → compuesto
            - Alta concavidad → compuesto
            3. Si compuesto, aplicar watershed local con más marcadores
            4. Iterar hasta que todos los objetos sean unitarios o límite de profundidad
        
        Criterios de división:
            - Número de máximos de distancia > 1
            - Relación área/convexidad > umbral
            - Solidez (solidity) < umbral
            - Múltiples picos en histograma de distancia
        
        Ventajas:
            - Maneja objetos de diferentes tamaños en misma imagen
            - Evita sobre-segmentación de objetos grandes
            - Evita sub-segmentación de grupos pequeños
            - Adaptativo a escala local
            - Progresivo: puede detenerse en cualquier nivel
        
        Desventajas:
            - Complejidad computacional mayor (múltiples pasadas)
            - Parámetros por nivel jerárquico
            - Puede fragmentar objetos legítimamente grandes
            - Difícil determinar cuándo un objeto es "unitario"
            - Riesgo de sobre-división en objetos complejos (neuritas)
        
        Usos típicos en microscopía:
            - Segmentación de colonias bacterianas con diferentes densidades
            - Tejidos con células de diferentes tamaños (epitelio + lumen)
            - Imágenes con agregados celulares y células individuales
            - Organoides con estructura interna compleja
            - Análisis de clustering de partículas
        
        Referencia:
            Meyer, F., & Beucher, S. (1990). Morphological segmentation.
            JVCIR, 1(1), 21-46.
    """
    nombre = "split_watershed"
    
    def __init__(self,
                niveles: int = 2,
                factor_marcadores: float = 2.0,
                criterio_division: Literal['maximos', 'concavidad', 'solidez'] = 'maximos',
                umbral_division: float = 0.5):
        """
            Args:
                niveles: Profundidad máxima de jerarquía (1 = watershed simple)
                
                factor_marcadores: Multiplicador de marcadores por nivel
                                (nivel 1: N marcadores, nivel 2: N*factor, etc.)
                
                criterio_division: Métrica para detectar objetos compuestos
                                'maximos': Múltiples máximos de distancia
                                'concavidad': Defectos de convexidad significativos
                                'solidez': Área / área convexa < umbral
                
                umbral_division: Umbral para criterio de división
                            (0.5-0.8 típico, depende de criterio)
        """
        self.niveles = niveles
        self.factor_marcadores = factor_marcadores
        self.criterio_division = criterio_division
        self.umbral_division = umbral_division
    
    def __call__(self, mascara: np.ndarray) -> np.ndarray:
        """
            Aplica división jerárquica por watershed.
            
            Args:
                mascara: Máscara binaria inicial
                
            Returns:
                Imagen etiquetada final (int32)
        """
        self._validar_imagen(mascara)
        
        mascara_bin = mascara > 0 if not mascara.dtype == bool else mascara
        
        # Etiquetado inicial
        etiquetas, num = label(mascara_bin)
        
        if num == 0:
            return etiquetas.astype(np.int32)
        
        # Procesar jerárquicamente
        for nivel in range(1, self.niveles + 1):
            etiquetas_nuevas = np.zeros_like(etiquetas)
            siguiente_id = 1
            
            for obj_id in range(1, num + 1):
                mascara_obj = (etiquetas == obj_id)
                
                if self._debe_dividir(mascara_obj, nivel):
                    # Subdividir este objeto
                    sub_etiquetas = self._subdividir(mascara_obj, nivel)
                    # Renumerar y agregar
                    for sub_id in range(1, sub_etiquetas.max() + 1):
                        etiquetas_nuevas[sub_etiquetas == sub_id] = siguiente_id
                        siguiente_id += 1
                else:
                    # Mantener como está
                    etiquetas_nuevas[mascara_obj] = siguiente_id
                    siguiente_id += 1
            
            etiquetas = etiquetas_nuevas
            num = siguiente_id - 1
            
            if num == 0:
                break
        
        return etiquetas.astype(np.int32)
    
    def _debe_dividir(self, mascara: np.ndarray, nivel: int) -> bool:
        """Determina si un objeto debe subdividirse."""
        if self.criterio_division == 'maximos':
            # Múltiples máximos de distancia
            distancia = distance_transform_edt(mascara)
            maximos = peak_local_max(distancia, 
                                    min_distance=int(5 / nivel),
                                    exclude_border=False)
            return len(maximos) > 1
        
        elif self.criterio_division == 'concavidad':
            # Defectos de convexidad
            from skimage.morphology import convex_hull_image
            convexo = convex_hull_image(mascara)
            area_convexo = convexo.sum()
            area_real = mascara.sum()
            return (area_convexo - area_real) / area_convexo > self.umbral_division
        
        else:  # solidez
            from skimage.measure import regionprops
            props = regionprops(mascara.astype(int))
            if len(props) == 0:
                return False
            return props[0].solidity < self.umbral_division
    
    def _subdividir(self, mascara: np.ndarray, nivel: int) -> np.ndarray:
        """Subdivide un objeto usando watershed con más marcadores."""
        distancia = distance_transform_edt(mascara)
        distancia = ndimage.gaussian_filter(distancia, sigma=1.0 / nivel)
        
        # Más marcadores en niveles profundos
        min_dist = int(5 / (self.factor_marcadores ** (nivel - 1)))
        min_dist = max(min_dist, 3)
        
        from skimage.feature import peak_local_max
        maximos = peak_local_max(distancia,
                                min_distance=min_dist,
                                exclude_border=False)
        
        marcadores = np.zeros_like(mascara, dtype=int)
        for i, (y, x) in enumerate(maximos, 1):
            marcadores[y, x] = i
        
        from skimage.segmentation import watershed
        return watershed(-distancia, marcadores, mask=mascara)

@registrar_en("segmentacion")
class MarcadorControlado(SegmentadorInstancial):
    """
        Watershed con control estricto de marcadores para casos difíciles.
        
        Extensión de WatershedMarcado con validación de marcadores y
        estrategias de fallback cuando los marcadores son insuficientes.
        
        Características:
            - Validación de marcadores (eliminar marcadores en fondo)
            - Expansión de marcadores pequeños (mínimo área garantizado)
            - Fallback a distancia si marcadores insuficientes
            - Post-procesamiento de líneas de separación
        
        Algoritmo:
            1. Validar marcadores: deben estar dentro de máscara, tamaño mínimo
            2. Si marcadores insuficientes (< objetos esperados), usar fallback
            3. Aplicar watershed con marcadores validados
            4. Post-procesar: eliminar segmentaciones pequeñas, fusionar fragmentos
        
        Ventajas:
            - Robusto a errores en detección de marcadores
            - Garantiza salida coherente incluso con entrada imperfecta
            - Control de calidad de segmentación
            - Flexible: acepta diferentes fuentes de marcadores
        
        Desventajas:
            - Complejidad adicional de validación
            - Parámetros de calidad necesarios
            - Puede ser conservador (prefiere fusión a sobre-segmentación)
        
        Usos típicos:
            - Segmentación en pipelines automatizados donde la calidad debe garantizarse
            - Casos donde los marcadores vienen de red neuronal (con errores)
            - Segmentación de estructuras donde la integridad es crítica
    """
    nombre = "marcador_controlado"
    
    def __init__(self,
                area_minima_marcador: int = 10,
                area_minima_segmento: int = 50,
                usar_fallback: bool = True,
                metodo_fallback: Literal['distancia', 'watershed'] = 'distancia'):
        """
            Args:
                area_minima_marcador: Área mínima para aceptar un marcador
                
                area_minima_segmento: Área mínima de segmento final válido
                
                usar_fallback: Si True, usa método alternativo si marcadores fallan
                
                metodo_fallback: Método si los marcadores son insuficientes
        """
        self.area_minima_marcador = area_minima_marcador
        self.area_minima_segmento = area_minima_segmento
        self.usar_fallback = usar_fallback
        self.metodo_fallback = metodo_fallback
    
    def __call__(self,
                img: np.ndarray,
                mascara: np.ndarray,
                marcadores: np.ndarray) -> np.ndarray:
        """
            Aplica watershed con control estricto de marcadores.
            
            Args:
                img: Imagen de intensidad
                mascara: Máscara binaria
                marcadores: Máscara binaria o etiquetada de marcadores propuestos
                
            Returns:
                Imagen etiquetada validada
        """
        self._validar_imagen(img)
        
        # Validar marcadores
        marcadores_validos = self._validar_marcadores(marcadores, mascara)
        
        num_marcadores = marcadores_validos.max()
        
        if num_marcadores == 0:
            if self.usar_fallback:
                # Fallback a método sin marcadores
                if self.metodo_fallback == 'distancia':
                    fallback = DistanciaWatershed()
                    return fallback(mascara)
                else:
                    fallback = Watershed()
                    return fallback(img, mascara)
            else:
                warnings.warn("No hay marcadores válidos y fallback deshabilitado")
                return self._etiquetar(mascara)
        
        # Preparar topografía
        from skimage.filters import sobel
        topografia = sobel(img.astype(np.float64))
        
        # Watershed
        from skimage.segmentation import watershed
        etiquetas = watershed(topografia, 
                            marcadores_validos, 
                            mask=mascara > 0)
        
        # Post-procesamiento: filtrar segmentos pequeños
        return self._filtrar_pequeños(etiquetas)
    
    def _validar_marcadores(self, 
                        marcadores: np.ndarray, 
                        mascara: np.ndarray) -> np.ndarray:
        """Valida y limpia marcadores."""
        # Asegurar que marcadores están dentro de máscara
        if marcadores.dtype == bool:
            marcadores = marcadores & (mascara > 0)
            marcadores, _ = label(marcadores)
        else:
            # Etiquetados: filtrar los que tocan fondo
            for i in range(1, marcadores.max() + 1):
                mask_i = (marcadores == i)
                if not np.any(mask_i & (mascara > 0)):
                    marcadores[mask_i] = 0
        
        # Filtrar por área mínima
        if self.area_minima_marcador > 0:
            for i in range(1, marcadores.max() + 1):
                if np.sum(marcadores == i) < self.area_minima_marcador:
                    marcadores[marcadores == i] = 0
        
        # Re-etiquetar consecutivamente
        marcadores, _ = label(marcadores > 0)
        
        return marcadores.astype(np.int32)
    
    def _filtrar_pequeños(self, etiquetas: np.ndarray) -> np.ndarray:
        """Elimina segmentos más pequeños que el umbral."""
        if self.area_minima_segmento <= 0:
            return etiquetas
        
        etiquetas_filtradas = etiquetas.copy()
        
        for i in range(1, etiquetas.max() + 1):
            if np.sum(etiquetas == i) < self.area_minima_segmento:
                etiquetas_filtradas[etiquetas == i] = 0
        
        # Re-etiquetar
        etiquetas_filtradas, _ = label(etiquetas_filtradas > 0)
        return etiquetas_filtradas.astype(np.int32)