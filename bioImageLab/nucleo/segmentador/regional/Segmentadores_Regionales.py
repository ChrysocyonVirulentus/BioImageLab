"""
Métodos de segmentación regional para agrupación de píxeles por similitud.

La segmentación regional agrupa píxeles conectados en regiones coherentes
basándose en criterios de homogeneidad (intensidad, color, textura) o
conectividad estructurada (grafos, caminos aleatorios). A diferencia de la
segmentación instancial, no requiere separación de objetos tocantes, sino
que busca regiones significativas del punto de vista semántico o funcional.

Principio fundamental:
Agrupar píxeles en regiones R_i tales que:
- Homogeneidad interna: los píxeles en R_i son similares según algún criterio
- Diferenciación externa: los píxeles de R_i son distintos a los de R_j (i≠j)

IMPORTANTE - Separación de responsabilidades:
- Estos métodos NO normalizan imágenes (ese rol es de normalizador.py)
- Asumen que la imagen de entrada ya está preprocesada (filtros, contraste)
- Trabajan con imágenes de intensidad, color o características según el método
- La inicialización (semillas, grafo) puede requerir pre-segmentación externa
- Estos métodos agrupan regiones, NO detectan objetos específicos (eso es rol del detector)
- Para separar objetos tocantes, usar Segmentadores_Instanciales.py

Tipos de segmentación regional:
- Crecimiento de regiones: Expansión desde semillas según criterio de homogeneidad
- Caminos aleatorios: Probabilidad de pertenencia basada en caminos estocásticos
- Cortes en grafos: Optimización global de fronteras mediante teoria de grafos
- Superpíxeles: Agrupación perceptual en "super-pixels" homogéneos
- Agrupación espectral: Segmentación basada en eigenvalores de matrices de similitud

Métodos disponibles:
- RegionGrowing: Crecimiento de regiones desde semillas con criterio de homogeneidad
- RandomWalk: Segmentación probabilística por caminos aleatorios
- CorteGrafico: Minimización de energía mediante corte mínimo en grafo
- SuperpixelSLIC: Superpíxeles compactos y homogéneos (Simple Linear Iterative Clustering)
- SuperpixelFelzenszwalb: Superpíxeles basados en partición de grafos
- WatershedRegiones: Watershed como segmentador regional (variante sin máscara binaria)
- MeanShiftSegmentacion: Segmentación por modo de densidad (clustering espacial-color)
"""

import numpy as np
import cv2
from typing import Optional, Tuple, List, Literal, Union, Callable
from scipy import ndimage
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from skimage import segmentation, color, filters, graph
from sklearn.cluster import MeanShift as SklearnMeanShift
import warnings


class SegmentadorRegional:
    """
        Clase base para segmentadores regionales.
        
        Los segmentadores regionales agrupan píxeles en regiones coherentes
        basándose en similitud local o conectividad estructurada.
        
        Conceptos clave:
            - Región: Conjunto conectado de píxeles similares
            - Homogeneidad: Criterio de similitud (intensidad, color, textura)
            - Frontera: Borde entre regiones con discontinuidad
            - Semilla: Punto inicial para crecimiento de región
            - Grafo: Representación de adyacencia y pesos entre píxeles/regiones
    """
    nombre = "segmentador_regional_base"
    
    def __call__(self, 
                img: np.ndarray,
                semillas: Optional[np.ndarray] = None) -> np.ndarray:
        """
            Aplica segmentación regional.
            
            Args:
                img: Imagen de entrada (intensidad, color o multicanal)
                semillas: Marcadores iniciales opcionales (formato según método)
                
            Returns:
                Imagen etiquetada (int32) donde cada región tiene ID único
        """
        raise NotImplementedError("Subclases deben implementar __call__")
    
    def _validar_imagen(self, img: np.ndarray, permitir_multicanal: bool = False):
        """Valida que la imagen sea 2D o 3D (multicanal)."""
        if img.ndim not in [2, 3]:
            raise ValueError(f"Imagen debe ser 2D o 3D, tiene {img.ndim} dimensiones")
        if img.ndim == 3 and not permitir_multicanal:
            raise ValueError("Este método no soporta imágenes multicanal")
    
    def _etiquetar(self, mascara: np.ndarray) -> np.ndarray:
        """Etiqueta componentes conectados."""
        etiquetas, num = ndimage.label(mascara, structure=np.ones((3,3)))
        return etiquetas.astype(np.int32)
    
    def _rgb2lab(self, img: np.ndarray) -> np.ndarray:
        """Convierte RGB a CIELAB para distancia perceptual uniforme."""
        if img.ndim == 3 and img.shape[2] == 3:
            return color.rgb2lab(img)
        return img

@registrar_en("segmentado")
class RegionGrowing(SegmentadorRegional):
    """
        Crecimiento de regiones desde semillas según criterio de homogeneidad.
        
        Algoritmo clásico de segmentación donde las regiones se expanden
        desde semillas iniciales agregando píxeles vecinos que satisfacen
        un criterio de similitud (umbral de intensidad, color o textura).
        
        Algoritmo:
            1. Inicializar cola con semillas (píxeles o regiones semilla)
            2. Mientras haya píxeles en cola:
            a. Extraer píxel de la cola
            b. Examinar vecinos no visitados
            c. Si vecino cumple criterio de homogeneidad:
                - Agregar a región
                - Agregar a cola
            3. Repetir hasta que no haya más expansiones posibles
        
        Criterios de homogeneidad:
            - Umbral absoluto: |I(p) - I(semilla)| < T
            - Umbral relativo: |I(p) - μ_región| < T·σ_región
            - Distancia vectorial: ||C(p) - C(semilla)|| < T (para color)
            - Textura: diferencia de descriptores de textura < T
        
        Ecuación:
            R(t+1) = R(t) ∪ {p ∉ R(t) | p ∈ N(R(t)) ∧ similitud(p, R(t)) < T}
            
            donde N(R) son vecinos de la región R
        
        Ventajas:
            - Conceptualmente simple e intuitivo
            - Flexible en criterio de homogeneidad
            - Permite control interactivo (usuario define semillas)
            - Bueno para objetos con fronteras débiles pero interior homogéneo
            - Extensible a múltiples características (intensidad + color + textura)
        
        Desventajas:
            - Sensibilidad a elección de semillas (resultado depende de inicio)
            - Sesgo de crecimiento: regiones grandes crecen más rápido
            - Problema de "leaking": puede atravesar fronteras débiles
            - Orden de procesamiento afecta resultado (no determinístico sin cuidado)
            - Ineficiente para grandes regiones (cola puede crecer mucho)
        
        Usos típicos en microscopía:
            - Segmentación interactiva de tejidos (médico define ROI)
            - Extracción de regiones de interés en imágenes histológicas
            - Segmentación de tumores con bordes difusos
            - Análisis de regiones funcionales en imágenes de ratiometría
            - Post-procesamiento de segmentaciones iniciales (refinamiento)
        
        Referencia:
            Adams, R., & Bischof, L. (1994). Seeded region growing.
            IEEE TPAMI, 16(6), 641-647.
    """
    nombre = "region_growing"
    
    def __init__(self,
                criterio: Literal['intensidad', 'color', 'textura'] = 'intensidad',
                umbral: float = 10.0,
                conectividad: Literal[4, 8] = 4,
                min_area: int = 10,
                max_iter: Optional[int] = None):
        """
            Args:
                criterio: Tipo de similitud a evaluar
                        'intensidad': Diferencia de intensidad gris
                        'color': Distancia en espacio de color (CIELAB)
                        'textura': Diferencia de descriptores de textura local
                
                umbral: Máxima diferencia permitida para agregar píxel
                    (en unidades de intensidad o distancia de color)
                    Valores típicos: 5-50 para intensidad (0-255)
                
                conectividad: 4 (vecinos cruz) o 8 (vecinos incluyendo diagonales)
                
                min_area: Área mínima de región válida (filtra ruido)
                
                max_iter: Máximo de iteraciones (None = hasta convergencia)
        """
        self.criterio = criterio
        self.umbral = umbral
        self.conectividad = conectividad
        self.min_area = min_area
        self.max_iter = max_iter
        
        # Definir vecinos según conectividad
        if conectividad == 4:
            self.vecinos = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        else:
            self.vecinos = [(-1, -1), (-1, 0), (-1, 1),
                            (0, -1),          (0, 1),
                            (1, -1),  (1, 0), (1, 1)]
    
    def __call__(self,
                img: np.ndarray,
                semillas: np.ndarray) -> np.ndarray:
        """
            Aplica crecimiento de regiones desde semillas.
            
            Args:
                img: Imagen de intensidad (2D) o color (3D si criterio='color')
                semillas: Máscara binaria (True=semillas) o etiquetada (int>0=semillas)
                
            Returns:
                Imagen etiquetada (int32) con regiones crecidas
        """
        self._validar_imagen(img, permitir_multicanal=(self.criterio == 'color'))
        
        # Preparar datos según criterio
        if self.criterio == 'color' and img.ndim == 3:
            datos = self._rgb2lab(img)
        else:
            datos = img.astype(np.float64)
            if datos.ndim == 3:
                datos = color.rgb2gray(datos)
        
        # Preparar semillas
        if semillas.dtype == bool:
            semillas_etq, num_semillas = ndimage.label(semillas)
        else:
            semillas_etq = semillas.astype(int)
            num_semillas = semillas_etq.max()
        
        if num_semillas == 0:
            raise ValueError("No se proporcionaron semillas válidas")
        
        # Inicializar resultado
        h, w = img.shape[:2]
        resultado = np.zeros((h, w), dtype=np.int32)
        
        # Crecer cada región desde su semilla
        for id_semilla in range(1, num_semillas + 1):
            mascara_semilla = (semillas_etq == id_semilla)
            region = self._crecer_region(datos, mascara_semilla)
            resultado[region] = id_semilla
        
        # Filtrar regiones pequeñas
        return self._filtrar_pequeñas(resultado)
    
    def _crecer_region(self, datos: np.ndarray, semilla: np.ndarray) -> np.ndarray:
        """Expande una región desde semilla inicial."""
        h, w = datos.shape[:2]
        
        # Inicializar
        region = semilla.copy()
        frontera = []
        
        # Agregar vecinos de semilla a frontera
        ys, xs = np.where(semilla)
        for y, x in zip(ys, xs):
            for dy, dx in self.vecinos:
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and not region[ny, nx]:
                    frontera.append((ny, nx))
        
        # Valor de referencia (media de la semilla)
        if self.criterio == 'intensidad':
            valor_ref = datos[semilla].mean()
        else:
            valor_ref = datos[semilla].mean(axis=0)
        
        # Crecer
        iteracion = 0
        while frontera and (self.max_iter is None or iteracion < self.max_iter):
            iteracion += 1
            
            # Evaluar frontera actual
            nueva_frontera = []
            
            for y, x in frontera:
                if region[y, x]:  # Ya agregado
                    continue
                
                # Evaluar criterio
                if self.criterio == 'intensidad':
                    diff = abs(datos[y, x] - valor_ref)
                elif self.criterio == 'color':
                    diff = np.linalg.norm(datos[y, x] - valor_ref)
                else:  # textura
                    diff = self._diferencia_textura(datos, region, y, x)
                
                if diff < self.umbral:
                    # Agregar a región
                    region[y, x] = True
                    
                    # Actualizar referencia (media móvil)
                    if self.criterio == 'intensidad':
                        valor_ref = datos[region].mean()
                    
                    # Agregar vecinos a nueva frontera
                    for dy, dx in self.vecinos:
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx < w and not region[ny, nx]:
                            nueva_frontera.append((ny, nx))
            
            frontera = list(set(nueva_frontera))  # Eliminar duplicados
        
        return region
    
    def _diferencia_textura(self, datos: np.ndarray, region: np.ndarray, y: int, x: int) -> float:
        """Calcula diferencia de textura local (desviación estándar en vecindario)."""
        ventana = 3
        y1, y2 = max(0, y-ventana), min(datos.shape[0], y+ventana+1)
        x1, x2 = max(0, x-ventana), min(datos.shape[1], x+ventana+1)
        
        std_local = datos[y1:y2, x1:x2].std()
        std_region = datos[region].std()
        
        return abs(std_local - std_region)
    
    def _filtrar_pequeñas(self, etiquetas: np.ndarray) -> np.ndarray:
        """Elimina regiones más pequeñas que el umbral."""
        resultado = etiquetas.copy()
        for i in range(1, etiquetas.max() + 1):
            if np.sum(etiquetas == i) < self.min_area:
                resultado[etiquetas == i] = 0
        resultado, _ = ndimage.label(resultado > 0)
        return resultado.astype(np.int32)

@registrar_en("segmentado")
class RandomWalk(SegmentadorRegional):
    """
        Segmentación por caminos aleatorios (Random Walker).
        
        Método probabilístico que resuelve el problema de segmentación
        mediante caminatas aleatorias desde píxeles no etiquetados hasta
        semillas etiquetadas. La probabilidad de pertenencia a una región
        es proporcional al número de caminos que llegan a la semilla de esa
        región.
        
        Algoritmo:
            1. Definir semillas etiquetadas (algunos píxeles con ID de región)
            2. Construir grafo de adyacencia con pesos basados en gradiente
            3. Resolver sistema lineal: para cada píxel no etiquetado,
            probabilidad de alcanzar cada semilla = promedio de probabilidades
            de vecinos ponderado por conductancia (inverso del peso)
            4. Asignar a cada píxel la región de mayor probabilidad
        
        Ecuación (proceso de difusión):
            ∇·(w(x)∇u(x)) = 0  con u|_semillas = fijo
            
            donde:
                w(x) = exp(-β||∇I(x)||²) : conductancia (peso del borde)
                u(x) : probabilidad de pertenencia
                β : parámetro de sensibilidad al borde
        
        Interpretación física:
            - La imagen es un medio conductor con resistencia inversa al gradiente
            - Las semillas son electrodos con potencial fijo (1 para su región, 0 para otras)
            - La solución es el potencial de equilibrio (Laplace equation)
            - Píxeles toman etiqueta del electrodo con mayor potencial influyente
        
        Ventajas:
            - Robusto a ruido (promedio de muchos caminos)
            - Maneja bien fronteras débiles o discontinuas
            - Propagación de información global (no solo local como region growing)
            - Resultados suaves y naturales (sin efectos de bloque)
            - Pocos parámetros que ajustar (principalmente β)
            - Puede usar pocas semillas (interactivo eficiente)
        
        Desventajas:
            - Costoso computacionalmente (resolver sistema lineal grande)
            - Requiere semillas en cada región deseada
            - Memoria intensiva para imágenes grandes
            - Puede "fugarse" por gaps en fronteras si β es alto
            - Convergencia lenta en regiones grandes homogéneas
        
        Usos típicos en microscopía:
            - Segmentación semi-automática con corrección de usuario (pocas semillas)
            - Segmentación de tejidos con bordes celulares débiles (gap junctions)
            - Análisis de imágenes médicas con ruido significativo (MRI, CT baja dosis)
            - Segmentación de células con membranas discontinuas (proteínas de adhesión)
            - Propagación de segmentaciones manuales en series temporales
        
        Referencia:
            Grady, L. (2006). Random walks for image segmentation.
            IEEE TPAMI, 28(11), 1768-1783.
    """
    nombre = "random_walk"
    
    def __init__(self,
                beta: float = 130.0,
                modo: Literal['bf', 'cg', 'cg_mg', 'cg_j'] = 'bf',
                tol: float = 1e-3,
                multichannel: bool = False):
        """
            Args:
                beta: Parámetro de penalización de bordes
                    Controla cuánto influye el gradiente en los pesos
                    w = exp(-β * ||grad||^2)
                    Valores típicos: 10-500
                    Bajo (10): Peso uniforme, caminos casi libres
                    Alto (500): Caminos siguen fuertemente bordes de bajo gradiente
                
                modo: Método de solución del sistema lineal
                    'bf': Bellman-Ford (lento pero exacto, para imágenes pequeñas)
                    'cg': Conjugate Gradient (recomendado, balance velocidad/precisión)
                    'cg_mg': CG con multigrid (más rápido para grandes)
                    'cg_j': CG con Jacobi preconditioner
                
                tol: Tolerancia de convergencia (1e-3 típico)
                
                multichannel: Si True, usa información de color (3 canales)
        """
        self.beta = beta
        self.modo = modo
        self.tol = tol
        self.multichannel = multichannel
    
    def __call__(self,
                img: np.ndarray,
                semillas: np.ndarray) -> np.ndarray:
        """
            Aplica segmentación por caminos aleatorios.
            
            Args:
                img: Imagen de intensidad (2D) o color (3D si multichannel=True)
                semillas: Array del mismo tamaño que img con:
                        0 = píxel no etiquetado (a segmentar)
                        1,2,3... = ID de región/semilla
                
            Returns:
                Imagen etiquetada (int32) con segmentación completa
        """
        self._validar_imagen(img, permitir_multicanal=self.multichannel)
        
        if semillas.shape != img.shape[:2]:
            raise ValueError("Las semillas deben tener mismo tamaño espacial que la imagen")
        
        # Verificar que hay semillas
        if semillas.max() == 0:
            raise ValueError("No hay semillas etiquetadas (todas son 0)")
        
        # Preparar datos
        if self.multichannel and img.ndim == 3:
            datos = img
        else:
            datos = img.astype(np.float64)
            if datos.ndim == 3:
                datos = color.rgb2gray(datos)
        
        # Aplicar random walker
        try:
            from skimage.segmentation import random_walker
            
            etiquetas = random_walker(
                datos,
                semillas,
                beta=self.beta,
                mode=self.modo,
                tol=self.tol,
                multichannel=self.multichannel
            )
            
            return etiquetas.astype(np.int32)
            
        except Exception as e:
            warnings.warn(f"Error en random_walker: {e}. Intentando con parámetros conservadores.")
            # Fallback con parámetros más conservadores
            etiquetas = random_walker(
                datos,
                semillas,
                beta=10.0,
                mode='bf',
                tol=1e-2,
                multichannel=self.multichannel
            )
            return etiquetas.astype(np.int32)

@registrar_en("segmentado")
class CorteGrafico(SegmentadorRegional):
    """
        Segmentación por corte mínimo en grafo (Graph Cut).
        
        Formulación de segmentación como problema de optimización global:
        encontrar la partición del grafo que minimiza la energía definida por
        términos de datos (afinidad a regiones) y términos de suavidad
        (penalización de fronteras).
        
        Algoritmo (Boykov-Kolmogorov):
            1. Construir grafo de píxeles (nodos) con aristas a vecinos
            2. Agregar nodo fuente (S) y sumidero (T) conectados a todos
            3. Pesos de aristas:
            - nodo-pixel: costo de asignar a región A o B (término de datos)
            - pixel-pixel: costo de corte (término de frontera/borde)
            4. Encontrar corte S-T de capacidad mínima (max-flow min-cut)
            5. Píxeles conectados a S = región A, a T = región B
        
        Energía (funcional a minimizar):
            E(L) = Σ D(p, L_p) + λ Σ V(p,q) · δ(L_p ≠ L_q)
            
            donde:
                L: Etiquetado de píxeles
                D(p, L_p): Costo de asignar píxel p a etiqueta L_p
                V(p,q): Potencial de frontera entre p y q (ej: 1/||I_p - I_q||)
                λ: Peso relativo de suavidad vs datos
                δ: Función indicadora (1 si diferentes, 0 si iguales)
        
        Ventajas:
            - Optimización global (mínimo garantizado para 2 regiones)
            - Maneja bien ruido (término de suavidad regulariza)
            - Puede incorporar información de forma (priors)
            - Eficiente para 2 regiones (binario) con max-flow algorithms
            - Resultados nítidos en fronteras (no difusos como random walker)
        
        Desventajas:
            - NP-hard para >2 regiones (aproximaciones necesarias)
            - Requiere inicialización (semillas o estimación inicial)
            - Memoria intensiva (grafo completo es costoso)
            - Término de datos debe diseñarse cuidadosamente
            - Puede producir soluciones "shrinkage" (favorece regiones pequeñas)
        
        Usos típicos en microscopía:
            - Segmentación binaria foreground/background (célula vs fondo)
            - Extracción de objetos con bordes bien definidos
            - Segmentación con modelos de forma (incorporar prior de elipticidad)
            - Video segmentation (propagación con temporal consistency)
            - Refinamiento de segmentaciones iniciales (gruesas a finas)
        
        Referencia:
            Boykov, Y., & Kolmogorov, V. (2004). An experimental comparison of
            min-cut/max-flow algorithms for energy minimization in vision.
            IEEE TPAMI, 26(9), 1124-1137.
    """
    nombre = "corte_grafico"
    
    def __init__(self,
                lambda_suavidad: float = 1.0,
                sigma_color: float = 10.0,
                n_iter: int = 5,
                algoritmo: Literal['max_flow', 'swap', 'expansion'] = 'max_flow'):
        """
            Args:
                lambda_suavidad: Peso del término de suavidad (frontera)
                                vs término de datos (0 = solo datos, ∞ = solo suavidad)
                                Valores típicos: 0.1-10
                
                sigma_color: Parámetro de escala para potencial de borde
                            V(p,q) = exp(-||I_p - I_q||² / (2σ²))
                            Valores típicos: 5-50 (depende de rango de intensidad)
                
                n_iter: Iteraciones para algoritmos multi-etiqueta (swap/expansion)
                
                algoritmo: Método de optimización
                        'max_flow': Para 2 regiones (exacto, rápido)
                        'swap': Alpha-expansion para multi-etiqueta
                        'expansion': Alpha-expansion (generalmente mejor que swap)
        """
        self.lambda_suavidad = lambda_suavidad
        self.sigma_color = sigma_color
        self.n_iter = n_iter
        self.algoritmo = algoritmo
    
    def __call__(self,
                img: np.ndarray,
                semillas: Optional[np.ndarray] = None,
                n_regiones: int = 2) -> np.ndarray:
        """
            Aplica segmentación por corte gráfico.
            
            Args:
                img: Imagen de intensidad (2D) o color (3D)
                semillas: Opcional, máscara binaria o etiquetada con inicialización
                n_regiones: Número de regiones a segmentar (2 para binario)
                
            Returns:
                Imagen etiquetada (int32)
        """
        self._validar_imagen(img, permitir_multicanal=True)
        
        # Para 2 regiones, usar implementación específica
        if n_regiones == 2:
            return self._corte_binario(img, semillas)
        else:
            return self._corte_multi_etiqueta(img, semillas, n_regiones)
    
    def _corte_binario(self, img: np.ndarray, semillas: Optional[np.ndarray]) -> np.ndarray:
        """Corte gráfico para 2 regiones usando max-flow."""
        from skimage.segmentation import slic, mark_boundaries
        
        # Simplificar: usar implementación basada en SLIC + graph cut
        # Para implementación completa de max-flow se necesitaría pygraphcut o similar
        
        # Alternativa usando skimage.graph.cut_threshold como aproximación
        # y luego refinamiento
        
        # Pre-segmentación con SLIC para reducir tamaño del problema
        n_segments = min(500, img.shape[0] * img.shape[1] // 100)
        superpixels = slic(img, n_segments=n_segments, compactness=30)
        
        # Construir grafo de regiones (RAG)
        g = graph.rag_mean_color(img, superpixels, mode='similarity')
        
        # Si hay semillas, incorporarlas como constraints
        if semillas is not None:
            # Modificar pesos del grafo según semillas
            for nodo in g.nodes():
                mascara_sp = (superpixels == nodo)
                # Verificar overlap con semillas
                if semillas[mascara_sp].mean() > 0.5:
                    # Forzar conexión a fuente (región 1)
                    pass  # Implementación específica depende de librería
        
        # Corte normalizado (aproximación a min-cut)
        etiquetas_sp = graph.cut_normalized(superpixels, g)
        
        # Expandir a píxeles
        from skimage.segmentation import relabel_sequential
        etiquetas = etiquetas_sp[superpixels]
        
        # Asegurar 2 regiones
        if etiquetas.max() != 1:
            # Binarizar por umbral de tamaño o intensidad media
            medias = [img[etiquetas == i].mean() for i in range(etiquetas.max() + 1)]
            # Asumir que la región más brillante es foreground
            fg_id = np.argmax(medias)
            etiquetas = (etiquetas == fg_id).astype(np.int32)
        
        return etiquetas
    
    def _corte_multi_etiqueta(self, img: np.ndarray, semillas: Optional[np.ndarray], n: int) -> np.ndarray:
        """Corte gráfico para múltiples regiones (alpha-expansion)."""
        # Usar SLIC + cut_normalized como aproximación
        n_segments = min(1000, img.shape[0] * img.shape[1] // 50)
        superpixels = slic(img, n_segments=n_segments, compactness=20)
        
        g = graph.rag_mean_color(img, superpixels, mode='similarity')
        
        # Corte en n regiones
        etiquetas_sp = graph.cut_threshold(superpixels, g, thresh=0.2)
        
        # Re-etiquetar para tener exactamente n regiones (aproximación)
        from skimage.segmentation import relabel_sequential
        etiquetas_sp, _, _ = relabel_sequential(etiquetas_sp)
        
        # Fusionar regiones si hay más de n
        while etiquetas_sp.max() >= n:
            # Encontrar regiones más similares y fusionar
            # Simplificación: fusionar las más pequeñas
            areas = [np.sum(etiquetas_sp == i) for i in range(etiquetas_sp.max() + 1)]
            id_min = np.argmin(areas[1:]) + 1  # No fusionar fondo (0)
            etiquetas_sp[etiquetas_sp == id_min] = 0
        
        etiquetas = etiquetas_sp[superpixels]
        return etiquetas.astype(np.int32)

@registrar_en("segmentado")
class SuperpixelSLIC(SegmentadorRegional):
    """
        Segmentación en superpíxeles compactos y homogéneos (SLIC).
        
        SLIC (Simple Linear Iterative Clustering) genera superpíxeles
        (grupos de píxeles perceptualmente coherentes) mediante clustering
        de píxeles en espacio 5D combinado: [L, a, b, x, y] (color + espacio).
        
        Algoritmo:
            1. Inicializar centroides de superpíxeles en grid regular
            2. Asignar píxeles al centroide más cercano en espacio 5D
            3. Recalcular centroides (media de píxeles asignados)
            4. Iterar 2-3 hasta convergencia
        
        Distancia SLIC:
            d = sqrt((d_lab/S)² + (d_xy/N)²)
            
            donde:
                d_lab = ||[L,a,b]_pixel - [L,a,b]_centroide|| (distancia color)
                d_xy = ||[x,y]_pixel - [x,y]_centroide|| (distancia espacial)
                S = √(N/K) : paso del grid (N=píxeles, K=superpíxeles)
                m = compactness : controla peso espacial vs color (típicamente 10)
        
        Ventajas:
            - Superpíxeles compactos (pocos bordes irregulares)
            - Homogéneos en color (siguen bordes de objeto)
            - Muy rápido (O(N), clustering local)
            - Reduce complejidad de imagen (N píxeles → K superpíxeles)
            - Mejora coherencia espacial vs clustering puro de color (K-means)
            - Control explícito de número y compactidad
        
        Desventajas:
            - Puede perder detalles finos (< tamaño de superpíxel)
            - Bordes de superpíxel no siempre coinciden con bordes de objeto
            - Parámetro m (compactness) requiere ajuste por imagen
            - Forma cuadrada preferida (sesgo de grid inicial)
            - No garantiza conectividad (puede haber superpíxeles disconexos, raro)
        
        Usos típicos en microscopía:
            - Pre-segmentación para reducir complejidad computacional
            - Segmentación de tejidos homogéneos (regiones funcionales)
            - Análisis de textura en escalas intermedias
            - Post-procesamiento: operaciones morfológicas en superpíxeles
            - Entrada para clasificadores de regiones (ej: célula vs fondo)
            - Compresión de imágenes manteniendo características
        
        Referencia:
            Achanta, R., et al. (2012). SLIC superpixels compared to state-of-the-art
            superpixel methods. IEEE TPAMI, 34(11), 2274-2282.
    """
    nombre = "superpixel_slic"
    
    def __init__(self,
                n_superpixels: int = 100,
                compactness: float = 10.0,
                sigma: float = 1.0,
                max_iter: int = 10,
                enforce_connectivity: bool = True):
        """
            Args:
                n_superpixels: Número aproximado de superpíxeles deseados
                            (resultado puede variar ±10%)
                
                compactness: Balance entre homogeneidad de color y compactidad espacial
                            1: Prioridad al color (bordes irregulares, siguen objeto)
                            10: Balance (recomendado)
                            100: Prioridad espacial (cuadrados regulares, ignoran bordes)
                
                sigma: Suavizado gaussiano previo (0 = deshabilitado)
                
                max_iter: Máximo de iteraciones de clustering (10 típico, converge rápido)
                
                enforce_connectivity: Si True, post-procesa para garantizar
                                    que cada superpíxel es conexo (elimina islas)
        """
        self.n_superpixels = n_superpixels
        self.compactness = compactness
        self.sigma = sigma
        self.max_iter = max_iter
        self.enforce_connectivity = enforce_connectivity
    
    def __call__(self, img: np.ndarray) -> np.ndarray:
        """
            Genera superpíxeles SLIC.
            
            Args:
                img: Imagen de color (RGB) o intensidad (se convertirá a RGB)
                
            Returns:
                Imagen etiquetada (int32) donde cada superpíxel tiene ID único
        """
        self._validar_imagen(img, permitir_multicanal=True)
        
        # Asegurar formato RGB
        if img.ndim == 2:
            img_rgb = color.gray2rgb(img)
        else:
            img_rgb = img
        
        # Aplicar SLIC
        from skimage.segmentation import slic
        
        etiquetas = slic(
            img_rgb,
            n_segments=self.n_superpixels,
            compactness=self.compactness,
            sigma=self.sigma,
            max_iter=self.max_iter,
            enforce_connectivity=self.enforce_connectivity,
            start_label=1  # Empezar en 1 (0 es fondo en algunas convenciones)
        )
        
        return etiquetas.astype(np.int32)
    
    def get_boundaries(self, 
                    img: np.ndarray, 
                    etiquetas: Optional[np.ndarray] = None) -> np.ndarray:
        """
            Devuelve imagen con límites de superpíxeles marcados.
            
            Args:
                img: Imagen original
                etiquetas: Si None, se recalcula SLIC
                
            Returns:
                Imagen RGB con bordes de superpíxeles en rojo
        """
        if etiquetas is None:
            etiquetas = self(img)
        
        from skimage.segmentation import mark_boundaries
        return (mark_boundaries(img, etiquetas) * 255).astype(np.uint8)

@registrar_en("segmentado")
class SuperpixelFelzenszwalb(SegmentadorRegional):
    """
        Superpíxeles por partición de grafos (Felzenszwalb-Huttenlocher).
        
        Algoritmo de segmentación basado en evidencia de borde en grafo de
        píxeles. Produce segmentaciones que preservan bordes significativos
        mientras agrupa regiones homogéneas.
        
        Algoritmo:
            1. Construir grafo donde nodos=píxeles, aristas=vecindad 4/8
            2. Peso de arista = |I(p) - I(q)| (diferencia de intensidad)
            3. Ordenar aristas por peso creciente
            4. Para cada arista, fusionar componentes si:
            w(e) ≤ min(Int(C1) + k/|C1|, Int(C2) + k/|C2|)
            donde Int(C) es máxima arista interna del componente C
            5. Resultado: forest de árboles de expansión mínima (regiones)
        
        Criterio de fusión (evidencia de borde interno):
            τ(C) = k / |C|  (umbral adaptativo por tamaño de región)
            
            Fusionar si: w(e) ≤ Int(C1) + τ(C1)  o  w(e) ≤ Int(C2) + τ(C2)
        
        Interpretación:
            - k controla tamaño de regiones (mayor k = regiones más grandes)
            - Umbral adaptativo: regiones pequeñas más fáciles de fusionar
            - Preserva bordes fuertes (aristas de alto peso no se cruzan)
            - Regiones pueden ser de cualquier forma (no forzadas a compactas)
        
        Ventajas:
            - Preserva bordes significativos (evidencia perceptual)
            - Umbral adaptativo por tamaño (no fragmenta regiones pequeñas)
            - Regiones de formas naturales (no cuadradas)
            - Un solo parámetro (k) que controla escala de segmentación
            - Rápido (casi lineal en número de píxeles)
            - No requiere número de regiones a priori
        
        Desventajas:
            - Regiones pueden ser muy alargadas o irregulares
            - No control directo de número de regiones
            - Puede producir regiones disconexas (raro pero posible)
            - Parámetro k depende de rango de intensidad de la imagen
            - Menos compacto que SLIC (útil para algunas aplicaciones, no para otras)
        
        Usos típicos en microscopía:
            - Segmentación de regiones funcionales en tejidos
            - Detección de áreas homogéneas en imágenes histológicas
            - Pre-segmentación para detección de bordes de objeto
            - Análisis de granularidad textural
            - Segmentación de fondo/textura en imágenes de contraste de fase
        
        Comparación con SLIC:
            - Felzenszwalb: Preserva bordes, formas naturales, escala variable
            - SLIC: Compacto, control de número, cuadrícula regular
        
        Referencia:
            Felzenszwalb, P. F., & Huttenlocher, D. P. (2004). Efficient graph-based
            image segmentation. IJCV, 59(2), 167-181.
    """
    nombre = "superpixel_felzenszwalb"
    
    def __init__(self,
                scale: float = 1.0,
                sigma: float = 0.8,
                min_size: int = 20):
        """
            Args:
                scale: Parámetro k de umbral de evidencia de borde
                    Controla tamaño de regiones (mayor = regiones más grandes)
                    Valores típicos: 0.5-5 (ajustar por rango de intensidad)
                
                sigma: Suavizado gaussiano previo (reduce ruido, evita oversegmentation)
                    Valores típicos: 0-1 (0 = sin suavizado)
                
                min_size: Tamaño mínimo de componente (en píxeles)
                        Filtra regiones muy pequeñas (ruido)
        """
        self.scale = scale
        self.sigma = sigma
        self.min_size = min_size
    
    def __call__(self, img: np.ndarray) -> np.ndarray:
        """
            Genera superpíxeles por partición de grafos.
            
            Args:
                img: Imagen de intensidad (2D) o color (3D, se usará luminancia)
                
            Returns:
                Imagen etiquetada (int32)
        """
        self._validar_imagen(img, permitir_multicanal=True)
        
        # Convertir a gris si es color
        if img.ndim == 3:
            img_proc = color.rgb2gray(img)
        else:
            img_proc = img.astype(np.float64)
        
        # Aplicar Felzenszwalb
        from skimage.segmentation import felzenszwalb
        
        etiquetas = felzenszwalb(
            img_proc,
            scale=self.scale,
            sigma=self.sigma,
            min_size=self.min_size
        )
        
        return etiquetas.astype(np.int32)

@registrar_en("segmentado")
class WatershedRegiones(SegmentadorRegional):
    """
        Watershed como segmentador regional sin máscara binaria previa.
        
        Versión del algoritmo watershed que opera directamente sobre la
        imagen de intensidad sin requerir una máscara binaria de objetos.
        Útil para segmentación regional topográfica completa.
        
        Algoritmo:
            1. Calcular gradiente o invertir intensidad para topografía
            2. Detectar todos los mínimos locales como cuencas iniciales
            3. Inundar desde mínimos hasta que cuencas se encuentren
            4. Líneas de watershed forman segmentación completa
        
        Diferencia con Watershed instancial:
            - Instancial: Requiere máscara binaria, separa objetos conocidos
            - Regional: Opera sobre imagen completa, encuentra regiones naturales
        
        Ventajas:
            - No requiere binarización previa (menos parámetros)
            - Segmentación completa de la imagen (no hay "fondo" no segmentado)
            - Regiones siguen topografía natural de la imagen
            - Útil para análisis de textura y patrones
        
        Desventajas:
            - Grave sobre-segmentación (cada mínimo local es una región)
            - Requiere pre-procesado agresivo (suavizado) para reducir mínimos
            - Sin control de qué es objeto vs fondo
            - Regionas pueden no corresponder a objetos semánticos
        
        Usos típicos:
            - Análisis de textura topográfica
            - Segmentación de patrones repetitivos
            - Pre-segmentación para análisis de regiones
            - Estudio de morfología de superficie (metrología)
    """
    nombre = "watershed_regiones"
    
    def __init__(self,
                sigma_gradiente: float = 2.0,
                marcadores: Optional[int] = None,
                compactness: float = 0.0):
        """
            Args:
                sigma_gradiente: Suavizado para reducir mínimos espurios
                
                marcadores: Si se especifica, número aproximado de regiones
                        (usa peak_local_max con min_distance calculado)
                        Si None, usa todos los mínimos locales
                
                compactness: Favorece regiones compactas
        """
        self.sigma_gradiente = sigma_gradiente
        self.marcadores = marcadores
        self.compactness = compactness
    
    def __call__(self, img: np.ndarray) -> np.ndarray:
        """
            Aplica watershed regional completo.
            
            Args:
                img: Imagen de intensidad
                
            Returns:
                Imagen etiquetada con todas las cuencas detectadas
        """
        self._validar_imagen(img)
        
        from skimage.filters import sobel, gaussian
        from skimage.segmentation import watershed
        from skimage.feature import peak_local_max
        
        # Topografía
        img_suave = gaussian(img.astype(np.float64), sigma=self.sigma_gradiente)
        gradiente = sobel(img_suave)
        topografia = gradiente
        
        if self.marcadores is None:
            # Todos los mínimos locales
            from skimage.morphology import local_minima
            minimos = local_minima(topografia)
            marcadores, _ = ndimage.label(minimos)
        else:
            # Número controlado de marcadores
            distancia = self.marcadores
            coords = peak_local_max(-topografia, 
                                    min_distance=int(np.sqrt(img.size / self.marcadores)),
                                    exclude_border=False)
            marcadores = np.zeros_like(img, dtype=int)
            for i, (y, x) in enumerate(coords, 1):
                marcadores[y, x] = i
        
        # Watershed sin máscara (segmenta todo)
        etiquetas = watershed(topografia, marcadores, compactness=self.compactness)
        
        return etiquetas.astype(np.int32)

@registrar_en("segmentado")
class MeanShiftSegmentacion(SegmentadorRegional):
    """
        Segmentación por modo de densidad (Mean Shift clustering).
        
        Algoritmo de clustering no paramétrico que encuentra modos (máximos
        locales) en la densidad de píxeles en espacio conjunto color-espacio.
        Cada píxel converge al modo más cercano formando regiones.
        
        Algoritmo:
            1. Para cada píxel, definir ventana (esfera) de radio h
            2. Calcular media de píxeles dentro de la ventana
            3. Mover centro de ventana a la media (shift)
            4. Repetir hasta convergencia (píxel llega a modo)
            5. Píxeles convergiendo al mismo modo forman región
        
        Espacio de características:
            x = [L, a, b, x, y]  (color CIELAB + coordenadas espaciales)
            o separado: color y espacio con bandwidths diferentes
        
        Ecuación del shift:
            m(x) = [Σ x_i K(x - x_i)] / [Σ K(x - x_i)] - x
            
            donde K es kernel (típicamente Epanechnikov o Gaussiano)
        
        Ventajas:
            - No requiere número de regiones a priori (lo encuentra automáticamente)
            - Adaptativo a densidad local (más modos donde hay más variación)
            - Robusto a outliers (ventana limitada)
            - Preserva discontinuidades (convergencia a modos separados)
            - Fundamento estadístico sólido (estimación de densidad)
        
        Desventajas:
            - Costoso computacionalmente (O(N²) naive, O(N) con aproximaciones)
            - Parámetros de ancho de banda (bandwidth) críticos y difíciles de fijar
            - Puede unir regiones distintas si h es muy grande
            - Puede fragmentar regiones homogéneas si h es muy pequeño
            - No garantiza conectividad espacial (regiones pueden ser disconexas)
        
        Usos típicos en microscopía:
            - Segmentación de regiones de color/textura similar
            - Análisis de imágenes multiespectrales (más de 3 canales)
            - Segmentación de tejidos con gradientes de intensidad (no bordes nítidos)
            - Análisis de imágenes de ratiometría o FRET
            - Clustering de características de textura (no solo intensidad)
        
        Referencia:
            Comaniciu, D., & Meer, P. (2002). Mean shift: A robust approach toward
            feature space analysis. IEEE TPAMI, 24(5), 603-619.
    """
    nombre = "mean_shift_segmentacion"
    
    def __init__(self,
                bandwidth_color: Optional[float] = None,
                bandwidth_space: float = 20.0,
                min_region_size: int = 50,
                bin_seeding: bool = True):
        """
            Args:
                bandwidth_color: Ancho de banda para componentes de color (L,a,b)
                            Si None, se estima automáticamente (scott's rule)
                            Valores típicos: 5-30 (en unidades CIELAB)
                
                bandwidth_space: Ancho de banda para coordenadas espaciales (x,y)
                                Controla compactidad espacial (mayor = regiones más grandes)
                                Valores típicos: 10-100 píxeles
                
                min_region_size: Tamaño mínimo de región válida (filtra modos espurios)
                
                bin_seeding: Si True, inicializa desde bins discretos (más rápido)
        """
        self.bandwidth_color = bandwidth_color
        self.bandwidth_space = bandwidth_space
        self.min_region_size = min_region_size
        self.bin_seeding = bin_seeding
    
    def __call__(self, img: np.ndarray) -> np.ndarray:
        """
            Aplica segmentación Mean Shift.
            
            Args:
                img: Imagen de color (RGB) o intensidad (convertida a RGB)
                
            Returns:
                Imagen etiquetada (int32)
        """
        self._validar_imagen(img, permitir_multicanal=True)
        
        # Convertir a CIELAB para distancia perceptual uniforme
        if img.ndim == 2:
            img_lab = color.rgb2lab(color.gray2rgb(img))
        else:
            img_lab = color.rgb2lab(img)
        
        h, w = img.shape[:2]
        
        # Preparar datos: [L, a, b, x, y] para cada píxel
        y_coords, x_coords = np.mgrid[0:h, 0:w]
        
        # Normalizar espacial para balance con color
        # L,a,b típicamente 0-100, -128-127; espacio 0-max(h,w)
        scale_space = max(h, w) / 100.0
        
        X = np.zeros((h * w, 5))
        X[:, 0] = img_lab[:, :, 0].ravel()  # L
        X[:, 1] = img_lab[:, :, 1].ravel()  # a
        X[:, 2] = img_lab[:, :, 2].ravel()  # b
        X[:, 3] = y_coords.ravel() / scale_space * self.bandwidth_space
        X[:, 4] = x_coords.ravel() / scale_space * self.bandwidth_space
        
        # Aplicar Mean Shift
        if self.bandwidth_color is None:
            # Estimar desde datos
            from sklearn.cluster import estimate_bandwidth
            bw = estimate_bandwidth(X, quantile=0.2, n_samples=500)
        else:
            # Construir bandwidth completo [color, color, color, space, space]
            bw = [self.bandwidth_color] * 3 + [self.bandwidth_space] * 2
        
        ms = SklearnMeanShift(
            bandwidth=bw if isinstance(bw, (int, float)) else np.mean(bw),
            bin_seeding=self.bin_seeding,
            min_bin_freq=self.min_region_size,
            cluster_all=False
        )
        
        ms.fit(X)
        
        # Reconstruir imagen etiquetada
        etiquetas = ms.labels_.reshape(h, w)
        
        # Re-etiquetar consecutivamente (Mean Shift puede dejar -1 para no asignados)
        etiquetas[etiquetas == -1] = 0
        etiquetas, _ = ndimage.label(etiquetas > 0)
        
        return etiquetas.astype(np.int32)

@registrar_en("segmentado")
class SegmentacionEspectral(SegmentadorRegional):
    """
        Segmentación espectral basada en eigenvalores de matriz de similitud.
        
        Método de clustering que utiliza el espectro (eigenvalores/eigenvectores)
        de la matriz de similitud entre píxeles para reducir dimensionalidad
        antes de clustering (típicamente K-means).
        
        Algoritmo (Normalized Cut):
            1. Construir grafo de píxeles con pesos de similitud
            W(i,j) = exp(-||I_i - I_j||² / σ²) si vecinos, 0 si no
            2. Calcular matriz Laplaciana: L = D - W (D = grados)
            3. Resolver eigenvalores generalizados: Lv = λDv
            4. Usar k eigenvectores menores como nueva representación (embedding)
            5. Aplicar K-means en este espacio reducido
        
        Ecuación:
            Normalized Cut: Ncut(A,B) = cut(A,B)/assoc(A,V) + cut(A,B)/assoc(B,V)
            
            donde cut(A,B) = Σ W(i,j) para i∈A, j∈B
                assoc(A,V) = Σ W(i,j) para i∈A, j∈V (todos)
        
        Interpretación:
            - Minimizar Ncut busca partición donde bordes tienen bajo peso
            (píxeles de diferentes regiones son distintos)
            y regiones tienen alta conectividad interna
            - Embedding espectral: píxeles similares se mapean cerca en espacio reducido
        
        Ventajas:
            - Optimización global (no greedy como region growing)
            - Maneja regiones de formas arbitrarias (no compactas)
            - Captura estructura global de la imagen (no solo local)
            - Fundamento teórico sólido (teoría espectral de grafos)
            - Flexible en definición de similitud (color, textura, espacio)
        
        Desventajas:
            - Costoso computacionalmente (eigenvalores de matriz grande)
            - Memoria intensiva (matriz de similitud N×N, aunque sparse)
            - Número de regiones k debe especificarse
            - Umbral de similitud σ difícil de fijar
            - Puede sobre-suavizar (preferencia por regiones grandes)
        
        Usos típicos en microscopía:
            - Segmentación de tejidos con estructura global compleja
            - Análisis de imágenes donde la conectividad es más importante que bordes nítidos
            - Segmentación de regiones funcionales en imágenes de conectividad neuronal
            - Agrupación de células con base en similitud de perfil de expresión (multicanal)
        
        Referencia:
            Shi, J., & Malik, J. (2000). Normalized cuts and image segmentation.
            IEEE TPAMI, 22(8), 888-905.
    """
    nombre = "segmentacion_espectral"
    
    def __init__(self,
                n_regiones: int = 2,
                sigma: float = 5.0,
                modo: Literal['ncut', 'kmeans'] = 'ncut',
                n_vecinos: int = 10):
        """
            Args:
                n_regiones: Número de regiones a segmentar (k)
                
                sigma: Escala para pesos de similitud gaussiana
                    W(i,j) = exp(-||I_i - I_j||² / (2σ²))
                
                modo: 'ncut' para Normalized Cut, 'kmeans' para clustering espectral simple
                
                n_vecinos: Número de vecinos cercanos para grafo sparse (reducir costo)
        """
        self.n_regiones = n_regiones
        self.sigma = sigma
        self.modo = modo
        self.n_vecinos = n_vecinos
    
    def __call__(self, img: np.ndarray) -> np.ndarray:
        """
            Aplica segmentación espectral.
            
            Args:
                img: Imagen de intensidad o color
                
            Returns:
                Imagen etiquetada (int32)
        """
        self._validar_imagen(img, permitir_multicanal=True)
        
        h, w = img.shape[:2]
        n_pixels = h * w
        
        # Preparar datos
        if img.ndim == 3:
            X = img.reshape(-1, 3).astype(np.float64)
        else:
            X = img.ravel().astype(np.float64).reshape(-1, 1)
        
        # Normalizar
        X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-10)
        
        # Construir grafo de k-vecinos más cercanos (sparse)
        from sklearn.neighbors import kneighbors_graph
        
        # Usar coordenadas espaciales + color para similitud
        y_coords, x_coords = np.mgrid[0:h, 0:w]
        coords = np.column_stack([y_coords.ravel(), x_coords.ravel()])
        
        # Combinar color y espacio (ponderado)
        X_combined = np.column_stack([
            X * self.sigma,  # Color escalado
            coords * 0.5     # Espacio con peso menor
        ])
        
        # Grafo de vecindad
        knn_graph = kneighbors_graph(X_combined, n_neighbors=self.n_vecinos, mode='distance')
        
        # Pesos gaussianos
        W = knn_graph.copy()
        W.data = np.exp(-W.data**2 / (2 * self.sigma**2))
        
        # Hacer simétrico
        W = np.maximum(W, W.T)
        
        # Laplaciana
        D = np.array(W.sum(axis=0)).ravel()
        D_inv_sqrt = np.diag(1.0 / np.sqrt(D + 1e-10))
        L_sym = np.eye(n_pixels) - D_inv_sqrt @ W @ D_inv_sqrt
        
        # Eigenvalores (usar sparse para eficiencia)
        from scipy.sparse.linalg import eigsh
        
        try:
            # k eigenvectores menores
            eigenvalues, eigenvectors = eigsh(L_sym, k=self.n_regiones, which='SM')
        except:
            # Fallback a dense si sparse falla (imágenes pequeñas)
            eigenvalues, eigenvectors = np.linalg.eigh(L_sym.toarray())
            idx = eigenvalues.argsort()
            eigenvectors = eigenvectors[:, idx[:self.n_regiones]]
        
        # Normalizar filas (embedding espectral)
        embedding = eigenvectors
        row_norms = np.linalg.norm(embedding, axis=1, keepdims=True)
        embedding = embedding / (row_norms + 1e-10)
        
        # K-means en embedding
        from sklearn.cluster import KMeans
        
        kmeans = KMeans(n_clusters=self.n_regiones, n_init=10, random_state=42)
        etiquetas = kmeans.fit_predict(embedding)
        
        return etiquetas.reshape(h, w).astype(np.int32)