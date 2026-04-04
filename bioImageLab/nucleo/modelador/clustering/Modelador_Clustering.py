"""
Clustering y segmentación de poblaciones celulares en datos multiparamétricos.

Este módulo recibe DataFrames con métricas cuantificadas (formato estándar del
pipeline: '{metrica}_{estadistico}') y aplica algoritmos de clustering para
identificar subpoblaciones, fenotipos discretos o estados celulares.

Estructura del DataFrame de entrada:
    ┌──────────┬────────────────┬──────────────┬─────────────┬──────────────┐
    │ imagen   │ grupo_exp      │ area_media   │ area_std    │ circularidad │
    ├──────────┼────────────────┼──────────────┼─────────────┼──────────────┤
    │ img_001  │ control        │ 312.5        │ 45.2        │ 0.85         │
    │ img_002  │ tratamiento_A  │ 289.1        │ 38.7        │ 0.72         │
    └──────────┴────────────────┴──────────────┴─────────────┴──────────────┘

IMPORTANTE — Principios de coherencia:
    Los algoritmos de clustering operan sobre la matriz X ya estandarizada.
    La estandarización debe aplicarse ANTES usando PreprocesamientoFeatures
    del módulo Modelador_dimensionalidad para garantizar:
        - KMeans: convergencia estable y centroides interpretables
        - DBSCAN: eps comparable en todas las dimensiones
        - HDBSCAN: densidades comparables en espacio transformado

Separación de responsabilidades:
    - NO realiza cuantificación de imágenes (eso es Cuantificadores_*)
    - NO reduce dimensionalidad (eso es Modelador_dimensionalidad)
    - NO visualiza (eso es Visualizador_*)
    - SÍ: clustering, evaluación de calidad, asignación de etiquetas
    - SÍ: exporta DataFrames con etiquetas y métricas de cluster
"""

import warnings
from typing import Dict, List, Literal, Optional, Tuple, Union, Any

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.neighbors import NearestNeighbors

# Intentar importar HDBSCAN (opcional, requiere instalación)
try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False
    warnings.warn(
        "hdbscan no instalado. Algunos métodos no estarán disponibles. "
        "Instalar con: pip install hdbscan",
        UserWarning
    )


# =============================================================================
# 1. EVALUACIÓN DE CALIDAD DE CLUSTERING
# =============================================================================
@registrar_en("modelado")
class EvaluadorClustering:
    """
    Métricas de evaluación interna y externa para clustering.

    Distingue entre:
        - Métricas internas (solo X y labels): silhouette, Davies-Bouldin, etc.
        - Métricas externas (requieren ground truth): ARI, NMI, V-measure

    Algoritmo Silhouette Score:
        s(i) = (b(i) - a(i)) / max(a(i), b(i))
        donde:
            a(i) = distancia promedio intra-cluster
            b(i) = distancia promedio al cluster más cercano (diferente)

        Rango: [-1, 1]
        1  → cluster compacto y bien separado
        0  → clusters superpuestos
        -1 → asignación incorrecta

    Algoritmo Davies-Bouldin Index:
        DB = (1/k) Σᵢ max_{j≠i} (σᵢ + σⱼ) / d(cᵢ, cⱼ)
        donde σ = dispersión intra-cluster, c = centroide, d = distancia

        Menor es mejor. Mide ratio dispersión/separación.

    Ventajas:
        - Silhouette: interpretable, no requiere ground truth
        - Davies-Bouldin: sensible a clusters convexos bien separados
    Desventajas:
        - Silhouette: costoso O(N²), asume clusters convexos
        - Davies-Bouldin: sensible a outliers, requiere centroides

    Usos microscopía:
        - Validar número óptimo de subpoblaciones celulares
        - Comparar algoritmos de clustering en datos fenotípicos
        - Detectar batch effects (silhouette bajo entre réplicas)
    """
    nombre = "evaluador_clustering"

    def __init__(self, metrica_distancia: str = 'euclidean'):
        """
        Args:
            metrica_distancia: Métrica para silhouette ('euclidean', 'cosine', etc.)
        """
        self.metrica_distancia = metrica_distancia

    def silhouette_score(self, X: np.ndarray, labels: np.ndarray) -> float:
        """
        Calcula el coeficiente de silhouette promedio.

        Args:
            X: Matriz de features (N, D)
            labels: Etiquetas de cluster (N,)

        Returns:
            Silhouette score promedio [-1, 1]
        """
        if len(set(labels)) < 2 or (labels == -1).all():
            return np.nan
        
        # Ignorar ruido (-1) para DBSCAN/HDBSCAN
        mask = labels != -1
        if mask.sum() < 2:
            return np.nan
        
        return float(metrics.silhouette_score(
            X[mask], labels[mask], metric=self.metrica_distancia
        ))

    def davies_bouldin_score(self, X: np.ndarray, labels: np.ndarray) -> float:
        """
        Calcula el índice de Davies-Bouldin (menor es mejor).

        Args:
            X: Matriz de features (N, D)
            labels: Etiquetas de cluster (N,)

        Returns:
            Davies-Bouldin index
        """
        mask = labels != -1
        if len(set(labels[mask])) < 2:
            return np.nan
        
        return float(metrics.davies_bouldin_score(X[mask], labels[mask]))

    def calinski_harabasz_score(self, X: np.ndarray, labels: np.ndarray) -> float:
        """
        Índice de Calinski-Harabasz (varianza entre / varianza intra).

        Mayor es mejor. Rápido de calcular.

        Args:
            X: Matriz de features (N, D)
            labels: Etiquetas de cluster (N,)

        Returns:
            Calinski-Harabasz score
        """
        mask = labels != -1
        if len(set(labels[mask])) < 2:
            return np.nan
        
        return float(metrics.calinski_harabasz_score(X[mask], labels[mask]))

    def adjusted_rand_index(self, labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
        """
        ARI: comparación con ground truth (cuando está disponible).

        Rango: [-1, 1], donde 1 es perfecto acuerdo, 0 es aleatorio.

        Args:
            labels_true: Etiquetas verdaderas (N,)
            labels_pred: Etiquetas predichas (N,)

        Returns:
            Adjusted Rand Index
        """
        return float(metrics.adjusted_rand_score(labels_true, labels_pred))

    def evaluar_completo(self, X: np.ndarray, labels: np.ndarray, 
                        labels_true: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Evaluación completa de calidad de clustering.

        Args:
            X: Matriz de features
            labels: Etiquetas predichas
            labels_true: Etiquetas verdaderas (opcional, para ARI)

        Returns:
            Dict con todas las métricas disponibles
        """
        resultados = {
            'silhouette_score': self.silhouette_score(X, labels),
            'davies_bouldin_index': self.davies_bouldin_score(X, labels),
            'calinski_harabasz_score': self.calinski_harabasz_score(X, labels),
            'n_clusters': len(set(labels)) - (1 if -1 in labels else 0),
            'n_ruido': int(np.sum(labels == -1)),
            'porcentaje_ruido': float(np.sum(labels == -1) / len(labels) * 100),
        }
        
        if labels_true is not None:
            resultados['adjusted_rand_index'] = self.adjusted_rand_index(labels_true, labels)
        
        return resultados


# =============================================================================
# 2. MÉTODOS DE SELECCIÓN DE K (K-MEANS)
# =============================================================================
@registrar_en("modelado")
class SelectorK:
    """
    Métodos para determinar el número óptimo de clusters en K-Means.

    Algoritmo Elbow Method (método del codo):
        Inercia: suma de distancias cuadradas de cada punto a su centroide
        WCSS(k) = Σᵢ Σ_{x∈Cᵢ} ||x - μᵢ||²
        
        El "codo" es donde la disminución de inercia se estabiliza.
        Indica punto de retorno decreciente al añadir más clusters.

    Algoritmo Silhouette Analysis:
        Para cada k, calcular silhouette_score promedio.
        Seleccionar k con máximo silhouette.

    Algoritmo Gap Statistic (Tibshirani et al.):
        Compara log(WCSS) con esperado bajo distribución uniforme de referencia.
        Gap(k) = Eₙ[log(WCSS)] - log(WCSS)
        Seleccionar k más pequeño donde Gap(k) ≥ Gap(k+1) - sₖ₊₁

    Ventajas:
        - Elbow: simple, visual, rápido
        - Silhouette: considera separación y cohesión
        - Gap: teóricamente fundamentado, robusto
    Desventajas:
        - Elbow: subjetivo, puede no haber codo claro
        - Silhouette: costoso para grandes N, favorece k medio
        - Gap: computacionalmente costoso (bootstrapping)

    Usos microscopía:
        - Determinar número de subpoblaciones celulares
        - Optimizar separación de fenotipos discretos
        - Evitar sobre-segmentación de estados continuos
    """
    nombre = "selector_k"

    def __init__(self, 
                 k_min: int = 2, 
                 k_max: int = 10,
                 metodo: Literal['elbow', 'silhouette', 'gap'] = 'silhouette',
                 n_refs_gap: int = 10):
        """
        Args:
            k_min: Mínimo número de clusters a probar
            k_max: Máximo número de clusters a probar
            metodo: 'elbow', 'silhouette', o 'gap'
            n_refs_gap: Número de datasets de referencia para Gap Statistic
        """
        self.k_min = k_min
        self.k_max = k_max
        self.metodo = metodo
        self.n_refs_gap = n_refs_gap

    def encontrar_k_optimo(self, X: np.ndarray, 
                          semilla: int = 42) -> Tuple[int, pd.DataFrame]:
        """
        Encuentra k óptimo según el método especificado.

        Args:
            X: Matriz de features estandarizada (N, D)
            semilla: Semilla para reproducibilidad

        Returns:
            Tupla (k_optimo, DataFrame con métricas para cada k)
        """
        if self.metodo == 'elbow':
            return self._metodo_elbow(X, semilla)
        elif self.metodo == 'silhouette':
            return self._metodo_silhouette(X, semilla)
        elif self.metodo == 'gap':
            return self._metodo_gap(X, semilla)
        else:
            raise ValueError(f"Método no reconocido: {self.metodo}")

    def _metodo_elbow(self, X: np.ndarray, semilla: int) -> Tuple[int, pd.DataFrame]:
        """Implementación del método del codo."""
        inercias = []
        
        for k in range(self.k_min, self.k_max + 1):
            kmeans = KMeans(n_clusters=k, random_state=semilla, n_init=10)
            kmeans.fit(X)
            inercias.append(kmeans.inertia_)
        
        # Heurística simple: punto de mayor curvatura (segunda diferencia)
        diffs = np.diff(inercias, 2)
        k_optimo = np.argmax(diffs) + self.k_min + 1 if len(diffs) > 0 else self.k_min
        
        df_resultados = pd.DataFrame({
            'k': range(self.k_min, self.k_max + 1),
            'inercia': inercias,
            'diff_1': [np.nan] + list(np.diff(inercias)),
            'diff_2': [np.nan, np.nan] + list(diffs) if len(diffs) > 0 else [np.nan] * len(inercias)
        })
        
        return k_optimo, df_resultados

    def _metodo_silhouette(self, X: np.ndarray, semilla: int) -> Tuple[int, pd.DataFrame]:
        """Implementación basada en silhouette score."""
        evaluador = EvaluadorClustering()
        silhouettes = []
        
        for k in range(self.k_min, self.k_max + 1):
            kmeans = KMeans(n_clusters=k, random_state=semilla, n_init=10)
            labels = kmeans.fit_predict(X)
            score = evaluador.silhouette_score(X, labels)
            silhouettes.append(score)
        
        k_optimo = np.argmax(silhouettes) + self.k_min
        
        df_resultados = pd.DataFrame({
            'k': range(self.k_min, self.k_max + 1),
            'silhouette_score': silhouettes
        })
        
        return k_optimo, df_resultados

    def _metodo_gap(self, X: np.ndarray, semilla: int) -> Tuple[int, pd.DataFrame]:
        """
        Gap Statistic (simplificado).
        
        Referencia: Tibshirani, Walther, Hastie (2001)
        """
        np.random.seed(semilla)
        n, d = X.shape
        
        log_wcss = []
        e_log_wcss = []
        
        for k in range(self.k_min, self.k_max + 1):
            # WCSS real
            kmeans = KMeans(n_clusters=k, random_state=semilla, n_init=10)
            kmeans.fit(X)
            log_wcss.append(np.log(kmeans.inertia_))
            
            # WCSS esperado (referencia uniforme)
            refs = []
            for _ in range(self.n_refs_gap):
                # Generar datos uniformes en el bounding box de X
                x_min, x_max = X.min(axis=0), X.max(axis=0)
                X_ref = np.random.uniform(x_min, x_max, size=(n, d))
                
                km_ref = KMeans(n_clusters=k, random_state=semilla, n_init=10)
                km_ref.fit(X_ref)
                refs.append(np.log(km_ref.inertia_))
            
            e_log_wcss.append(np.mean(refs))
        
        gaps = np.array(e_log_wcss) - np.array(log_wcss)
        
        # Seleccionar k más pequeño donde Gap(k) >= Gap(k+1) - sd(k+1)
        k_optimo = self.k_min
        for i in range(len(gaps) - 1):
            sd = np.std([np.log(kmeans.inertia_)])  # Simplificado
            if gaps[i] >= gaps[i+1] - sd:
                k_optimo = i + self.k_min
                break
        
        df_resultados = pd.DataFrame({
            'k': range(self.k_min, self.k_max + 1),
            'log_wcss': log_wcss,
            'e_log_wcss': e_log_wcss,
            'gap': gaps
        })
        
        return k_optimo, df_resultados


# =============================================================================
# 3. ALGORITMOS DE CLUSTERING
# =============================================================================

class ClusterizadorBase:
    """
    Clase base para algoritmos de clustering.

    Todos los clusterizadores reciben X ya estandarizada y retornan:
        - labels: array (N,) con etiquetas de cluster
        - DataFrame con asignaciones y probabilidades (si aplica)
        - Métricas de calidad del clustering
    """
    nombre = "clusterizador_base"

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _validar_X(self, X: np.ndarray) -> None:
        if not isinstance(X, np.ndarray) or X.ndim != 2:
            raise ValueError("X debe ser np.ndarray 2D (N, D).")
        if X.shape[0] < 2:
            raise ValueError("X debe tener al menos 2 muestras.")
        if np.isnan(X).any():
            raise ValueError("X contiene NaN. Imputar antes de clusterizar.")

    def get_dataframe_resultado(self, 
                                 X: np.ndarray, 
                                 labels: np.ndarray,
                                 indices: Optional[List] = None) -> pd.DataFrame:
        """
        Crea DataFrame con resultados del clustering.

        Args:
            X: Matriz de features
            labels: Etiquetas asignadas
            indices: Identificadores de muestras (si None, usa range(N))

        Returns:
            DataFrame con columnas: índice, cluster, coordenadas (si 2D/3D)
        """
        n = len(labels)
        indices = indices or [f"muestra_{i}" for i in range(n)]
        
        df = pd.DataFrame({
            'id': indices,
            'cluster': labels,
            'es_ruido': labels == -1
        })
        
        # Añadir coordenadas si X es 2D o 3D (útil para visualización)
        if X.shape[1] == 2:
            df['x'] = X[:, 0]
            df['y'] = X[:, 1]
        elif X.shape[1] == 3:
            df['x'] = X[:, 0]
            df['y'] = X[:, 1]
            df['z'] = X[:, 2]
        
        return df

@registrar_en("modelado")
class KMeansClustering(ClusterizadorBase):
    """
    K-Means clustering con selección automática de k.

    Algoritmo (Lloyd):
        1. Inicializar k centroides aleatoriamente
        2. Asignar cada punto al centroide más cercano
        3. Recalcular centroides como media de sus puntos
        4. Repetir 2-3 hasta convergencia

    Complejidad: O(N × k × D × iteraciones)

    Ventajas:
        - Eficiente para grandes datasets (O(N))
        - Fácil de interpretar (centroides = "fenotipo promedio")
        - Determinístico dada la inicialización
        - transform() disponible para nuevas muestras

    Desventajas:
        - Asume clusters esféricos y de tamaño similar
        - Sensible a outliers (media no es robusta)
        - Requiere especificar k (resuelto con SelectorK)
        - Converge a mínimo local (n_init > 1 recomendado)

    Usos microscopía:
        - Clasificación de fenotipos celulares discretos
        - Segmentación de fases del ciclo celular
        - Agrupamiento de perfiles de respuesta a fármacos
    """
    nombre = "kmeans"

    def __init__(self,
                 n_clusters: Optional[int] = None,
                 seleccionar_k_auto: bool = False,
                 selector_k: Optional[SelectorK] = None,
                 n_init: int = 10,
                 max_iter: int = 300,
                 semilla: int = 42):
        """
        Args:
            n_clusters: Número de clusters. Si None y seleccionar_k_auto=True,
                       se determina automáticamente.
            seleccionar_k_auto: Si True, usa SelectorK para encontrar k óptimo.
            selector_k: Configuración de SelectorK (si None, usa default).
            n_init: Número de inicializaciones (mejor resultado se queda).
            max_iter: Máximo de iteraciones por ejecución.
            semilla: Semilla para reproducibilidad.
        """
        self.n_clusters = n_clusters
        self.seleccionar_k_auto = seleccionar_k_auto
        self.selector_k = selector_k or SelectorK()
        self.n_init = n_init
        self.max_iter = max_iter
        self.semilla = semilla
        self._modelo = None
        self.inercia_: Optional[float] = None

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """
        Ajusta K-Means y retorna etiquetas.

        Si seleccionar_k_auto=True, primero encuentra k óptimo.

        Args:
            X: Matriz (N, D) estandarizada.

        Returns:
            Array (N,) con etiquetas de cluster.
        """
        self._validar_X(X)

        # Determinar k si es automático
        if self.seleccionar_k_auto or self.n_clusters is None:
            k_optimo, _ = self.selector_k.encontrar_k_optimo(X, self.semilla)
            self.n_clusters = k_optimo
            print(f"K óptimo seleccionado: {self.n_clusters}")

        self._modelo = KMeans(
            n_clusters=self.n_clusters,
            n_init=self.n_init,
            max_iter=self.max_iter,
            random_state=self.semilla
        )
        
        labels = self._modelo.fit_predict(X)
        self.inercia_ = self._modelo.inertia_
        
        return labels

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transforma a espacio de distancias a centroides."""
        if self._modelo is None:
            raise RuntimeError("Llamar fit_predict() primero.")
        return self._modelo.transform(X)

    def get_centroides(self) -> np.ndarray:
        """Retorna centroides de clusters (en espacio estandarizado)."""
        if self._modelo is None:
            raise RuntimeError("Llamar fit_predict() primero.")
        return self._modelo.cluster_centers_

    def get_inercia_por_cluster(self, X: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
        """
        Calcula inercia (suma de cuadrados intra-cluster) por cluster.

        Args:
            X: Matriz de features
            labels: Etiquetas asignadas

        Returns:
            DataFrame con inercia por cluster
        """
        inercias = []
        for k in range(self.n_clusters):
            mask = labels == k
            if mask.sum() > 0:
                puntos = X[mask]
                centroide = puntos.mean(axis=0)
                inercia = np.sum((puntos - centroide) ** 2)
                inercias.append({
                    'cluster': k,
                    'n_puntos': int(mask.sum()),
                    'inercia': float(inercia),
                    'inercia_promedio': float(inercia / mask.sum()) if mask.sum() > 0 else 0
                })
        
        return pd.DataFrame(inercias)

@registrar_en("modelado")
class DBSCANClustering(ClusterizadorBase):
    """
    Density-Based Spatial Clustering of Applications with Noise.

    Algoritmo:
        1. Para cada punto, encontrar vecinos dentro de eps
        2. Marcar como core point si tiene ≥ min_samples vecinos
        3. Crear cluster conectando core points mutuamente alcanzables
        4. Asignar border points a clusters de core points vecinos
        5. Puntos no asignados = ruido (-1)

    Parámetros críticos:
        eps (ε): radio de vecindad. Determina con k-distance graph.
        min_samples: mínimo de puntos para ser core point.
                    Regla: min_samples ≥ D + 1 (dimensiones + 1)

    Ventajas:
        - No requiere especificar número de clusters
        - Identifica outliers como ruido (biológicamente relevante)
        - Clusters de forma arbitraria (no solo esféricos)
        - Robusto a outliers

    Desventajas:
        - Sensibilidad a eps (crítico y dataset-dependiente)
        - Dificultad con densidades variables (resuelto por HDBSCAN)
        - Costoso para grandes N sin indexación espacial

    Usos microscopía:
        - Detección de subpoblaciones raras (outliers = células anómalas)
        - Clustering espacial de células en tejido (coordenadas x,y)
        - Filtrado de artefactos de segmentación (ruido)
    """
    nombre = "dbscan"

    def __init__(self,
                 eps: Optional[float] = None,
                 min_samples: Optional[int] = None,
                 metric: str = 'euclidean',
                 calcular_eps_auto: bool = False,
                 k_vecinos: int = 4,
                 percentil_eps: float = 90.0):
        """
        Args:
            eps: Distancia máxima para vecindad. Si None, calcula automáticamente.
            min_samples: Mínimo de puntos para core point. Si None, usa D+1.
            metric: Métrica de distancia ('euclidean', 'cosine', etc.).
            calcular_eps_auto: Si True, usa k-distance graph para estimar eps.
            k_vecinos: k para k-distance graph (si calcular_eps_auto=True).
            percentil_eps: Percentil de distancias para seleccionar eps.
        """
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self.calcular_eps_auto = calcular_eps_auto
        self.k_vecinos = k_vecinos
        self.percentil_eps = percentil_eps
        self._modelo = None
        self.eps_calculado_: Optional[float] = None

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """
        Ajusta DBSCAN y retorna etiquetas.

        Args:
            X: Matriz (N, D) estandarizada.

        Returns:
            Array (N,) con etiquetas (-1 para ruido).
        """
        self._validar_X(X)
        n, d = X.shape

        # Determinar min_samples si no se especificó
        if self.min_samples is None:
            self.min_samples = d + 1
            print(f"min_samples ajustado a D+1 = {self.min_samples}")

        # Calcular eps automáticamente si se solicitó
        if self.calcular_eps_auto or self.eps is None:
            self.eps = self._calcular_eps_k_distance(X)
            self.eps_calculado_ = self.eps
            print(f"eps calculado automáticamente: {self.eps:.4f}")

        self._modelo = DBSCAN(
            eps=self.eps,
            min_samples=self.min_samples,
            metric=self.metric
        )
        
        labels = self._modelo.fit_predict(X)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        print(f"DBSCAN encontró {n_clusters} clusters + {np.sum(labels == -1)} puntos de ruido")
        
        return labels

    def _calcular_eps_k_distance(self, X: np.ndarray) -> float:
        """
        Calcula eps usando el k-distance graph.

        Algoritmo:
            1. Para cada punto, calcular distancia a su k-ésimo vecino más cercano
            2. Ordenar distancias
            3. Encontrar "codo" (punto de máxima curvatura)
            4. Usar distancia en el codo como eps

        Args:
            X: Matriz de features

        Returns:
            Valor de eps estimado
        """
        neigh = NearestNeighbors(n_neighbors=self.k_vecinos + 1, metric=self.metric)
        neigh.fit(X)
        
        # Distancias al k-ésimo vecino (excluyendo el punto mismo)
        distances, _ = neigh.kneighbors(X)
        k_distances = distances[:, self.k_vecinos]
        
        # Ordenar y encontrar codo (simplificado: usar percentil)
        k_distances_sorted = np.sort(k_distances)
        
        # Heurística: usar percentil especificado (default 90%)
        eps_estimado = np.percentile(k_distances_sorted, self.percentil_eps)
        
        return float(eps_estimado)

    def get_core_samples(self) -> np.ndarray:
        """Retorna índices de core samples (puntos densos)."""
        if self._modelo is None:
            raise RuntimeError("Llamar fit_predict() primero.")
        return self._modelo.core_sample_indices_

@registrar_en("modelado")
class HDBSCANClustering(ClusterizadorBase):
    """
    Hierarchical DBSCAN: clustering jerárquico basado en densidad.

    Generaliza DBSCAN para encontrar clusters de densidades variables.
    Construye un árbol de clusters jerárquico y extrae los más estables.

    Algoritmo (simplificado):
        1. Calcular distancia de alcance mutuo (mutual reachability distance)
        2. Construir Minimum Spanning Tree (MST) en espacio transformado
        3. Crear jerarquía de clusters via single linkage en MST
        4. Condensar árbol según min_cluster_size
        5. Seleccionar clusters más estables (máxima persistencia)

    Parámetros clave:
        min_cluster_size: mínimo de puntos para considerar un cluster.
        min_samples: robustez del clustering (análogo a DBSCAN).
                    Mayor valor → más puntos marcados como ruido.

    Ventajas:
        - No requiere eps (parámetro difícil de DBSCAN)
        - Maneja clusters de densidades variables (crítico en biología)
        - Jerarquía explorable (dendrograma)
        - Probabilidades de membresía (soft clustering)
        - Outlier scores para cada punto

    Desventajas:
        - Más lento que DBSCAN (O(N log N) vs O(N log N) con indexación)
        - Requiere instalación adicional (hdbscan)
        - min_cluster_size puede ser difícil de estimar

    Usos microscopía:
        - Descubrimiento de subpoblaciones en cáncer heterogéneo
        - Jerarquía de diferenciación celular (stem → progenitor → diferenciada)
        - Detección de estados transitorios raros
        - Análisis de continuidad fenotípica (pseudo-time)
    """
    nombre = "hdbscan"

    def __init__(self,
                 min_cluster_size: int = 5,
                 min_samples: Optional[int] = None,
                 cluster_selection_method: Literal['eom', 'leaf'] = 'eom',
                 allow_single_cluster: bool = False,
                 metric: str = 'euclidean',
                 gen_min_span_tree: bool = True):
        """
        Args:
            min_cluster_size: Mínimo de puntos para formar un cluster.
            min_samples: Robustez (si None, usa min_cluster_size).
            cluster_selection_method: 'eom' (Excess of Mass) o 'leaf'.
            allow_single_cluster: Permitir que todo sea un solo cluster.
            metric: Métrica de distancia.
            gen_min_span_tree: Generar árbol de expansión mínima (para visualización).
        """
        if not HDBSCAN_AVAILABLE:
            raise ImportError(
                "hdbscan no está instalado. "
                "Instalar con: pip install hdbscan"
            )
        
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples or min_cluster_size
        self.cluster_selection_method = cluster_selection_method
        self.allow_single_cluster = allow_single_cluster
        self.metric = metric
        self.gen_min_span_tree = gen_min_span_tree
        self._modelo = None

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """
        Ajusta HDBSCAN y retorna etiquetas.

        Args:
            X: Matriz (N, D) estandarizada.

        Returns:
            Array (N,) con etiquetas (-1 para ruido).
        """
        self._validar_X(X)

        self._modelo = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            cluster_selection_method=self.cluster_selection_method,
            allow_single_cluster=self.allow_single_cluster,
            metric=self.metric,
            gen_min_span_tree=self.gen_min_span_tree
        )
        
        labels = self._modelo.fit_predict(X)
        
        # Información adicional
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        print(f"HDBSCAN encontró {n_clusters} clusters")
        
        if hasattr(self._modelo, 'cluster_persistence_'):
            print(f"Persistencia de clusters: {self._modelo.cluster_persistence_}")
        
        return labels

    def get_probabilidades(self) -> np.ndarray:
        """
        Probabilidades de membresía a cluster asignado.

        Returns:
            Array (N,) con probabilidades [0, 1].
        """
        if self._modelo is None:
            raise RuntimeError("Llamar fit_predict() primero.")
        return self._modelo.probabilities_

    def get_outlier_scores(self) -> np.ndarray:
        """
        Scores de outlier para cada punto (mayor = más anómalo).

        Returns:
            Array (N,) con outlier scores.
        """
        if self._modelo is None:
            raise RuntimeError("Llamar fit_predict() primero.")
        return self._modelo.outlier_scores_

    def get_arbol_condensado(self):
        """Retorna el árbol de clusters condensado (para visualización)."""
        if self._modelo is None:
            raise RuntimeError("Llamar fit_predict() primero.")
        return self._modelo.condensed_tree_

    def get_dataframe_resultado(self, 
                                 X: np.ndarray, 
                                 labels: np.ndarray,
                                 indices: Optional[List] = None) -> pd.DataFrame:
        """
        DataFrame extendido con probabilidades y outlier scores.
        """
        df = super().get_dataframe_resultado(X, labels, indices)
        
        # Añadir información específica de HDBSCAN
        df['probabilidad_cluster'] = self.get_probabilidades()
        df['outlier_score'] = self.get_outlier_scores()
        df['es_outlier'] = df['outlier_score'] > np.percentile(df['outlier_score'], 95)
        
        return df


# =============================================================================
# 4. PIPELINE DE CLUSTERING COMPLETO
# =============================================================================

def pipeline_clustering_completo(
    df: pd.DataFrame,
    columnas_features: List[str],
    metodo: Literal['kmeans', 'dbscan', 'hdbscan'] = 'kmeans',
    params_clustering: Optional[Dict] = None,
    columna_id: str = 'imagen',
    estandarizar: bool = True,
    evaluar: bool = True
) -> Dict[str, Any]:
    """
    Pipeline completo de clustering desde DataFrame.

    Args:
        df: DataFrame con métricas por imagen/objeto
        columnas_features: Lista de columnas a usar para clustering
        metodo: Algoritmo de clustering ('kmeans', 'dbscan', 'hdbscan')
        params_clustering: Parámetros específicos del algoritmo
        columna_id: Columna con identificador de muestra
        estandarizar: Si True, aplica StandardScaler antes de clustering
        evaluar: Si True, calcula métricas de calidad

    Returns:
        Dict con:
            - 'labels': etiquetas asignadas
            - 'dataframe': DataFrame con resultados
            - 'metricas': métricas de evaluación (si evaluar=True)
            - 'modelo': objeto clusterizador ajustado
            - 'X_procesada': matriz de features usada
    """
    # Extraer X
    X = df[columnas_features].values
    
    # Manejar NaN
    if np.isnan(X).any():
        warnings.warn("X contiene NaN. Imputando con media...", UserWarning)
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy='mean')
        X = imputer.fit_transform(X)
    
    # Estandarizar
    if estandarizar:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    
    # Seleccionar y configurar clusterizador
    params = params_clustering or {}
    
    if metodo == 'kmeans':
        clusterizador = KMeansClustering(**params)
    elif metodo == 'dbscan':
        clusterizador = DBSCANClustering(**params)
    elif metodo == 'hdbscan':
        clusterizador = HDBSCANClustering(**params)
    else:
        raise ValueError(f"Método no reconocido: {metodo}")
    
    # Ajustar
    labels = clusterizador.fit_predict(X)
    
    # Crear DataFrame de resultados
    indices = df[columna_id].values if columna_id in df.columns else None
    df_resultado = clusterizador.get_dataframe_resultado(X, labels, indices)
    
    # Añadir información original
    for col in df.columns:
        if col not in df_resultado.columns and col != columna_id:
            df_resultado[col] = df[col].values
    
    # Evaluar
    metricas = {}
    if evaluar:
        evaluador = EvaluadorClustering()
        metricas = evaluador.evaluar_completo(X, labels)
    
    return {
        'labels': labels,
        'dataframe': df_resultado,
        'metricas': metricas,
        'modelo': clusterizador,
        'X_procesada': X,
        'metodo': metodo
    }


def comparar_metodos_clustering(
    X: np.ndarray,
    metodos: List[Literal['kmeans', 'dbscan', 'hdbscan']] = None,
    k_min: int = 2,
    k_max: int = 8
) -> pd.DataFrame:
    """
    Compara múltiples algoritmos de clustering en los mismos datos.

    Args:
        X: Matriz de features estandarizada
        metodos: Lista de métodos a comparar
        k_min, k_max: Rango de k para K-Means

    Returns:
        DataFrame comparativo con métricas de cada método
    """
    metodos = metodos or ['kmeans', 'dbscan', 'hdbscan']
    resultados = []
    
    evaluador = EvaluadorClustering()
    
    for metodo in metodos:
        try:
            if metodo == 'kmeans':
                # Probar múltiples k
                for k in range(k_min, k_max + 1):
                    km = KMeansClustering(n_clusters=k, n_init=10)
                    labels = km.fit_predict(X)
                    metricas = evaluador.evaluar_completo(X, labels)
                    metricas['metodo'] = f'kmeans_k{k}'
                    metricas['parametros'] = f'n_clusters={k}'
                    resultados.append(metricas)
            
            elif metodo == 'dbscan':
                db = DBSCANClustering(calcular_eps_auto=True)
                labels = db.fit_predict(X)
                metricas = evaluador.evaluar_completo(X, labels)
                metricas['metodo'] = 'dbscan'
                metricas['parametros'] = f'eps={db.eps:.3f}, min_samples={db.min_samples}'
                resultados.append(metricas)
            
            elif metodo == 'hdbscan':
                if not HDBSCAN_AVAILABLE:
                    continue
                hdb = HDBSCANClustering(min_cluster_size=5)
                labels = hdb.fit_predict(X)
                metricas = evaluador.evaluar_completo(X, labels)
                metricas['metodo'] = 'hdbscan'
                metricas['parametros'] = f'min_cluster_size={hdb.min_cluster_size}'
                resultados.append(metricas)
        
        except Exception as e:
            warnings.warn(f"Error en {metodo}: {e}", UserWarning)
    
    return pd.DataFrame(resultados)