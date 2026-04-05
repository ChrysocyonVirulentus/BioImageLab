"""
Reducción de dimensionalidad, clustering y visualización para datos
multiparamétricos de microscopía.

En esta etapa del pipeline, cada imagen fue cuantificada y sus métricas
se almacenaron en un DataFrame con la convención:

    '{nombre_metrica}_{nombre_estadistico}'
    Ejemplos: 'intensidad_media', 'area_iqr', 'excentricidad_std'

    Estructura del DataFrame de entrada (ejemplo):
    ┌──────────┬────────────────┬──────────────┬─────────────┬──────────────┐
    │ imagen   │ grupo_exp      │ intens_media │ intens_std  │ area_mediana │
    ├──────────┼────────────────┼──────────────┼─────────────┼──────────────┤
    │ img_001  │ control        │ 1243.2       │ 87.4        │ 312.5        │
    │ img_002  │ tratamiento_A  │ 2187.6       │ 154.3       │ 289.1        │
    └──────────┴────────────────┴──────────────┴─────────────┴──────────────┘

Principio de coherencia estadística:
    No es válido combinar libremente estimadores de tendencia central con
    estimadores de dispersión de distinta familia. Las combinaciones válidas:

        Media       ←→  std, var, se, cv
        Mediana     ←→  iqr, mad
        Percentiles ←→  entre sí (p5↔p95, p10↔p90, etc.)
        Forma       :   skewness, kurtosis (autónomos)

    El SelectorCoherente aplica estas reglas al construir la matriz X.

Módulos del archivo:
    1. Coherencia estadística       verificar_coherencia(), SelectorCoherente
    2. Preprocesamiento (sklearn)   PreprocesamientoFeatures
    3. Reductores                   PCA, UMAP, tSNE
    4. Pipeline de alto nivel       reducir_desde_dataframe()
    5. Visualización                VisualizadorDimensionalidad
            - grafico_varianza_pca()
            - grafico_biplot_pca()
            - grafico_dendrograma()
            - grafico_embedding_2d()
            - grafico_scatter_features()
            - grafico_raincloud()

IMPORTANTE — Separación de responsabilidades:
    Estos métodos NO realizan cuantificación ni normalización de imágenes.
    Reciben DataFrames ya construidos por la etapa de cuantificación.
    La estandarización de features se realiza internamente via sklearn.
"""

import warnings
from typing import Dict, List, Literal, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from ...gestorLab.Registro_Metodos import registrar_en

# ─────────────────────────────────────────────────────────────
# 1. Coherencia estadística
# ─────────────────────────────────────────────────────────────

_TODOS_ESTADISTICOS = [
    'media', 'mediana', 'std', 'var', 'mad', 'iqr',
    'min', 'max', 'rango', 'cv', 'skewness', 'kurtosis',
    'p5', 'p10', 'p25', 'p50', 'p75', 'p90', 'p95', 'n_muestras', 'se',
]

_DISPERSION_PARAMETRICA = {'std', 'var', 'se'}
_DISPERSION_ROBUSTA     = {'iqr', 'mad'}
_FORMA                  = {'skewness', 'kurtosis'}
_MAGNITUD               = {'min', 'max', 'rango', 'cv', 'n_muestras'}
_PERCENTILES            = {'p5', 'p10', 'p25', 'p50', 'p75', 'p90', 'p95'}


def verificar_coherencia(estadisticos: List[str]) -> Tuple[bool, str]:
    """
    Verifica que la lista de estadísticos sea internamente coherente.

    Reglas:
        R1. 'media'   sin {'std','var','se'} → incoherente.
        R2. 'mediana' sin {'iqr','mad'}      → incoherente.
        R3. Dispersión paramétrica sola, sin tendencia central → incoherente.
        R4. Dispersión robusta sola, sin tendencia central     → incoherente.

    Args:
        estadisticos: Lista de nombres de estadísticos seleccionados.

    Returns:
        Tupla (es_coherente: bool, mensaje: str).
    """
    est = set(estadisticos)
    tiene_media   = 'media'   in est
    tiene_mediana = 'mediana' in est
    tiene_param   = bool(est & _DISPERSION_PARAMETRICA)
    tiene_robusta = bool(est & _DISPERSION_ROBUSTA)

    if tiene_media and not tiene_param:
        return False, (
            f"'media' sin dispersión paramétrica ({_DISPERSION_PARAMETRICA}). "
            "Agregar 'std', 'var' o 'se'."
        )
    if tiene_mediana and not tiene_robusta:
        return False, (
            f"'mediana' sin dispersión robusta ({_DISPERSION_ROBUSTA}). "
            "Agregar 'iqr' o 'mad'."
        )
    if tiene_param and not tiene_media and not tiene_mediana:
        return False, (
            f"Dispersión paramétrica {est & _DISPERSION_PARAMETRICA} "
            "sin tendencia central. Agregar 'media'."
        )
    if tiene_robusta and not tiene_mediana and not tiene_media:
        return False, (
            f"Dispersión robusta {est & _DISPERSION_ROBUSTA} "
            "sin tendencia central. Agregar 'mediana'."
        )
    return True, "Combinación coherente."


class SelectorCoherente:
    """
    Construye la matriz de features X desde el DataFrame de métricas,
    aplicando las reglas de coherencia estadística.

    Modos de uso:

        Modo A — metricas + estadisticos:
            Genera columnas '{metrica}_{estadistico}'.
            Ejemplo: metricas=['intensidad'], estadisticos=['mediana','iqr']
            → columnas: 'intensidad_mediana', 'intensidad_iqr'

        Modo B — columnas_directas:
            Lista de nombres exactos de columnas del DataFrame.
            Ejemplo: columnas_directas=['mediana', 'cv_mediana_v2', 'mad']
            Útil cuando las columnas ya tienen nombres arbitrarios.
    """

    def __init__(
        self,
        metricas: Optional[List[str]] = None,
        estadisticos: Optional[List[str]] = None,
        columnas_directas: Optional[List[str]] = None,
        forzar_coherencia: bool = False,
    ):
        """
        Args:
            metricas: Nombres de las métricas cuantificadas.
            estadisticos: Estadísticos para cada métrica (deben ser coherentes).
            columnas_directas: Alternativa a metricas+estadisticos.
                              Lista de columnas exactas del DataFrame.
            forzar_coherencia: True → ValueError si incoherente.
                              False → UserWarning (por defecto).
        """
        if columnas_directas is None and (metricas is None or estadisticos is None):
            raise ValueError(
                "Especificar (metricas + estadisticos) o columnas_directas."
            )

        self.metricas          = metricas
        self.estadisticos      = estadisticos
        self.columnas_directas = columnas_directas
        self.forzar_coherencia = forzar_coherencia

        if estadisticos is not None:
            invalidos = set(estadisticos) - set(_TODOS_ESTADISTICOS)
            if invalidos:
                raise ValueError(
                    f"Estadísticos no reconocidos: {invalidos}. "
                    f"Disponibles: {_TODOS_ESTADISTICOS}"
                )
            es_coherente, mensaje = verificar_coherencia(estadisticos)
            if not es_coherente:
                if forzar_coherencia:
                    raise ValueError(f"Coherencia violada: {mensaje}")
                warnings.warn(
                    f"Advertencia de coherencia estadística: {mensaje}",
                    UserWarning, stacklevel=2,
                )

    def construir_matriz_features(
        self,
        df: pd.DataFrame,
        columna_grupo: Optional[str] = 'grupo_experimental',
    ) -> Tuple[np.ndarray, List[str], Optional[pd.Series]]:
        """
        Extrae la matriz X y las etiquetas de grupo del DataFrame.

        Args:
            df: DataFrame con una fila por imagen.
            columna_grupo: Columna con etiquetas de grupo. None → sin grupos.

        Returns:
            Tupla (X, nombres_columnas, grupos):
                X               : Array (N, D) float64
                nombres_columnas: Lista de columnas del DataFrame usadas
                grupos          : Serie de etiquetas o None
        """
        if self.columnas_directas is not None:
            columnas_solicitadas = self.columnas_directas
        else:
            columnas_solicitadas = [
                f"{m}_{e}" for m in self.metricas for e in self.estadisticos
            ]

        disponibles = [c for c in columnas_solicitadas if c     in df.columns]
        faltantes   = [c for c in columnas_solicitadas if c not in df.columns]

        if not disponibles:
            raise ValueError(
                f"Ninguna columna encontrada. "
                f"Pedidas: {columnas_solicitadas[:8]}. "
                f"Disponibles: {list(df.columns[:8])}..."
            )
        if faltantes:
            warnings.warn(
                f"Columnas no encontradas (ignoradas): {faltantes}",
                UserWarning, stacklevel=2,
            )

        X = df[disponibles].to_numpy(dtype=np.float64)

        n_nan = np.isnan(X).sum()
        if n_nan > 0:
            warnings.warn(
                f"Matriz de features contiene {n_nan} NaN "
                f"({100*n_nan/X.size:.1f}%). Imputar antes de reducir.",
                UserWarning, stacklevel=2,
            )

        grupos = (
            df[columna_grupo]
            if (columna_grupo and columna_grupo in df.columns)
            else None
        )
        return X, disponibles, grupos


# ─────────────────────────────────────────────────────────────
# 2. Preprocesamiento con sklearn
# ─────────────────────────────────────────────────────────────

class PreprocesamientoFeatures:
    """
    Estandarización de features usando scalers de sklearn.

    Elección recomendada según familia estadística:

        Familia media   → 'zscore'  (StandardScaler: x' = (x-μ)/σ)
        Familia mediana → 'robusto' (RobustScaler: x' = (x-mediana)/IQR)
        Mixta o dudosa  → 'robusto' (más conservador)

    El scaler ajustado puede aplicarse a nuevas muestras con transform(),
    y convertirse a DataFrame para inspección con to_dataframe().
    """

    def __init__(
        self,
        metodo: Literal['zscore', 'robusto', 'ninguno'] = 'zscore',
        rango_robusto: Tuple[float, float] = (25.0, 75.0),
    ):
        """
        Args:
            metodo: Método de estandarización.
                   'zscore'  : StandardScaler.
                   'robusto' : RobustScaler.
                   'ninguno' : sin transformación.

            rango_robusto: Percentiles para RobustScaler.
                          (25, 75) = IQR estándar (por defecto).
                          (10, 90) = mayor robustez ante outliers extremos.
        """
        if metodo not in ('zscore', 'robusto', 'ninguno'):
            raise ValueError("metodo debe ser 'zscore', 'robusto' o 'ninguno'")
        self.metodo        = metodo
        self.rango_robusto = rango_robusto
        self._scaler       = None
        self._ajustado     = False

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Ajusta el scaler sobre X y devuelve X estandarizada.

        Args:
            X: Matriz (N, D)

        Returns:
            X estandarizada (N, D)
        """
        if self.metodo == 'ninguno':
            self._ajustado = True
            return X.copy()

        if self.metodo == 'zscore':
            from sklearn.preprocessing import StandardScaler
            self._scaler = StandardScaler()
        else:  # robusto
            from sklearn.preprocessing import RobustScaler
            self._scaler = RobustScaler(quantile_range=self.rango_robusto)

        X_scaled = self._scaler.fit_transform(X)
        self._ajustado = True
        return X_scaled

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Aplica el scaler ya ajustado a nuevas muestras."""
        if not self._ajustado:
            raise RuntimeError("Llamar fit_transform() antes de transform().")
        if self.metodo == 'ninguno':
            return X.copy()
        return self._scaler.transform(X)

    def to_dataframe(
        self,
        X_scaled: np.ndarray,
        nombres_columnas: List[str],
    ) -> pd.DataFrame:
        """
        Convierte la matriz estandarizada a DataFrame con sufijo '_scaled'.

        Args:
            X_scaled:         Array (N, D) estandarizado
            nombres_columnas: Nombres de las D columnas originales

        Returns:
            DataFrame con columnas '{nombre}_scaled'
        """
        return pd.DataFrame(
            X_scaled,
            columns=[f"{c}_scaled" for c in nombres_columnas],
        )

    @staticmethod
    def recomendar_metodo(estadisticos: List[str]) -> str:
        """
        Recomienda el método de estandarización según los estadísticos elegidos.

        Lógica:
            Familia robusta presente  → 'robusto'
            Solo familia paramétrica  → 'zscore'
            Mezcla de ambas familias  → 'robusto'  (conservador)
            Sin estadísticos claros   → 'zscore'

        Args:
            estadisticos: Lista de estadísticos del SelectorCoherente.

        Returns:
            'zscore' o 'robusto'
        """
        est = set(estadisticos)
        usa_robusta     = bool(est & _DISPERSION_ROBUSTA)
        usa_parametrica = bool(est & _DISPERSION_PARAMETRICA)

        if usa_robusta:
            return 'robusto'
        if usa_parametrica:
            return 'zscore'
        return 'zscore'


# ─────────────────────────────────────────────────────────────
# 3. Reductores de dimensionalidad
# ─────────────────────────────────────────────────────────────

class ReduccionDimensionalidad:
    """
    Clase base para reductores de dimensionalidad.

    La estandarización se delega a PreprocesamientoFeatures y debe
    aplicarse antes de llamar a fit_transform(). Los reductores reciben
    X ya estandarizada.
    """
    nombre = "reduccion_base"

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def transform(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError(
            f"{self.nombre} no soporta transform() sobre nuevas muestras."
        )

    def _validar_X(self, X: np.ndarray) -> None:
        if not isinstance(X, np.ndarray) or X.ndim != 2:
            raise ValueError("X debe ser np.ndarray 2D (N, D).")
        if X.shape[0] < 2:
            raise ValueError("X debe tener al menos 2 muestras.")
        if np.isnan(X).any():
            raise ValueError("X contiene NaN. Imputar antes de reducir.")

@registrar_en("modelado")
class PCADimensional(ReduccionDimensionalidad):
    """
    Análisis de Componentes Principales via sklearn.decomposition.PCA.

    Encuentra la proyección lineal que maximiza la varianza explicada.
    Cada componente principal (CP) es una combinación lineal ortogonal
    de las features originales.

    Algoritmo (SVD):
        X = U · Σ · Vᵀ
        Loadings (V): dirección en el espacio original
        Scores (Z = X·V): coordenadas en el espacio reducido
        Varianza k: λₖ = σₖ² / (N-1)

    Ventajas:
        - Completamente determinista y reproducible
        - transform() estable para nuevas muestras
        - Varianza explicada e interpretabilidad de loadings
        - Elimina redundancia por correlación entre features

    Desventajas:
        - Solo captura estructura lineal
        - Sensible a outliers (minimiza varianza)
        - No preserva estructura local ni clusters no lineales

    Usos típicos en microscopía:
        - Análisis exploratorio inicial de datos multiparamétricos
        - Detección de batch effects entre días/experimentos
        - Preprocesamiento para UMAP/t-SNE (PCA→50D→UMAP→2D)
        - Identificación de features redundantes (alta correlación)
    """
    nombre = "pca"

    def __init__(
        self,
        n_componentes: int = 2,
        n_componentes_varianza: Optional[float] = None,
        svd_solver: str = 'auto',
        semilla: Optional[int] = 42,
    ):
        """
        Args:
            n_componentes: Número de CPs a retener.

            n_componentes_varianza: Si se especifica (0 < valor ≤ 1),
                                selecciona el mínimo K que explica ese
                                porcentaje de varianza acumulada.
                                Tiene prioridad sobre n_componentes.
                                Ejemplo: 0.95 → CPs que explican 95%.

            svd_solver: 'auto', 'full', 'randomized', 'arpack'.

            semilla: Semilla para reproducibilidad.
        """
        if n_componentes_varianza is not None and not (0 < n_componentes_varianza <= 1):
            raise ValueError("n_componentes_varianza debe estar en (0, 1]")

        self.n_componentes          = n_componentes
        self.n_componentes_varianza = n_componentes_varianza
        self.svd_solver             = svd_solver
        self.semilla                = semilla
        self._modelo                = None

        # Atributos expuestos tras fit_transform
        self.componentes_:              Optional[np.ndarray] = None
        self.varianza_explicada_ratio_: Optional[np.ndarray] = None
        self.varianza_acumulada_:       Optional[np.ndarray] = None
        self.n_componentes_:            Optional[int]        = None

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Ajusta PCA y proyecta X al espacio de componentes principales.

        Args:
            X: Matriz (N, D) ya estandarizada.

        Returns:
            Array (N, K) con las coordenadas en el espacio PCA.
            Inspeccionar .varianza_explicada_ratio_ y .componentes_ para
            varianza explicada y loadings respectivamente.
        """
        from sklearn.decomposition import PCA as _PCA

        self._validar_X(X)

        n_comp = (
            self.n_componentes_varianza
            if self.n_componentes_varianza is not None
            else min(self.n_componentes, min(X.shape) - 1)
        )

        self._modelo = _PCA(
            n_components=n_comp,
            svd_solver=self.svd_solver,
            random_state=self.semilla,
        )
        Z = self._modelo.fit_transform(X)

        self.componentes_              = self._modelo.components_
        self.varianza_explicada_ratio_ = self._modelo.explained_variance_ratio_
        self.varianza_acumulada_       = np.cumsum(self.varianza_explicada_ratio_)
        self.n_componentes_            = Z.shape[1]
        return Z

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Proyecta nuevas muestras con el modelo ya ajustado."""
        if self._modelo is None:
            raise RuntimeError("Llamar fit_transform() antes de transform().")
        return self._modelo.transform(X)

    def tabla_varianza(self) -> pd.DataFrame:
        """
        DataFrame con varianza explicada por componente.

        Returns:
            DataFrame: columnas 'componente', 'varianza_%', 'acumulada_%'
        """
        if self.varianza_explicada_ratio_ is None:
            raise RuntimeError("Llamar fit_transform() primero.")
        return pd.DataFrame({
            'componente':  [f"CP{i+1}" for i in range(self.n_componentes_)],
            'varianza_%':  self.varianza_explicada_ratio_ * 100,
            'acumulada_%': self.varianza_acumulada_ * 100,
        })

    def tabla_loadings(self, nombres_features: List[str]) -> pd.DataFrame:
        """
        DataFrame con loadings de cada feature para cada CP.

        Args:
            nombres_features: Nombres de las D features originales.

        Returns:
            DataFrame (D × K). Índice = features, columnas = 'CP1', 'CP2', ...
        """
        if self.componentes_ is None:
            raise RuntimeError("Llamar fit_transform() primero.")
        cols = [f"CP{i+1}" for i in range(self.n_componentes_)]
        return pd.DataFrame(
            self.componentes_.T,
            index=nombres_features,
            columns=cols,
        )

@registrar_en("modelado")
class UMAPDimensional(ReduccionDimensionalidad):
    """
    Uniform Manifold Approximation and Projection (McInnes et al. 2018).

    Construye un grafo fuzzy de vecinos más cercanos en alta dimensión
    y optimiza un embedding de baja dimensión que preserve esa topología.

    Parámetros clave:
        n_vecinos: balance global/local.
                Bajo (5–15)  → estructura local fina, clusters pequeños.
                Alto (30–100) → estructura global, gradientes continuos.
        min_dist:  compacidad de clusters.
                Bajo (0.0–0.1)  → clusters densos (clasificación).
                Alto (0.5–1.0)  → distribución continua (gradientes).

    Ventajas:
        - Preserva estructura global mejor que t-SNE
        - Más rápido: O(N log N) vs O(N²) de t-SNE
        - transform() disponible para nuevas muestras (aproximado)

    Desventajas:
        - Resultados dependen de n_vecinos y min_dist
        - No determinista sin semilla fija
        - Requiere: pip install umap-learn

    Usos típicos en microscopía:
        - Visualización de poblaciones celulares heterogéneas
        - Descubrimiento de subpoblaciones (combinado con HDBSCAN)
        - Atlas de fenotipos celulares (HCS)
    """
    nombre = "umap"

    def __init__(
        self,
        n_componentes: int = 2,
        n_vecinos: int = 15,
        min_dist: float = 0.1,
        metrica: str = 'euclidean',
        n_epocas: Optional[int] = None,
        semilla: Optional[int] = 42,
        preprocesar_pca: Optional[int] = None,
    ):
        """
        Args:
            n_componentes: Dimensiones del embedding. Típico: 2 o 3.
            n_vecinos: Vecinos para construir el grafo (5–50 según N).
            min_dist: Distancia mínima en el embedding [0.0, 1.0].
            metrica: 'euclidean', 'cosine', 'manhattan', entre otras.
            n_epocas: Iteraciones de optimización. None → automático.
            semilla: Semilla para reproducibilidad.
            preprocesar_pca: Si D > este valor, aplicar PCA previo.
                            Recomendado cuando D > 50.
        """
        if n_vecinos < 2:
            raise ValueError("n_vecinos debe ser >= 2")
        if not (0.0 <= min_dist <= 1.0):
            raise ValueError("min_dist debe estar en [0.0, 1.0]")

        self.n_componentes   = n_componentes
        self.n_vecinos       = n_vecinos
        self.min_dist        = min_dist
        self.metrica         = metrica
        self.n_epocas        = n_epocas
        self.semilla         = semilla
        self.preprocesar_pca = preprocesar_pca
        self._modelo         = None
        self._pca_prep: Optional[PCA] = None

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Ajusta UMAP y genera el embedding de baja dimensión.

        Args:
            X: Matriz (N, D) ya estandarizada.

        Returns:
            Array (N, n_componentes).
        """
        try:
            import umap as umap_lib
        except ImportError:
            raise ImportError("Instalar umap-learn: pip install umap-learn")

        self._validar_X(X)

        n_vec = min(self.n_vecinos, X.shape[0] - 1)
        if n_vec < self.n_vecinos:
            warnings.warn(
                f"n_vecinos ajustado de {self.n_vecinos} a {n_vec} (N={X.shape[0]}).",
                UserWarning, stacklevel=2,
            )

        X_entrada = X
        if self.preprocesar_pca and X.shape[1] > self.preprocesar_pca:
            self._pca_prep = PCA(n_componentes=self.preprocesar_pca)
            X_entrada = self._pca_prep.fit_transform(X)

        self._modelo = umap_lib.UMAP(
            n_components=self.n_componentes,
            n_neighbors=n_vec,
            min_dist=self.min_dist,
            metric=self.metrica,
            n_epochs=self.n_epocas,
            random_state=self.semilla,
        )
        return self._modelo.fit_transform(X_entrada)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Proyecta nuevas muestras (resultado aproximado)."""
        if self._modelo is None:
            raise RuntimeError("Llamar fit_transform() antes.")
        X_e = self._pca_prep.transform(X) if self._pca_prep else X
        return self._modelo.transform(X_e)

@registrar_en("modelado")
class tSNE(ReduccionDimensionalidad):
    """
    t-distributed Stochastic Neighbor Embedding (van der Maaten & Hinton 2008).

    Minimiza la divergencia KL entre similitudes modeladas con gaussiana
    en alta dimensión y con distribución t de Student en baja dimensión.
    La cola pesada de la distribución t resuelve el crowding problem.

    Parámetros críticos:
        perplexidad: número efectivo de vecinos (5 ≤ ⊥ ≤ min(50, N/4)).
        n_iter:      iteraciones de optimización (mínimo recomendado: 1000).

    Ventajas:
        - Excelente para revelar clusters locales y subpoblaciones
        - Muy visual e intuitivo para exploración inicial

    Desventajas:
        - No preserva distancias globales entre clusters
        - Lento para N > 10000 (O(N log N) con Barnes-Hut)
        - No soporta transform() para nuevas muestras
        - Sensible a perplexidad y learning_rate

    Usos típicos en microscopía:
        - Visualización exploratoria de poblaciones celulares
        - Validación visual de clustering (colorear por cluster conocido)
        - Comparación cualitativa de condiciones en espacio 2D
    """
    nombre = "tsne"

    def __init__(
        self,
        n_componentes: int = 2,
        perplexidad: float = 30.0,
        learning_rate: Union[float, str] = 'auto',
        n_iter: int = 1000,
        n_iter_sin_progreso: int = 300,
        metodo: str = 'barnes_hut',
        semilla: Optional[int] = 42,
        preprocesar_pca: Optional[int] = 50,
    ):
        """
        Args:
            n_componentes: Dimensiones de salida. Casi siempre 2.
            perplexidad: Número efectivo de vecinos (5–50).
            learning_rate: Tasa de aprendizaje. 'auto' → max(N/200, 50).
            n_iter: Iteraciones. Recomendado: 1000–5000.
            n_iter_sin_progreso: Iteraciones sin mejora antes de parar.
            metodo: 'barnes_hut' (rápido, n_comp≤3) o 'exact' (O(N²)).
            semilla: Semilla para reproducibilidad.
            preprocesar_pca: CPs de PCA previos al t-SNE.
                            Convención estándar: 50. None → sin PCA previo.
        """
        if n_componentes > 3 and metodo == 'barnes_hut':
            raise ValueError("barnes_hut solo soporta n_componentes ≤ 3.")
        if perplexidad < 1:
            raise ValueError("perplexidad debe ser >= 1")

        self.n_componentes       = n_componentes
        self.perplexidad         = perplexidad
        self.learning_rate       = learning_rate
        self.n_iter              = n_iter
        self.n_iter_sin_progreso = n_iter_sin_progreso
        self.metodo              = metodo
        self.semilla             = semilla
        self.preprocesar_pca     = preprocesar_pca
        self._pca_prep: Optional[PCA] = None
        self.kl_divergencia_: Optional[float] = None

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Ajusta t-SNE y genera el embedding de baja dimensión.

        Args:
            X: Matriz (N, D) ya estandarizada.

        Returns:
            Array (N, n_componentes).
            Inspeccionar .kl_divergencia_ para calidad del ajuste.
            Valores > 5 sugieren aumentar n_iter o ajustar perplexidad.
        """
        from sklearn.manifold import TSNE as _TSNE

        self._validar_X(X)

        perp = min(self.perplexidad, (X.shape[0] - 1) / 3.0 - 1)
        if perp < self.perplexidad:
            warnings.warn(
                f"perplexidad ajustada a {perp:.0f} (N={X.shape[0]}).",
                UserWarning, stacklevel=2,
            )

        X_entrada = X
        if self.preprocesar_pca:
            n_pca = min(self.preprocesar_pca, X.shape[1], X.shape[0] - 1)
            self._pca_prep = PCA(n_componentes=n_pca)
            X_entrada = self._pca_prep.fit_transform(X)

        modelo = _TSNE(
            n_components=self.n_componentes,
            perplexity=perp,
            learning_rate=self.learning_rate,
            n_iter=self.n_iter,
            n_iter_without_progress=self.n_iter_sin_progreso,
            method=self.metodo,
            random_state=self.semilla,
            init='pca',
        )
        Z = modelo.fit_transform(X_entrada)
        self.kl_divergencia_ = float(modelo.kl_divergence_)

        if self.kl_divergencia_ > 5.0:
            warnings.warn(
                f"KL divergencia alta ({self.kl_divergencia_:.2f}). "
                "Aumentar n_iter o ajustar perplexidad.",
                UserWarning, stacklevel=2,
            )
        return Z

    def transform(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError(
            "t-SNE no soporta transform(). Incluir todas las muestras "
            "en fit_transform(), o usar UMAP para proyección de nuevas muestras."
        )


# ─────────────────────────────────────────────────────────────
# 4. Pipeline de alto nivel
# ─────────────────────────────────────────────────────────────

def reducir_desde_dataframe(
    df: pd.DataFrame,
    metricas: Optional[List[str]] = None,
    estadisticos: Optional[List[str]] = None,
    columnas_directas: Optional[List[str]] = None,
    metodo: Literal['pca', 'umap', 'tsne'] = 'umap',
    estandarizar: Optional[Literal['zscore', 'robusto', 'ninguno', 'auto']] = 'auto',
    columna_grupo: str = 'grupo_experimental',
    kwargs_selector: Optional[Dict] = None,
    kwargs_preprocesamiento: Optional[Dict] = None,
    kwargs_reductor: Optional[Dict] = None,
) -> Tuple[np.ndarray, List[str], Optional[pd.Series],
           ReduccionDimensionalidad, PreprocesamientoFeatures]:
    """
    Pipeline completo: selección coherente → estandarización → reducción.

    Ejemplo de uso equivalente al código original del usuario:

        # ── Equivalente a StandardScaler + PCA de sklearn ──
        Z, nombres, grupos, modelo, scaler = reducir_desde_dataframe(
            df=df_datos_combinado_2_10,
            columnas_directas=['mediana', 'cv_mediana_v2', 'mad'],
            metodo='pca',
            estandarizar='robusto',   # equivalente a RobustScaler()
            columna_grupo='Genotipo',
            kwargs_reductor={'n_componentes': 2},
        )
        df_pca = pd.DataFrame(Z, columns=['PC1', 'PC2'])
        df_pca['Cluster_H'] = df_datos_combinado_2_10['Cluster_H'].values
        df_pca['Genotipo']  = grupos.values

    Args:
        df: DataFrame con métricas cuantificadas (una fila por imagen).
        metricas: Nombres de métricas (modo A del SelectorCoherente).
        estadisticos: Estadísticos para cada métrica (modo A).
        columnas_directas: Columnas exactas del DataFrame (modo B).
        metodo: 'pca', 'umap' o 'tsne'.
        estandarizar: 'auto' → elige según estadísticos (zscore o robusto).
                    'zscore', 'robusto', 'ninguno' → fuerza el método.
        columna_grupo: Columna con etiquetas de grupo.
        kwargs_selector: Argumentos extra para SelectorCoherente.
        kwargs_preprocesamiento: Argumentos extra para PreprocesamientoFeatures.
        kwargs_reductor: Argumentos extra para el reductor.
                        PCA:  {'n_componentes': 3}
                        UMAP: {'n_vecinos': 30, 'min_dist': 0.05}
                        tSNE: {'perplexidad': 20, 'n_iter': 2000}

    Returns:
        Tupla (Z, nombres_features, grupos, reductor, scaler):
            Z              : Array (N, n_componentes)
            nombres_features: Columnas del DataFrame usadas como features
            grupos         : Serie con etiquetas de grupo (o None)
            reductor       : Instancia ajustada del reductor
            scaler         : Instancia ajustada del preprocesamiento
    """
    kwargs_selector         = kwargs_selector or {}
    kwargs_preprocesamiento = kwargs_preprocesamiento or {}
    kwargs_reductor         = kwargs_reductor or {}

    selector = SelectorCoherente(
        metricas=metricas,
        estadisticos=estadisticos,
        columnas_directas=columnas_directas,
        **kwargs_selector,
    )
    X, nombres_features, grupos = selector.construir_matriz_features(
        df, columna_grupo=columna_grupo
    )

    if estandarizar == 'auto':
        metodo_std = PreprocesamientoFeatures.recomendar_metodo(
            estadisticos or []
        )
    else:
        metodo_std = estandarizar or 'zscore'

    prep     = PreprocesamientoFeatures(metodo=metodo_std, **kwargs_preprocesamiento)
    X_scaled = prep.fit_transform(X)

    _reductores = {'pca': PCA, 'umap': UMAP, 'tsne': tSNE}
    if metodo not in _reductores:
        raise ValueError(
            f"Método '{metodo}' no reconocido. Opciones: {list(_reductores)}"
        )

    reductor = _reductores[metodo](**kwargs_reductor)
    Z        = reductor.fit_transform(X_scaled)

    return Z, nombres_features, grupos, reductor, prep


# ─────────────────────────────────────────────────────────────
# 5. Visualización
# ─────────────────────────────────────────────────────────────

class VisualizadorDimensionalidad:
    """
    Suite completa de gráficos para exploración de datos de alta dimensión.

    Todos los métodos:
        - Son métodos estáticos: no requieren instancia.
        - Devuelven (fig, ax) para permitir retoque posterior.
        - Tienen parámetros explícitos para todos los aspectos visuales.
        - Son independientes entre sí.

    Gráficos disponibles:
        grafico_varianza_pca()    Scree plot + varianza acumulada
        grafico_biplot_pca()      Scores + loadings en plano CP1–CP2
        grafico_dendrograma()     Dendrograma de clustering jerárquico
        grafico_embedding_2d()    Scatter del embedding (PCA/UMAP/tSNE)
        grafico_scatter_features()Scatter de dos features con color de cluster
        grafico_raincloud()       Violin + Boxplot + Strip con anotaciones N
    """

    @staticmethod
    def _estilo_base(
        ax: Axes,
        titulo: str,
        xlabel: str,
        ylabel: str,
        grid: bool = True,
        grid_axis: str = 'both',
        fuente_titulo: int = 14,
        fuente_ejes: int = 12,
        estilo_fuente: str = 'serif',
    ) -> None:
        """Aplica formato estándar a un Axes."""
        ax.set_title(titulo, fontsize=fuente_titulo, fontweight='bold',
                    fontfamily=estilo_fuente)
        ax.set_xlabel(xlabel, fontsize=fuente_ejes, fontfamily=estilo_fuente)
        ax.set_ylabel(ylabel, fontsize=fuente_ejes, fontfamily=estilo_fuente)
        if grid:
            ax.grid(True, linestyle='--', alpha=0.5, axis=grid_axis)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # ── 5.1 Scree plot ────────────────────────────────────────

    @staticmethod
    def grafico_varianza_pca(
        modelo_pca: 'PCA',
        titulo: str = 'Varianza Explicada — PCA',
        color_barras: str = 'steelblue',
        color_acumulada: str = 'tomato',
        umbral_varianza: Optional[float] = 0.90,
        figsize: Tuple[int, int] = (9, 5),
        fuente_titulo: int = 14,
        fuente_ejes: int = 12,
        mostrar: bool = True,
    ) -> Tuple[Figure, Axes]:
        """
        Scree plot: varianza individual por componente + curva acumulada.

        La intersección de la curva con el umbral indica el número mínimo
        de CPs necesarios para explicar ese porcentaje de la varianza total.

        Args:
            modelo_pca: Instancia de PCA ajustada con fit_transform().
            titulo: Título del gráfico.
            color_barras: Color de las barras de varianza individual.
            color_acumulada: Color de la curva de varianza acumulada.
            umbral_varianza: Línea horizontal al X% de varianza acumulada.
                            None → sin línea. Ejemplo: 0.90 → 90%.
            figsize: Tamaño de figura.
            fuente_titulo: Tamaño de fuente del título.
            fuente_ejes: Tamaño de fuente de los ejes.
            mostrar: Si True, llama plt.show().

        Returns:
            Tupla (fig, ax)
        """
        if modelo_pca.varianza_explicada_ratio_ is None:
            raise RuntimeError("El modelo PCA debe estar ajustado (fit_transform).")

        ratios    = modelo_pca.varianza_explicada_ratio_ * 100
        acumulada = modelo_pca.varianza_acumulada_ * 100
        n_comp    = len(ratios)
        x         = np.arange(1, n_comp + 1)

        fig, ax = plt.subplots(figsize=figsize)
        ax.bar(x, ratios, color=color_barras, alpha=0.8, label='Varianza individual')
        ax.plot(x, acumulada, 'o-', color=color_acumulada, linewidth=2,
                label='Varianza acumulada')

        if umbral_varianza is not None:
            ax.axhline(umbral_varianza * 100, linestyle='--', color='gray',
                    alpha=0.8, label=f'{umbral_varianza*100:.0f}% umbral')

        for xi, r in zip(x, ratios):
            ax.text(xi, r + 0.5, f'{r:.1f}%', ha='center', va='bottom',
                    fontsize=9, color='black')

        ax.set_xticks(x)
        ax.set_xticklabels([f'CP{i}' for i in x])
        ax.set_ylim(0, 110)
        ax.legend(frameon=True)

        VisualizadorDimensionalidad._estilo_base(
            ax, titulo, 'Componente Principal', 'Varianza Explicada (%)',
            fuente_titulo=fuente_titulo, fuente_ejes=fuente_ejes,
        )
        plt.tight_layout()
        if mostrar:
            plt.show()
        return fig, ax

    # ── 5.2 Biplot PCA ────────────────────────────────────────

    @staticmethod
    def grafico_biplot_pca(
        Z: np.ndarray,
        modelo_pca: 'PCA',
        nombres_features: List[str],
        grupos: Optional[pd.Series] = None,
        cp_x: int = 1,
        cp_y: int = 2,
        escala_flechas: float = 1.0,
        n_flechas_max: int = 10,
        paleta: str = 'tab10',
        tamaño_punto: int = 60,
        alpha_puntos: float = 0.8,
        titulo: str = 'Biplot PCA',
        figsize: Tuple[int, int] = (10, 8),
        mostrar: bool = True,
    ) -> Tuple[Figure, Axes]:
        """
        Biplot: scores (imágenes como puntos) + loadings (features como flechas).

        Permite identificar:
            - Qué imágenes/grupos son similares (puntos cercanos).
            - Qué features contribuyen más a cada CP (flechas largas).
            - Qué features están correlacionadas (flechas paralelas).
            - Qué features están anticorrelacionadas (flechas opuestas).

        Args:
            Z: Array (N, K) de scores PCA (salida de PCA.fit_transform()).
            modelo_pca: Instancia de PCA ajustada.
            nombres_features: Nombres de las D features originales.
            grupos: Serie con etiquetas de grupo para colorear puntos.
            cp_x: Número del componente para eje X (base 1).
            cp_y: Número del componente para eje Y (base 1).
            escala_flechas: Factor de escala para longitud de flechas.
            n_flechas_max: Máximo de flechas (selecciona las de mayor magnitud).
            paleta: Paleta de colores para grupos.
            tamaño_punto: Tamaño de los puntos de score.
            alpha_puntos: Transparencia de los puntos.
            titulo: Título.
            figsize: Tamaño de figura.
            mostrar: Si True, llama plt.show().

        Returns:
            Tupla (fig, ax)
        """
        if modelo_pca.componentes_ is None:
            raise RuntimeError("El modelo PCA debe estar ajustado.")

        ix = cp_x - 1
        iy = cp_y - 1

        fig, ax = plt.subplots(figsize=figsize)

        # Scores
        if grupos is not None:
            cats = grupos.unique()
            colores = sns.color_palette(paleta, len(cats))
            for cat, color in zip(cats, colores):
                mask = grupos == cat
                ax.scatter(Z[mask, ix], Z[mask, iy], c=[color], label=str(cat),
                        s=tamaño_punto, alpha=alpha_puntos, zorder=3)
            ax.legend(title=grupos.name or 'Grupo', frameon=True,
                    bbox_to_anchor=(1.02, 1), loc='upper left')
        else:
            ax.scatter(Z[:, ix], Z[:, iy], color='steelblue',
                    s=tamaño_punto, alpha=alpha_puntos, zorder=3)

        # Loadings (flechas)
        loadings   = modelo_pca.componentes_[[ix, iy], :].T  # (D, 2)
        magnitudes = np.linalg.norm(loadings, axis=1)
        idx_top    = np.argsort(magnitudes)[::-1][:n_flechas_max]

        rango_scores = max(np.abs(Z[:, ix]).max(), np.abs(Z[:, iy]).max())
        rango_load   = magnitudes[idx_top].max() if len(idx_top) > 0 else 1.0
        factor       = (rango_scores * 0.8 * escala_flechas) / (rango_load + 1e-10)

        for i in idx_top:
            dx = loadings[i, 0] * factor
            dy = loadings[i, 1] * factor
            ax.annotate('', xy=(dx, dy), xytext=(0, 0),
                        arrowprops=dict(arrowstyle='->', color='crimson', lw=1.5))
            ax.text(dx * 1.08, dy * 1.08, nombres_features[i],
                    fontsize=8, color='crimson', ha='center')

        ax.axhline(0, color='gray', linewidth=0.8, linestyle='--')
        ax.axvline(0, color='gray', linewidth=0.8, linestyle='--')

        ve     = modelo_pca.varianza_explicada_ratio_
        xlabel = f'CP{cp_x} ({ve[ix]*100:.1f}% var.)'
        ylabel = f'CP{cp_y} ({ve[iy]*100:.1f}% var.)'

        VisualizadorDimensionalidad._estilo_base(ax, titulo, xlabel, ylabel)
        plt.tight_layout()
        if mostrar:
            plt.show()
        return fig, ax

    # ── 5.3 Dendrograma ───────────────────────────────────────

    @staticmethod
    def grafico_dendrograma(
        X_scaled: np.ndarray,
        metodo_linkage: str = 'ward',
        metrica_distancia: str = 'euclidean',
        truncar_p: Optional[int] = 30,
        titulo: str = 'Dendrograma — Clustering Jerárquico',
        xlabel: str = 'Índice de muestra (o tamaño del cluster)',
        ylabel_prefijo: str = 'Distancia',
        color_umbral: Optional[float] = None,
        color_encima: str = 'gray',
        orientacion: str = 'top',
        rotacion_etiquetas: float = 90.0,
        fuente_etiquetas: float = 8.0,
        figsize: Tuple[int, int] = (14, 6),
        fuente_titulo: int = 14,
        mostrar: bool = True,
    ) -> Tuple[Figure, Axes]:
        """
        Dendrograma de clustering jerárquico sobre la matriz estandarizada.

        El salto vertical más largo sin cruzar ninguna línea horizontal
        indica el número natural de clusters en los datos.

        Args:
            X_scaled: Matriz (N, D) ya estandarizada.
            metodo_linkage: 'ward' (minimiza varianza, recomendado),
                        'complete', 'average', 'single'.
            metrica_distancia: Métrica de distancia. Solo 'euclidean' con 'ward'.
            truncar_p: Muestra solo los últimos p nodos del dendrograma.
                    Útil para N grande. None → completo.
            titulo: Título del gráfico.
            xlabel: Etiqueta del eje X.
            ylabel_prefijo: Prefijo del eje Y.
            color_umbral: Umbral de color. Clusters por encima → color_encima.
                        None → umbrales automáticos de scipy.
            color_encima: Color para clusters sobre el umbral.
            orientacion: 'top', 'bottom', 'left', 'right'.
            rotacion_etiquetas: Rotación de etiquetas de hojas.
            fuente_etiquetas: Tamaño de fuente de etiquetas.
            figsize: Tamaño de figura.
            fuente_titulo: Tamaño de fuente del título.
            mostrar: Si True, llama plt.show().

        Returns:
            Tupla (fig, ax)
        """
        from scipy.cluster.hierarchy import linkage, dendrogram

        Z_link = linkage(X_scaled, method=metodo_linkage,
                         metric=metrica_distancia)

        fig, ax = plt.subplots(figsize=figsize)

        kwargs_dend: Dict = dict(
            orientation=orientacion,
            leaf_rotation=rotacion_etiquetas,
            leaf_font_size=fuente_etiquetas,
            show_contracted=True,
            ax=ax,
            above_threshold_color=color_encima,
        )
        if truncar_p is not None:
            kwargs_dend['truncate_mode'] = 'lastp'
            kwargs_dend['p'] = truncar_p
        if color_umbral is not None:
            kwargs_dend['color_threshold'] = color_umbral

        dendrogram(Z_link, **kwargs_dend)

        ax.set_title(titulo, fontsize=fuente_titulo, fontweight='bold')
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(
            f'{ylabel_prefijo} ({metodo_linkage.capitalize()})', fontsize=12
        )
        ax.grid(axis='y', linestyle='--', alpha=0.4)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.tight_layout()
        if mostrar:
            plt.show()
        return fig, ax

    # ── 5.4 Embedding 2D ─────────────────────────────────────

    @staticmethod
    def grafico_embedding_2d(
        Z: np.ndarray,
        grupos: Optional[pd.Series] = None,
        columna_estilo: Optional[pd.Series] = None,
        etiquetas_puntos: Optional[pd.Series] = None,
        nombre_metodo: str = 'Embedding',
        varianza_explicada: Optional[Tuple[float, float]] = None,
        paleta: str = 'viridis',
        tamaño_punto: int = 70,
        alpha: float = 0.8,
        titulo: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 8),
        añadir_convex_hull: bool = False,
        fuente_titulo: int = 15,
        fuente_ejes: int = 13,
        mostrar: bool = True,
    ) -> Tuple[Figure, Axes]:
        """
        Scatter 2D del embedding de dimensionalidad reducida.

        Válido para cualquier salida de fit_transform() (PCA, UMAP, tSNE).
        Colorea por grupo y diferencia por estilo de marcador.
        Equivalente parametrizado al scatter de Cluster × Genotipo del ejemplo.

        Args:
            Z: Array (N, 2+) con coordenadas del embedding.
            grupos: Serie con etiquetas de color (p.ej. 'Cluster_H').
            columna_estilo: Serie para estilo de marcador (p.ej. 'Genotipo').
            etiquetas_puntos: Anotaciones por punto. Solo para N pequeño (< 30).
            nombre_metodo: Nombre del método para etiquetas de ejes.
            varianza_explicada: (ve1, ve2) en fracción. Solo para PCA.
                            Se añade al label de cada eje.
            paleta: Paleta de colores.
            tamaño_punto: Tamaño de los puntos.
            alpha: Transparencia.
            titulo: Título. None → generado automáticamente.
            figsize: Tamaño de figura.
            añadir_convex_hull: Dibuja envoltura convexa de cada grupo.
            fuente_titulo: Tamaño de fuente del título.
            fuente_ejes: Tamaño de fuente de los ejes.
            mostrar: Si True, llama plt.show().

        Returns:
            Tupla (fig, ax)
        """
        fig, ax = plt.subplots(figsize=figsize)

        plot_df   = pd.DataFrame({'x': Z[:, 0], 'y': Z[:, 1]})
        hue_col   = None
        style_col = None

        if grupos is not None:
            plot_df['_grupo'] = grupos.values
            hue_col = '_grupo'
        if columna_estilo is not None:
            plot_df['_estilo'] = columna_estilo.values
            style_col = '_estilo'

        sns.scatterplot(
            data=plot_df, x='x', y='y',
            hue=hue_col, style=style_col,
            palette=paleta, s=tamaño_punto, alpha=alpha,
            legend='full', ax=ax,
        )

        if añadir_convex_hull and grupos is not None:
            from scipy.spatial import ConvexHull
            cats     = plot_df['_grupo'].unique()
            colores  = sns.color_palette(paleta, len(cats))
            for cat, color in zip(cats, colores):
                pts = plot_df[plot_df['_grupo'] == cat][['x', 'y']].values
                if len(pts) >= 3:
                    try:
                        hull     = ConvexHull(pts)
                        vertices = np.append(hull.vertices, hull.vertices[0])
                        ax.plot(pts[vertices, 0], pts[vertices, 1],
                                '--', color=color, alpha=0.5, linewidth=1.5)
                    except Exception:
                        pass

        if etiquetas_puntos is not None:
            for xi, yi, lbl in zip(Z[:, 0], Z[:, 1], etiquetas_puntos):
                ax.annotate(str(lbl), (xi, yi), fontsize=7, alpha=0.7,
                            xytext=(3, 3), textcoords='offset points')

        if varianza_explicada is not None:
            xlabel = f'{nombre_metodo} 1 ({varianza_explicada[0]*100:.1f}% var.)'
            ylabel = f'{nombre_metodo} 2 ({varianza_explicada[1]*100:.1f}% var.)'
        else:
            xlabel = f'{nombre_metodo} — Eje 1'
            ylabel = f'{nombre_metodo} — Eje 2'

        titulo_final = titulo or f'Clusters — Proyección {nombre_metodo}'
        VisualizadorDimensionalidad._estilo_base(
            ax, titulo_final, xlabel, ylabel,
            fuente_titulo=fuente_titulo, fuente_ejes=fuente_ejes,
        )
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left',
                frameon=True, fontsize=10)
        plt.tight_layout()
        if mostrar:
            plt.show()
        return fig, ax

    # ── 5.5 Scatter de features ───────────────────────────────

    @staticmethod
    def grafico_scatter_features(
        df: pd.DataFrame,
        columna_x: str,
        columna_y: str,
        columna_color: Optional[str] = None,
        columna_estilo: Optional[str] = None,
        mapa_etiquetas: Optional[Dict] = None,
        paleta: str = 'Spectral',
        tamaño_punto: int = 100,
        alpha: float = 0.7,
        titulo: str = 'Separación por Features',
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        titulo_leyenda: str = 'Subpoblación',
        figsize: Tuple[int, int] = (12, 8),
        fuente_titulo: int = 16,
        fuente_ejes: int = 12,
        mostrar: bool = True,
    ) -> Tuple[Figure, Axes]:
        """
        Scatter de dos features del DataFrame coloreado por cluster/grupo.

        Útil para validar la separación de clusters en el espacio de features
        originales (no reducidas), interpretando qué features separan los grupos.
        Equivalente parametrizado al gráfico media vs. CV del ejemplo original.

        Args:
            df: DataFrame con las columnas de features y de grupos.
            columna_x: Columna para el eje X.
            columna_y: Columna para el eje Y.
            columna_color: Columna para el color de los puntos.
            columna_estilo: Columna para el estilo de marcador.
            mapa_etiquetas: Dict para renombrar valores de columna_color.
                        Ejemplo: {0: 'Baja', 1: 'Alta', 2: 'Media'}.
            paleta: Paleta de colores.
            tamaño_punto: Tamaño de los puntos.
            alpha: Transparencia.
            titulo: Título del gráfico.
            xlabel: Etiqueta eje X. None → usa nombre de la columna.
            ylabel: Etiqueta eje Y. None → usa nombre de la columna.
            titulo_leyenda: Título de la leyenda.
            figsize: Tamaño de figura.
            fuente_titulo: Tamaño de fuente del título.
            fuente_ejes: Tamaño de fuente de los ejes.
            mostrar: Si True, llama plt.show().

        Returns:
            Tupla (fig, ax)
        """
        df_plot = df.copy()
        if mapa_etiquetas and columna_color:
            df_plot[columna_color] = df_plot[columna_color].map(mapa_etiquetas)

        fig, ax = plt.subplots(figsize=figsize)
        sns.scatterplot(
            data=df_plot,
            x=columna_x, y=columna_y,
            hue=columna_color, style=columna_estilo,
            palette=paleta, s=tamaño_punto, alpha=alpha,
            legend='full', ax=ax,
        )
        ax.legend(title=titulo_leyenda, bbox_to_anchor=(1.05, 1),
                  loc='upper left', frameon=True)

        VisualizadorDimensionalidad._estilo_base(
            ax, titulo,
            xlabel or columna_x,
            ylabel or columna_y,
            fuente_titulo=fuente_titulo, fuente_ejes=fuente_ejes,
        )
        plt.tight_layout()
        if mostrar:
            plt.show()
        return fig, ax

    # ── 5.6 Raincloud ─────────────────────────────────────────

    @staticmethod
    def grafico_raincloud(
        df: pd.DataFrame,
        columna_y: str,
        columna_x: str,
        columna_hue: str,
        mapa_etiquetas: Optional[Dict] = None,
        titulo: str = 'Raincloud Plot',
        ylabel: Optional[str] = None,
        xlabel: Optional[str] = None,
        titulo_leyenda: str = 'Subpoblación (Cluster)',
        paleta: str = 'Set1',
        alpha_violin: float = 0.6,
        alpha_strip: float = 0.8,
        alpha_box: float = 0.8,
        tamaño_punto_strip: int = 7,
        jitter_strip: float = 0.1,
        ancho_violin: float = 0.8,
        ancho_box: float = 0.8,
        mostrar_conteos: bool = True,
        posicion_conteo_offset: float = 1.05,
        fuente_conteo: int = 11,
        fuente_titulo: int = 18,
        fuente_ejes: int = 14,
        fuente_eje_x: int = 11,
        estilo_fuente: str = 'serif',
        figsize: Tuple[int, int] = (16, 9),
        hue_shift: float = 0.25,
        mostrar: bool = True,
    ) -> Tuple[Figure, Axes]:
        """
        Raincloud plot completo: Violin + Boxplot + Stripplot con conteos N.

        Combina tres capas superpuestas para la vista más informativa de
        la distribución de cada grupo:
            1. Violín  : densidad completa de la distribución
            2. Boxplot : mediana, cuartiles y bigotes (sin outliers)
            3. Strip   : todos los puntos individuales con jitter
            4. N=k     : anotación del tamaño de muestra por grupo

        Es el gráfico estándar para comparar distribuciones entre grupos
        con posible heterogeneidad o subpoblaciones.

        Args:
            df: DataFrame con los datos.
            columna_y: Variable dependiente (eje Y).
            columna_x: Eje X (p.ej. 'Genotipo').
            columna_hue: Subgrupos por color (p.ej. 'Etiqueta_Cluster').
            mapa_etiquetas: Dict para renombrar valores de columna_hue.
                        Ejemplo: {0: 'Baja', 1: 'Alta', 2: 'Media'}.
            titulo: Título del gráfico.
            ylabel: Etiqueta del eje Y. None → usa nombre de columna_y.
            xlabel: Etiqueta del eje X. None → usa nombre de columna_x.
            titulo_leyenda: Título de la leyenda.
            paleta: Paleta de colores base para los grupos hue.
            alpha_violin: Transparencia del violín.
            alpha_strip: Transparencia de los puntos.
            alpha_box: Transparencia del boxplot.
            tamaño_punto_strip: Tamaño de los puntos.
            jitter_strip: Magnitud del jitter horizontal.
            ancho_violin: Ancho relativo de los violines.
            ancho_box: Ancho relativo de los boxplots.
            mostrar_conteos: Si True, anota 'N=k' encima de cada grupo.
            posicion_conteo_offset: Multiplicador de max_y para posición de N=k.
                                Típico: 1.03–1.10.
            fuente_conteo: Tamaño de fuente de las anotaciones N=k.
            fuente_titulo: Tamaño de fuente del título.
            fuente_ejes: Tamaño de fuente de los ejes.
            fuente_eje_x: Tamaño de fuente de las etiquetas del eje X.
            estilo_fuente: 'serif', 'sans-serif', etc.
            figsize: Tamaño de figura.
            hue_shift: Separación horizontal entre subgrupos dentro de cada x.
            mostrar: Si True, llama plt.show().

        Returns:
            Tupla (fig, ax)

        Ejemplo de uso (equivalente al código original del usuario):
            df_plot = df_datos_combinado_2_10.copy()
            df_plot['Etiqueta_Cluster'] = df_plot['Cluster_H'].map({
                0: 'Baja Fluorescencia - Alta Variabilidad',
                1: 'Alta Fluorescencia - Media Variabilidad',
                2: 'Media Fluorescencia - Media Variabilidad',
            })
            fig, ax = VisualizadorDimensionalidad.grafico_raincloud(
                df=df_plot,
                columna_y='media',
                columna_x='Genotipo',
                columna_hue='Etiqueta_Cluster',
                titulo='Raincloud Plot: Mediana de Fluorescencia '
                    'por Genotipo y Subpoblación — 2/10',
                ylabel='Mediana de Fluorescencia (Intensidad del Gusano)',
                xlabel='Genotipo',
                paleta='Set1',
                figsize=(16, 9),
            )
        """
        sns.set_style("whitegrid")
        plt.rcParams['font.family'] = estilo_fuente
        plt.rcParams['font.size']   = 12

        df_plot = df.copy()
        if mapa_etiquetas and columna_hue in df_plot.columns:
            df_plot[columna_hue] = df_plot[columna_hue].map(mapa_etiquetas)

        cats_x   = list(df_plot[columna_x].unique())
        cats_hue = list(df_plot[columna_hue].unique())
        n_hue    = len(cats_hue)

        paleta_base  = sns.color_palette(paleta, n_colors=n_hue)
        paleta_strip = [sns.desaturate(c, 0.9) for c in paleta_base]

        counts = (
            df_plot.groupby([columna_x, columna_hue])
            .size()
            .reset_index(name='N')
        )

        fig, ax = plt.subplots(figsize=figsize)

        # A. Violin
        sns.violinplot(
            data=df_plot, x=columna_x, y=columna_y, hue=columna_hue,
            split=False, inner=None, width=ancho_violin,
            palette=paleta_base, alpha=alpha_violin,
            saturation=0.8, linewidth=0.8, ax=ax,
        )

        # B. Boxplot
        sns.boxplot(
            data=df_plot, x=columna_x, y=columna_y, hue=columna_hue,
            width=ancho_box, palette=paleta_base, ax=ax,
            boxprops={'zorder': 2, 'alpha': alpha_box,
                    'edgecolor': 'black', 'linewidth': 1.5},
            medianprops={'color': 'black', 'linewidth': 3},
            whiskerprops={'color': 'black', 'linewidth': 1.5},
            capprops={'color': 'black', 'linewidth': 1.5},
            showfliers=False,
        )

        # C. Stripplot
        sns.stripplot(
            data=df_plot, x=columna_x, y=columna_y, hue=columna_hue,
            dodge=True, palette=paleta_strip,
            s=tamaño_punto_strip, alpha=alpha_strip,
            linewidth=0, jitter=jitter_strip, ax=ax,
        )

        # D. Anotaciones N=k
        if mostrar_conteos:
            for _, fila in counts.iterrows():
                gen     = fila[columna_x]
                cluster = fila[columna_hue]
                n_val   = fila['N']
                if n_val == 0:
                    continue

                gen_idx = cats_x.index(gen)
                hue_idx = cats_hue.index(cluster)
                shift   = (hue_idx - (n_hue - 1) / 2.0) * hue_shift

                subset = df_plot[
                    (df_plot[columna_x]   == gen) &
                    (df_plot[columna_hue] == cluster)
                ][columna_y]

                if subset.empty:
                    continue

                text_y = subset.max() * posicion_conteo_offset
                ax.text(
                    gen_idx + shift, text_y, f'N={n_val}',
                    ha='center', va='bottom',
                    fontsize=fuente_conteo, fontweight='bold', color='black',
                )

        # Leyenda y estilo
        ax.set_title(titulo, fontsize=fuente_titulo, fontweight='bold',
                     fontfamily=estilo_fuente)
        ax.set_xlabel(xlabel or columna_x, fontsize=fuente_ejes,
                      fontweight='bold', fontfamily=estilo_fuente)
        ax.set_ylabel(ylabel or columna_y, fontsize=fuente_ejes,
                      fontweight='bold', fontfamily=estilo_fuente)
        ax.tick_params(axis='x', labelsize=fuente_eje_x)
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Deduplicar leyenda (violin + box + strip generan triplicados)
        handles, labels = ax.get_legend_handles_labels()
        legend = ax.legend(
            handles[:n_hue], labels[:n_hue],
            title=titulo_leyenda,
            loc='upper right', frameon=True, fontsize='large',
        )
        legend.get_title().set_fontweight('bold')

        plt.tight_layout()
        if mostrar:
            plt.show()
        return fig, ax