"""
Cuantificadores estadísticos para análisis de poblaciones de objetos.

Los cuantificadores estadísticos operan sobre colecciones de métricas
extraídas de múltiples imágenes/objetos para inferir propiedades de
la población, distribuciones y correlaciones.

Principio fundamental:
    A partir de muestras {x₁, x₂, ..., xₙ} de métricas morfométricas
    o topológicas, estimar parámetros poblacionales, distribuciones
    y relaciones estadísticas.

IMPORTANTE - Separación de responsabilidades:
- NO procesan imágenes directamente (reciben arrays/dicts de métricas)
- NO normalizan datos (ese rol es del preprocesamiento previo)
- Trabajan con pandas DataFrames para organización tabular
- Retornan estadísticos descriptivos e inferenciales
- Soportan múltiples imágenes (N samples) organizadas por columnas

Métricas disponibles:
- Estadísticos descriptivos: media, mediana, std, MAD, IQR, percentiles
- Distribuciones: ajuste de PDFs, tests de normalidad, KDE
- Correlaciones: Pearson, Spearman, matrices de correlación
- Inferencia: intervalos de confianza, tests de hipótesis
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, List, Union, Literal, Dict, Callable
from scipy import stats
from scipy.stats import kurtosis, skew, iqr as scipy_iqr
from collections import defaultdict
import warnings
from ...gestorLab.Registro_Metodos import registrar_en

class CuantificadorEstadistico:
    """Clase base para cuantificadores estadísticos."""
    nombre = "cuantificador_estadistico_base"
    
    def __call__(self, datos: Union[Dict, pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        """
        Args:
            datos: Diccionario de métricas, DataFrame o array 2D
        
        Returns:
            DataFrame con resultados estadísticos
        """
        raise NotImplementedError
    
    def _normalizar_a_dataframe(self, datos: Union[Dict, pd.DataFrame, np.ndarray],
                            nombre_indice: str = "metrica") -> pd.DataFrame:
        """Convierte diversos inputs a DataFrame estandarizado."""
        if isinstance(datos, pd.DataFrame):
            return datos.copy()
        
        elif isinstance(datos, dict):
            # Diccionario: {imagen_id: {metrica: valor, ...}, ...}
            # o {imagen_id: [valores], ...}
            registros = []
            for img_id, valores in datos.items():
                if isinstance(valores, dict):
                    for metrica, valor in valores.items():
                        if isinstance(valor, (list, np.ndarray)):
                            for v in valor:
                                registros.append({
                                    'imagen': img_id,
                                    'metrica': metrica,
                                    'valor': v
                                })
                        else:
                            registros.append({
                                'imagen': img_id,
                                'metrica': metrica,
                                'valor': valor
                            })
                else:
                    # Array de valores únicos por imagen
                    registros.append({
                        'imagen': img_id,
                        'valor': valores
                    })
            
            df = pd.DataFrame(registros)
            if 'metrica' in df.columns:
                return df.pivot(index='metrica', columns='imagen', values='valor')
            else:
                return df.set_index('imagen').T
        
        elif isinstance(datos, np.ndarray):
            if datos.ndim == 1:
                datos = datos.reshape(-1, 1)
            return pd.DataFrame(datos, columns=[f"imagen_{i}" for i in range(datos.shape[1])])
        
        else:
            raise ValueError(f"Tipo de datos no soportado: {type(datos)}")

@registrar_en("cuantificacion")
class EstadisticosDescriptivos(CuantificadorEstadistico):
    """
        Estadísticos descriptivos clásicos de población de objetos.
        
        Calcula métricas de tendencia central, dispersión y forma
        para cada imagen/columna del input.
        
        Algoritmo (para cada columna/imagen con n observaciones):
            Media: x̄ = (1/n) Σ xᵢ
            Mediana: Q₂ = percentil 50
            Std: s = √[Σ(xᵢ - x̄)²/(n-1)]
            MAD: median(|xᵢ - mediana|) × 1.4826 (estimador consistente σ)
            IQR: Q₃ - Q₁ = P₇₅ - P₂₅
            CV: s / |x̄| (coeficiente de variación)
            Skewness: asimetría de la distribución
            Kurtosis: "colas" de la distribución
        
        Propiedades:
            - Robustez: mediana/MAD > media/std ante outliers
            - Invarianza: estadísticos de escala/ubicación tienen
            comportamientos diferentes ante transformaciones lineales
        
        Ventajas:
            - Resumen completo de población en una tabla,
            - permite comparación entre imágenes/condiciones,
            - base para selección de thresholds
        Desventajas:
            - Asume independencia entre observaciones,
            - estadísticos clásicos sensibles a outliers,
            - no captura multimodalidad
        
        Usos microscopía:
            - Caracterización de heterogeneidad celular,
            - control de calidad entre réplicas experimentales,
            - detección de subpoblaciones (bimodalidad),
            - normalización entre placas/lotes
    """
    nombre = "estadisticos_descriptivos"
    
    def __init__(self, 
                estadisticos: Optional[List[str]] = None,
                percentiles: Optional[List[float]] = None,
                ignorar_nan: bool = True):
        """
        Args:
            estadisticos: Lista de estadísticos a calcular. None = todos
            percentiles: Lista de percentiles adicionales (0-100)
            ignorar_nan: Si True, ignora NaN en cálculos
        """
        self.estadisticos_disponibles = [
            'media', 'mediana', 'std', 'var', 'mad', 'iqr', 
            'min', 'max', 'rango', 'cv', 'skewness', 'kurtosis',
            'p5', 'p10', 'p90', 'p95', 'n_muestras', 'se'
        ]
        
        self.estadisticos = estadisticos or self.estadisticos_disponibles
        self.percentiles = percentiles or []
        self.ignorar_nan = ignorar_nan
        
        # Validar estadísticos solicitados
        invalidos = set(self.estadisticos) - set(self.estadisticos_disponibles)
        if invalidos:
            raise ValueError(f"Estadísticos inválidos: {invalidos}")
    
    def __call__(self, 
                datos: Union[Dict, pd.DataFrame, np.ndarray],
                agrupar_por: Optional[str] = None) -> pd.DataFrame:
        """
        Calcula estadísticos descriptivos por imagen/columna.
        
        Args:
            datos: Diccionario {img_id: {metrica: valores}} o DataFrame
            agrupar_por: Si datos es largo, agrupa por esta columna
        
        Returns:
            DataFrame con estadísticos (filas) × imágenes/métricas (columnas)
            Formato: índice = estadístico, columnas = imagen_o_metrica
        """
        df = self._normalizar_a_dataframe(datos)
        
        if agrupar_por and agrupar_por in df.columns:
            df = df.groupby(agrupar_por).agg('mean')  # Promedio por grupo
        
        # Calcular estadísticos para cada columna
        resultados = {}
        
        for col in df.columns:
            valores = df[col].values
            
            if self.ignorar_nan:
                valores = valores[~np.isnan(valores)]
            
            if len(valores) == 0:
                resultados[col] = {est: np.nan for est in self.estadisticos}
                continue
            
            stats_col = {}
            
            for est in self.estadisticos:
                stats_col[est] = self._calcular_estadistico(valores, est)
            
            # Percentiles personalizados
            for p in self.percentiles:
                stats_col[f'p{int(p)}'] = np.percentile(valores, p)
            
            resultados[col] = stats_col
        
        # Convertir a DataFrame: estadísticos × columnas
        df_resultado = pd.DataFrame(resultados).T
        
        # Reorganizar para que estadísticos sean filas
        if df_resultado.shape[0] == len(df.columns):
            df_resultado = df_resultado.T
        
        # Añadir metadatos
        df_resultado.attrs['tipo_analisis'] = 'estadisticos_descriptivos'
        df_resultado.attrs['estadisticos_calculados'] = self.estadisticos
        df_resultado.attrs['n_muestras_total'] = df.shape[0]
        
        return df_resultado
    
    def _calcular_estadistico(self, valores: np.ndarray, estadistico: str) -> float:
        """Calcula un estadístico específico."""
        n = len(valores)
        
        if estadistico == 'media':
            return float(np.mean(valores))
        
        elif estadistico == 'mediana':
            return float(np.median(valores))
        
        elif estadistico == 'std':
            return float(np.std(valores, ddof=1))
        
        elif estadistico == 'var':
            return float(np.var(valores, ddof=1))
        
        elif estadistico == 'mad':
            # Median Absolute Deviation (estimador robusto de escala)
            med = np.median(valores)
            mad = np.median(np.abs(valores - med))
            return float(mad * 1.4826)  # Factor de consistencia para normal
        
        elif estadistico == 'iqr':
            return float(scipy_iqr(valores))
        
        elif estadistico == 'min':
            return float(np.min(valores))
        
        elif estadistico == 'max':
            return float(np.max(valores))
        
        elif estadistico == 'rango':
            return float(np.max(valores) - np.min(valores))
        
        elif estadistico == 'cv':
            # Coeficiente de variación (%)
            media = np.mean(valores)
            return float(np.std(valores, ddof=1) / abs(media) * 100) if media != 0 else np.nan
        
        elif estadistico == 'skewness':
            return float(skew(valores))
        
        elif estadistico == 'kurtosis':
            return float(kurtosis(valores))
        
        elif estadistico == 'p5':
            return float(np.percentile(valores, 5))
        
        elif estadistico == 'p10':
            return float(np.percentile(valores, 10))
        
        elif estadistico == 'p90':
            return float(np.percentile(valores, 90))
        
        elif estadistico == 'p95':
            return float(np.percentile(valores, 95))
        
        elif estadistico == 'n_muestras':
            return int(n)
        
        elif estadistico == 'se':
            # Error estándar de la media
            return float(np.std(valores, ddof=1) / np.sqrt(n))
        
        else:
            return np.nan
    
    def comparar_grupos(self, 
                    datos: Dict[str, Dict[str, np.ndarray]],
                    metrica: str,
                    grupo_control: Optional[str] = None) -> pd.DataFrame:
        """
        Compara estadísticos entre grupos experimentales.
        
        Args:
            datos: {grupo: {imagen: valores_array}}
            metrica: Métrica específica a comparar
            grupo_control: Nombre del grupo control (para fold-change)
        
        Returns:
            DataFrame con estadísticos por grupo y comparaciones
        """
        resultados_grupos = {}
        
        for grupo, imagenes in datos.items():
            # Extraer valores de la métrica específica
            todos_valores = []
            for img_id, metricas in imagenes.items():
                if metrica in metricas:
                    val = metricas[metrica]
                    if isinstance(val, (list, np.ndarray)):
                        todos_valores.extend(val)
                    else:
                        todos_valores.append(val)
            
            if todos_valores:
                arr = np.array(todos_valores)
                stats_dict = {est: self._calcular_estadistico(arr, est) 
                            for est in self.estadisticos}
                resultados_grupos[grupo] = stats_dict
        
        df_comparacion = pd.DataFrame(resultados_grupos).T
        
        # Calcular fold-change vs control si se especifica
        if grupo_control and grupo_control in resultados_grupos:
            control_media = resultados_grupos[grupo_control]['media']
            if control_media != 0:
                df_comparacion['fold_change'] = (
                    df_comparacion['media'] / control_media
                )
                df_comparacion['log2_fc'] = np.log2(df_comparacion['fold_change'])
        
        return df_comparacion

@registrar_en("cuantificacion")
class Distribuciones(CuantificadorEstadistico):
    """
        Análisis de distribuciones y ajuste de modelos probabilísticos.
        
        Caracteriza la forma de las distribuciones de métricas y
        ajusta modelos paramétricos (normal, log-normal, gamma, etc).
        
        Algoritmo:
            1. Estimación no paramétrica: KDE (Kernel Density Estimation)
            f̂(x) = (1/nh) Σ K((x-xᵢ)/h)
            
            2. Ajuste paramétrico: MLE (Maximum Likelihood Estimation)
            para distribuciones candidatas
            
            3. Selección de modelo: AIC, BIC, test Kolmogorov-Smirnov
            
            4. Tests de bondad de ajuste:
            - Shapiro-Wilk (normalidad)
            - Anderson-Darling
            - D'Agostino-Pearson
        
        Ventajas:
            - Identifica distribuciones subyacentes,
            - detecta desviaciones de normalidad (común en biología),
            - permite simulaciones Monte Carlo
        Desventajas:
            - Requiere muestras grandes (n>30) para confianza,
            - múltiples distribuciones pueden ajustar igual,
            - sensibilidad a bins en histogramas
        
        Usos microscopía:
            - Modelado de distribuciones de tamaño celular,
            - detección de subpoblaciones (mixturas gaussianas),
            - control de calidad (detección de anomalías),
            - predicción de rangos esperados
    """
    nombre = "distribuciones"
    
    def __init__(self,
                distribuciones: List[str] = None,
                test_normalidad: bool = True,
                bins_histograma: int = 50):
        """
        Args:
            distribuciones: Lista de distribuciones a ajustar
            test_normalidad: Si True, realiza tests de normalidad
            bins_histograma: Número de bins para histogramas
        """
        self.distribuciones = distribuciones or ['norm', 'lognorm', 'gamma', 'expon']
        self.test_normalidad = test_normalidad
        self.bins_histograma = bins_histograma
        
        # Mapeo de nombres scipy
        self.dist_map = {
            'norm': stats.norm,
            'lognorm': stats.lognorm,
            'gamma': stats.gamma,
            'expon': stats.expon,
            'weibull': stats.weibull_min,
            'beta': stats.beta,
            'uniform': stats.uniform
        }
    
    def __call__(self,
                datos: Union[Dict, pd.DataFrame, np.ndarray],
                columna: Optional[str] = None) -> pd.DataFrame:
        """
        Analiza distribuciones de los datos.
        
        Args:
            datos: Diccionario o DataFrame con métricas
            columna: Columna específica a analizar (si None, todas)
        
        Returns:
            DataFrame con análisis de distribución por imagen/métrica
        """
        df = self._normalizar_a_dataframe(datos)
        
        if columna:
            df = df[[columna]]
        
        resultados = []
        
        for col in df.columns:
            valores = df[col].dropna().values
            
            if len(valores) < 5:
                warnings.warn(f"Muestra muy pequeña para {col}: n={len(valores)}")
                continue
            
            analisis = self._analizar_distribucion(valores, col)
            resultados.append(analisis)
        
        return pd.DataFrame(resultados).set_index('columna')
    
    def _analizar_distribucion(self, valores: np.ndarray, nombre: str) -> Dict:
        """Analiza la distribución de un conjunto de valores."""
        resultado = {
            'columna': nombre,
            'n_muestras': len(valores),
            'media': np.mean(valores),
            'std': np.std(valores, ddof=1),
        }
        
        # Tests de normalidad
        if self.test_normalidad and len(valores) >= 8:
            # Shapiro-Wilk (mejor para n < 5000)
            if len(valores) <= 5000:
                shapiro_stat, shapiro_p = stats.shapiro(valores)
                resultado['shapiro_stat'] = shapiro_stat
                resultado['shapiro_p'] = shapiro_p
                resultado['es_normal_95'] = shapiro_p > 0.05
            
            # D'Agostino-Pearson (mejor para n > 20)
            if len(valores) >= 20:
                dagostino_stat, dagostino_p = stats.normaltest(valores)
                resultado['dagostino_stat'] = dagostino_stat
                resultado['dagostino_p'] = dagostino_p
        
        # Ajuste de distribuciones paramétricas
        ajustes = {}
        for dist_name in self.distribuciones:
            try:
                dist = self.dist_map[dist_name]
                params = dist.fit(valores)
                
                # Test KS
                ks_stat, ks_p = stats.kstest(valores, 
                                            lambda x: dist.cdf(x, *params))
                
                # Log-likelihood
                loglik = np.sum(dist.logpdf(valores, *params))
                
                # AIC = 2k - 2ln(L)
                k = len(params)
                aic = 2 * k - 2 * loglik
                
                ajustes[dist_name] = {
                    'params': params,
                    'ks_stat': ks_stat,
                    'ks_p': ks_p,
                    'aic': aic,
                    'loglik': loglik
                }
            except Exception as e:
                ajustes[dist_name] = {'error': str(e)}
        
        # Seleccionar mejor ajuste (menor AIC)
        if ajustes:
            validos = {k: v for k, v in ajustes.items() if 'aic' in v}
            if validos:
                mejor = min(validos.items(), key=lambda x: x[1]['aic'])
                resultado['mejor_distribucion'] = mejor[0]
                resultado['mejor_aic'] = mejor[1]['aic']
                resultado['mejor_ks_p'] = mejor[1]['ks_p']
        
        # Estadísticos de forma
        resultado['skewness'] = skew(valores)
        resultado['kurtosis'] = kurtosis(valores)
        
        # Rangos
        resultado['rango_95_inf'] = np.percentile(valores, 2.5)
        resultado['rango_95_sup'] = np.percentile(valores, 97.5)
        
        return resultado
    
    def detectar_multimodalidad(self,
                            datos: Union[pd.DataFrame, np.ndarray],
                            columna: str,
                            max_modas: int = 3) -> Dict:
        """
        Detecta si la distribución es multimodal usando Gaussian Mixture.
        
        Args:
            datos: DataFrame con los datos
            columna: Columna a analizar
            max_modas: Máximo número de componentes a probar
        
        Returns:
            Dict con información de la mejor mezcla gaussiana
        """
        try:
            from sklearn.mixture import GaussianMixture
        except ImportError:
            raise ImportError("Se requiere scikit-learn para análisis de mixturas")
        
        valores = datos[columna].dropna().values.reshape(-1, 1)
        
        mejor_bic = np.inf
        mejor_gmm = None
        mejor_n = 1
        
        for n in range(1, max_modas + 1):
            gmm = GaussianMixture(n_components=n, random_state=42)
            gmm.fit(valores)
            bic = gmm.bic(valores)
            
            if bic < mejor_bic:
                mejor_bic = bic
                mejor_gmm = gmm
                mejor_n = n
        
        return {
            'n_componentes_optimo': mejor_n,
            'bic': mejor_bic,
            'medias': mejor_gmm.means_.flatten().tolist(),
            'pesos': mejor_gmm.weights_.tolist(),
            'covarianzas': mejor_gmm.covariances_.flatten().tolist(),
            'es_multimodal': mejor_n > 1
        }

@registrar_en("cuantificacion")
class Correlaciones(CuantificadorEstadistico):
    """
        Análisis de correlaciones entre métricas e imágenes.
        
        Cuantifica relaciones lineales y monotónicas entre variables
        morfométricas y detecta redundancias.
        
        Algoritmo:
            Pearson r = cov(X,Y) / (σₓ σᵧ)
            [-1, 1], lineal, sensible a outliers
            
            Spearman ρ = Pearson(rango(X), rango(Y))
            [-1, 1], monotónica, robusta
            
            Kendall τ = (concordantes - discordantes) / (n(n-1)/2)
            [-1, 1], ordinal, robusta para muestras pequeñas
            
            Matriz de correlación: R donde Rᵢⱼ = corr(Xᵢ, Xⱼ)
        
        Propiedades:
            - Simetría: corr(X,Y) = corr(Y,X)
            - Invarianza a escala (Pearson) o rank (Spearman)
            - No implica causalidad
        
        Ventajas:
            - Detecta redundancias entre métricas,
            - identifica variables predictoras,
            - base para PCA y reducción de dimensionalidad
        Desventajas:
            - Solo detecta relaciones monotónicas,
            - no captura interacciones no lineales,
            - sensible a rangos restringidos
        
        Usos microscopía:
            - Redundancia entre área y diámetro equivalente,
            - correlación forma-función (ej: circularidad vs viabilidad),
            - selección de features para ML (eliminar correlacionadas),
            - análisis de co-variación en time-lapse
    """
    nombre = "correlaciones"
    
    def __init__(self,
                metodo: Literal['pearson', 'spearman', 'kendall'] = 'pearson',
                umbral_significancia: float = 0.05,
                corregir_multiple: bool = True):
        """
        Args:
            metodo: Tipo de correlación ('pearson', 'spearman', 'kendall')
            umbral_significancia: p-value para significancia
            corregir_multiple: Si True, aplica corrección Bonferroni
        """
        self.metodo = metodo
        self.umbral_significancia = umbral_significancia
        self.corregir_multiple = corregir_multiple
    
    def __call__(self,
                datos: Union[pd.DataFrame, Dict],
                variables: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Calcula matriz de correlación entre variables.
        
        Args:
            datos: DataFrame con métricas (filas=obs, cols=variables)
            variables: Subconjunto de variables a correlacionar
        
        Returns:
            DataFrame con matriz de correlación
        """
        df = self._normalizar_a_dataframe(datos)
        
        if variables:
            df = df[variables]
        
        # Asegurar que todas las columnas son numéricas
        df = df.select_dtypes(include=[np.number])
        
        if df.shape[1] < 2:
            raise ValueError("Se necesitan al menos 2 variables numéricas")
        
        # Calcular correlaciones
        corr_matrix = df.corr(method=self.metodo)
        
        # Calcular p-values
        pvalues = self._calcular_pvalues(df)
        
        # Añadir metadatos
        corr_matrix.attrs['metodo'] = self.metodo
        corr_matrix.attrs['n_observaciones'] = len(df)
        corr_matrix.attrs['p_values'] = pvalues
        
        return corr_matrix
    
    def _calcular_pvalues(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcula p-values para cada par de correlación."""
        cols = df.columns
        n = len(cols)
        pvalues = pd.DataFrame(np.ones((n, n)), columns=cols, index=cols)
        
        for i, col1 in enumerate(cols):
            for j, col2 in enumerate(cols):
                if i != j:
                    valores1 = df[col1].dropna()
                    valores2 = df[col2].dropna()
                    
                    # Alinear índices
                    idx_comun = valores1.index.intersection(valores2.index)
                    if len(idx_comun) > 2:
                        x = valores1.loc[idx_comun]
                        y = valores2.loc[idx_comun]
                        
                        if self.metodo == 'pearson':
                            _, p = stats.pearsonr(x, y)
                        elif self.metodo == 'spearman':
                            _, p = stats.spearmanr(x, y)
                        else:
                            _, p = stats.kendalltau(x, y)
                        
                        pvalues.loc[col1, col2] = p
        
        # Corrección Bonferroni
        if self.corregir_multiple:
            n_tests = n * (n - 1) / 2
            pvalues = pvalues * n_tests
            pvalues = pvalues.clip(upper=1.0)
        
        return pvalues
    
    def correlacion_imagenes(self,
                            datos: Dict[str, Dict[str, float]],
                            metodo_agregacion: Literal['media', 'mediana'] = 'media'
                            ) -> pd.DataFrame:
        """
        Correlación entre imágenes basada en sus perfiles de métricas.
        
        Args:
            datos: {imagen_id: {metrica: valor_promedio}}
            metodo_agregacion: Cómo agregar si hay múltiples valores
        
        Returns:
            DataFrame con matriz de correlación entre imágenes
        """
        # Convertir a DataFrame: imágenes × métricas
        df = pd.DataFrame(datos).T
        
        # Calcular correlación entre filas (imágenes)
        corr_img = df.T.corr(method=self.metodo)
        
        return corr_img
    
    def detectar_colinealidad(self,
                            matriz_corr: pd.DataFrame,
                            umbral: float = 0.8) -> List[Tuple[str, str, float]]:
        """
        Detecta pares de variables altamente correlacionadas.
        
        Args:
            matriz_corr: Matriz de correlación
            umbral: Umbral para considerar colinealidad
        
        Returns:
            Lista de tuplas (var1, var2, correlación)
        """
        colineales = []
        
        for i in range(len(matriz_corr.columns)):
            for j in range(i+1, len(matriz_corr.columns)):
                corr_val = abs(matriz_corr.iloc[i, j])
                if corr_val >= umbral:
                    colineales.append((
                        matriz_corr.columns[i],
                        matriz_corr.columns[j],
                        corr_val
                    ))
        
        return sorted(colineales, key=lambda x: x[2], reverse=True)
    
    def analisis_canonico(self,
                        datos_x: pd.DataFrame,
                        datos_y: pd.DataFrame) -> Dict:
        """
        Análisis de correlación canónica (CCA) entre dos conjuntos de variables.
        
        Útil para relacionar morfometría (X) con topología (Y), por ejemplo.
        
        Args:
            datos_x: DataFrame con primer conjunto de variables
            datos_y: DataFrame con segundo conjunto
        
        Returns:
            Dict con correlaciones canónicas y coeficientes
        """
        try:
            from sklearn.cross_decomposition import CCA
        except ImportError:
            raise ImportError("Se requiere scikit-learn para CCA")
        
        # Alinear índices
        idx_comun = datos_x.index.intersection(datos_y.index)
        X = datos_x.loc[idx_comun].values
        Y = datos_y.loc[idx_comun].values
        
        # Remover NaN
        mask = ~(np.isnan(X).any(axis=1) | np.isnan(Y).any(axis=1))
        X = X[mask]
        Y = Y[mask]
        
        n_components = min(X.shape[1], Y.shape[1], 5)
        cca = CCA(n_components=n_components)
        cca.fit(X, Y)
        
        X_c, Y_c = cca.transform(X, Y)
        
        # Correlaciones canónicas
        corr_canonicas = [np.corrcoef(X_c[:, i], Y_c[:, i])[0, 1] 
                        for i in range(n_components)]
        
        return {
            'correlaciones_canonicas': corr_canonicas,
            'n_componentes': n_components,
            'coeficientes_x': cca.x_weights_.tolist(),
            'coeficientes_y': cca.y_weights_.tolist(),
            'varianza_x_explicada': cca.explained_variance_ratio_.tolist() if hasattr(cca, 'explained_variance_ratio_') else None
        }

# FUNCIONES DE UTILIDAD Y USO PARA PIPELINE COMPLETO

def crear_dataframe_metricas(lista_metricas_por_imagen: List[Dict[str, Union[float, List]]],
                            imagen_ids: Optional[List[str]] = None) -> pd.DataFrame:
    """
        Crea DataFrame organizado desde lista de diccionarios de métricas.
        
        Args:
            lista_metricas_por_imagen: [{metrica: valor_o_lista}, ...]
            imagen_ids: Identificadores de imágenes (si None, genera automático)
        
        Returns:
            DataFrame largo (formato tidy) con columnas:
            [imagen, metrica, valor, tipo_valor]
    """
    if imagen_ids is None:
        imagen_ids = [f"img_{i:04d}" for i in range(len(lista_metricas_por_imagen))]
    
    registros = []
    
    for img_id, metricas in zip(imagen_ids, lista_metricas_por_imagen):
        for metrica, valor in metricas.items():
            if isinstance(valor, (list, np.ndarray)):
                # Métricas por objeto (múltiples valores)
                for i, v in enumerate(valor):
                    registros.append({
                        'imagen': img_id,
                        'metrica': metrica,
                        'valor': v,
                        'objeto_id': i,
                        'tipo': 'por_objeto'
                    })
            else:
                # Métrica única por imagen
                registros.append({
                    'imagen': img_id,
                    'metrica': metrica,
                    'valor': valor,
                    'objeto_id': None,
                    'tipo': 'global'
                })
    
    return pd.DataFrame(registros)


def pipeline_estadistico_completo(datos_por_imagen: Dict[str, Dict[str, Union[float, List]]],
                                grupos: Optional[Dict[str, List[str]]] = None
                                ) -> Dict[str, pd.DataFrame]:
    """
        Pipeline completo de análisis estadístico.
        
        Ejecuta todos los análisis estadísticos y retorna DataFrames organizados.
        
        Args:
            datos_por_imagen: {imagen_id: {metrica: valores}}
            grupos: Opcional: {grupo_nombre: [imagen_ids]}
        
        Returns:
            Dict con DataFrames: 'descriptivos', 'distribuciones', 'correlaciones'
    """
    # Crear DataFrame tidy
    df_tidy = crear_dataframe_metricas(
        [datos_por_imagen[k] for k in datos_por_imagen.keys()],
        list(datos_por_imagen.keys())
    )
    
    # Pivotar para análisis (métricas × imágenes, agregando por media si es necesario)
    df_pivot = df_tidy.groupby(['imagen', 'metrica'])['valor'].mean().unstack()
    
    resultados = {}
    
    # 1. Estadísticos descriptivos por métrica
    print("Calculando estadísticos descriptivos...")
    est_desc = EstadisticosDescriptivos()
    
    # Por cada métrica, calcular estadísticos entre imágenes
    stats_por_metrica = {}
    for metrica in df_tidy['metrica'].unique():
        df_metrica = df_tidy[df_tidy['metrica'] == metrica]
        # Pivot: objetos × imágenes
        df_pivot_metrica = df_metrica.pivot_table(
            index='objeto_id', 
            columns='imagen', 
            values='valor',
            aggfunc='first'
        )
        stats_por_metrica[metrica] = est_desc(df_pivot_metrica)
    
    resultados['descriptivos'] = pd.concat(stats_por_metrica, names=['metrica'])
    
    # 2. Análisis de distribuciones
    print("Analizando distribuciones...")
    dist = Distribuciones()
    dist_por_metrica = {}
    for metrica in df_tidy['metrica'].unique():
        valores = df_tidy[df_tidy['metrica'] == metrica]['valor'].dropna()
        if len(valores) > 10:
            dist_por_metrica[metrica] = dist._analizar_distribucion(valores.values, metrica)
    
    resultados['distribuciones'] = pd.DataFrame(dist_por_metrica).T
    
    # 3. Correlaciones entre métricas (promediando por imagen)
    print("Calculando correlaciones...")
    corr = Correlaciones(metodo='pearson')
    df_metricas_promedio = df_tidy.groupby(['imagen', 'metrica'])['valor'].mean().unstack()
    resultados['correlaciones'] = corr(df_metricas_promedio)
    
    # 4. Comparación entre grupos si se especifican
    if grupos:
        print("Comparando grupos...")
        comparaciones = {}
        for metrica in df_tidy['metrica'].unique():
            datos_grupo = {}
            for grupo, imgs in grupos.items():
                datos_grupo[grupo] = {
                    img: datos_por_imagen[img] 
                    for img in imgs if img in datos_por_imagen
                }
            
            if datos_grupo:
                comp = est_desc.comparar_grupos(datos_grupo, metrica)
                comparaciones[metrica] = comp
        
        resultados['comparacion_grupos'] = pd.concat(comparaciones, names=['metrica'])
    
    # Metadatos globales
    resultados['_metadatos'] = {
        'n_imagenes': len(datos_por_imagen),
        'n_metricas': df_tidy['metrica'].nunique(),
        'n_observaciones_totales': len(df_tidy),
        'metricas_analizadas': df_tidy['metrica'].unique().tolist()
    }
    
    return resultados


def exportar_resultados(resultados: Dict[str, pd.DataFrame],
                    prefijo: str = "analisis",
                    formato: Literal['csv', 'excel', 'json'] = 'excel'):
    """
    Exporta DataFrames de resultados a archivos.
    
    Args:
        resultados: Dict con DataFrames del pipeline
        prefijo: Prefijo para nombres de archivo
        formato: Formato de exportación
    """
    if formato == 'excel':
        with pd.ExcelWriter(f"{prefijo}_estadisticos.xlsx", engine='openpyxl') as writer:
            for nombre, df in resultados.items():
                if isinstance(df, pd.DataFrame):
                    # Limitar nombre de hoja a 31 caracteres
                    hoja = nombre[:31]
                    df.to_excel(writer, sheet_name=hoja)
    elif formato == 'csv':
        for nombre, df in resultados.items():
            if isinstance(df, pd.DataFrame):
                df.to_csv(f"{prefijo}_{nombre}.csv")
    elif formato == 'json':
        for nombre, df in resultados.items():
            if isinstance(df, pd.DataFrame):
                df.to_json(f"{prefijo}_{nombre}.json", orient='records')
    
    print(f"Resultados exportados con prefijo: {prefijo}")