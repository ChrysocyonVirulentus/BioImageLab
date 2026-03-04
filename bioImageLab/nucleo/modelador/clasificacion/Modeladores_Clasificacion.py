"""
Clasificación supervisada de fenotipos celulares en datos multiparamétricos.

Este módulo recibe DataFrames con métricas cuantificadas (formato estándar del
pipeline: '{metrica}_{estadistico}') y aplica algoritmos de clasificación supervisada
para predecir etiquetas fenotípicas, estados celulares o respuestas a tratamientos.

Estructura del DataFrame de entrada:
    ┌──────────┬────────────────┬──────────────┬─────────────┬──────────────┬─────────────┐
    │ imagen   │ grupo_exp      │ area_media   │ area_std    │ circularidad │ fenotipo    │
    ├──────────┼────────────────┼──────────────┼─────────────┼──────────────┼─────────────┤
    │ img_001  │ control        │ 312.5        │ 45.2        │ 0.85         │ apoptótica  │
    │ img_002  │ tratamiento_A  │ 289.1        │ 38.7        │ 0.72         │ viable      │
    └──────────┴────────────────┴──────────────┴─────────────┴──────────────┴─────────────┘

IMPORTANTE — Principios de coherencia:
    Los clasificadores operan sobre la matriz X ya estandarizada.
    La estandarización es OBLIGATORIA para:
        - SVM: kernel RBF sensible a escalas (evita dominancia de features grandes)
        - LogisticRegression: convergencia estable del gradiente
        - RandomForest: aunque es invariante a escala, la estandarización mejora
          la comparabilidad de importancias de features y la consistencia del pipeline
    
    El desbalanceo de clases es común en microscopía (ej: apoptóticas vs viables).
    Se debe manejar via class_weight o técnicas de resampling.

Separación de responsabilidades:
    - NO realiza cuantificación de imágenes (eso es Cuantificadores_*)
    - NO reduce dimensionalidad (eso es Modelador_dimensionalidad)
    - NO visualiza (eso es Visualizador_*)
    - SÍ: entrenamiento, predicción, evaluación de modelos, selección de hiperparámetros
    - SÍ: manejo de desbalanceo, validación cruzada, análisis de importancia de features
    - SÍ: exporta métricas de rendimiento, probabilidades de predicción, curvas ROC/PR
"""

import warnings
from typing import Dict, List, Literal, Optional, Tuple, Union, Any, Callable

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.calibration import CalibratedClassifierCV

# Intentar importar imbalanced-learn (opcional, para manejo avanzado de desbalanceo)
try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False
    warnings.warn(
        "imbalanced-learn no instalado. Algunas técnicas de resampling no estarán disponibles. "
        "Instalar con: pip install imbalanced-learn",
        UserWarning
    )


# =============================================================================
# 1. EVALUACIÓN DE RENDIMIENTO DE CLASIFICACIÓN
# =============================================================================

class EvaluadorClasificacion:
    """
    Métricas de evaluación para clasificación supervisada.

    Distingue entre:
        - Métricas globales: accuracy, balanced_accuracy, log_loss
        - Métricas por clase: precision, recall (sensitivity), specificity, f1
        - Métricas de ranking: ROC-AUC, PR-AUC (Average Precision)
        - Métricas de calibración: Brier score, expected calibration error

    Algoritmo Matriz de Confusión:
        ┌─────────────────┬─────────────────┐
        │ Verdaderos      │ Falsos          │
        │ Positivos (TP)  │ Positivos (FP)  │
        ├─────────────────┼─────────────────┤
        │ Falsos          │ Verdaderos      │
        │ Negativos (FN)  │ Negativos (TN)  │
        └─────────────────┴─────────────────┘

    Métricas derivadas:
        - Precision = TP / (TP + FP)  → confianza en predicciones positivas
        - Recall (Sensitivity) = TP / (TP + FN)  → capacidad de detectar positivos
        - Specificity = TN / (TN + FP)  → capacidad de detectar negativos
        - F1 = 2 × (Precision × Recall) / (Precision + Recall)  → balance P/R
        - Balanced Accuracy = (Sensitivity + Specificity) / 2  → robusta a desbalanceo

    Algoritmo ROC-AUC:
        AUC = P(score_pos > score_neg) donde score_pos ~ positivos, score_neg ~ negativos
        Interpretación: probabilidad de que un positivo aleatorio tenga mayor score
        que un negativo aleatorio.
        Rango: [0, 1], donde 0.5 = aleatorio, 1.0 = perfecto.

    Algoritmo PR-AUC (Average Precision):
        AP = Σₙ (Recallₙ - Recallₙ₋₁) × Precisionₙ
        Más informativo que ROC-AUC cuando hay desbalanceo severo de clases.

    Ventajas:
        - Accuracy: intuitiva, pero engañosa con clases desbalanceadas
        - F1: balance robusto entre precision y recall
        - ROC-AUC: invariante a umbrales de decisión, buena para ranking
        - PR-AUC: sensible a rendimiento en clase minoritaria

    Desventajas:
        - Accuracy: inútil cuando clase mayoritaria > 90%
        - F1: no considera verdaderos negativos (especificidad)
        - ROC-AUC: puede ser optimista con desbalanceo extremo
        - Todas: requieren suficientes muestras por clase para ser estables

    Usos microscopía:
        - Validar clasificación de fenotipos celulares (viable vs apoptótica)
        - Evaluar detección de subpoblaciones raras (células tumorales en tejido sano)
        - Comparar modelos para screening de fármacos (priorizar sensitivity vs specificity)
        - Calibrar probabilidades para toma de decisiones clínicas
    """
    nombre = "evaluador_clasificacion"

    def __init__(self, promedio: Literal['binary', 'micro', 'macro', 'weighted'] = 'weighted'):
        """
        Args:
            promedio: Estrategia de promedio para métricas multiclase.
                     'binary': solo para 2 clases.
                     'micro': global, agregando contribuciones.
                     'macro': promedio no ponderado de clases.
                     'weighted': promedio ponderado por soporte de clase.
        """
        self.promedio = promedio

    def accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Accuracy global: (TP + TN) / total."""
        return float(metrics.accuracy_score(y_true, y_pred))

    def balanced_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Balanced accuracy: promedio de recall por clase.
        Robusta a desbalanceo de clases.
        """
        return float(metrics.balanced_accuracy_score(y_true, y_pred))

    def precision_recall_f1(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Precision, Recall y F1-score.
        
        Returns:
            Dict con precision, recall, f1 (macro/weighted según self.promedio)
        """
        precision, recall, f1, _ = metrics.precision_recall_fscore_support(
            y_true, y_pred, average=self.promedio, zero_division=0
        )
        return {
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1)
        }

    def matriz_confusion(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Matriz de confusión.
        
        Returns:
            Array (n_clases, n_clases) donde C[i,j] = predichos como j siendo i.
        """
        return metrics.confusion_matrix(y_true, y_pred)

    def roc_auc(self, y_true: np.ndarray, y_score: np.ndarray, 
                multi_class: Literal['raise', 'ovr', 'ovo'] = 'ovr') -> float:
        """
        Area Under the ROC Curve.
        
        Args:
            y_true: Etiquetas verdaderas
            y_score: Probabilidades de clase positiva (binario) o todas las clases (multiclase)
            multi_class: Estrategia multiclase ('ovr': One-vs-Rest, 'ovo': One-vs-One)
        
        Returns:
            ROC-AUC score
        """
        try:
            if len(np.unique(y_true)) == 2:
                return float(metrics.roc_auc_score(y_true, y_score[:, 1] if y_score.ndim > 1 else y_score))
            else:
                return float(metrics.roc_auc_score(y_true, y_score, multi_class=multi_class, average=self.promedio))
        except ValueError:
            return np.nan

    def average_precision(self, y_true: np.ndarray, y_score: np.ndarray) -> float:
        """
        Average Precision (PR-AUC).
        Más informativo que ROC-AUC con clases desbalanceadas.
        """
        try:
            if len(np.unique(y_true)) == 2:
                return float(metrics.average_precision_score(y_true, y_score[:, 1] if y_score.ndim > 1 else y_score))
            else:
                # Para multiclase, retornar macro promedio
                return float(metrics.average_precision_score(
                    y_true, y_score, average=self.promedio
                ))
        except ValueError:
            return np.nan

    def log_loss(self, y_true: np.ndarray, y_proba: np.ndarray) -> float:
        """
        Log-loss (entropía cruzada).
        Penaliza predicciones confiadas pero incorrectas.
        """
        return float(metrics.log_loss(y_true, y_proba))

    def brier_score(self, y_true: np.ndarray, y_proba: np.ndarray) -> float:
        """
        Brier score: mean squared error de probabilidades.
        Mide calibración del modelo (qué tan cerca están probs de frecuencias reales).
        """
        if len(np.unique(y_true)) == 2:
            # Para binario, usar probabilidad de clase positiva
            proba_pos = y_proba[:, 1] if y_proba.ndim > 1 else y_proba
            return float(metrics.brier_score_loss(y_true, proba_pos))
        return np.nan  # No implementado para multiclase en sklearn

    def evaluar_completo(self, 
                        y_true: np.ndarray, 
                        y_pred: np.ndarray,
                        y_proba: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Evaluación completa del rendimiento de clasificación.

        Args:
            y_true: Etiquetas verdaderas (N,)
            y_pred: Etiquetas predichas (N,)
            y_proba: Probabilidades predichas (N, n_clases) o (N,) para binario

        Returns:
            Dict con todas las métricas disponibles, matriz de confusión y reporte por clase
        """
        resultados = {
            'accuracy': self.accuracy(y_true, y_pred),
            'balanced_accuracy': self.balanced_accuracy(y_true, y_pred),
            'n_muestras': len(y_true),
            'n_clases': len(np.unique(y_true)),
        }
        
        # Métricas P/R/F1
        resultados.update(self.precision_recall_f1(y_true, y_pred))
        
        # Matriz de confusión
        resultados['matriz_confusion'] = self.matriz_confusion(y_true, y_pred).tolist()
        
        # Reporte detallado por clase
        reporte = metrics.classification_report(
            y_true, y_pred, output_dict=True, zero_division=0
        )
        resultados['reporte_por_clase'] = reporte
        
        # Métricas que requieren probabilidades
        if y_proba is not None:
            resultados['roc_auc'] = self.roc_auc(y_true, y_proba)
            resultados['average_precision'] = self.average_precision(y_true, y_proba)
            resultados['log_loss'] = self.log_loss(y_true, y_proba)
            
            if resultados['n_clases'] == 2:
                resultados['brier_score'] = self.brier_score(y_true, y_proba)
        
        return resultados

    def curva_roc(self, y_true: np.ndarray, y_score: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Calcula puntos para graficar curva ROC.

        Returns:
            Dict con 'fpr', 'tpr', 'thresholds'
        """
        if len(np.unique(y_true)) != 2:
            raise ValueError("Curva ROC binaria solo para 2 clases. Use roc_auc multiclase.")
        
        scores = y_score[:, 1] if y_score.ndim > 1 else y_score
        fpr, tpr, thresholds = metrics.roc_curve(y_true, scores)
        return {'fpr': fpr, 'tpr': tpr, 'thresholds': thresholds}

    def curva_precision_recall(self, y_true: np.ndarray, y_score: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Calcula puntos para graficar curva Precision-Recall.

        Returns:
            Dict con 'precision', 'recall', 'thresholds'
        """
        if len(np.unique(y_true)) != 2:
            raise ValueError("Curva PR binaria solo para 2 clases.")
        
        scores = y_score[:, 1] if y_score.ndim > 1 else y_score
        precision, recall, thresholds = metrics.precision_recall_curve(y_true, scores)
        return {'precision': precision, 'recall': recall, 'thresholds': thresholds}


# =============================================================================
# 2. MANEJO DE DESBALANCEO DE CLASES
# =============================================================================

class ManejadorDesbalanceo:
    """
    Estrategias para manejar clases desbalanceadas en microscopía.

    Problema: En imágenes celulares, algunos fenotipos son raros
    (ej: células apoptóticas ~5%, viables ~95%). Los clasificadores
    tienden a predecir siempre la clase mayoritaria.

    Estrategias implementadas:

    1. Class Weighting (pesos de clase):
        Asignar peso inversamente proporcional a frecuencia.
        w_j = n_muestras / (n_clases × n_muestras_clase_j)
        
        Ventajas: simple, no requiere oversampling (mantiene datos originales)
        Desventajas: puede causar overfitting en clase minoritaria si pesos muy altos

    2. SMOTE (Synthetic Minority Over-sampling Technique):
        Generar muestras sintéticas interpolando vecinos cercanos de la
        clase minoritaria.
        
        Algoritmo:
            Para cada muestra minoritaria x:
                1. Encontrar k vecinos más cercanos (default k=5)
                2. Elegir un vecino aleatorio x̃
                3. Generar nueva muestra: x_new = x + rand(0,1) × (x̃ - x)
        
        Ventajas: aumenta datos sin duplicación simple, ayuda a decision boundaries
        Desventajas: puede generar ruido si outliers en minoritaria, no usar en test

    3. Threshold Moving (ajuste de umbral):
        Para clasificación binaria, ajustar umbral de decisión para maximizar
        F1 o balanced accuracy, en lugar de usar 0.5 por defecto.

    Usos microscopía:
        - Detectar células tumorales raras en biopsias (sensitivity > 95%)
        - Clasificar eventos raros (mitosis, muerte celular)
        - Evitar falsos negativos en screening de fármacos
    """
    nombre = "manejador_desbalanceo"

    def __init__(self, estrategia: Literal['class_weight', 'smote', 'threshold', 'ninguna'] = 'class_weight'):
        """
        Args:
            estrategia: 'class_weight', 'smote', 'threshold', o 'ninguna'
        """
        self.estrategia = estrategia

    def calcular_class_weights(self, y: np.ndarray) -> Dict[int, float]:
        """
        Calcula pesos de clase balanceados inversamente a frecuencia.

        Args:
            y: Array de etiquetas (N,)

        Returns:
            Dict {clase: peso}
        """
        clases, counts = np.unique(y, return_counts=True)
        total = len(y)
        n_clases = len(clases)
        
        # Fórmula: n_muestras / (n_clases * n_muestras_clase)
        pesos = {int(c): float(total / (n_clases * count)) for c, count in zip(clases, counts)}
        return pesos

    def crear_pipeline_smote(self, 
                            clasificador: Any, 
                            random_state: int = 42) -> Any:
        """
        Crea pipeline con SMOTE + clasificador.

        Args:
            clasificador: Estimador sklearn compatible
            random_state: Semilla para reproducibilidad

        Returns:
            Pipeline de imbalanced-learn o sklearn
        """
        if not IMBLEARN_AVAILABLE:
            warnings.warn("SMOTE no disponible. Retornando clasificador original.", UserWarning)
            return clasificador
        
        if self.estrategia == 'smote':
            pipeline = ImbPipeline([
                ('smote', SMOTE(random_state=random_state)),
                ('clasificador', clasificador)
            ])
            return pipeline
        
        return clasificador

    def encontrar_umbral_optimo(self, 
                                 y_true: np.ndarray, 
                                 y_scores: np.ndarray,
                                 metrica: Literal['f1', 'balanced_accuracy', 'youden'] = 'f1') -> Tuple[float, float]:
        """
        Encuentra umbral óptimo de decisión para clasificación binaria.

        Args:
            y_true: Etiquetas verdaderas (binario)
            y_scores: Probabilidades de clase positiva
            metrica: Métrica a optimizar ('f1', 'balanced_accuracy', 'youden')

        Returns:
            Tupla (umbral_optimo, valor_metrica)
        """
        if metrica == 'youden':
            fpr, tpr, thresholds = metrics.roc_curve(y_true, y_scores)
            youden_scores = tpr - fpr
            idx_optimo = np.argmax(youden_scores)
        else:
            precision, recall, thresholds = metrics.precision_recall_curve(y_true, y_scores)
            # precision_recall_curve retorna un threshold menos que puntos
            thresholds = np.append(thresholds, 1.0)
            
            if metrica == 'f1':
                f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
                idx_optimo = np.argmax(f1_scores)
            else:  # balanced_accuracy
                # Aproximación usando precision y recall
                balanced_scores = (precision + recall) / 2
                idx_optimo = np.argmax(balanced_scores)
        
        return float(thresholds[idx_optimo]), float(youden_scores[idx_optimo] if metrica == 'youden' else 
                                                     (2 * (precision[idx_optimo] * recall[idx_optimo]) / 
                                                      (precision[idx_optimo] + recall[idx_optimo] + 1e-10) if metrica == 'f1' else
                                                      (precision[idx_optimo] + recall[idx_optimo]) / 2))


# =============================================================================
# 3. CLASIFICADORES
# =============================================================================

class ClasificadorBase:
    """
    Clase base para clasificadores supervisados.

    Todos los clasificadores reciben (X, y) y retornan:
        - modelo entrenado
        - predicciones
        - probabilidades (si aplica)
        - métricas de rendimiento
        - importancia de features (si aplica)
    """
    nombre = "clasificador_base"

    def __init__(self, 
                 manejar_desbalanceo: Literal['class_weight', 'smote', 'ninguna'] = 'class_weight',
                 calibrar_probabilidades: bool = False,
                 semilla: int = 42):
        """
        Args:
            manejar_desbalanceo: Estrategia para clases desbalanceadas
            calibrar_probabilidades: Si True, aplica Platt scaling (sigmoid) o isotónica
            semilla: Semilla para reproducibilidad
        """
        self.manejar_desbalanceo = manejar_desbalanceo
        self.calibrar_probabilidades = calibrar_probabilidades
        self.semilla = semilla
        self._modelo = None
        self._label_encoder = None
        self._manejador = ManejadorDesbalanceo(manejar_desbalanceo)
        self.classes_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'ClasificadorBase':
        """Entrena el clasificador. Debe implementarse en subclases."""
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predice etiquetas para X."""
        if self._modelo is None:
            raise RuntimeError("Llamar fit() primero.")
        return self._modelo.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predice probabilidades de clase para X.
        Si el modelo no soporta predict_proba, retorna one-hot de predicciones.
        """
        if self._modelo is None:
            raise RuntimeError("Llamar fit() primero.")
        
        if hasattr(self._modelo, 'predict_proba'):
            return self._modelo.predict_proba(X)
        else:
            # Fallback: crear matriz one-hot de predicciones
            preds = self.predict(X)
            n_clases = len(self.classes_)
            proba = np.zeros((len(preds), n_clases))
            for i, pred in enumerate(preds):
                proba[i, pred] = 1.0
            return proba

    def _preparar_y(self, y: np.ndarray) -> np.ndarray:
        """
        Prepara etiquetas: codifica strings a enteros si es necesario.
        
        Returns:
            Array de enteros y guarda el encoder para decode posterior
        """
        if y.dtype == object or y.dtype.kind in ['U', 'S']:
            self._label_encoder = LabelEncoder()
            y_encoded = self._label_encoder.fit_transform(y)
            self.classes_ = self._label_encoder.classes_
        else:
            y_encoded = y.astype(int)
            self.classes_ = np.unique(y)
        
        return y_encoded

    def decode_labels(self, y: np.ndarray) -> np.ndarray:
        """Decodifica etiquetas enteras a originales si se usó LabelEncoder."""
        if self._label_encoder is not None:
            return self._label_encoder.inverse_transform(y)
        return y

    def _validar_X_y(self, X: np.ndarray, y: np.ndarray) -> None:
        """Valida dimensiones y consistencia de X e y."""
        if not isinstance(X, np.ndarray) or X.ndim != 2:
            raise ValueError("X debe ser np.ndarray 2D (N, D).")
        if not isinstance(y, np.ndarray) or y.ndim != 1:
            raise ValueError("y debe ser np.ndarray 1D (N,).")
        if X.shape[0] != len(y):
            raise ValueError(f"X ({X.shape[0]}) e y ({len(y)}) deben tener mismo número de muestras.")
        if np.isnan(X).any():
            raise ValueError("X contiene NaN. Imputar antes de entrenar.")

    def get_importancia_features(self) -> Optional[pd.DataFrame]:
        """
        Retorna importancia de features si el modelo la soporta.
        
        Returns:
            DataFrame con 'feature_idx', 'importancia', o None si no aplica
        """
        return None

    def evaluar(self, X: np.ndarray, y_true: np.ndarray) -> Dict[str, Any]:
        """
        Evalúa el modelo en datos de test.

        Args:
            X: Features
            y_true: Etiquetas verdaderas

        Returns:
            Dict con métricas de EvaluadorClasificacion
        """
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)
        
        # Si y_true es string y tenemos encoder, codificar para comparar
        if y_true.dtype == object and self._label_encoder is not None:
            y_true_encoded = self._label_encoder.transform(y_true)
        else:
            y_true_encoded = y_true
        
        evaluador = EvaluadorClasificacion()
        return evaluador.evaluar_completo(y_true_encoded, y_pred, y_proba)


class SVMClasificador(ClasificadorBase):
    """
    Support Vector Machine para clasificación de fenotipos celulares.

    Algoritmo (C-SVC):
        Minimizar: (1/2)||w||² + C × Σ ξᵢ
        Sujeto a: yᵢ(w·xᵢ + b) ≥ 1 - ξᵢ, ξᵢ ≥ 0
        
        donde w es el vector normal al hiperplano, C penaliza errores,
        y ξᵢ son variables de holgura para puntos mal clasificados.

    Kernel RBF (Radial Basis Function):
        K(xᵢ, xⱼ) = exp(-γ × ||xᵢ - xⱼ||²)
        
        Mapea datos a espacio de alta dimensionalidad donde son linealmente
        separables. γ controla el "alcance" de influencia de cada muestra.

    Parámetros críticos:
        C: trade-off entre margen suave y clasificación correcta.
           C pequeño → margen amplio, más tolerante a outliers (underfitting)
           C grande → margen estrecho, menos tolerante (overfitting)
        gamma: coeficiente del kernel RBF.
               'scale': 1 / (n_features × X.var()) - recomendado
               'auto': 1 / n_features - legacy
               Valor alto → modelo complejo, riesgo de overfitting
               Valor bajo → modelo simple, riesgo de underfitting

    Complejidad: O(N² × D) a O(N³ × D) dependiendo de implementación y C.

    Ventajas:
        - Efectivo en espacios de alta dimensionalidad (D > N)
        - Versátil via kernels no lineales (RBF, polinomial)
        - Robusto a overfitting en alta dimensión (maximiza margen)
        - Soporta clasificación multiclase (one-vs-one por defecto)

    Desventajas:
        - No escala bien a N > 10,000 (cuadrático en muestras)
        - Requiere estandarización estricta (sensible a escalas)
        - Probabilidades requieren calibración adicional (Platt scaling)
        - Difícil de interpretar (no es tree-based)

    Usos microscopía:
        - Clasificación de fenotipos con alta dimensionalidad (miles de features)
        - Separación de poblaciones con fronteras no lineales complejas
        - Clasificación binaria de alta precisión (ej: célula sana vs tumoral)
        - Cuando N es moderado (< 5,000 células) y D es alto
    """
    nombre = "svm"

    def __init__(self,
                 C: float = 1.0,
                 kernel: Literal['linear', 'rbf', 'poly', 'sigmoid'] = 'rbf',
                 gamma: Union[Literal['scale', 'auto'], float] = 'scale',
                 class_weight: Union[Literal['balanced'], Dict, None] = 'balanced',
                 probability: bool = True,  # Necesario para predict_proba
                 **kwargs):
        """
        Args:
            C: Parámetro de regularización (positivo).
            kernel: Tipo de kernel.
            gamma: Coeficiente para kernels no lineales.
            class_weight: 'balanced' para auto-ajustar, dict manual, o None.
            probability: Si True, habilita predict_proba (más lento, usa Platt).
            **kwargs: Parámetros adicionales de SVC.
        """
        super().__init__(manejar_desbalanceo='class_weight' if class_weight else 'ninguna')
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.class_weight_param = class_weight
        self.probability_param = probability
        self.kwargs = kwargs

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'SVMClasificador':
        """
        Entrena SVM.

        Args:
            X: Matriz (N, D) estandarizada.
            y: Etiquetas (N,) - pueden ser strings o enteros.

        Returns:
            self
        """
        self._validar_X_y(X, y)
        y_encoded = self._preparar_y(y)
        
        # Determinar class_weight
        class_weight = self.class_weight_param
        if self.manejar_desbalanceo == 'class_weight' and class_weight == 'balanced':
            class_weight = self._manejador.calcular_class_weights(y_encoded)
        
        # Crear y entrenar modelo
        self._modelo = SVC(
            C=self.C,
            kernel=self.kernel,
            gamma=self.gamma,
            class_weight=class_weight,
            probability=self.probability_param,
            random_state=self.semilla,
            **self.kwargs
        )
        
        self._modelo.fit(X, y_encoded)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Probabilidades calibradas via Platt scaling (sigmoid).
        Solo disponible si probability=True en constructor.
        """
        if not self.probability_param:
            raise RuntimeError("SVC inicializado con probability=False. No hay predict_proba.")
        return super().predict_proba(X)

    def get_vectores_soporte(self) -> np.ndarray:
        """Retorna índices de vectores de soporte."""
        if self._modelo is None:
            raise RuntimeError("Llamar fit() primero.")
        return self._modelo.support_

    def get_n_vectores_soporte(self) -> int:
        """Número de vectores de soporte (indica complejidad del modelo)."""
        if self._modelo is None:
            raise RuntimeError("Llamar fit() primero.")
        return len(self._modelo.support_)


class RandomForestClasificador(ClasificadorBase):
    """
    Random Forest para clasificación de fenotipos celulares.

    Algoritmo (Breiman, 2001):
        1. Bootstrap: crear B muestras bootstrap del dataset (muestreo con reemplazo)
        2. Para cada muestra bootstrap:
           a. Construir árbol de decisión
           b. En cada nodo, seleccionar random subset de m features (m << D)
           c. Dividir nodo usando mejor feature del subset (criterio Gini o entropía)
        3. Predicción: votación mayoritaria entre todos los árboles

    Criterios de impureza:
        Gini: 1 - Σ(pᵢ²) donde pᵢ es proporción de clase i en nodo
              Mide probabilidad de clasificación incorrecta si se etiqueta aleatoriamente
              según distribución del nodo.
        
        Entropía: -Σ pᵢ × log(pᵢ)
                  Mide desorden/información del nodo.
        
        Ambos son similares; Gini ligeramente más rápido de calcular.

    Parámetros clave:
        n_estimators: número de árboles. Más árboles = mejor generalización hasta
                      punto de rendimientos decrecientes (default 100).
        max_depth: profundidad máxima. None = expandir hasta hojas puras.
                   Limitar reduce overfitting.
        min_samples_split: mínimo de muestras para dividir nodo interno.
                           Aumentar reduce overfitting.
        max_features: número de features a considerar en cada split.
                      'sqrt': √D (default, bueno para clasificación)
                      'log2': log₂(D)
                      None: D (bagging puro, no random subspace)

    Complejidad:
        Entrenamiento: O(B × N × log(N) × D) donde B = n_estimators
        Predicción: O(B × log(N))

    Ventajas:
        - No requiere estandarización (invariante a escalas, robusto a outliers)
        - Maneja bien datos faltantes (imputación implícita via surrogates)
        - Proporciona importancia de features nativa (MDI: Mean Decrease Impurity)
        - Rápido en predicción y entrenamiento paralelizable
        - Bajo riesgo de overfitting (promedio de muchos modelos)

    Desventajas:
        - Puede sobreajustar con ruido si árboles muy profundos
        - Sesgo hacia features con más categorías (en variables categóricas)
        - No extrapola bien fuera del rango de entrenamiento
        - Importancia MDI puede ser sesgada con features correlacionadas

    Usos microscopía:
        - Selección de features relevantes (qué métricas morfológicas predicen fenotipo)
        - Clasificación robusta con datos ruidosos (segmentación imperfecta)
        - Detección de interacciones no lineales entre features
        - Cuando se necesita interpretabilidad (árboles individuales inspeccionables)
        - Grandes datasets (N > 10,000 células) por eficiencia computacional
    """
    nombre = "random_forest"

    def __init__(self,
                 n_estimators: int = 100,
                 max_depth: Optional[int] = None,
                 min_samples_split: Union[int, float] = 2,
                 min_samples_leaf: Union[int, float] = 1,
                 max_features: Union[Literal['sqrt', 'log2'], int, float] = 'sqrt',
                 class_weight: Union[Literal['balanced', 'balanced_subsample'], Dict, None] = 'balanced',
                 criterio: Literal['gini', 'entropy', 'log_loss'] = 'gini',
                 bootstrap: bool = True,
                 oob_score: bool = False,  # Out-of-bag error
                 **kwargs):
        """
        Args:
            n_estimators: Número de árboles en el bosque.
            max_depth: Profundidad máxima de árboles.
            min_samples_split: Mínimo de muestras para dividir nodo interno.
            min_samples_leaf: Mínimo de muestras en hoja.
            max_features: Features a considerar en cada split.
            class_weight: Manejo de desbalanceo ('balanced', 'balanced_subsample').
            criterio: Función de impureza ('gini', 'entropy').
            bootstrap: Usar muestreo bootstrap.
            oob_score: Calcular error out-of-bag (requiere bootstrap=True).
            **kwargs: Parámetros adicionales de RandomForestClassifier.
        """
        super().__init__(manejar_desbalanceo='class_weight' if class_weight else 'ninguna')
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.class_weight_param = class_weight
        self.criterio = criterio
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.kwargs = kwargs
        self.oob_score_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RandomForestClasificador':
        """
        Entrena Random Forest.

        Args:
            X: Matriz (N, D). No requiere estandarización.
            y: Etiquetas (N,).

        Returns:
            self
        """
        self._validar_X_y(X, y)
        y_encoded = self._preparar_y(y)
        
        # Determinar class_weight
        class_weight = self.class_weight_param
        if self.manejar_desbalanceo == 'class_weight' and class_weight == 'balanced':
            class_weight = self._manejador.calcular_class_weights(y_encoded)
        
        self._modelo = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            class_weight=class_weight,
            criterion=self.criterio,
            bootstrap=self.bootstrap,
            oob_score=self.oob_score,
            random_state=self.semilla,
            n_jobs=-1,  # Usar todos los cores
            **self.kwargs
        )
        
        self._modelo.fit(X, y_encoded)
        
        if self.oob_score:
            self.oob_score_ = self._modelo.oob_score_
        
        return self

    def get_importancia_features(self) -> pd.DataFrame:
        """
        Importancia de features basada en Mean Decrease in Impurity (MDI).

        Returns:
            DataFrame con 'feature_idx', 'importancia', 'importancia_normalizada'
        """
        if self._modelo is None:
            raise RuntimeError("Llamar fit() primero.")
        
        importancias = self._modelo.feature_importances_
        df = pd.DataFrame({
            'feature_idx': range(len(importancias)),
            'importancia': importancias,
            'importancia_normalizada': importancias / importancias.sum()
        })
        df = df.sort_values('importancia', ascending=False).reset_index(drop=True)
        return df

    def get_importancia_permutacion(self, X: np.ndarray, y: np.ndarray, 
                                     n_repeats: int = 10) -> pd.DataFrame:
        """
        Importancia de features por permutación (más robusta que MDI).

        Algoritmo:
            1. Calcular score base en X, y
            2. Para cada feature j:
               a. Permutar aleatoriamente feature j en X
               b. Calcular score en X_permutado
               c. Importancia = score_base - score_permutado
            3. Repetir n_repeats veces para estabilidad

        Args:
            X: Features de validación
            y: Etiquetas de validación
            n_repeats: Número de repeticiones de permutación

        Returns:
            DataFrame con importancias y desviaciones estándar
        """
        if self._modelo is None:
            raise RuntimeError("Llamar fit() primero.")
        
        # Preparar y si es necesario
        if y.dtype == object and self._label_encoder is not None:
            y = self._label_encoder.transform(y)
        
        resultados = metrics.permutation_importance(
            self._modelo, X, y, n_repeats=n_repeats, random_state=self.semilla, n_jobs=-1
        )
        
        df = pd.DataFrame({
            'feature_idx': range(len(resultados.importances_mean)),
            'importancia': resultados.importances_mean,
            'std': resultados.importances_std
        })
        df = df.sort_values('importancia', ascending=False).reset_index(drop=True)
        return df

    def get_oob_score(self) -> Optional[float]:
        """Retorna Out-of-Bag score si fue calculado durante entrenamiento."""
        return self.oob_score_

    def predecir_con_intervalo(self, X: np.ndarray, 
                                nivel_confianza: float = 0.95) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predicción con intervalo de confianza basado en votación de árboles.

        Args:
            X: Features
            nivel_confianza: Nivel para intervalo (ej: 0.95 para 95%)

        Returns:
            Tupla (predicciones, limite_inferior, limite_superior) de proporciones
        """
        if self._modelo is None:
            raise RuntimeError("Llamar fit() primero.")
        
        # Obtener predicciones de todos los árboles
        todas_predicciones = np.array([tree.predict(X) for tree in self._modelo.estimators_])
        
        # Para cada muestra, calcular proporción de votos por clase
        predicciones = []
        limite_inf = []
        limite_sup = []
        
        alpha = 1 - nivel_confianza
        
        for i in range(X.shape[0]):
            votos = todas_predicciones[:, i]
            clases, counts = np.unique(votos, return_counts=True)
            props = counts / len(votos)
            
            # Clase ganadora
            pred_clase = clases[np.argmax(props)]
            predicciones.append(pred_clase)
            
            # Intervalo de confianza aproximado (percentiles de bootstrap)
            # Para la clase ganadora
            prop_ganadora = props[np.argmax(props)]
            # Intervalo binomial aproximado (Wilson)
            n = len(votos)
            z = 1.96 if nivel_confianza == 0.95 else 2.576  # 95% o 99%
            p = prop_ganadora
            denom = 1 + z**2/n
            centre = (p + z**2/(2*n)) / denom
            half_width = z * np.sqrt((p*(1-p) + z**2/(4*n)) / n) / denom
            
            limite_inf.append(max(0, centre - half_width))
            limite_sup.append(min(1, centre + half_width))
        
        return np.array(predicciones), np.array(limite_inf), np.array(limite_sup)


class LogisticRegressionClasificador(ClasificadorBase):
    """
    Regresión Logística para clasificación de fenotipos celulares.

    Algoritmo (Optimización):
        Maximizar verosimilitud logarítmica:
        L(w) = Σ [yᵢ log(σ(w·xᵢ + b)) + (1-yᵢ) log(1 - σ(w·xᵢ + b))]
        
        donde σ(z) = 1 / (1 + exp(-z)) es la función sigmoid.
        
        Con regularización L2 (Ridge):
        L_reg(w) = L(w) - (λ/2) ||w||²
        
        donde λ = 1/C (C es parámetro de inversa de regularización).

    Interpretación:
        log(p/(1-p)) = w·x + b  (log-odds o logit)
        p = 1 / (1 + exp(-(w·x + b)))
        
        Los coeficientes w indican cambio en log-odds por unidad de feature.
        Son directamente interpretables: w_j > 0 aumenta probabilidad de clase 1.

    Solvers:
        'lbfgs': Limited-memory BFGS, bueno para datasets pequeños/medios
        'liblinear': Coordinada descendente, soporta L1, bueno para sparse
        'sag': Stochastic Average Gradient, rápido para grandes datasets
        'saga': Extensión de SAG, soporta L1, ElasticNet, multinomial

    Regularización:
        L2 (Ridge): penaliza ||w||², evita overfitting, coeficientes pequeños
        L1 (Lasso): penaliza ||w||₁, produce sparse solutions (selección de features)
        ElasticNet: combinación α||w||₁ + (1-α)||w||²/2

    Ventajas:
        - Probabilidades bien calibradas nativamente (no requiere Platt scaling)
        - Rápido en entrenamiento y predicción (O(N × D))
        - Interpretable (coeficientes = impacto de cada feature en log-odds)
        - Funciona bien con pocos datos si D no es excesivo
        - Multiclase nativo (softmax para >2 clases)

    Desventajas:
        - Asume relación log-lineal (no captura interacciones complejas sin engineering)
        - Sensible a outliers (función logarítmica penaliza fuertemente predicciones erróneas)
        - Requiere estandarización para comparabilidad de coeficientes
        - Colinealidad perfecta causa inestabilidad numérica
        - Límites de decisión lineales (a menos que se expandan features)

    Usos microscopía:
        - Clasificación interpretable (reportar odds ratios de features morfológicas)
        - Screening inicial de features relevantes (L1 regularization)
        - Calibración de probabilidades para decisiones clínicas
        - Cuando se necesita entender qué métricas predicen el fenotipo
        - Baseline rápido antes de modelos complejos (SVM, RF)
    """
    nombre = "logistic_regression"

    def __init__(self,
                 C: float = 1.0,
                 penalty: Literal['l1', 'l2', 'elasticnet'] = 'l2',
                 solver: Literal['lbfgs', 'liblinear', 'sag', 'saga'] = 'lbfgs',
                 class_weight: Union[Literal['balanced'], Dict, None] = 'balanced',
                 max_iter: int = 1000,
                 l1_ratio: Optional[float] = None,  # Solo para elasticnet
                 multi_class: Literal['auto', 'ovr', 'multinomial'] = 'auto',
                 **kwargs):
        """
        Args:
            C: Inverso de fuerza de regularización (menor = más regularización).
            penalty: Tipo de regularización ('l1', 'l2', 'elasticnet').
            solver: Algoritmo de optimización.
            class_weight: Manejo de desbalanceo.
            max_iter: Máximo de iteraciones para convergencia.
            l1_ratio: Mezcla L1/L2 para elasticnet (0=puro L2, 1=puro L1).
            multi_class: Estrategia multiclase.
            **kwargs: Parámetros adicionales de LogisticRegression.
        """
        super().__init__(manejar_desbalanceo='class_weight' if class_weight else 'ninguna')
        self.C = C
        self.penalty = penalty
        self.solver = solver
        self.class_weight_param = class_weight
        self.max_iter = max_iter
        self.l1_ratio = l1_ratio
        self.multi_class = multi_class
        self.kwargs = kwargs
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LogisticRegressionClasificador':
        """
        Entrena Regresión Logística.

        Args:
            X: Matriz (N, D) estandarizada (crítico para convergencia e interpretación).
            y: Etiquetas (N,).

        Returns:
            self
        """
        self._validar_X_y(X, y)
        y_encoded = self._preparar_y(y)
        
        # Ajustar solver si es necesario según penalty
        if self.penalty == 'l1' and self.solver not in ['liblinear', 'saga']:
            self.solver = 'saga'
        if self.penalty == 'elasticnet' and self.solver != 'saga':
            self.solver = 'saga'
        
        # Determinar class_weight
        class_weight = self.class_weight_param
        if self.manejar_desbalanceo == 'class_weight' and class_weight == 'balanced':
            class_weight = self._manejador.calcular_class_weights(y_encoded)
        
        self._modelo = LogisticRegression(
            C=self.C,
            penalty=self.penalty,
            solver=self.solver,
            class_weight=class_weight,
            max_iter=self.max_iter,
            l1_ratio=self.l1_ratio,
            multi_class=self.multi_class,
            random_state=self.semilla,
            **self.kwargs
        )
        
        self._modelo.fit(X, y_encoded)
        self.coef_ = self._modelo.coef_
        self.intercept_ = self._modelo.intercept_
        
        return self

    def get_coeficientes(self) -> pd.DataFrame:
        """
        Retorna coeficientes del modelo (interpretables como log-odds ratios).

        Returns:
            DataFrame con 'feature_idx', 'coeficiente', 'odds_ratio', 'abs_coef'
        """
        if self.coef_ is None:
            raise RuntimeError("Llamar fit() primero.")
        
        # Para multiclase, coef_ tiene shape (n_classes, n_features)
        # Para binario, shape (1, n_features)
        coefs = self.coef_
        
        if coefs.shape[0] == 1:
            # Binario
            df = pd.DataFrame({
                'feature_idx': range(coefs.shape[1]),
                'coeficiente': coefs[0],
                'odds_ratio': np.exp(coefs[0]),
                'abs_coef': np.abs(coefs[0])
            })
        else:
            # Multiclase - retornar para cada clase
            data = []
            for i, clase in enumerate(self.classes_):
                for j in range(coefs.shape[1]):
                    data.append({
                        'clase': clase,
                        'feature_idx': j,
                        'coeficiente': coefs[i, j],
                        'odds_ratio': np.exp(coefs[i, j]),
                        'abs_coef': np.abs(coefs[i, j])
                    })
            df = pd.DataFrame(data)
        
        return df.sort_values('abs_coef', ascending=False).reset_index(drop=True)

    def get_prediccion_explicada(self, X: np.ndarray, 
                                  idx_muestra: int = 0) -> pd.DataFrame:
        """
        Descompone la predicción de una muestra en contribuciones de cada feature.

        Útil para explicar por qué el modelo predijo cierta clase.

        Args:
            X: Features (puede ser matriz completa, se usa idx_muestra)
            idx_muestra: Índice de la muestra a explicar

        Returns:
            DataFrame con contribuciones de cada feature al logit
        """
        if self.coef_ is None:
            raise RuntimeError("Llamar fit() primero.")
        
        x = X[idx_muestra]
        
        if self.coef_.shape[0] == 1:
            # Binario
            contribuciones = x * self.coef_[0]
            df = pd.DataFrame({
                'feature_idx': range(len(x)),
                'valor_feature': x,
                'coeficiente': self.coef_[0],
                'contribucion': contribuciones,
                'contribucion_abs': np.abs(contribuciones)
            })
            df['intercept'] = self.intercept_[0]
            df['logit_total'] = df['contribucion'].sum() + self.intercept_[0]
            df['probabilidad'] = 1 / (1 + np.exp(-df['logit_total'].iloc[0]))
        else:
            # Multiclase - calcular para cada clase
            data = []
            for i, clase in enumerate(self.classes_):
                contribuciones = x * self.coef_[i]
                logit = contribuciones.sum() + self.intercept_[i]
                data.append({
                    'clase': clase,
                    'logit': logit,
                    'probabilidad': None  # Se calculará después con softmax
                })
                for j, (val, coef, contrib) in enumerate(zip(x, self.coef_[i], contribuciones)):
                    data.append({
                        'clase': clase,
                        'feature_idx': j,
                        'valor_feature': val,
                        'coeficiente': coef,
                        'contribucion': contrib
                    })
            df = pd.DataFrame(data)
            # Calcular softmax para probabilidades
            logits = [d['logit'] for d in data if 'logit' in d]
            exp_logits = np.exp(logits - np.max(logits))
            probs = exp_logits / exp_logits.sum()
            # Añadir probabilidades al dataframe...
        
        return df.sort_values('contribucion_abs', ascending=False).reset_index(drop=True)


# =============================================================================
# 4. VALIDACIÓN CRUZADA Y SELECCIÓN DE HIPERPARÁMETROS
# =============================================================================

class ValidadorCruzado:
    """
    Validación cruzada estratificada y selección de hiperparámetros.

    Estratificación: mantiene proporción de clases en cada fold.
    Crítico para datos desbalanceados (evita folds sin clase minoritaria).

    Algoritmo StratifiedK-Fold:
        1. Dividir datos en K folds manteniendo distribución de clases
        2. Para cada fold i:
           a. Entrenar en K-1 folds
           b. Evaluar en fold i
        3. Promediar métricas de los K folds

    Grid Search:
        1. Definir grid de hiperparámetros
        2. Para cada combinación:
           a. Ejecutar validación cruzada
           b. Calcular métrica promedio
        3. Seleccionar combinación con mejor métrica

    Ventajas:
        - Estimación robusta de rendimiento generalizable
        - Uso eficiente de datos limitados
        - Comparación justa de modelos/hipérparámetros
        - Detección de overfitting (alta varianza entre folds)

    Desventajas:
        - Costoso computacionalmente (K × N_combinaciones entrenamientos)
        - Puede ser optimista si hay leakage entre muestras relacionadas
          (ej: células de la misma imagen en train y test)

    Usos microscopía:
        - Estimar rendimiento en nuevos experimentos/lotes
        - Seleccionar hiperparámetros robustos a variación biológica
        - Comparar SVM vs RF vs LR en datos específicos
        - Detectar si modelo memoriza artefactos técnicos (batch effects)
    """
    nombre = "validador_cruzado"

    def __init__(self,
                 n_splits: int = 5,
                 semilla: int = 42,
                 scoring: str = 'f1_weighted'):
        """
        Args:
            n_splits: Número de folds (K).
            semilla: Semilla para reproducibilidad.
            scoring: Métrica para optimización ('accuracy', 'f1_weighted', 'roc_auc', etc.)
        """
        self.n_splits = n_splits
        self.semilla = semilla
        self.scoring = scoring
        self.cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=semilla)

    def cross_val_score(self, 
                        modelo: Any, 
                        X: np.ndarray, 
                        y: np.ndarray) -> Dict[str, Any]:
        """
        Ejecuta validación cruzada estratificada.

        Args:
            modelo: Clasificador sklearn-compatible o instancia de ClasificadorBase
            X: Features (N, D)
            y: Etiquetas (N,)

        Returns:
            Dict con scores por fold, promedio, std, y predicciones OOF (out-of-fold)
        """
        # Si es nuestro clasificador base, extraer el modelo sklearn subyacente
        if isinstance(modelo, ClasificadorBase):
            # Necesitamos crear un wrapper o usar el modelo directamente
            # Para CV, usamos el modelo sklearn interno si ya está entrenado,
            # o mejor, creamos uno nuevo con los mismos parámetros
            sklearn_model = self._extract_sklearn_model(modelo)
        else:
            sklearn_model = modelo
        
        # Calcular scores
        scores = cross_val_score(sklearn_model, X, y, cv=self.cv, scoring=self.scoring, n_jobs=-1)
        
        # Obtener predicciones out-of-fold para análisis detallado
        oof_predictions = np.zeros(len(y))
        oof_probabilities = np.zeros((len(y), len(np.unique(y))))
        
        for fold_idx, (train_idx, val_idx) in enumerate(self.cv.split(X, y)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train = y[train_idx]
            
            # Clonar modelo para este fold
            fold_model = self._clone_model(sklearn_model)
            fold_model.fit(X_train, y_train)
            
            oof_predictions[val_idx] = fold_model.predict(X_val)
            if hasattr(fold_model, 'predict_proba'):
                oof_probabilities[val_idx] = fold_model.predict_proba(X_val)
        
        return {
            'scores_por_fold': scores.tolist(),
            'score_promedio': float(scores.mean()),
            'score_std': float(scores.std()),
            'oof_predictions': oof_predictions,
            'oof_probabilities': oof_probabilities,
            'n_splits': self.n_splits
        }

    def grid_search(self,
                    modelo: ClasificadorBase,
                    X: np.ndarray,
                    y: np.ndarray,
                    param_grid: Dict[str, List],
                    refit: bool = True) -> Dict[str, Any]:
        """
        Búsqueda exhaustiva de hiperparámetros con validación cruzada.

        Args:
            modelo: Instancia de clasificador base
            X: Features
            y: Etiquetas
            param_grid: Dict {parametro: [valores]}
            refit: Si True, reentrena con mejor params en todo el dataset

        Returns:
            Dict con mejores parámetros, score, y modelo reentrenado
        """
        sklearn_model = self._extract_sklearn_model(modelo)
        
        grid_search = GridSearchCV(
            sklearn_model,
            param_grid,
            cv=self.cv,
            scoring=self.scoring,
            n_jobs=-1,
            refit=refit,
            return_train_score=True
        )
        
        grid_search.fit(X, y)
        
        resultados = {
            'mejores_params': grid_search.best_params_,
            'mejor_score': float(grid_search.best_score_),
            'cv_results': pd.DataFrame(grid_search.cv_results_),
            'modelo_optimo': grid_search.best_estimator_ if refit else None
        }
        
        return resultados

    def _extract_sklearn_model(self, modelo: ClasificadorBase) -> Any:
        """Extrae el estimador sklearn subyacente de nuestros clasificadores."""
        # Crear nuevo modelo sklearn con mismos parámetros
        if isinstance(modelo, SVMClasificador):
            return SVC(
                C=modelo.C, kernel=modelo.kernel, gamma=modelo.gamma,
                class_weight='balanced' if modelo.manejar_desbalanceo == 'class_weight' else None,
                probability=True, random_state=modelo.semilla
            )
        elif isinstance(modelo, RandomForestClasificador):
            return RandomForestClassifier(
                n_estimators=modelo.n_estimators,
                max_depth=modelo.max_depth,
                class_weight='balanced' if modelo.manejar_desbalanceo == 'class_weight' else None,
                random_state=modelo.semilla, n_jobs=-1
            )
        elif isinstance(modelo, LogisticRegressionClasificador):
            return LogisticRegression(
                C=modelo.C, penalty=modelo.penalty, solver=modelo.solver,
                class_weight='balanced' if modelo.manejar_desbalanceo == 'class_weight' else None,
                max_iter=modelo.max_iter, random_state=modelo.semilla
            )
        else:
            raise ValueError("Tipo de modelo no soportado")

    def _clone_model(self, modelo: Any) -> Any:
        """Clona un modelo sklearn."""
        from sklearn.base import clone
        return clone(modelo)


# =============================================================================
# 5. PIPELINE DE CLASIFICACIÓN COMPLETO
# =============================================================================

def pipeline_clasificacion_completo(
    df: pd.DataFrame,
    columnas_features: List[str],
    columna_target: str,
    columna_id: str = 'imagen',
    metodo: Literal['svm', 'random_forest', 'logistic_regression'] = 'random_forest',
    params_clasificador: Optional[Dict] = None,
    test_size: float = 0.2,
    validacion_cruzada: bool = True,
    n_splits_cv: int = 5,
    estandarizar: bool = True,
    manejar_desbalanceo: Literal['class_weight', 'smote', 'ninguna'] = 'class_weight',
    semilla: int = 42
) -> Dict[str, Any]:
    """
    Pipeline completo de clasificación desde DataFrame.

    Args:
        df: DataFrame con features y etiquetas
        columnas_features: Lista de columnas predictoras
        columna_target: Columna con etiquetas a predecir
        columna_id: Columna identificadora de muestras
        metodo: Algoritmo de clasificación
        params_clasificador: Parámetros específicos del clasificador
        test_size: Proporción para test set (si no hay CV)
        validacion_cruzada: Si True, usa CV en lugar de train/test split
        n_splits_cv: Número de folds para CV
        estandarizar: Si True, aplica StandardScaler
        manejar_desbalanceo: Estrategia para clases desbalanceadas
        semilla: Semilla para reproducibilidad

    Returns:
        Dict con:
            - 'modelo': clasificador entrenado
            - 'metricas_train': métricas en entrenamiento
            - 'metricas_test': métricas en test (si aplica)
            - 'resultados_cv': resultados de validación cruzada (si aplica)
            - 'importancia_features': importancia si aplica
            - 'dataframe_predicciones': DataFrame con IDs y predicciones
            - 'X_procesada': matriz de features usada
            - 'y': etiquetas
    """
    from sklearn.model_selection import train_test_split
    
    # Preparar datos
    X = df[columnas_features].values
    y = df[columna_target].values
    
    # Manejar NaN
    if np.isnan(X).any():
        warnings.warn("X contiene NaN. Imputando con media...", UserWarning)
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy='mean')
        X = imputer.fit_transform(X)
    
    # Estandarizar
    scaler = None
    if estandarizar:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    
    # Dividir train/test si no hay CV
    if not validacion_cruzada:
        X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
            X, y, range(len(y)), test_size=test_size, random_state=semilla, stratify=y
        )
    else:
        X_train, y_train = X, y
        X_test, y_test = None, None
    
    # Crear clasificador
    params = params_clasificador or {}
    params['semilla'] = semilla
    
    if metodo == 'svm':
        clasificador = SVMClasificador(**params)
    elif metodo == 'random_forest':
        clasificador = RandomForestClasificador(**params)
    elif metodo == 'logistic_regression':
        clasificador = LogisticRegressionClasificador(**params)
    else:
        raise ValueError(f"Método no reconocido: {metodo}")
    
    resultados = {
        'metodo': metodo,
        'X_procesada': X,
        'y': y,
        'scaler': scaler,
        'columnas_features': columnas_features
    }
    
    # Validación cruzada
    if validacion_cruzada:
        validador = ValidadorCruzado(n_splits=n_splits_cv, semilla=semilla)
        cv_resultados = validador.cross_val_score(clasificador, X, y)
        resultados['resultados_cv'] = cv_resultados
        
        # Reentrenar en todo el dataset para modelo final
        clasificador.fit(X, y)
        resultados['metricas_train'] = clasificador.evaluar(X, y)
    else:
        # Train/test split tradicional
        clasificador.fit(X_train, y_train)
        resultados['metricas_train'] = clasificador.evaluar(X_train, y_train)
        resultados['metricas_test'] = clasificador.evaluar(X_test, y_test)
        
        # Predicciones en test
        y_pred = clasificador.predict(X_test)
        df_pred = pd.DataFrame({
            columna_id: df.iloc[idx_test][columna_id].values if columna_id in df.columns else idx_test,
            'y_true': y_test,
            'y_pred': clasificador.decode_labels(y_pred),
            'split': 'test'
        })
        resultados['dataframe_predicciones'] = df_pred
    
    # Importancia de features si aplica
    if hasattr(clasificador, 'get_importancia_features'):
        resultados['importancia_features'] = clasificador.get_importancia_features()
    
    # Coeficientes si es LR
    if isinstance(clasificador, LogisticRegressionClasificador):
        resultados['coeficientes'] = clasificador.get_coeficientes()
    
    resultados['modelo'] = clasificador
    return resultados


def comparar_clasificadores(
    df: pd.DataFrame,
    columnas_features: List[str],
    columna_target: str,
    clasificadores: List[Literal['svm', 'random_forest', 'logistic_regression']] = None,
    n_splits: int = 5,
    estandarizar: bool = True,
    semilla: int = 42
) -> pd.DataFrame:
    """
    Compara múltiples clasificadores con validación cruzada estratificada.

    Args:
        df: DataFrame con datos
        columnas_features: Features predictoras
        columna_target: Variable objetivo
        clasificadores: Lista de métodos a comparar
        n_splits: Folds para CV
        estandarizar: Si estandarizar features
        semilla: Semilla

    Returns:
        DataFrame comparativo con métricas de cada clasificador
    """
    clasificadores = clasificadores or ['svm', 'random_forest', 'logistic_regression']
    
    # Preparar datos
    X = df[columnas_features].values
    y = df[columna_target].values
    
    if np.isnan(X).any():
        from sklearn.impute import SimpleImputer
        X = SimpleImputer(strategy='mean').fit_transform(X)
    
    if estandarizar:
        X = StandardScaler().fit_transform(X)
    
    validador = ValidadorCruzado(n_splits=n_splits, semilla=semilla)
    resultados = []
    
    for nombre in clasificadores:
        try:
            if nombre == 'svm':
                modelo = SVMClasificador(probability=True)
            elif nombre == 'random_forest':
                modelo = RandomForestClasificador()
            elif nombre == 'logistic_regression':
                modelo = LogisticRegressionClasificador()
            else:
                continue
            
            cv_resultados = validador.cross_val_score(modelo, X, y)
            
            resultados.append({
                'clasificador': nombre,
                'accuracy_mean': cv_resultados['score_promedio'],
                'accuracy_std': cv_resultados['score_std'],
                'scores_folds': cv_resultados['scores_por_fold']
            })
            
        except Exception as e:
            warnings.warn(f"Error con {nombre}: {e}", UserWarning)
    
    return pd.DataFrame(resultados)