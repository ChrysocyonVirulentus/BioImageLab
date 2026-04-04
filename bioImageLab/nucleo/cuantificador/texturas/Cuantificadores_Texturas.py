"""
Cuantificadores de textura para objetos segmentados en microscopía.

La textura describe la distribución espacial de intensidades dentro de
un objeto: no sólo qué valores están presentes, sino cómo se organizan
en el espacio. Dos objetos con idéntico histograma pueden tener texturas
completamente distintas.

Convención de entrada (igual que Cuantificadores_Intensidades.py):

    img_segmentada : np.ndarray 2D (uint8, 0/255)
        Máscara binaria resultado de la etapa de segmentación.
        Define QUÉ región del objeto se analiza.

    img_procesada  : np.ndarray 2D (cualquier dtype numérico)
        Imagen de intensidades resultado de normalización/filtrado.
        Provee la SEÑAL sobre la que se calcula la textura.

Pipeline esperado:
    Adquisición → Normalización → Filtrado/Realce → Segmentación
                        ↓                ↓                 ↓
                    img_procesada    img_procesada    img_segmentada

Jerarquía de métodos implementados:

    Basados en estadística de segundo orden (relaciones entre pares de píxeles):
        - GLCM           : Matriz de Co-ocurrencia de Niveles de Gris
        - CaracteristicasHaralick : 13 features de Haralick derivadas de GLCM

    Basados en análisis de estructura local:
        - LBP            : Local Binary Pattern (histograma de patrones locales)

    Basados en filtros orientados en frecuencia:
        - FiltrosGabor   : Banco de filtros Gabor (frecuencia × orientación)

    Basados en estadísticas de corridas:
        - GLRLM          : Matriz de Longitud de Corrida de Niveles de Gris

    Basados en convolución con máscaras estructurales:
        - EnergiaLaws    : Energía de textura de Laws (5 máscaras × 5 máscaras)

IMPORTANTE - Separación de responsabilidades:
    - Estos métodos NO realizan normalización, filtrado ni segmentación.
    - La cuantización de niveles de gris (reducción de bins) se realiza
        internamente donde es necesaria, como paso previo al análisis.
    - El análisis siempre queda restringido a la región de la máscara.
"""

import numpy as np
import warnings
from typing import Dict, List, Optional, Tuple, Union


# Clase base


class CuantificadorTextura:
    """
        Clase base para cuantificadores de textura.

        Define la interfaz común y los métodos de validación/preprocesamiento
        compartidos por todos los cuantificadores derivados.

        Nota sobre cuantización:
            La mayoría de métodos de textura requieren imágenes con un número
            reducido de niveles de gris (8–256 bins) para que las matrices
            sean manejables. Internamente se discretiza img_procesada al rango
            [0, num_niveles-1] usando escalado min-max dentro de la máscara.
            Esto garantiza que la cuantización use toda la dinámica de la señal
            real del objeto, sin influencia del fondo.
    """
    nombre = "cuantificador_textura_base"

    def __call__(
        self,
        img_segmentada: np.ndarray,
        img_procesada: np.ndarray,
    ):
        """
            Aplica el cuantificador de textura.

            Args:
                img_segmentada: Máscara binaria 2D (0=fondo, >0=objeto)
                img_procesada:  Imagen de intensidades 2D a cuantificar

            Returns:
                Escalar, array o diccionario según el cuantificador (ver subclase)
        """
        raise NotImplementedError("Subclases deben implementar __call__")

    # ── Helpers compartidos ───────────────────────────────────

    def _validar_entradas(
        self,
        img_segmentada: np.ndarray,
        img_procesada: np.ndarray,
    ) -> None:
        """Valida dimensiones, forma y presencia de al menos un objeto."""
        if img_segmentada.ndim != 2:
            raise ValueError(
                f"img_segmentada debe ser 2D, tiene {img_segmentada.ndim} dims"
            )
        if img_procesada.ndim != 2:
            raise ValueError(
                f"img_procesada debe ser 2D, tiene {img_procesada.ndim} dims"
            )
        if img_segmentada.shape != img_procesada.shape:
            raise ValueError(
                f"Las imágenes deben tener la misma forma. "
                f"Segmentada: {img_segmentada.shape}, "
                f"Procesada:  {img_procesada.shape}"
            )
        if not np.any(img_segmentada > 0):
            raise ValueError(
                "img_segmentada no contiene píxeles de objeto (todos son 0). "
                "Verificar etapa de segmentación."
            )

    def _cuantizar_en_mascara(
        self,
        img_procesada: np.ndarray,
        img_segmentada: np.ndarray,
        num_niveles: int,
    ) -> np.ndarray:
        """
            Discretiza img_procesada a [0, num_niveles-1] usando min-max dentro
            de la máscara. Los píxeles de fondo quedan en 0 (sin efecto en análisis).

            La cuantización min-max dentro de la máscara asegura que toda la
            dinámica de señal del objeto sea representada uniformemente en los
            niveles disponibles, independientemente del rango absoluto.

            Args:
                img_procesada:  Imagen de intensidades 2D
                img_segmentada: Máscara binaria 2D
                num_niveles:    Número de niveles de gris destino (típico: 8, 16, 32, 256)

            Returns:
                Imagen 2D cuantizada (dtype uint8 o uint16 según num_niveles)
        """
        mask = img_segmentada > 0
        valores = img_procesada[mask].astype(np.float64)

        v_min = valores.min()
        v_max = valores.max()

        if v_max == v_min:
            # Imagen constante dentro de la máscara → nivel 0
            cuantizada = np.zeros_like(img_procesada, dtype=np.int32)
            return cuantizada

        img_float = np.where(
            mask,
            (img_procesada.astype(np.float64) - v_min) / (v_max - v_min),
            0.0,
        )

        cuantizada = np.floor(img_float * (num_niveles - 1)).astype(np.int32)
        cuantizada = np.clip(cuantizada, 0, num_niveles - 1)
        # Fondo siempre a 0; no participa en análisis
        cuantizada[~mask] = 0
        return cuantizada

    def _bbox_mascara(
        self,
        img_segmentada: np.ndarray,
    ) -> Tuple[int, int, int, int]:
        """
            Devuelve el bounding box de la máscara para recortar la región de interés.

            Returns:
                (fila_min, fila_max, col_min, col_max) con límites inclusivos
        """
        filas = np.any(img_segmentada > 0, axis=1)
        cols = np.any(img_segmentada > 0, axis=0)
        f_min, f_max = np.where(filas)[0][[0, -1]]
        c_min, c_max = np.where(cols)[0][[0, -1]]
        return int(f_min), int(f_max), int(c_min), int(c_max)


# GLCM — Matriz de Co-ocurrencia de Niveles de Gris
@registrar_en("cuantificacion")
class GLCM(CuantificadorTextura):
    """
        Matriz de Co-ocurrencia de Niveles de Gris (Gray-Level Co-occurrence Matrix).

        La GLCM, introducida por Haralick et al. (1973), captura la distribución
        estadística de pares de píxeles con una separación espacial y orientación
        definidas. Es el fundamento de la mayoría de descriptores de textura de
        segundo orden.

        Definición formal:
            Dada una imagen I cuantizada a G niveles y un desplazamiento (Δr, Δc):

            GLCM[i, j] = #{(r,c) : I(r,c)=i  ∧  I(r+Δr, c+Δc)=j  ∧  ambos ∈ máscara}
                        ─────────────────────────────────────────────────────────────
                                            N_pares

            donde N_pares es el número total de pares válidos (normalización a
            probabilidad conjunta).

        Desplazamientos y ángulos estándar (distancia d=1):
            0°   : (Δr=0,  Δc=1)
            45°  : (Δr=-1, Δc=1)
            90°  : (Δr=1,  Δc=0)
            135° : (Δr=1,  Δc=1)

        Simetría:
            Si simetrica=True se promedia GLCM con su transpuesta:
            GLCM_sim = (GLCM + GLCM^T) / 2
            Esto hace la matriz invariante a la dirección del par (i→j = j→i).

        Promediado angular:
            Si promediar_angulos=True, se devuelve el promedio de las GLCMs de
            todos los ángulos solicitados, obteniendo un descriptor isótropo
            (invariante a rotación aproximadamente).

        Ventajas:
            - Captura relaciones espaciales de segundo orden entre píxeles
            - Base teórica sólida (Haralick 1973, ampliamente validado)
            - Permite análisis direccional (anisotropía de textura)
            - Escalable a múltiples distancias y ángulos

        Desventajas:
            - Costo computacional O(N × G²): alto para G grande
            - Sensible al número de niveles de gris (G debe elegirse con cuidado)
            - Requiere suficientes píxeles en la máscara (N >> G²)
            - No captura información de orden superior

        Usos típicos en microscopía:
            - Análisis de cromatina nuclear (eucromatina vs. heterocromatina)
            - Clasificación de tipos celulares por textura de citoplasma
            - Caracterización de matrices extracelulares
            - Distinguir células en distintas fases del ciclo celular
            - Feature extraction para clasificadores de fenotipo celular (HCS)
    """
    nombre = "glcm"

    def __init__(
        self,
        distancias: List[int] = None,
        angulos: List[float] = None,
        num_niveles: int = 32,
        simetrica: bool = True,
        promediar_angulos: bool = False,
        normalizar: bool = True,
    ):
        """
            Args:
                distancias: Lista de distancias de desplazamiento en píxeles.
                        Por defecto: [1].
                        Distancias mayores capturan variaciones de textura a
                        escala más grande.

                angulos: Lista de ángulos en radianes.
                        Por defecto: [0, π/4, π/2, 3π/4] (los 4 estándar de Haralick).
                        Usar un solo ángulo para análisis direccional.

                num_niveles: Número de niveles de gris (G) tras cuantización.
                            Valores típicos: 8, 16, 32.
                            Regla práctica: G² << N_píxeles_objeto.
                            G pequeño → GLCM más estable, menos discriminativa.
                            G grande → más discriminativa, requiere más píxeles.

                simetrica: Si True, la GLCM se simetriza (GLCM + GLCM^T) / 2.
                        Hace las features independientes del sentido de la dirección.
                        Recomendado True para análisis de textura general.

                promediar_angulos: Si True, promedia las GLCMs de todos los ángulos
                                en una sola matriz isótropa. Si False, devuelve
                                una matriz por cada combinación (distancia, ángulo).

                normalizar: Si True, divide por el total de pares (probabilidades).
                        Si False, devuelve conteos enteros.
        """
        self.distancias = distancias if distancias is not None else [1]
        self.angulos = angulos if angulos is not None else [
            0, np.pi / 4, np.pi / 2, 3 * np.pi / 4
        ]
        if num_niveles < 2:
            raise ValueError("num_niveles debe ser >= 2")
        self.num_niveles = num_niveles
        self.simetrica = simetrica
        self.promediar_angulos = promediar_angulos
        self.normalizar = normalizar

    def __call__(
        self,
        img_segmentada: np.ndarray,
        img_procesada: np.ndarray,
    ) -> np.ndarray:
        """
            Calcula la(s) GLCM dentro de la máscara segmentada.

            Args:
                img_segmentada: Máscara binaria 2D
                img_procesada:  Imagen de intensidades 2D

            Returns:
                Si promediar_angulos=True:
                    Array (G, G, len(distancias)) con una GLCM isótropa por distancia.
                Si promediar_angulos=False:
                    Array (G, G, len(distancias), len(angulos)) con una GLCM por
                    cada combinación (distancia, ángulo).

                G = num_niveles
        """
        self._validar_entradas(img_segmentada, img_procesada)

        img_q = self._cuantizar_en_mascara(
            img_procesada, img_segmentada, self.num_niveles
        )
        mask = img_segmentada > 0
        G = self.num_niveles
        nd = len(self.distancias)
        na = len(self.angulos)

        matrices = np.zeros((G, G, nd, na), dtype=np.float64)

        for di, d in enumerate(self.distancias):
            for ai, angulo in enumerate(self.angulos):
                # Desplazamiento (Δfila, Δcol) para este ángulo y distancia
                delta_fila = int(round(-d * np.sin(angulo)))
                delta_col  = int(round( d * np.cos(angulo)))

                glcm = self._calcular_glcm_par(
                    img_q, mask, G, delta_fila, delta_col
                )

                if self.simetrica:
                    glcm = (glcm + glcm.T) / 2.0

                if self.normalizar:
                    total = glcm.sum()
                    if total > 0:
                        glcm /= total

                matrices[:, :, di, ai] = glcm

        if self.promediar_angulos:
            return matrices.mean(axis=3)  # (G, G, nd)

        return matrices  # (G, G, nd, na)

    @staticmethod
    def _calcular_glcm_par(
        img_q: np.ndarray,
        mask: np.ndarray,
        G: int,
        delta_fila: int,
        delta_col: int,
    ) -> np.ndarray:
        """
            Calcula la GLCM para un único desplazamiento (Δfila, Δcol).

            Solo cuenta pares donde AMBOS píxeles están dentro de la máscara.
            Esto asegura que el análisis no se contamina con pares objeto-fondo
            en los bordes de la segmentación.

            Args:
                img_q:       Imagen cuantizada 2D
                mask:        Máscara booleana 2D
                G:           Número de niveles de gris
                delta_fila:  Desplazamiento en filas
                delta_col:   Desplazamiento en columnas

            Returns:
                Matriz GLCM (G × G) de conteos
        """
        H, W = img_q.shape
        glcm = np.zeros((G, G), dtype=np.float64)

        # Calcular ventanas de origen y destino considerando el desplazamiento
        f0_ini = max(0, -delta_fila)
        f0_fin = min(H, H - delta_fila)
        c0_ini = max(0, -delta_col)
        c0_fin = min(W, W - delta_col)

        f1_ini = f0_ini + delta_fila
        f1_fin = f0_fin + delta_fila
        c1_ini = c0_ini + delta_col
        c1_fin = c0_fin + delta_col

        # Recortar regiones
        origen  = img_q[f0_ini:f0_fin, c0_ini:c0_fin]
        destino = img_q[f1_ini:f1_fin, c1_ini:c1_fin]
        mask_origen  = mask[f0_ini:f0_fin, c0_ini:c0_fin]
        mask_destino = mask[f1_ini:f1_fin, c1_ini:c1_fin]

        # Solo pares donde ambos están en la máscara
        ambos_en_mascara = mask_origen & mask_destino

        if not np.any(ambos_en_mascara):
            return glcm

        i_vals = origen[ambos_en_mascara]
        j_vals = destino[ambos_en_mascara]

        np.add.at(glcm, (i_vals, j_vals), 1)
        return glcm

# Características de Haralick
@registrar_en("cuantificacion")
class CaracteristicasHaralick(CuantificadorTextura):
    """
        Las 13 características de textura de Haralick derivadas de la GLCM.

        Haralick et al. (1973) propusieron un conjunto de medidas estadísticas
        calculadas sobre la GLCM para describir propiedades perceptuales de la
        textura: uniformidad, contraste, correlación, entropía, etc.

        Sean P(i,j) los elementos de la GLCM normalizada (probabilidades conjuntas),
        con G niveles de gris, μ la media, σ la desviación estándar, y:
            μₓ = Σᵢ Σⱼ i·P(i,j)     (media marginal fila)
            μᵧ = Σᵢ Σⱼ j·P(i,j)     (media marginal columna)
            σₓ = sqrt(Σᵢ Σⱼ (i-μₓ)²·P(i,j))
            σᵧ = sqrt(Σᵢ Σⱼ (j-μᵧ)²·P(i,j))

        Características implementadas:

        1. Energía (Angular Second Moment, ASM):
            ASM = Σᵢ Σⱼ P(i,j)²
            Alta → textura uniforme o con patrones repetitivos regulares

        2. Contraste:
            Contraste = Σᵢ Σⱼ (i-j)² · P(i,j)
            Alta → grandes diferencias locales de intensidad (textura rugosa)

        3. Correlación:
            Correlación = [Σᵢ Σⱼ i·j·P(i,j) - μₓ·μᵧ] / (σₓ·σᵧ)
            Alta → dependencia lineal entre pares de píxeles

        4. Homogeneidad (Inverse Difference Moment):
            IDM = Σᵢ Σⱼ P(i,j) / (1 + (i-j)²)
            Alta → pares similares predominan (textura suave)

        5. Suma de cuadrados (Varianza):
            Varianza = Σᵢ Σⱼ (i - μ)² · P(i,j)
            Mide dispersión de la distribución de intensidades

        6. Entropía:
            Entropía = -Σᵢ Σⱼ P(i,j) · log₂(P(i,j) + ε)
            Alta → distribución de pares muy variada (textura compleja)

        7. Media de suma (Sum Average):
            SumAvg = Σₖ k · Pₓ₊ᵧ(k),  k = 2..2G
            donde Pₓ₊ᵧ(k) = Σᵢ₊ⱼ₌ₖ P(i,j)

        8. Varianza de suma (Sum Variance):
            SumVar = Σₖ (k - SumAvg)² · Pₓ₊ᵧ(k)

        9. Entropía de suma (Sum Entropy):
            SumEnt = -Σₖ Pₓ₊ᵧ(k) · log₂(Pₓ₊ᵧ(k) + ε)

        10. Varianza de diferencia (Difference Variance):
            DiffVar = Σₖ k² · Pₓ₋ᵧ(k),  k = 0..G-1
            donde Pₓ₋ᵧ(k) = Σ|ᵢ₋ⱼ|₌ₖ P(i,j)

        11. Entropía de diferencia (Difference Entropy):
            DiffEnt = -Σₖ Pₓ₋ᵧ(k) · log₂(Pₓ₋ᵧ(k) + ε)

        12. Información de correlación 1 (IMC1):
            IMC1 = (HXY - HXY1) / max(HX, HY)
            donde HX, HY son entropías marginales y HXY1 es una medida cruzada

        13. Información de correlación 2 (IMC2):
            IMC2 = sqrt(1 - exp(-2(HXY2 - HXY)))

        Ventajas:
            - 13 features en un único cálculo sobre la GLCM
            - Cada feature tiene interpretación semántica clara
            - Estándar de facto en análisis de textura biomédica
            - Permiten análisis multidireccional (anisotropía)
            - Ampliamente validados en clasificación celular

        Desventajas:
            - Correlacionadas entre sí (no son 13 dimensiones independientes)
            - Sensibles a la elección de G (num_niveles) y distancia
            - Requieren suficientes píxeles para GLCM estable (N >> G²)
            - IMC1 e IMC2 son matemáticamente complejas y menos usadas

        Usos típicos en microscopía:
            - Clasificación de tipos celulares (HCS, High Content Screening)
            - Análisis de cromatina: condensada vs. descondensada
            - Detección de apoptosis (reorganización de cromatina)
            - Caracterización de textura citoplasmática
            - Features para clasificadores SVM, Random Forest, redes neuronales
    """
    nombre = "haralick"

    def __init__(
        self,
        distancias: List[int] = None,
        angulos: List[float] = None,
        num_niveles: int = 32,
        promediar_angulos: bool = True,
        features: Optional[List[str]] = None,
    ):
        """
            Args:
                distancias: Distancias de desplazamiento (ver GLCM). Por defecto: [1].
                angulos:    Ángulos en radianes (ver GLCM). Por defecto: 4 estándar.
                num_niveles: Niveles de gris para cuantización. Por defecto: 32.
                promediar_angulos: Si True (recomendado), devuelve el promedio de
                                features sobre todos los ángulos (descriptor isótropo).
                                Si False, devuelve features por cada ángulo.
                features: Lista de nombres de features a calcular. Si None, calcula
                        las 13 de Haralick. Opciones: 'energia', 'contraste',
                        'correlacion', 'homogeneidad', 'varianza', 'entropia',
                        'suma_media', 'suma_varianza', 'suma_entropia',
                        'diff_varianza', 'diff_entropia', 'imc1', 'imc2'.
        """
        self.distancias = distancias if distancias is not None else [1]
        self.angulos = angulos if angulos is not None else [
            0, np.pi / 4, np.pi / 2, 3 * np.pi / 4
        ]
        if num_niveles < 2:
            raise ValueError("num_niveles debe ser >= 2")
        self.num_niveles = num_niveles
        self.promediar_angulos = promediar_angulos

        _todas = [
            'energia', 'contraste', 'correlacion', 'homogeneidad', 'varianza',
            'entropia', 'suma_media', 'suma_varianza', 'suma_entropia',
            'diff_varianza', 'diff_entropia', 'imc1', 'imc2',
        ]
        if features is None:
            self.features = _todas
        else:
            invalidas = set(features) - set(_todas)
            if invalidas:
                raise ValueError(
                    f"Features no reconocidas: {invalidas}. "
                    f"Opciones válidas: {_todas}"
                )
            self.features = features

        # Calculador de GLCM interno
        self._glcm_calc = GLCM(
            distancias=self.distancias,
            angulos=self.angulos,
            num_niveles=self.num_niveles,
            simetrica=True,
            promediar_angulos=False,
            normalizar=True,
        )

    def __call__(
        self,
        img_segmentada: np.ndarray,
        img_procesada: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """
            Calcula las características de Haralick dentro de la máscara.

            Args:
                img_segmentada: Máscara binaria 2D
                img_procesada:  Imagen de intensidades 2D

            Returns:
                Diccionario {nombre_feature: valor_o_array}.
                Si promediar_angulos=True, cada valor es un array de shape
                (len(distancias),) — una media sobre ángulos por distancia.
                Si promediar_angulos=False, shape (len(distancias), len(angulos)).
        """
        self._validar_entradas(img_segmentada, img_procesada)

        # matrices: (G, G, nd, na)
        matrices = self._glcm_calc(img_segmentada, img_procesada)
        G  = self.num_niveles
        nd = len(self.distancias)
        na = len(self.angulos)

        # Índices i, j como matrices (G×G) para operaciones vectorizadas
        ii = np.arange(G, dtype=np.float64)
        jj = np.arange(G, dtype=np.float64)
        I, J = np.meshgrid(ii, jj, indexing='ij')  # (G, G)
        eps = np.finfo(np.float64).tiny  # evitar log(0)

        resultados: Dict[str, np.ndarray] = {}

        # Calcular cada feature solicitada
        for di in range(nd):
            for ai in range(na):
                P = matrices[:, :, di, ai]  # (G, G)

                # Distribuciones marginales
                px = P.sum(axis=1)  # (G,) marginal fila
                py = P.sum(axis=0)  # (G,) marginal columna

                mu_x = np.sum(ii * px)
                mu_y = np.sum(jj * py)
                sig_x = np.sqrt(np.sum((ii - mu_x) ** 2 * px))
                sig_y = np.sqrt(np.sum((jj - mu_y) ** 2 * py))

                # Pₓ₊ᵧ: distribución de suma i+j, k ∈ [0, 2(G-1)]
                p_suma = np.zeros(2 * G, dtype=np.float64)
                for i_idx in range(G):
                    for j_idx in range(G):
                        p_suma[i_idx + j_idx] += P[i_idx, j_idx]

                # Pₓ₋ᵧ: distribución de diferencia |i-j|, k ∈ [0, G-1]
                p_diff = np.zeros(G, dtype=np.float64)
                for i_idx in range(G):
                    for j_idx in range(G):
                        p_diff[abs(i_idx - j_idx)] += P[i_idx, j_idx]

                calcs = self._calcular_features_una_glcm(
                    P, I, J, px, py, mu_x, mu_y, sig_x, sig_y,
                    p_suma, p_diff, G, eps,
                )

                for nombre in self.features:
                    key = nombre
                    val = calcs[nombre]
                    if key not in resultados:
                        resultados[key] = np.zeros((nd, na), dtype=np.float64)
                    resultados[key][di, ai] = val

        if self.promediar_angulos:
            resultados = {k: v.mean(axis=1) for k, v in resultados.items()}

        return resultados

    def _calcular_features_una_glcm(
        self,
        P, I, J, px, py, mu_x, mu_y, sig_x, sig_y,
        p_suma, p_diff, G, eps,
    ) -> Dict[str, float]:
        """Calcula todas las features para una única GLCM (G×G)."""
        ii_k = np.arange(G, dtype=np.float64)
        calcs = {}

        if 'energia' in self.features:
            calcs['energia'] = float(np.sum(P ** 2))

        if 'contraste' in self.features:
            calcs['contraste'] = float(np.sum((I - J) ** 2 * P))

        if 'correlacion' in self.features:
            if sig_x < eps or sig_y < eps:
                calcs['correlacion'] = 0.0
            else:
                calcs['correlacion'] = float(
                    (np.sum(I * J * P) - mu_x * mu_y) / (sig_x * sig_y)
                )

        if 'homogeneidad' in self.features:
            calcs['homogeneidad'] = float(np.sum(P / (1 + (I - J) ** 2)))

        if 'varianza' in self.features:
            mu = np.sum(I * P)
            calcs['varianza'] = float(np.sum((I - mu) ** 2 * P))

        if 'entropia' in self.features:
            calcs['entropia'] = float(-np.sum(P * np.log2(P + eps)))

        k_suma = np.arange(2 * G, dtype=np.float64)
        if 'suma_media' in self.features:
            calcs['suma_media'] = float(np.sum(k_suma * p_suma))

        if 'suma_varianza' in self.features:
            sm = np.sum(k_suma * p_suma)
            calcs['suma_varianza'] = float(np.sum((k_suma - sm) ** 2 * p_suma))

        if 'suma_entropia' in self.features:
            calcs['suma_entropia'] = float(-np.sum(p_suma * np.log2(p_suma + eps)))

        k_diff = np.arange(G, dtype=np.float64)
        if 'diff_varianza' in self.features:
            calcs['diff_varianza'] = float(np.sum(k_diff ** 2 * p_diff))

        if 'diff_entropia' in self.features:
            calcs['diff_entropia'] = float(-np.sum(p_diff * np.log2(p_diff + eps)))

        if 'imc1' in self.features or 'imc2' in self.features:
            HXY  = -np.sum(P * np.log2(P + eps))
            HX   = -np.sum(px * np.log2(px + eps))
            HY   = -np.sum(py * np.log2(py + eps))
            # HXY1: Σᵢ Σⱼ P(i,j) log(px(i)*py(j))
            outer = np.outer(px, py)
            HXY1 = -np.sum(P * np.log2(outer + eps))
            HXY2 = -np.sum(outer * np.log2(outer + eps))

            if 'imc1' in self.features:
                denom = max(HX, HY)
                calcs['imc1'] = float(
                    (HXY - HXY1) / denom if denom > eps else 0.0
                )
            if 'imc2' in self.features:
                arg = -2.0 * (HXY2 - HXY)
                calcs['imc2'] = float(np.sqrt(max(0.0, 1.0 - np.exp(arg))))

        return calcs


# LBP — Local Binary Pattern
@registrar_en("cuantificacion")
class LBP(CuantificadorTextura):
    """
        Patron Binario Local (Local Binary Pattern) dentro de la máscara.

        LBP, propuesto por Ojala et al. (1994, extendido 2002), es un descriptor
        de textura que codifica la microestructura local de la imagen comparando
        cada píxel con sus vecinos en un círculo de radio R con P puntos.

        Algoritmo (versión uniforme y circular):
            Para cada píxel central c con valor Ic:
            1. Muestrear P vecinos en un círculo de radio R mediante interpolación bilineal:
                xₚ = xc + R·cos(2πp/P)
                yₚ = yc - R·sin(2πp/P)
            2. Comparar cada vecino con el centro:
                bₚ = 1 si I(xₚ,yₚ) ≥ Ic, else 0
            3. Formar el código binario circular:
                LBP_{P,R}(c) = Σₚ bₚ · 2ᵖ

        Patrones uniformes (Ojala 2002):
            Un patrón es "uniforme" si tiene ≤ 2 transiciones 0→1 o 1→0 en el
            código circular. Son los patrones más frecuentes y estables en
            imágenes naturales (P+2 patrones uniformes, 1 patrón "no-uniforme").
            Reducen dimensionalidad y aumentan robustez al ruido.

        Invarianza a rotación:
            En modo 'nri_uniform' (no-rotation-invariant uniform), el código se
            mantiene tal cual → sensible a orientación.
            En modo 'uniform', se usa el código mínimo de rotación → invariante.

        Resultado:
            Histograma de códigos LBP dentro de la máscara, que actúa como
            descriptor de la microestructura local de la textura.

        Ventajas:
            - Invariante a cambios de iluminación monotónicos (escala de grises)
            - Computacionalmente eficiente
            - Invariante a rotación (modo uniform)
            - Robusto al ruido en modo uniforme
            - No requiere parámetros de textura a priori

        Desventajas:
            - Solo captura microestructura (vecindad pequeña)
            - No captura relaciones de largo alcance
            - Pérdida de información espacial al resumir en histograma
            - Sensible al radio R y número de puntos P elegidos

        Usos típicos en microscopía:
            - Reconocimiento de patrones de cromatina (condensada/laxa)
            - Clasificación de tipos celulares por textura
            - Detección de mitosis (patrón de cromatina condensada)
            - Análisis de textura citoplasmática (gránulos, retículo)
            - Descriptor local para redes neuronales convolucionales (features)
    """
    nombre = "lbp"

    def __init__(
        self,
        radio: int = 1,
        num_puntos: int = 8,
        metodo: str = 'uniform',
        normalizar_histograma: bool = True,
    ):
        """
            Args:
                radio: Radio R del círculo de vecindad en píxeles.
                    Valores típicos:
                        R=1, P=8  : microestructura fina (el más común)
                        R=2, P=16 : escala intermedia
                        R=3, P=24 : escala más gruesa
                    R mayor = captura patrones de textura más grandes.

                num_puntos: Número de puntos P en el círculo.
                        Debe ser múltiplo de 8 por convención.
                        Mayor P → más discriminativo pero más sensible a ruido.

                metodo: Método de codificación LBP.
                    'default'    : LBP estándar (2^P bins)
                    'uniform'    : Solo patrones uniformes (P+2 bins), invariante rotación
                    'nri_uniform': Uniforme no-invariante a rotación (P+2 bins)
                    'var'        : Varianza del vecindario (escalar, no histograma)

                normalizar_histograma: Si True, el histograma se normaliza a suma=1
                                    (distribución de probabilidad).
                                    Si False, se devuelven conteos enteros.
        """
        if radio < 1:
            raise ValueError("radio debe ser >= 1")
        if num_puntos < 4:
            raise ValueError("num_puntos debe ser >= 4")
        if metodo not in ('default', 'uniform', 'nri_uniform', 'var'):
            raise ValueError(
                "metodo debe ser 'default', 'uniform', 'nri_uniform' o 'var'"
            )
        self.radio = radio
        self.num_puntos = num_puntos
        self.metodo = metodo
        self.normalizar_histograma = normalizar_histograma

    def __call__(
        self,
        img_segmentada: np.ndarray,
        img_procesada: np.ndarray,
    ) -> np.ndarray:
        """
            Calcula el histograma LBP dentro de la máscara segmentada.

            Args:
                img_segmentada: Máscara binaria 2D
                img_procesada:  Imagen de intensidades 2D

            Returns:
                Array 1D con el histograma de códigos LBP de los píxeles del objeto.
                Longitud del histograma según metodo:
                    'default'    : 2^P bins
                    'uniform'    : P+2 bins
                    'nri_uniform': P+2 bins
                    'var'        : 1 escalar (varianza media de vecindarios)

            Nota:
                Usa skimage.feature.local_binary_pattern que implementa LBP
                circular con interpolación bilineal (Ojala 2002).
        """
        from skimage.feature import local_binary_pattern

        self._validar_entradas(img_segmentada, img_procesada)

        # Normalizar img_procesada a uint8 para LBP
        img_q = self._cuantizar_en_mascara(
            img_procesada, img_segmentada, num_niveles=256
        ).astype(np.uint8)

        # Calcular mapa LBP completo
        lbp_map = local_binary_pattern(
            img_q, self.num_puntos, self.radio, method=self.metodo
        )

        mask = img_segmentada > 0
        valores_lbp = lbp_map[mask]

        if self.metodo == 'var':
            # En modo 'var', LBP devuelve varianza local (no código binario)
            return np.array([float(np.mean(valores_lbp))])

        # Determinar número de bins según método
        if self.metodo == 'default':
            num_bins = 2 ** self.num_puntos
        else:
            # uniform y nri_uniform: P+2 bins (P uniformes + 1 no-uniforme)
            num_bins = self.num_puntos + 2

        histograma, _ = np.histogram(
            valores_lbp,
            bins=num_bins,
            range=(0, num_bins),
        )

        if self.normalizar_histograma:
            total = histograma.sum()
            if total > 0:
                histograma = histograma.astype(np.float64) / total

        return histograma.astype(np.float64)


# Filtros de Gabor
@registrar_en("cuantificacion")
class FiltrosGabor(CuantificadorTextura):
    """
        Banco de filtros de Gabor 2D para análisis de textura orientada en frecuencia.

        Un filtro de Gabor es una función gaussiana modulada por una onda sinusoidal.
        Responde selectivamente a contenido de frecuencia espacial en una orientación
        y escala específicas, siendo el modelo más cercano a las células simples de
        la corteza visual primaria (V1).

        Función de Gabor (en el espacio):
            g(x,y; λ, θ, ψ, σ, γ) = exp(-(x'²+γ²y'²)/(2σ²)) · cos(2πx'/λ + ψ)
            donde:
                x' = x·cos(θ) + y·sin(θ)    (coordenada rotada)
                y' = -x·sin(θ) + y·cos(θ)
                λ  = longitud de onda de la portadora sinusoidal (píxeles)
                θ  = orientación del filtro (radianes)
                ψ  = fase de la portadora (0 → simétrico, π/2 → antisimétrico)
                σ  = desviación estándar de la envolvente gaussiana
                γ  = relación de aspecto espacial (elipticidad)

        Banco de filtros:
            Se evalúan múltiples combinaciones de (frecuencia, orientación).
            La respuesta de energía de Gabor en cada punto es:
                E(x,y; λ,θ) = sqrt(R_par(x,y)² + R_impar(x,y)²)
            donde R_par usa ψ=0 (parte real) y R_impar usa ψ=π/2 (parte imaginaria).

        Para cada filtro (λ, θ) se extraen estadísticas sobre la máscara:
            media(E), desviación(E) → 2 valores por filtro → 2 × Nλ × Nθ features totales

        Ventajas:
            - Multi-escala y multi-orientación en un único descriptor
            - Sensible a anisotropía (diferente respuesta según dirección)
            - Buena localización simultánea en espacio y frecuencia
            - Robusto al ruido de alta frecuencia

        Desventajas:
            - Computacionalmente costoso para bancos grandes
            - Múltiples hiperparámetros (λ, θ, σ, γ) que requieren ajuste
            - La superposición de filtros puede generar features redundantes
            - No invariante a rotación por defecto (requiere promediado explícito)

        Usos típicos en microscopía:
            - Detección de estructuras orientadas (filamentos, microtúbulos)
            - Análisis de orientación preferencial en tejidos
            - Clasificación de tipos celulares por textura orientada
            - Segmentación de fibras en tejido conectivo
            - Extracción de features multi-escala para deep learning
    """
    nombre = "filtros_gabor"

    def __init__(
        self,
        frecuencias: List[float] = None,
        orientaciones: List[float] = None,
        sigma_x: float = 1.0,
        sigma_y: float = 1.0,
        n_std: float = 3.0,
    ):
        """
            Args:
                frecuencias: Lista de frecuencias espaciales (ciclos/píxel).
                            Por defecto: [0.1, 0.2, 0.3].
                            Rango válido: (0, 0.5] (Nyquist a 0.5).
                            Frecuencias bajas → texturas gruesas.
                            Frecuencias altas → texturas finas.

                orientaciones: Lista de orientaciones en radianes.
                            Por defecto: [0, π/4, π/2, 3π/4].
                            Si se usan todas las orientaciones y se promedian
                            las features, el descriptor es isótropo.

                sigma_x: Desviación estándar de la gaussiana en el eje x (paralelo a onda).
                        Controla la extensión espacial del filtro.
                        Mayor sigma_x → mayor selectividad de frecuencia.

                sigma_y: Desviación estándar en el eje y (perpendicular a onda).
                        Mayor sigma_y → menor selectividad de orientación.

                n_std: Número de desviaciones estándar para el tamaño del kernel.
                    Valor típico: 3.0. Aumentar si los filtros están cortados.
        """
        self.frecuencias = frecuencias if frecuencias is not None else [0.1, 0.2, 0.3]
        self.orientaciones = orientaciones if orientaciones is not None else [
            0, np.pi / 4, np.pi / 2, 3 * np.pi / 4
        ]
        if sigma_x <= 0 or sigma_y <= 0:
            raise ValueError("sigma_x y sigma_y deben ser > 0")
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y
        self.n_std = n_std

    def __call__(
        self,
        img_segmentada: np.ndarray,
        img_procesada: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """
            Calcula respuestas del banco de filtros de Gabor dentro de la máscara.

            Args:
                img_segmentada: Máscara binaria 2D
                img_procesada:  Imagen de intensidades 2D

            Returns:
                Diccionario con:
                    'media'    : Array (Nf, Nθ) con media de la energía de Gabor
                    'desviacion': Array (Nf, Nθ) con σ de la energía de Gabor
                Donde Nf = len(frecuencias), Nθ = len(orientaciones).
                Aplanar ambos arrays para obtener el vector de features de Gabor:
                    features = np.concatenate([res['media'].ravel(),
                                            res['desviacion'].ravel()])
        """
        from scipy.ndimage import convolve

        self._validar_entradas(img_segmentada, img_procesada)

        img_float = img_procesada.astype(np.float64)
        # Normalizar amplitud globalmente para que las respuestas sean comparables
        rango = img_float.max() - img_float.min()
        if rango > 0:
            img_norm = (img_float - img_float.min()) / rango
        else:
            img_norm = np.zeros_like(img_float)

        mask = img_segmentada > 0
        nf = len(self.frecuencias)
        no = len(self.orientaciones)

        medias      = np.zeros((nf, no), dtype=np.float64)
        desviaciones = np.zeros((nf, no), dtype=np.float64)

        for fi, freq in enumerate(self.frecuencias):
            for oi, theta in enumerate(self.orientaciones):
                kernel_real, kernel_imag = self._construir_kernel_gabor(
                    freq, theta
                )
                # Respuesta par (coseno) e impar (seno)
                resp_real = convolve(img_norm, kernel_real, mode='reflect')
                resp_imag = convolve(img_norm, kernel_imag, mode='reflect')

                # Energía: módulo de la respuesta compleja
                energia = np.sqrt(resp_real ** 2 + resp_imag ** 2)

                valores = energia[mask]
                medias[fi, oi]       = np.mean(valores)
                desviaciones[fi, oi] = np.std(valores, ddof=1) if len(valores) > 1 else 0.0

        return {'media': medias, 'desviacion': desviaciones}

    def _construir_kernel_gabor(
        self, frecuencia: float, theta: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
            Construye los kernels de Gabor par (coseno) e impar (seno).

            Args:
                frecuencia: Frecuencia espacial (ciclos/píxel)
                theta:      Orientación (radianes)

            Returns:
                Tupla (kernel_real, kernel_imag) — partes coseno y seno
        """
        sigma_x = self.sigma_x
        sigma_y = self.sigma_y

        # Tamaño del kernel: n_std desviaciones estándar en ambas direcciones
        half = int(np.ceil(self.n_std * max(sigma_x, sigma_y)))
        y_range = np.arange(-half, half + 1, dtype=np.float64)
        x_range = np.arange(-half, half + 1, dtype=np.float64)
        X, Y = np.meshgrid(x_range, y_range)

        # Rotación de coordenadas
        X_rot =  X * np.cos(theta) + Y * np.sin(theta)
        Y_rot = -X * np.sin(theta) + Y * np.cos(theta)

        # Envolvente gaussiana
        gauss = np.exp(
            -(X_rot ** 2 / (2 * sigma_x ** 2) + Y_rot ** 2 / (2 * sigma_y ** 2))
        )

        # Normalización DC para respuesta cero ante campo uniforme
        omega = 2 * np.pi * frecuencia
        kernel_real = gauss * np.cos(omega * X_rot)
        kernel_imag = gauss * np.sin(omega * X_rot)

        # Eliminar componente DC del kernel real
        kernel_real -= kernel_real.mean()

        return kernel_real, kernel_imag


# GLRLM — Matriz de Longitud de Corrida
@registrar_en("cuantificacion")
class GLRLM(CuantificadorTextura):
    """
        Matriz de Longitud de Corrida de Niveles de Gris (Gray-Level Run-Length Matrix).

        Propuesta por Galloway (1975), la GLRLM describe la distribución de
        corridas — secuencias de píxeles consecutivos con el mismo nivel de gris
        a lo largo de una dirección definida.

        Definición formal:
            R(i, l; θ) = número de corridas de nivel i con longitud l en dirección θ

        La GLRLM es una matriz (G × L_max) donde:
            G     = número de niveles de gris
            L_max = longitud máxima de corrida (acotado por la dimensión de la imagen)

        Features derivadas (Galloway 1975; Chu et al. 1990):

        1. SRE (Short Run Emphasis, énfasis en corridas cortas):
            SRE = Σᵢ Σₗ R(i,l) / l²  / N_r
            Alta → predominan corridas cortas (textura fina o irregular)

        2. LRE (Long Run Emphasis, énfasis en corridas largas):
            LRE = Σᵢ Σₗ R(i,l) · l²  / N_r
            Alta → predominan corridas largas (textura gruesa o periódica)

        3. GLN (Gray-Level Non-uniformity):
            GLN = Σᵢ (Σₗ R(i,l))² / N_r
            Baja → distribución uniforme entre niveles de gris

        4. RLN (Run-Length Non-uniformity):
            RLN = Σₗ (Σᵢ R(i,l))² / N_r
            Baja → longitudes de corrida distribuidas uniformemente

        5. RP (Run Percentage, fracción de corridas sobre total de píxeles):
            RP = N_r / N_p
            Alta → muchas corridas cortas (textura fina); Baja → pocas largas (homogénea)

        6. LGRE (Low Gray-Level Run Emphasis):
            LGRE = Σᵢ Σₗ R(i,l) / i² / N_r
            Alta → corridas de baja intensidad dominan

        7. HGRE (High Gray-Level Run Emphasis):
            HGRE = Σᵢ Σₗ R(i,l) · i² / N_r
            Alta → corridas de alta intensidad dominan

        Ventajas:
            - Captura periodicidad y continuidad de la textura
            - Diferencia textura fina (SRE) de textura gruesa (LRE)
            - Computacionalmente más rápida que GLCM para imágenes grandes
            - Sensible a dirección (análisis de anisotropía)

        Desventajas:
            - Sensible al número de niveles de gris G
            - La longitud máxima de corrida depende del tamaño del objeto
            - Menos información de relaciones entre píxeles que GLCM
            - Escasa validación en algunos dominios respecto a Haralick

        Usos típicos en microscopía:
            - Análisis de dirección preferencial en tejidos fibrilares
            - Caracterización de homogeneidad de distribución nuclear
            - Detección de estructuras lineales (filamentos de actina, microtúbulos)
            - Análisis de periodicidad en patrones de cromatina
            - Features complementarias a Haralick para clasificadores
    """
    nombre = "glrlm"

    def __init__(
        self,
        angulos: List[float] = None,
        num_niveles: int = 16,
        promediar_angulos: bool = True,
    ):
        """
            Args:
                angulos: Ángulos de dirección en radianes.
                        Por defecto: [0, π/4, π/2, 3π/4] (cuatro direcciones).
                        Usar un único ángulo para análisis direccional.

                num_niveles: Niveles de gris para cuantización.
                            Por defecto: 16. Valores menores → GLRLM más estable.

                promediar_angulos: Si True, promedia las 7 features sobre todos
                                los ángulos (descriptor isótropo).
                                Si False, devuelve features por ángulo.
        """
        self.angulos = angulos if angulos is not None else [
            0, np.pi / 4, np.pi / 2, 3 * np.pi / 4
        ]
        if num_niveles < 2:
            raise ValueError("num_niveles debe ser >= 2")
        self.num_niveles = num_niveles
        self.promediar_angulos = promediar_angulos

    def __call__(
        self,
        img_segmentada: np.ndarray,
        img_procesada: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """
            Calcula las features GLRLM dentro de la máscara segmentada.

            Args:
                img_segmentada: Máscara binaria 2D
                img_procesada:  Imagen de intensidades 2D

            Returns:
                Diccionario con 7 features: 'sre', 'lre', 'gln', 'rln', 'rp',
                'lgre', 'hgre'. Cada valor es un array de shape (na,) si
                promediar_angulos=False, o un escalar si promediar_angulos=True.
        """
        self._validar_entradas(img_segmentada, img_procesada)

        img_q = self._cuantizar_en_mascara(
            img_procesada, img_segmentada, self.num_niveles
        )
        mask = img_segmentada > 0
        na = len(self.angulos)

        nombres_features = ['sre', 'lre', 'gln', 'rln', 'rp', 'lgre', 'hgre']
        resultados = {k: np.zeros(na, dtype=np.float64) for k in nombres_features}

        for ai, angulo in enumerate(self.angulos):
            glrlm_mat = self._construir_glrlm(img_q, mask, angulo)
            feats = self._calcular_features_glrlm(glrlm_mat)
            for nombre in nombres_features:
                resultados[nombre][ai] = feats[nombre]

        if self.promediar_angulos:
            resultados = {k: float(v.mean()) for k, v in resultados.items()}

        return resultados

    def _construir_glrlm(
        self,
        img_q: np.ndarray,
        mask: np.ndarray,
        angulo: float,
    ) -> np.ndarray:
        """
            Construye la GLRLM para un ángulo dado.

            Recorre la imagen en la dirección del ángulo, extrayendo corridas
            de píxeles consecutivos con el mismo nivel de gris dentro de la máscara.

            Args:
                img_q:  Imagen cuantizada 2D
                mask:   Máscara booleana 2D
                angulo: Ángulo de barrido en radianes

            Returns:
                Matriz GLRLM (num_niveles × L_max)
        """
        H, W = img_q.shape
        G = self.num_niveles

        # Dirección de desplazamiento unitario para el ángulo dado
        d_col = int(round(np.cos(angulo)))
        d_fila = int(round(-np.sin(angulo)))

        # Si ambos son 0 (ángulo no estándar), usar horizontal por defecto
        if d_col == 0 and d_fila == 0:
            d_col = 1

        L_max = max(H, W)
        glrlm = np.zeros((G, L_max), dtype=np.float64)

        # Encontrar píxeles de inicio: dentro de la máscara y sin predecesor en máscara
        for f in range(H):
            for c in range(W):
                if not mask[f, c]:
                    continue

                # Verificar si es inicio de corrida (no tiene predecesor en máscara)
                f_prev = f - d_fila
                c_prev = c - d_col
                tiene_predecesor = (
                    0 <= f_prev < H and
                    0 <= c_prev < W and
                    mask[f_prev, c_prev]
                )
                if tiene_predecesor:
                    continue

                # Seguir la corrida desde este píxel
                nivel_inicio = img_q[f, c]
                longitud = 1
                f_cur, c_cur = f + d_fila, c + d_col

                while (
                    0 <= f_cur < H and
                    0 <= c_cur < W and
                    mask[f_cur, c_cur] and
                    img_q[f_cur, c_cur] == nivel_inicio
                ):
                    longitud += 1
                    f_cur += d_fila
                    c_cur += d_col

                if longitud <= L_max:
                    glrlm[nivel_inicio, longitud - 1] += 1

        return glrlm

    @staticmethod
    def _calcular_features_glrlm(
        R: np.ndarray,
    ) -> Dict[str, float]:
        """
            Calcula las 7 features de Galloway a partir de la GLRLM.

            Args:
                R: Matriz GLRLM (G × L_max), sin normalizar

            Returns:
                Diccionario con las 7 features
        """
        G, L_max = R.shape
        eps = np.finfo(np.float64).tiny

        ii = np.arange(1, G + 1, dtype=np.float64)[:, np.newaxis]      # (G,1) niveles base 1
        ll = np.arange(1, L_max + 1, dtype=np.float64)[np.newaxis, :]   # (1,L) longitudes base 1

        N_r = R.sum()      # total de corridas
        N_p = (R * ll).sum()  # total de píxeles en corridas

        if N_r == 0:
            return {k: 0.0 for k in ['sre', 'lre', 'gln', 'rln', 'rp', 'lgre', 'hgre']}

        sre  = np.sum(R / (ll ** 2)) / N_r
        lre  = np.sum(R * (ll ** 2)) / N_r
        gln  = np.sum(R.sum(axis=1) ** 2) / N_r
        rln  = np.sum(R.sum(axis=0) ** 2) / N_r
        rp   = N_r / (N_p + eps)
        lgre = np.sum(R / (ii ** 2)) / N_r
        hgre = np.sum(R * (ii ** 2)) / N_r

        return {
            'sre': float(sre),
            'lre': float(lre),
            'gln': float(gln),
            'rln': float(rln),
            'rp':  float(rp),
            'lgre': float(lgre),
            'hgre': float(hgre),
        }


# Energía de Laws
@registrar_en("cuantificacion")
class EnergiaLaws(CuantificadorTextura):
    """
        Energía de textura de Laws (Laws' Texture Energy Measures).

        Laws (1980) propuso un banco de pequeños filtros 2D construidos como
        productos externos de 5 vectores 1D que detectan distintas propiedades
        locales de la textura: niveles (L), bordes (E), manchas (S), ondas (W)
        y rizado (R).

        Vectores 1D base (de 5 elementos):
            L5 = [ 1,  4,  6,  4,  1]  → Nivel (suavizado gaussiano)
            E5 = [-1, -2,  0,  2,  1]  → Borde (gradiente)
            S5 = [-1,  0,  2,  0, -1]  → Mancha (Laplaciano)
            W5 = [-1,  2,  0, -2,  1]  → Onda (oscilación)
            R5 = [ 1, -4,  6, -4,  1]  → Rizado (alta frecuencia)

        Filtros 2D (producto externo F = v₁ᵀ · v₂):
            Se generan 25 filtros 5×5. En la práctica se usan 14 combinaciones
            simétricamente únicas: L5E5, L5S5, L5R5, E5S5, E5E5, S5S5, etc.

        Cálculo de energía de textura:
            1. Convolucionar imagen con cada filtro F_{ab}: response_{ab} = I * F_{ab}
            2. Calcular la energía local con una ventana de suavizado W:
            E_{ab}(x,y) = (|response_{ab}| * W)(x,y)
            3. Extraer estadísticas de E_{ab} dentro de la máscara

        Interpretación de filtros clave:
            L5E5 / E5L5 : Bordes horizontales / verticales
            L5S5 / S5L5 : Manchas horizontales / verticales
            E5E5        : Puntos de esquina (edges en ambas direcciones)
            S5S5        : Manchas isótropas (blobs)
            R5R5        : Rizado 2D de alta frecuencia

        Ventajas:
            - Filtros muy pequeños (5×5) → rápidos y eficientes
            - Interpretación física clara de cada filtro
            - Captura múltiples tipos de primitiva de textura
            - Ampliamente usados en análisis de textura de tejidos
            - Robustos al ruido si se usa suavizado de energía adecuado

        Desventajas:
            - Kernels fijos (no adaptativos a la imagen)
            - Sólo capturan textura a escala de 5×5 píxeles
            - Muchas features correlacionadas entre sí
            - No invariante a rotación sin combinar filtros transpuestos

        Usos típicos en microscopía:
            - Análisis de textura de tejido histológico (H&E)
            - Clasificación de grado tumoral
            - Caracterización de distribución de colágeno
            - Segmentación de regiones de textura similar
            - Features de textura para análisis de imagen computacional clínica
    """
    nombre = "energia_laws"

    # Vectores 1D base de Laws (normalizados)
    _L5 = np.array([ 1,  4,  6,  4,  1], dtype=np.float64)
    _E5 = np.array([-1, -2,  0,  2,  1], dtype=np.float64)
    _S5 = np.array([-1,  0,  2,  0, -1], dtype=np.float64)
    _W5 = np.array([-1,  2,  0, -2,  1], dtype=np.float64)
    _R5 = np.array([ 1, -4,  6, -4,  1], dtype=np.float64)

    _VECTORES = {'L5': _L5, 'E5': _E5, 'S5': _S5, 'W5': _W5, 'R5': _R5}

    def __init__(
        self,
        filtros: Optional[List[str]] = None,
        tamaño_ventana_energia: int = 15,
        combinar_transpuestos: bool = True,
        estadisticas: Optional[List[str]] = None,
    ):
        """
            Args:
                filtros: Lista de nombres de filtros 2D a calcular.
                        Formato: 'AB' donde A,B ∈ {L5,E5,S5,W5,R5}.
                        Por defecto (None): usa los 9 filtros más usados:
                        ['L5E5','L5S5','L5R5','E5S5','E5E5','S5S5','E5L5','S5L5','R5R5']

                tamaño_ventana_energia: Tamaño del kernel de suavizado para calcular
                                    la energía local (media del valor absoluto).
                                    Debe ser impar. Valores típicos: 7–21.
                                    Mayor ventana → energía más suavizada.

                combinar_transpuestos: Si True, para filtros no-simétricos (ej. L5E5
                                    y E5L5) promedia la energía del filtro y su
                                    transpuesto para obtener un descriptor isótropo.
                                    Si False, se tratan como filtros independientes.

                estadisticas: Lista de estadísticas a extraer de la energía dentro de
                            la máscara. Por defecto: ['media', 'desviacion'].
                            Opciones: 'media', 'desviacion', 'maximo', 'entropia'.
        """
        _filtros_default = [
            'L5E5', 'L5S5', 'L5R5', 'E5S5', 'E5E5', 'S5S5', 'E5L5', 'S5L5', 'R5R5'
        ]
        self.filtros = filtros if filtros is not None else _filtros_default

        # Validar nombres de filtros
        nombres_validos = {a + b for a in self._VECTORES for b in self._VECTORES}
        for f in self.filtros:
            if f not in nombres_validos:
                raise ValueError(
                    f"Filtro '{f}' no reconocido. Formato: 'AB' donde "
                    f"A,B ∈ {set(self._VECTORES.keys())}"
                )

        if tamaño_ventana_energia < 1 or tamaño_ventana_energia % 2 == 0:
            raise ValueError("tamaño_ventana_energia debe ser impar y >= 1")
        self.tamaño_ventana_energia = tamaño_ventana_energia
        self.combinar_transpuestos = combinar_transpuestos

        _stats_validas = {'media', 'desviacion', 'maximo', 'entropia'}
        self.estadisticas = estadisticas if estadisticas is not None else ['media', 'desviacion']
        invalidas = set(self.estadisticas) - _stats_validas
        if invalidas:
            raise ValueError(
                f"Estadísticas no válidas: {invalidas}. Opciones: {_stats_validas}"
            )

    def __call__(
        self,
        img_segmentada: np.ndarray,
        img_procesada: np.ndarray,
    ) -> Dict[str, Dict[str, float]]:
        """
            Calcula la energía de textura de Laws dentro de la máscara.

            Args:
                img_segmentada: Máscara binaria 2D
                img_procesada:  Imagen de intensidades 2D

            Returns:
                Diccionario anidado: {nombre_filtro: {estadistica: valor}}
                Ejemplo:
                    {
                        'L5E5': {'media': 12.3, 'desviacion': 4.1},
                        'E5E5': {'media': 8.7,  'desviacion': 2.3},
                        ...
                    }
        """
        from scipy.ndimage import uniform_filter, convolve

        self._validar_entradas(img_segmentada, img_procesada)

        img_float = img_procesada.astype(np.float64)
        rango = img_float.max() - img_float.min()
        img_norm = (img_float - img_float.min()) / rango if rango > 0 else img_float * 0.0

        # Eliminar media local para invarianza a iluminación uniforme
        img_media_removida = img_norm - uniform_filter(
            img_norm, size=self.tamaño_ventana_energia
        )

        mask = img_segmentada > 0
        resultados = {}

        for nombre_filtro in self.filtros:
            nombre_a = nombre_filtro[:2]  # ej: 'L5'
            nombre_b = nombre_filtro[2:]  # ej: 'E5'

            v_a = self._VECTORES[nombre_a]
            v_b = self._VECTORES[nombre_b]

            # Kernel 2D: producto externo v_a ⊗ v_b (filas × columnas)
            kernel = np.outer(v_a, v_b)

            respuesta = convolve(img_media_removida, kernel, mode='reflect')

            if self.combinar_transpuestos and nombre_a != nombre_b:
                # Transpuesto: intercambiar vectores fila/columna
                kernel_t = np.outer(v_b, v_a)
                respuesta_t = convolve(img_media_removida, kernel_t, mode='reflect')
                respuesta = (respuesta + respuesta_t) / 2.0

            # Energía local: suavizar |respuesta|
            energia = uniform_filter(
                np.abs(respuesta), size=self.tamaño_ventana_energia
            )

            valores = energia[mask]
            stats = {}

            if 'media' in self.estadisticas:
                stats['media'] = float(np.mean(valores))
            if 'desviacion' in self.estadisticas:
                stats['desviacion'] = float(
                    np.std(valores, ddof=1) if len(valores) > 1 else 0.0
                )
            if 'maximo' in self.estadisticas:
                stats['maximo'] = float(np.max(valores))
            if 'entropia' in self.estadisticas:
                # Entropía aproximada via histograma de 32 bins
                hist, _ = np.histogram(valores, bins=32, density=True)
                hist = hist[hist > 0]
                stats['entropia'] = float(-np.sum(hist * np.log2(hist + np.finfo(float).tiny)))

            resultados[nombre_filtro] = stats

        return resultados