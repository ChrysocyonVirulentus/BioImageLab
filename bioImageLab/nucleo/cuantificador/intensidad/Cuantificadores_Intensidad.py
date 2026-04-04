"""
Cuantificadores de intensidad para objetos segmentados en microscopía.

Estos cuantificadores miden propiedades de la distribución de intensidades
dentro de las regiones definidas por una máscara de segmentación binaria.
Cada método recibe dos imágenes:

    - img_segmentada: Máscara binaria (0/255) resultado de la etapa de
        segmentación (binarización). Define QUÉ píxeles pertenecen a objetos.
    - img_procesada: Imagen de intensidades resultado de la etapa de
        normalización, filtrado o transformación. Define la SEÑAL a medir.

Esta separación es fundamental: la segmentación define la geometría,
y la imagen procesada provee la señal cuantificable.

Flujo de pipeline esperado:
    Adquisición → Normalización → Filtrado/Realce → Segmentación
                        ↓                ↓                 ↓
                    img_procesada    img_procesada     img_segmentada

Concepto de máscara booleana:
    mask = img_segmentada > 0       # True en píxeles de objetos
    pixeles_objeto = img_procesada[mask]  # Extrae sólo esos valores

Cuantificadores disponibles:
    - MediaIntensidad      : Media aritmética dentro de la máscara
    - IntensidadIntegrada  : Suma total de intensidades (densidad óptica)
    - MaximoIntensidad     : Valor máximo dentro de la máscara
    - MinimoIntensidad     : Valor mínimo dentro de la máscara
    - MedianaIntensidad    : Mediana (robusta a outliers)
    - DesviacionEstandar   : Dispersión de intensidades
    - CoeficienteVariacion : CV = σ/μ × 100, variabilidad relativa
    - PercentilIntensidad  : Percentil arbitrario de la distribución
    - RelacionSenialRuido  : SNR = μ_objeto / σ_fondo
    - AsimetriaIntensidad  : Skewness, forma de la distribución
    - CurtosisIntensidad   : Kurtosis, "picudez" de la distribución
    - PerfilLineal         : Perfil de intensidad a lo largo de una línea

IMPORTANTE - Separación de responsabilidades:
    - Estos métodos NO realizan normalización, filtrado ni segmentación.
    - Asumen que img_segmentada es binaria (0 y 255) o al menos 0/no-0.
    - Asumen que img_procesada tiene los valores de señal a cuantificar.
"""

import numpy as np
from typing import Optional, Tuple, Union
import warnings


# Clase base

class CuantificadorIntensidad:
    """
        Clase base para cuantificadores de intensidad.

        Define la interfaz común y métodos de validación compartidos
        por todos los cuantificadores derivados.

        Convención de entrada:
            img_segmentada : np.ndarray 2D, dtype uint8
                Máscara binaria. Valores esperados: 0 (fondo) y 255 (objeto).
                También acepta cualquier imagen donde 0 = fondo y != 0 = objeto.
            img_procesada  : np.ndarray 2D, cualquier dtype numérico
                Imagen de intensidades sobre la que se mide.
                Debe tener la misma forma (shape) que img_segmentada.
    """
    nombre = "cuantificador_intensidad_base"

    def __call__(
        self,
        img_segmentada: np.ndarray,
        img_procesada: np.ndarray,
    ):
        """
            Aplica el cuantificador.

            Args:
                img_segmentada: Máscara binaria 2D (0 = fondo, >0 = objeto)
                img_procesada:  Imagen de intensidades 2D a cuantificar

            Returns:
                Escalar o estructura según el cuantificador (ver subclase)
        """
        raise NotImplementedError("Subclases deben implementar __call__")

    # Helpers de validación

    def _validar_entradas(
        self,
        img_segmentada: np.ndarray,
        img_procesada: np.ndarray,
    ) -> None:
        """Valida dimensiones, forma y presencia de objetos."""
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
                f"Procesada: {img_procesada.shape}"
            )

    def _extraer_pixeles_objeto(
        self,
        img_segmentada: np.ndarray,
        img_procesada: np.ndarray,
    ) -> np.ndarray:
        """
            Extrae los valores de img_procesada en posiciones de objeto.

            Args:
                img_segmentada: Máscara binaria
                img_procesada:  Imagen de intensidades

            Returns:
                Array 1D con valores de intensidad de los píxeles de objeto.

            Raises:
                ValueError: Si la máscara no contiene ningún píxel de objeto.
        """
        mask = img_segmentada > 0
        if not np.any(mask):
            raise ValueError(
                "img_segmentada no contiene píxeles de objeto (todos son 0). "
                "Verificar etapa de segmentación."
            )
        return img_procesada[mask].astype(np.float64)


# Cuantificadores de tendencia central
@registrar_en("cuantificacion")
class MediaIntensidad(CuantificadorIntensidad):
    """
        Media aritmética de intensidades en la región segmentada.

        La media de intensidad es el estimador más común de la señal
        central de un objeto. En fluorescencia, es proporcional a la
        concentración promedio del fluoróforo dentro del objeto.

        Ecuación:
            μ = (1/N) Σᵢ I(xᵢ, yᵢ)
            donde la suma recorre los N píxeles del objeto (mask > 0)

        Ventajas:
            - Intuitiva y universalmente utilizada
            - Sensible a toda la distribución de intensidades
            - Base para otros cálculos estadísticos
            - Computacionalmente eficiente

        Desventajas:
            - Sensible a outliers (píxeles saturados o muy oscuros)
            - No captura la heterogeneidad intra-objeto
            - Puede ser engañosa para distribuciones bimodales
            - Depende de normalización previa

        Usos típicos en microscopía:
            - Cuantificación de expresión proteica media por célula
            - Comparación de niveles de señal entre condiciones
            - Normalización de intensidad integrada por área
            - Estimación de concentración local de fluoróforo
            - Análisis de activación nuclear media (p.ej. NF-kB translocation)
    """
    nombre = "media_intensidad"

    def __init__(self):
        """Inicializa el cuantificador (sin parámetros)."""
        pass

    def __call__(
        self,
        img_segmentada: np.ndarray,
        img_procesada: np.ndarray,
    ) -> float:
        """
            Calcula la media de intensidad dentro de la máscara.

            Args:
                img_segmentada: Máscara binaria 2D
                img_procesada:  Imagen de intensidades 2D

            Returns:
                Media aritmética de intensidades (float)
        """
        self._validar_entradas(img_segmentada, img_procesada)
        pixeles = self._extraer_pixeles_objeto(img_segmentada, img_procesada)
        return float(np.mean(pixeles))

@registrar_en("cuantificacion")
class MedianaIntensidad(CuantificadorIntensidad):
    """
        Mediana de intensidades en la región segmentada.

        Estimador robusto de la tendencia central. A diferencia de la media,
        la mediana no se ve afectada por valores extremos (saturación, hot pixels,
        ruido impulsional). Es el percentil 50 de la distribución.

        Propiedad de robustez:
            Si hasta el 50% de los píxeles son atípicos (outliers), la mediana
            sigue siendo un estimador válido del valor típico. La media, en cambio,
            puede desplazarse significativamente con un solo outlier.

        Ecuación:
            mediana = valor tal que P(I ≤ mediana) = 0.5
            (valor central cuando los píxeles están ordenados)

        Ventajas:
            - Robusta ante saturación y hot pixels
            - No requiere suposición de distribución simétrica
            - Estable ante contaminación por ruido impulsional
            - Útil cuando la distribución es asimétrica

        Desventajas:
            - Menos eficiente estadísticamente que la media (distribución normal)
            - Computacionalmente más costosa (requiere ordenamiento)
            - Ignora información de los extremos de la distribución
            - Menos sensible a cambios sutiles de señal

        Usos típicos en microscopía:
            - Imágenes con posible saturación de píxeles
            - Fluorescencia con hot pixels o artefactos puntuales
            - Cuando se sospecha contaminación de señal en bordes
            - Análisis de núcleos con estructuras internas brillantes
            - Estimador alternativo cuando media es no representativa
    """
    nombre = "mediana_intensidad"

    def __init__(self):
        """Inicializa el cuantificador (sin parámetros)."""
        pass

    def __call__(
        self,
        img_segmentada: np.ndarray,
        img_procesada: np.ndarray,
    ) -> float:
        """
            Calcula la mediana de intensidad dentro de la máscara.

            Args:
                img_segmentada: Máscara binaria 2D
                img_procesada:  Imagen de intensidades 2D

            Returns:
                Mediana de intensidades (float)
        """
        self._validar_entradas(img_segmentada, img_procesada)
        pixeles = self._extraer_pixeles_objeto(img_segmentada, img_procesada)
        return float(np.median(pixeles))

# Cuantificadores de magnitud total
@registrar_en("cuantificacion")
class IntensidadIntegrada(CuantificadorIntensidad):
    """
        Intensidad integrada (suma total) de la región segmentada.

        También conocida como "Integrated Density" en ImageJ/FIJI. Es la suma
        de todos los valores de intensidad dentro de la máscara. Equivale
        conceptualmente a la densidad óptica total o la cantidad total
        de fluoróforo en el objeto.

        Fundamento físico:
            En microscopía de fluorescencia, la intensidad de cada píxel es
            proporcional al número de fotones detectados, que a su vez es
            proporcional a la concentración local de fluoróforo y al volumen
            del vóxel. La intensidad integrada es por lo tanto proporcional
            a la cantidad total de fluoróforo (o proteína marcada) dentro del
            objeto, independientemente de su forma o tamaño.

        Ecuación:
            IntDen = Σᵢ I(xᵢ, yᵢ)    (suma sobre todos los píxeles del objeto)

        Relación con media:
            IntDen = μ × N
            donde N es el número de píxeles (área) del objeto

        Ventajas:
            - Proporcional a cantidad total de señal/fluoróforo
            - Independiente de la forma del objeto
            - Estándar en cuantificación de Western blots digitales
            - Permite comparación de contenido total entre células de distinto tamaño

        Desventajas:
            - Fuertemente dependiente del área: objetos más grandes dan más IntDen
            - Requiere segmentación precisa (bordes imprecisos afectan mucho)
            - Sensible a señal de fondo no segmentada incluida en la máscara
            - No informativa sin normalizar por área si los objetos varían en tamaño

        Usos típicos en microscopía:
            - Cuantificación de Western blots y dot blots digitalizados
            - Estimación de contenido de DNA (tinción de Feulgen)
            - Medición de contenido proteico total por célula
            - Comparación de carga de fluorescencia entre células
            - Análisis de intensidad de spots (FISH, smFISH)
    """
    nombre = "intensidad_integrada"

    def __init__(self, normalizar_por_area: bool = False):
        """
            Args:
                normalizar_por_area: Si True, divide la suma por el número de
                                    píxeles del objeto (equivale a la media).
                                    Útil para separar cantidad total de densidad.
        """
        self.normalizar_por_area = normalizar_por_area

    def __call__(
        self,
        img_segmentada: np.ndarray,
        img_procesada: np.ndarray,
    ) -> float:
        """
            Calcula la intensidad integrada dentro de la máscara.

            Args:
                img_segmentada: Máscara binaria 2D
                img_procesada:  Imagen de intensidades 2D

            Returns:
                Suma de intensidades (float). Si normalizar_por_area=True,
                devuelve la suma dividida por el número de píxeles (= media).
        """
        self._validar_entradas(img_segmentada, img_procesada)
        pixeles = self._extraer_pixeles_objeto(img_segmentada, img_procesada)
        suma = float(np.sum(pixeles))
        if self.normalizar_por_area:
            return suma / len(pixeles)
        return suma

# Cuantificadores de valores extremos

@registrar_en("cuantificacion")
class MaximoIntensidad(CuantificadorIntensidad):
    """
        Valor máximo de intensidad dentro de la región segmentada.

        Cuantifica el pico de señal del objeto. En fluorescencia, corresponde
        al píxel de mayor concentración local de fluoróforo, o bien al punto
        de máxima activación/expresión.

        Nota sobre saturación:
            Si el máximo alcanza el valor límite del tipo de dato (255 para uint8,
            65535 para uint16), la imagen puede estar saturada. En ese caso este
            cuantificador devuelve un valor subestimado y se emite una advertencia.

        Ecuación:
            I_max = max{ I(xᵢ, yᵢ) : (xᵢ,yᵢ) ∈ objeto }

        Ventajas:
            - Detecta picos de señal locales
            - Útil para caracterizar el "punto caliente" del objeto
            - Rápido de calcular
            - Sensible a estructuras internas brillantes (nucléolo, centrosoma)

        Desventajas:
            - Extremadamente sensible a ruido y hot pixels
            - Un solo píxel ruidoso puede dominar el resultado
            - No representativo de la señal global del objeto
            - Puede indicar saturación del detector

        Usos típicos en microscopía:
            - Detección de picos de concentración (gránulos, vesículas)
            - Control de calidad de adquisición (detección de saturación)
            - Segmentación de orgánulos dentro de células ya segmentadas
            - Análisis de estructuras con señal focal (centrosomas, cuerpos PML)
            - Comparación de expresión máxima entre condiciones
    """
    nombre = "maximo_intensidad"

    def __init__(self, advertir_saturacion: bool = True):
        """
            Args:
                advertir_saturacion: Si True, emite una advertencia cuando el
                                    máximo coincide con el límite del tipo de dato,
                                    indicando posible saturación del detector.
        """
        self.advertir_saturacion = advertir_saturacion

    def __call__(
        self,
        img_segmentada: np.ndarray,
        img_procesada: np.ndarray,
    ) -> float:
        """
            Calcula el valor máximo de intensidad dentro de la máscara.

            Args:
                img_segmentada: Máscara binaria 2D
                img_procesada:  Imagen de intensidades 2D

            Returns:
                Valor máximo de intensidad (float). Emite UserWarning si el
                valor sugiere saturación del detector.
        """
        self._validar_entradas(img_segmentada, img_procesada)
        pixeles = self._extraer_pixeles_objeto(img_segmentada, img_procesada)
        maximo = float(np.max(pixeles))

        if self.advertir_saturacion:
            dtype = img_procesada.dtype
            if np.issubdtype(dtype, np.integer):
                limite = np.iinfo(dtype).max
                if maximo >= limite:
                    warnings.warn(
                        f"El máximo ({maximo}) alcanza el límite del tipo "
                        f"{dtype} ({limite}). La imagen puede estar saturada. "
                        "Verificar configuración de adquisición.",
                        UserWarning,
                        stacklevel=2,
                    )
        return maximo

@registrar_en("cuantificacion")
class MinimoIntensidad(CuantificadorIntensidad):
    """
        Valor mínimo de intensidad dentro de la región segmentada.

        Cuantifica el piso de señal dentro del objeto. Útil para estimar
        el nivel de fondo residual dentro de la máscara o detectar
        zonas de exclusión de señal.

        Ecuación:
            I_min = min{ I(xᵢ, yᵢ) : (xᵢ,yᵢ) ∈ objeto }

        Relación con rango dinámico:
            Rango = I_max - I_min
            Expresa la heterogeneidad de señal dentro del objeto.

        Ventajas:
            - Cuantifica el piso de señal intra-objeto
            - Útil como estimador de fondo si la máscara contiene área vacía
            - Permite calcular el rango dinámico intra-objeto

        Desventajas:
            - Sensible a sombras y artefactos oscuros dentro del objeto
            - Un píxel defectuoso (dead pixel) puede dominar el resultado
            - Poco informativo para señal distribuida uniformemente

        Usos típicos en microscopía:
            - Estimación de fondo residual en citoplasma
            - Detección de zonas de exclusión nuclear
            - Cálculo de rango dinámico intra-objeto
            - Control de calidad: detección de píxeles muertos
            - Corrección de fondo local (sustracción del mínimo)
    """
    nombre = "minimo_intensidad"

    def __init__(self):
        """Inicializa el cuantificador (sin parámetros)."""
        pass

    def __call__(
        self,
        img_segmentada: np.ndarray,
        img_procesada: np.ndarray,
    ) -> float:
        """
            Calcula el valor mínimo de intensidad dentro de la máscara.

            Args:
                img_segmentada: Máscara binaria 2D
                img_procesada:  Imagen de intensidades 2D

            Returns:
                Valor mínimo de intensidad (float)
        """
        self._validar_entradas(img_segmentada, img_procesada)
        pixeles = self._extraer_pixeles_objeto(img_segmentada, img_procesada)
        return float(np.min(pixeles))

# Cuantificadores de dispersión
@registrar_en("cuantificacion")
class DesviacionEstandar(CuantificadorIntensidad):
    """
        Desviación estándar de intensidades en la región segmentada.

        Cuantifica la heterogeneidad o variabilidad de la señal dentro del
        objeto. Una σ alta indica distribución de señal no uniforme; una σ
        baja indica señal homogénea.

        Fundamento estadístico:
            La desviación estándar es la raíz de la varianza, que es el
            segundo momento central de la distribución de intensidades.
            Para N píxeles:

            σ = sqrt[ (1/(N-1)) Σᵢ (Iᵢ - μ)² ]

            Se usa N-1 (corrección de Bessel) para estimador insesgado de
            la varianza poblacional a partir de la muestra de píxeles.

        Ventajas:
            - Cuantifica heterogeneidad de señal intra-objeto
            - En las mismas unidades que la señal (a diferencia de la varianza)
            - Sensible a distribuciones no uniformes
            - Permite construir intervalos de confianza (μ ± σ)

        Desventajas:
            - Sensible a outliers (hot pixels elevan artificialmente σ)
            - No distingue entre heterogeneidad real y ruido de adquisición
            - Depende de la escala de la imagen (no es adimensional)
            - Para comparar entre experimentos usar CoeficienteVariacion

        Usos típicos en microscopía:
            - Caracterización de heterogeneidad intra-celular
            - Comparación de uniformidad de expresión entre células
            - Detección de estructuras inhomogéneas (gránulos de estrés)
            - Input para clasificadores de fenotipo celular
            - Textura de señal como feature de machine learning
    """
    nombre = "desviacion_estandar"

    def __init__(self, ddof: int = 1):
        """
            Args:
                ddof: Delta grados de libertad para el cálculo de σ.
                    ddof=1: estimador insesgado de la población (recomendado)
                    ddof=0: desviación estándar de la muestra (descriptivo puro)
        """
        if ddof not in (0, 1):
            raise ValueError("ddof debe ser 0 o 1")
        self.ddof = ddof

    def __call__(
        self,
        img_segmentada: np.ndarray,
        img_procesada: np.ndarray,
    ) -> float:
        """
            Calcula la desviación estándar de intensidad dentro de la máscara.

            Args:
                img_segmentada: Máscara binaria 2D
                img_procesada:  Imagen de intensidades 2D

            Returns:
                Desviación estándar de intensidades (float)
        """
        self._validar_entradas(img_segmentada, img_procesada)
        pixeles = self._extraer_pixeles_objeto(img_segmentada, img_procesada)
        return float(np.std(pixeles, ddof=self.ddof))

@registrar_en("cuantificacion")
class CoeficienteVariacion(CuantificadorIntensidad):
    """
        Coeficiente de variación (CV) de intensidades en la región segmentada.

        El CV es la desviación estándar normalizada por la media, expresada
        generalmente como porcentaje. Es una medida adimensional de variabilidad
        relativa que permite comparar dispersiones entre objetos con distintas
        intensidades medias.

        Ecuación:
            CV (%) = (σ / μ) × 100
            donde σ = desviación estándar, μ = media

        Ejemplo interpretativo:
            Objeto A: μ=100, σ=20 → CV=20%
            Objeto B: μ=500, σ=20 → CV=4%
            Aunque tienen la misma σ absoluta, B es mucho más homogéneo
            en términos relativos. La media sola no revela esto.

        Ventajas:
            - Adimensional: permite comparación directa entre experimentos
            - Independiente de la escala de intensidad
            - Estándar en control de calidad analítico (CV < 5% = buena reproducibilidad)
            - Robusto a cambios de ganancia del detector si estos son multiplicativos

        Desventajas:
            - No definido si μ = 0 (imagen completamente negra en la máscara)
            - Inestable para valores de μ muy pequeños (amplificación de errores)
            - No distingue entre variabilidad biológica real y ruido de adquisición
            - Asume que la variabilidad es proporcional a la media (modelo multiplicativo)

        Usos típicos en microscopía:
            - Control de calidad de tinción (CV alto → tinción irregular)
            - Comparación de homogeneidad entre lotes de experimentos
            - Clasificación de células por heterogeneidad de señal
            - Análisis de distribución de señal en núcleos (eucromatina/heterocromatina)
            - Normalización de variabilidad entre canales o días de adquisición
    """
    nombre = "coeficiente_variacion"

    def __init__(self, como_porcentaje: bool = True, ddof: int = 1):
        """
            Args:
                como_porcentaje: Si True, multiplica por 100 el resultado.
                                Si False, devuelve la fracción decimal (0-1).
                ddof: Delta grados de libertad para σ (ver DesviacionEstandar).
        """
        if ddof not in (0, 1):
            raise ValueError("ddof debe ser 0 o 1")
        self.como_porcentaje = como_porcentaje
        self.ddof = ddof

    def __call__(
        self,
        img_segmentada: np.ndarray,
        img_procesada: np.ndarray,
    ) -> float:
        """
                Calcula el coeficiente de variación dentro de la máscara.

                Args:
                    img_segmentada: Máscara binaria 2D
                    img_procesada:  Imagen de intensidades 2D

                Returns:
                    CV en porcentaje (si como_porcentaje=True) o fracción decimal.

                Raises:
                    ValueError: Si la media es cero (CV indefinido).
        """
        self._validar_entradas(img_segmentada, img_procesada)
        pixeles = self._extraer_pixeles_objeto(img_segmentada, img_procesada)

        media = np.mean(pixeles)
        if media == 0:
            raise ValueError(
                "La media de intensidad dentro de la máscara es 0. "
                "El CV no está definido. Verificar segmentación e imagen procesada."
            )

        sigma = np.std(pixeles, ddof=self.ddof)
        cv = sigma / media
        return float(cv * 100 if self.como_porcentaje else cv)

# Cuantificadores de forma de distribución
@registrar_en("cuantificacion")
class PercentilIntensidad(CuantificadorIntensidad):
    """
        Percentil arbitrario de la distribución de intensidades en la máscara.

        Permite caracterizar cualquier punto de la distribución acumulada
        de intensidades. Los percentiles son estimadores robustos que no
        requieren suposiciones sobre la forma de la distribución.

        Definición:
            El percentil p es el valor v tal que el p% de los píxeles
            del objeto tienen intensidad ≤ v.

        Percentiles estándar de uso frecuente:
            p=25 → Q1 (cuartil inferior)
            p=50 → mediana (Q2)
            p=75 → Q3 (cuartil superior)
            IQR = P75 - P25  (rango intercuartílico, medida robusta de dispersión)
            p=5, p=95        → rango robusto sin outliers extremos

        Ventajas:
            - No asume forma de distribución
            - Robusto ante outliers (especialmente percentiles centrales)
            - Permite reconstruir la distribución completa si se usan múltiples
            - Independiente de la escala absoluta de intensidad

        Desventajas:
            - Interpretación depende del percentil elegido
            - Computacionalmente más costoso que media o máximo
            - Puede dar falsa sensación de precisión si N es pequeño

        Usos típicos en microscopía:
            - Percentil 95 como máximo robusto (excluyendo hot pixels)
            - IQR como dispersión robusta para células con gránulos
            - Percentil 5 como estimador de fondo residual dentro del objeto
            - Descripción completa de la distribución de señal para clasificación
            - Análisis de distribución de señal en poblaciones celulares
    """
    nombre = "percentil_intensidad"

    def __init__(self, percentil: float = 95.0):
        """
            Args:
                percentil: Percentil a calcular (0–100).
                        Valores típicos: 5, 25, 50, 75, 95.
        """
        if not (0.0 <= percentil <= 100.0):
            raise ValueError("percentil debe estar en [0, 100]")
        self.percentil = percentil

    def __call__(
        self,
        img_segmentada: np.ndarray,
        img_procesada: np.ndarray,
    ) -> float:
        """
            Calcula el percentil de intensidad dentro de la máscara.

            Args:
                img_segmentada: Máscara binaria 2D
                img_procesada:  Imagen de intensidades 2D

            Returns:
                Valor del percentil especificado (float)
        """
        self._validar_entradas(img_segmentada, img_procesada)
        pixeles = self._extraer_pixeles_objeto(img_segmentada, img_procesada)
        return float(np.percentile(pixeles, self.percentil))

@registrar_en("cuantificacion")
class AsimetriaIntensidad(CuantificadorIntensidad):
    """
        Asimetría (skewness) de la distribución de intensidades en la máscara.

        El skewness es el tercer momento estandarizado de la distribución.
        Describe si la "cola" de la distribución se extiende más hacia
        valores altos (asimetría positiva) o bajos (asimetría negativa).

        Ecuación:
            γ₁ = [1/N Σᵢ (Iᵢ - μ)³] / σ³

        Interpretación:
            γ₁ > 0 : Cola hacia valores altos (la mayoría de píxeles son oscuros,
                    pocos píxeles muy brillantes → típico en fluorescencia)
            γ₁ = 0 : Distribución simétrica (normal ideal)
            γ₁ < 0 : Cola hacia valores bajos (la mayoría de píxeles son brillantes,
                    pocos muy oscuros → típico en imágenes invertidas)

        Ventajas:
            - Caracteriza la forma de la distribución más allá de media y σ
            - Útil para distinguir objetos con misma media pero diferente forma
            - Indicador de presencia de estructuras sub-resueltas (gránulos)
            - Puede diferenciar tinción difusa vs. focal

        Desventajas:
            - Muy sensible a outliers (tercer momento)
            - Requiere N suficientemente grande (N > 20 como mínimo)
            - Difícil de interpretar en contextos complejos
            - Cambia con transformaciones no lineales de intensidad

        Usos típicos en microscopía:
            - Clasificación de distribución difusa vs. focal de proteína
            - Detección de gránulos de estrés (señal focal → skewness alto)
            - Descripción de distribución de señal en análisis de textura
            - Feature para clasificación fenotípica por ML
            - Control de calidad de segmentación (skewness inesperado = error)
    """
    nombre = "asimetria_intensidad"

    def __init__(self):
        """Inicializa el cuantificador (sin parámetros)."""
        pass

    def __call__(
        self,
        img_segmentada: np.ndarray,
        img_procesada: np.ndarray,
    ) -> float:
        """
            Calcula el skewness de intensidades dentro de la máscara.

            Args:
                img_segmentada: Máscara binaria 2D
                img_procesada:  Imagen de intensidades 2D

            Returns:
                Skewness de la distribución de intensidades (float).
                Implementación Fisher-Pearson: insesgada y comparable con scipy.
        """
        self._validar_entradas(img_segmentada, img_procesada)
        pixeles = self._extraer_pixeles_objeto(img_segmentada, img_procesada)

        n = len(pixeles)
        if n < 3:
            warnings.warn(
                f"Solo {n} píxeles en la máscara. El skewness no es fiable con N < 3.",
                UserWarning,
                stacklevel=2,
            )
            return float("nan")

        media = np.mean(pixeles)
        sigma = np.std(pixeles, ddof=1)

        if sigma == 0:
            return 0.0  # Distribución constante: sin asimetría

        # Coeficiente de asimetría de Fisher-Pearson
        skewness = np.mean(((pixeles - media) / sigma) ** 3)
        return float(skewness)

@registrar_en("cuantificacion")
class CurtosisIntensidad(CuantificadorIntensidad):
    """
        Curtosis de la distribución de intensidades en la región segmentada.

        La curtosis es el cuarto momento estandarizado. Describe la "picudez"
        de la distribución respecto a una distribución normal (exceso de curtosis).
        Indica si los valores están muy concentrados alrededor de la media
        (distribución leptocúrtica) o dispersos con colas pesadas.

        Ecuación (exceso de curtosis de Fisher):
            κ = [1/N Σᵢ (Iᵢ - μ)⁴] / σ⁴  -  3

        El "-3" centra el resultado respecto a la distribución normal:
            κ > 0 : Leptocúrtica (pico pronunciado, colas pesadas → señal concentrada
                    con algunos outliers muy brillantes; típico en fluorescencia)
            κ = 0 : Mesocúrtica (equivalente a distribución normal)
            κ < 0 : Platicúrtica (pico suave, distribución más uniforme)

        Ventajas:
            - Sensible a distribuciones con colas pesadas (hot pixels, gránulos)
            - Complementa al skewness en caracterización de la distribución
            - Útil como feature de textura para clasificación celular
            - Puede detectar bimodalidad residual (κ negativa)

        Desventajas:
            - Extremadamente sensible a outliers (cuarto momento)
            - Requiere N grande para ser estable (N > 50 recomendado)
            - Interpretación no trivial en contextos biológicos
            - Puede ser dominada por un solo píxel saturado

        Usos típicos en microscopía:
            - Análisis de textura intra-nuclear (eucromatina vs. heterocromatina)
            - Detección de distribuciones bimodales dentro de objetos segmentados
            - Feature de ML para clasificación fenotípica de alta dimensión
            - Caracterización de distribución de gránulos secretores
            - Comparación de homogeneidad entre condiciones experimentales
    """
    nombre = "curtosis_intensidad"

    def __init__(self, exceso: bool = True):
        """
            Args:
                exceso: Si True (por defecto), devuelve exceso de curtosis (Fisher),
                    restando 3 para que la distribución normal tenga κ=0.
                    Si False, devuelve curtosis de Pearson sin restar 3.
        """
        self.exceso = exceso

    def __call__(
        self,
        img_segmentada: np.ndarray,
        img_procesada: np.ndarray,
    ) -> float:
        """
            Calcula la curtosis de intensidades dentro de la máscara.

            Args:
                img_segmentada: Máscara binaria 2D
                img_procesada:  Imagen de intensidades 2D

            Returns:
                Curtosis de la distribución (float). Si exceso=True, resta 3
                (referencia: distribución normal = 0).
        """
        self._validar_entradas(img_segmentada, img_procesada)
        pixeles = self._extraer_pixeles_objeto(img_segmentada, img_procesada)

        n = len(pixeles)
        if n < 4:
            warnings.warn(
                f"Solo {n} píxeles en la máscara. La curtosis no es fiable con N < 4.",
                UserWarning,
                stacklevel=2,
            )
            return float("nan")

        media = np.mean(pixeles)
        sigma = np.std(pixeles, ddof=1)

        if sigma == 0:
            return 0.0

        curtosis_pearson = np.mean(((pixeles - media) / sigma) ** 4)
        if self.exceso:
            return float(curtosis_pearson - 3.0)
        return float(curtosis_pearson)

# Cuantificador de calidad de señal
@registrar_en("cuantificacion")
class RelacionSenialRuido(CuantificadorIntensidad):
    """
        Relación señal/ruido (SNR) entre objeto segmentado y fondo.

        Cuantifica cuántas veces es más intensa la señal del objeto respecto
        al ruido (desviación estándar) del fondo. Es la métrica estándar de
        calidad de imagen en microscopía de fluorescencia.

        Definiciones de SNR en microscopía:
            El fondo se estima a partir de los píxeles fuera de la máscara
            (img_segmentada == 0). Existen múltiples definiciones; aquí:

            SNR = μ_objeto / σ_fondo

            donde:
                μ_objeto = media de intensidad dentro de la máscara
                σ_fondo  = desviación estándar de los píxeles fuera de la máscara

        Nota sobre fondo:
            Esta implementación estima el ruido del fondo como la σ de los
            píxeles excluidos de la máscara. Esto es válido cuando el fondo
            es aproximadamente constante. Si el fondo tiene gradiente,
            pre-corregirlo con normalización antes de calcular SNR.

        Interpretación práctica:
            SNR < 2  : Señal apenas distinguible del ruido
            SNR 2–5  : Señal detectable, análisis cuantitativo limitado
            SNR 5–10 : Señal buena para análisis cuantitativo
            SNR > 10 : Señal excelente (fluorescencia bien optimizada)

        Ventajas:
            - Métrica estándar y universalmente reconocida en microscopía
            - Cuantifica la calidad de la adquisición de manera interpretable
            - Permite comparar experimentos/condiciones
            - Diagnóstico de sobreexposición o subexposición

        Desventajas:
            - Requiere que el fondo sea representativo del ruido real
            - Sensible a heterogeneidad del fondo
            - No válido si la máscara cubre casi toda la imagen (poco fondo)
            - Diferentes definiciones de SNR en la literatura (cuidado al comparar)

        Usos típicos en microscopía:
            - Control de calidad de adquisición (¿es la señal suficiente?)
            - Comparación de condiciones de tinción (anticuerpo, concentración)
            - Optimización de parámetros de microscopio
            - Criterio de inclusión/exclusión de imágenes en análisis
            - Validación de protocolo de preparación de muestras
    """
    nombre = "relacion_senal_ruido"

    def __init__(self, usar_media_fondo: bool = False):
        """
            Args:
                usar_media_fondo: Si True, calcula:
                                    SNR = (μ_objeto - μ_fondo) / σ_fondo
                                que descuenta la señal de fondo media antes de
                                dividir por el ruido. Más conservador y preciso
                                cuando el fondo no es cero.
                                Si False (por defecto): SNR = μ_objeto / σ_fondo
        """
        self.usar_media_fondo = usar_media_fondo

    def __call__(
        self,
        img_segmentada: np.ndarray,
        img_procesada: np.ndarray,
    ) -> float:
        """
            Calcula el SNR entre objeto segmentado y fondo.

            Args:
                img_segmentada: Máscara binaria 2D (0=fondo, >0=objeto)
                img_procesada:  Imagen de intensidades 2D

            Returns:
                SNR como escalar (float). Retorna inf si σ_fondo = 0.

            Raises:
                ValueError: Si no hay píxeles de fondo (máscara cubre todo).
        """
        self._validar_entradas(img_segmentada, img_procesada)
        pixeles_objeto = self._extraer_pixeles_objeto(img_segmentada, img_procesada)

        mask_fondo = img_segmentada == 0
        if not np.any(mask_fondo):
            raise ValueError(
                "No hay píxeles de fondo (img_segmentada cubre toda la imagen). "
                "No es posible estimar σ_fondo."
            )

        pixeles_fondo = img_procesada[mask_fondo].astype(np.float64)
        sigma_fondo = float(np.std(pixeles_fondo, ddof=1))

        if sigma_fondo == 0:
            warnings.warn(
                "σ_fondo = 0. El SNR es indefinido (fondo perfectamente uniforme). "
                "Se devuelve inf.",
                UserWarning,
                stacklevel=2,
            )
            return float("inf")

        mu_objeto = float(np.mean(pixeles_objeto))

        if self.usar_media_fondo:
            mu_fondo = float(np.mean(pixeles_fondo))
            return (mu_objeto - mu_fondo) / sigma_fondo
        else:
            return mu_objeto / sigma_fondo

# Cuantificador espacial
@registrar_en("cuantificacion")
class PerfilLineal(CuantificadorIntensidad):
    """
        Perfil de intensidad a lo largo de una línea dentro de la máscara.

        Extrae los valores de intensidad de img_procesada a lo largo de
        una línea definida por dos puntos, restringida opcionalmente a los
        píxeles dentro de la máscara. Útil para analizar gradientes espaciales,
        distribuciones radiales y transiciones de señal.

        Fundamento geométrico:
            La línea entre p0=(x0,y0) y p1=(x1,y1) se parametriza:
                x(t) = x0 + t*(x1-x0),  t ∈ [0,1]
                y(t) = y0 + t*(y1-y0),  t ∈ [0,1]
            con N puntos equiespaciados a lo largo de la línea.

        Interpolación:
            Los puntos de la línea raramente coinciden con centros de píxeles.
            Se usa interpolación bilineal (orden=1) sobre img_procesada para
            estimar la intensidad en posiciones sub-pixel.

        Ventajas:
            - Captura variación espacial de señal (gradientes, polaridad)
            - Permite medir ancho de objetos filamentosos
            - Diagnóstico visual de distribuciones de señal
            - Útil para validar segmentación en bordes de objeto

        Desventajas:
            - Resultado depende fuertemente de la posición y ángulo de la línea
            - No provee un único escalar (devuelve array, menos automatizable)
            - Sensible a orientación del objeto si no se alinea con el eje de interés
            - Requiere especificación manual de los puntos de la línea

        Usos típicos en microscopía:
            - Análisis de distribución radial de señal en núcleos
            - Medición de gradiente de concentración en procesos celulares
            - Verificación de colocalización en dos canales (perfil superpuesto)
            - Medición de ancho de filamentos (actina, microtúbulos)
            - Análisis de polarización celular (señal frontal vs. trasera)
    """
    nombre = "perfil_lineal"

    def __init__(
        self,
        punto_inicio: Tuple[int, int] = (0, 0),
        punto_fin: Tuple[int, int] = (10, 10),
        num_puntos: int = 100,
        solo_en_mascara: bool = True,
        orden_interpolacion: int = 1,
    ):
        """
            Args:
                punto_inicio: (col, fila) = (x, y) del punto de inicio de la línea.
                            Coordenadas en píxeles de la imagen.
                punto_fin:    (col, fila) = (x, y) del punto de fin de la línea.
                num_puntos:   Número de puntos equiespaciados a muestrear
                            a lo largo de la línea.
                            Recomendado: ~2 × longitud de la línea en píxeles.
                solo_en_mascara: Si True, devuelve solo valores donde la máscara
                                es positiva. Si False, devuelve todos los puntos
                                de la línea (incluyendo fondo).
                orden_interpolacion: Orden de la interpolación spline:
                                    0 = vecino más cercano
                                    1 = bilineal (recomendado)
                                    3 = cúbica (más suave, más lento)
        """
        if num_puntos < 2:
            raise ValueError("num_puntos debe ser >= 2")
        if orden_interpolacion not in (0, 1, 2, 3):
            raise ValueError("orden_interpolacion debe ser 0, 1, 2 o 3")

        self.punto_inicio = punto_inicio
        self.punto_fin = punto_fin
        self.num_puntos = num_puntos
        self.solo_en_mascara = solo_en_mascara
        self.orden_interpolacion = orden_interpolacion

    def __call__(
        self,
        img_segmentada: np.ndarray,
        img_procesada: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
            Extrae el perfil de intensidad a lo largo de la línea definida.

            Args:
                img_segmentada: Máscara binaria 2D
                img_procesada:  Imagen de intensidades 2D

            Returns:
                Tupla (distancias, intensidades):
                    - distancias: Array 1D con distancias en píxeles desde punto_inicio
                    - intensidades: Array 1D con valores de intensidad interpolados

                Si solo_en_mascara=True, ambos arrays están filtrados a los puntos
                donde la máscara es positiva.

            Nota:
                scipy.ndimage.map_coordinates usa convención (fila, col) = (y, x).
        """
        from scipy.ndimage import map_coordinates

        self._validar_entradas(img_segmentada, img_procesada)

        x0, y0 = self.punto_inicio
        x1, y1 = self.punto_fin

        # Construir coordenadas de la línea
        cols = np.linspace(x0, x1, self.num_puntos)  # eje X = columna
        filas = np.linspace(y0, y1, self.num_puntos)  # eje Y = fila

        # Distancia acumulada a lo largo de la línea
        longitud = np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
        distancias = np.linspace(0, longitud, self.num_puntos)

        # Interpolación sobre img_procesada (convención scipy: [fila, col])
        intensidades = map_coordinates(
            img_procesada.astype(np.float64),
            [filas, cols],
            order=self.orden_interpolacion,
            mode="nearest",
        )

        if self.solo_en_mascara:
            # Muestrear también la máscara para filtrar puntos de fondo
            mask_valores = map_coordinates(
                img_segmentada.astype(np.float64),
                [filas, cols],
                order=0,  # vecino más cercano para máscara binaria
                mode="nearest",
            )
            en_objeto = mask_valores > 0
            if not np.any(en_objeto):
                warnings.warn(
                    "La línea definida no pasa por ningún píxel de objeto. "
                    "Verificar punto_inicio y punto_fin respecto a la máscara.",
                    UserWarning,
                    stacklevel=2,
                )
            distancias = distancias[en_objeto]
            intensidades = intensidades[en_objeto]

        return distancias, intensidades

# FUNCIONES DE UTILIDAD Y PIPELINE

def extraer_todas_metricas(img_segmentada: np.ndarray,
                        img_procesada: np.ndarray, 
                        por_region: bool = True) -> dict():
    """
        Extrae todas las métricas morfométricas disponibles.
        
        Args:
            img_segmentada: Imagen binaria o etiquetada
            por_region: Si True, calcula por cada región etiquetada
        
        Returns:
            Diccionario con todas las métricas
    """
    metricas = {}

    # Instanciar cuantificadores
    cuantificadores = {
        'mediaIntensidad': MediaIntensidad(),
        'intensidadIntegrada': IntensidadIntegrada(),
        'maximoIntensidad': MaximoIntensidad(),
        'minimoIntensidad': MinimoIntensidad(),
        'medianaIntensidad': MedianaIntensidad(),
        'desviacionEstandar': DesviacionEstandar(),
        'coeficienteVariacion': CoeficienteVariacion(),
        'percentilIntensidad': PercentilIntensidad(),
        'relacionSR': RelacionSenialRuido(),
        'asimetriaIntensidad': AsimetriaIntensidad(),
        'curtosisIntensidad': CurtosisIntensidad(),
        'perfilLineal': PerfilLineal(),
    }
    
    for nombre, cuantificador in cuantificadores.items():
        try:
            metricas[nombre] = cuantificador(img_segmentada, img_procesada)
        except Exception as e:
            warnings.warn(f"Error calculando {nombre}: {e}")
            metricas[nombre] = None
    
    return metricas