"""
Configuración estética centralizada para todos los módulos de visualización.

Este módulo define EsteticaGrafico, un dataclass que agrupa todos los
parámetros visuales que son comunes a los distintos tipos de gráfico del
pipeline: Plots_Estadisticos, Plots_Modelos, Plots_Imagen y
VisualizadorDimensionalidad.

Uso típico:

    from Estetica import EsteticaGrafico, ESTILOS_PREDEFINIDOS

    # Usar estilo predefinido
    est = ESTILOS_PREDEFINIDOS['publicacion']

    # O personalizar desde cero
    est = EsteticaGrafico(
        paleta='viridis',
        figsize=(12, 7),
        fuente_familia='sans-serif',
        fuente_titulo=16,
        alpha_puntos=0.9,
    )

    # Pasar a cualquier función de plots
    fig, ax = Plots_Estadisticos.histograma(df, columna='media', estetica=est)

Separación de responsabilidades:
    - EsteticaGrafico controla CÓMO se ve el gráfico (colores, fuentes, tamaños).
    - Las funciones de cada módulo controlan QUÉ se grafica (datos, ejes, capas).
    - Los parámetros semánticos (qué columna, qué modelo, etc.) siempre van
      directamente a la función, nunca dentro de EsteticaGrafico.

Nota sobre seaborn:
    EsteticaGrafico.aplicar() llama sns.set_style() y plt.rcParams para
    establecer el estilo globalmente antes de crear la figura. Si se crean
    múltiples figuras en un notebook, llamar .aplicar() antes de cada una
    para garantizar consistencia.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
import seaborn as sns


@dataclass
class EsteticaGrafico:
    """
        Parámetros estéticos unificados para todas las funciones de visualización.

        Agrupa en un solo objeto todos los argumentos que habitualmente se
        repiten en cada llamada a matplotlib/seaborn: paletas, fuentes, tamaños,
        transparencias y estilo de fondo.

        Puede instanciarse directamente, construirse desde un preset con
        ESTILOS_PREDEFINIDOS, o copiarse y modificarse con dataclasses.replace().

        Ejemplo de modificación puntual:
            from dataclasses import replace
            est_grande = replace(est_base, figsize=(16, 10), fuente_titulo=20)
    """

    # ── Figura ─────────────────────────────────────────────────
    figsize: Tuple[int, int] = (10, 6)
    """Tamaño de figura (ancho, alto) en pulgadas."""

    dpi: int = 100
    """Resolución de la figura. 100 para pantalla, 300 para publicación."""

    # ── Paletas de color ───────────────────────────────────────
    paleta: str = 'tab10'
    """
    Paleta principal para grupos/categorías discretas.
    Opciones frecuentes: 'tab10', 'Set1', 'Set2', 'Spectral', 'viridis'.
    """

    paleta_continua: str = 'viridis'
    """
    Paleta para variables continuas (heatmaps, superficies).
    Opciones: 'viridis', 'plasma', 'coolwarm', 'RdBu_r'.
    """

    paleta_divergente: str = 'RdBu_r'
    """
    Paleta para variables con punto neutro (correlaciones, residuos).
    Opciones: 'RdBu_r', 'coolwarm', 'bwr'.
    """

    # ── Fuentes ────────────────────────────────────────────────
    fuente_familia: str = 'serif'
    """Familia tipográfica: 'serif', 'sans-serif', 'monospace'."""

    fuente_titulo: int = 14
    """Tamaño de fuente del título principal."""

    fuente_ejes: int = 12
    """Tamaño de fuente de las etiquetas de los ejes X e Y."""

    fuente_ticks: int = 10
    """Tamaño de fuente de los valores de los ticks en los ejes."""

    fuente_leyenda: int = 10
    """Tamaño de fuente de la leyenda."""

    fuente_anotaciones: int = 9
    """Tamaño de fuente de las anotaciones de texto dentro del gráfico."""

    negrita_titulo: bool = True
    """Si True, el título se muestra en negrita (fontweight='bold')."""

    negrita_ejes: bool = False
    """Si True, los labels de los ejes se muestran en negrita."""

    # ── Puntos y marcadores ────────────────────────────────────
    tamaño_punto: int = 60
    """Tamaño de los marcadores en scatter plots."""

    alpha_puntos: float = 0.8
    """Transparencia de los puntos (0=invisible, 1=opaco)."""

    alpha_relleno: float = 0.3
    """
    Transparencia de áreas rellenas (violines, barras, kernels KDE).
    Generalmente menor que alpha_puntos para no tapar los datos.
    """

    grosor_linea: float = 1.5
    """Grosor de líneas (bordes de boxplot, contornos, líneas de tendencia)."""

    # ── Grid y fondo ───────────────────────────────────────────
    estilo_fondo: str = 'whitegrid'
    """
    Estilo de seaborn para el fondo:
        'whitegrid'  : fondo blanco con grid (recomendado para publicación).
        'darkgrid'   : fondo gris con grid (presentaciones).
        'white'      : fondo blanco sin grid.
        'ticks'      : solo ticks, sin grid.
    """

    grid: bool = True
    """Si True, muestra grid en el gráfico."""

    grid_axis: str = 'both'
    """Eje del grid: 'both', 'x', 'y'."""

    grid_alpha: float = 0.4
    """Transparencia del grid."""

    grid_linestyle: str = '--'
    """Estilo de línea del grid: '--', '-', ':', '-.'."""

    quitar_spine_top: bool = True
    """Si True, elimina el borde superior del gráfico."""

    quitar_spine_right: bool = True
    """Si True, elimina el borde derecho del gráfico."""

    # ── Leyenda ────────────────────────────────────────────────
    leyenda_fuera: bool = False
    """Si True, coloca la leyenda fuera del área de plot (bbox_to_anchor)."""

    leyenda_loc: str = 'best'
    """Posición de la leyenda: 'best', 'upper right', 'lower left', etc."""

    leyenda_marco: bool = True
    """Si True, dibuja el recuadro de la leyenda."""

    # ── Colores específicos ────────────────────────────────────
    color_mediana: str = 'black'
    """Color de la línea de mediana en boxplots."""

    color_media: str = 'red'
    """Color del marcador de media (si se muestra)."""

    color_referencia: str = 'gray'
    """Color de líneas de referencia (x=0, y=0, umbrales)."""

    color_outlier: str = 'crimson'
    """Color de outliers o puntos atípicos."""

    # ── Opciones de guardado ───────────────────────────────────
    bbox_inches: str = 'tight'
    """Ajuste del bounding box al guardar: 'tight' elimina márgenes vacíos."""

    formato_guardado: str = 'png'
    """Formato por defecto al guardar: 'png', 'pdf', 'svg', 'tiff'."""

    def aplicar(self) -> None:
        """
            Aplica el estilo globalmente via seaborn y matplotlib.rcParams.

            Debe llamarse una vez antes de crear la figura para garantizar
            que todos los parámetros estéticos se apliquen correctamente.
            Llamar de nuevo entre figuras si se cambia el estilo a mitad del
            notebook.
        """
        sns.set_style(self.estilo_fondo)
        plt.rcParams.update({
            'font.family':    self.fuente_familia,
            'font.size':      self.fuente_ticks,
            'axes.titlesize': self.fuente_titulo,
            'axes.labelsize': self.fuente_ejes,
            'xtick.labelsize':self.fuente_ticks,
            'ytick.labelsize':self.fuente_ticks,
            'legend.fontsize':self.fuente_leyenda,
            'figure.dpi':     self.dpi,
            'lines.linewidth':self.grosor_linea,
        })

    def aplicar_a_ax(self, ax, titulo: str = '', xlabel: str = '',
                    ylabel: str = '') -> None:
        """
            Aplica estilo estándar a un Axes ya creado.

            Llamar al final de cada función de plot para garantizar coherencia
            visual entre todos los gráficos del pipeline.

            Args:
                ax:     Axes de matplotlib sobre el que aplicar el estilo.
                titulo: Título del gráfico.
                xlabel: Etiqueta del eje X.
                ylabel: Etiqueta del eje Y.
        """
        titulo_kw = {'fontweight': 'bold'} if self.negrita_titulo else {}
        ejes_kw   = {'fontweight': 'bold'} if self.negrita_ejes   else {}

        if titulo:
            ax.set_title(titulo, fontsize=self.fuente_titulo,
                         fontfamily=self.fuente_familia, **titulo_kw)
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=self.fuente_ejes,
                          fontfamily=self.fuente_familia, **ejes_kw)
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=self.fuente_ejes,
                          fontfamily=self.fuente_familia, **ejes_kw)

        if self.grid:
            ax.grid(True, linestyle=self.grid_linestyle,
                    alpha=self.grid_alpha, axis=self.grid_axis)
        else:
            ax.grid(False)

        if self.quitar_spine_top:
            ax.spines['top'].set_visible(False)
        if self.quitar_spine_right:
            ax.spines['right'].set_visible(False)

    def kwargs_leyenda(self, titulo_leyenda: str = '') -> Dict:
        """
        Devuelve dict de kwargs para ax.legend() coherente con este estilo.

        Args:
            titulo_leyenda: Título de la leyenda.

        Returns:
            Dict listo para desempacar en ax.legend(**kwargs).
        """
        kwargs: Dict = {
            'frameon':   self.leyenda_marco,
            'fontsize':  self.fuente_leyenda,
        }
        if titulo_leyenda:
            kwargs['title'] = titulo_leyenda
        if self.leyenda_fuera:
            kwargs['bbox_to_anchor'] = (1.02, 1)
            kwargs['loc']            = 'upper left'
        else:
            kwargs['loc'] = self.leyenda_loc
        return kwargs


# ─────────────────────────────────────────────────────────────
# Estilos predefinidos
# ─────────────────────────────────────────────────────────────

ESTILOS_PREDEFINIDOS: Dict[str, EsteticaGrafico] = {

    'default': EsteticaGrafico(),

    'publicacion': EsteticaGrafico(
        figsize=(8, 5),
        dpi=300,
        paleta='tab10',
        paleta_continua='viridis',
        fuente_familia='serif',
        fuente_titulo=12,
        fuente_ejes=11,
        fuente_ticks=10,
        fuente_leyenda=10,
        alpha_puntos=0.85,
        alpha_relleno=0.25,
        grosor_linea=1.2,
        estilo_fondo='white',
        grid=True,
        grid_axis='y',
        grid_alpha=0.3,
        negrita_titulo=True,
        negrita_ejes=False,
    ),

    'presentacion': EsteticaGrafico(
        figsize=(14, 8),
        dpi=120,
        paleta='Set2',
        paleta_continua='plasma',
        fuente_familia='sans-serif',
        fuente_titulo=18,
        fuente_ejes=14,
        fuente_ticks=12,
        fuente_leyenda=12,
        alpha_puntos=0.9,
        alpha_relleno=0.4,
        grosor_linea=2.0,
        estilo_fondo='darkgrid',
        grid=True,
        grid_axis='both',
        grid_alpha=0.3,
        negrita_titulo=True,
        negrita_ejes=True,
        leyenda_fuera=False,
    ),

    'exploratorio': EsteticaGrafico(
        figsize=(10, 6),
        dpi=90,
        paleta='Spectral',
        fuente_familia='sans-serif',
        fuente_titulo=13,
        fuente_ejes=11,
        alpha_puntos=0.7,
        alpha_relleno=0.35,
        estilo_fondo='whitegrid',
    ),

    'microscopio': EsteticaGrafico(
        figsize=(10, 8),
        dpi=150,
        paleta='tab20',
        paleta_continua='inferno',
        paleta_divergente='coolwarm',
        fuente_familia='sans-serif',
        fuente_titulo=13,
        fuente_ejes=11,
        fuente_ticks=10,
        alpha_puntos=0.75,
        alpha_relleno=0.3,
        estilo_fondo='white',
        grid=True,
        grid_axis='both',
        grid_alpha=0.25,
        leyenda_fuera=True,
    ),
}