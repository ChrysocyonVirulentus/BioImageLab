# === filtrador/Controlador_Filtrador.py ===
from __future__ import annotations

from typing import Union, Optional, Dict, Any

# Core
from .Controlador_Base import Controlador_Base

# Sistema
from .Resultado_Either import Resultado
from .Controlador_BioImagen import BioImagenData, ErrorBioImagen
from ..gestorLab.Categoria_Operacion import CategoriaOperacion
from ..gestorLab.Operacion import Operacion, TipoDato

# Métodos
from ..filtrador.locales.Filtros_Locales import (
    CajaBlur, Gaussiano, Bilateral, Mediana, DifusionAnisotropica
)
from ..filtrador.espectrales.Filtros_Ffts import (
    FFTPasaBajo, FFTPasaAlto, FFTPasaBanda, FFTBandStop, FiltracionNotch
)
from ..filtrador.multiescala.Filtros_Multiescala import (
    DiferenciaLaplaciana, DiferenciaGaussiana, WaveletTransform, PiramideLaplaciana
)
from ..filtrador.noLocales.Filtros_NoLocales import (
    NonLocalMeans, BlockMatching3D
)

# Estrategias
from .Estrategias_Aplicacion import (
    TipoAplicacion,
    Global,
    PorCorteZ,
    PorTimepoint,
    PorCorteEspaciotemporal,
    PorVolumen3D,
    ConReferencia,
)


MetodoFiltro = Union[
    CajaBlur, Gaussiano, Bilateral, Mediana, DifusionAnisotropica,
    FFTPasaBajo, FFTPasaAlto, FFTPasaBanda, FFTBandStop, FiltracionNotch,
    DiferenciaLaplaciana, DiferenciaGaussiana, WaveletTransform, PiramideLaplaciana,
    NonLocalMeans, BlockMatching3D
]


class Controlador_Filtrador((Controlador_Base[BioImagenData, BioImagenData])):
    """
    Controlador de filtrado.

    Dos modos de uso:
      - YAML / CLI : crear_operacion(...) → Operacion  [flujo principal]
      - Programático: crear_gaussiano(...) → Callable, aplicar(...) → Resultado
    """

    def __init__(self):
        # dominio="filtracion" debe coincidir con @registrar_en("filtracion")
        super().__init__(etapa="procesamiento", dominio="filtracion")
        self._ultimo_metodo: Optional[MetodoFiltro] = None

    # =========================================================
    # HELPER INTERNO
    # =========================================================

    def _crear(
        self,
        metodo: MetodoFiltro,
        tipo: TipoAplicacion,
        canal: int = 0,
    ):
        """
        Callable canal-capturado: (BioImagenData) → Resultado.
        El canal queda fijo en el cierre — no se pasa en la llamada.
        """
        self._ultimo_metodo = metodo
        return self.crear_operador(
            metodo=metodo,
            tipo_aplicacion=tipo,
            canal=canal,           # ← capturado en el cierre del base
        )

    def _crear_multicanal(
        self,
        metodo: MetodoFiltro,
        tipo: TipoAplicacion,
    ):
        self._ultimo_metodo = metodo
        return self.crear_operador_multicanal(
            metodo=metodo,
            tipo_aplicacion=tipo,
        )

    # =========================================================
    # FACTORIES PARA YAML / CLI
    # Devuelven Operacion lista para el PipelineBuilder.
    # El YAML resuelve nombre_metodo + params → crear_operacion del base.
    # Estos métodos son el contrato explícito por si se instancia directo.
    # =========================================================

    def crear_operacion_gaussiano(
        self,
        sigma: float = 1.0,
        tipo: TipoAplicacion = Global(),
        canal: Optional[int] = 0,
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self.crear_operacion(
            nombre_metodo="gaussiano",
            categoria=CategoriaOperacion.FILTRACION,
            tipo_aplicacion=tipo,
            canal=canal,
            nombre=nombre,
            params={"sigma": sigma},
        )

    def crear_operacion_mediana(
        self,
        tamano: int = 3,
        tipo: TipoAplicacion = Global(),
        canal: Optional[int] = 0,
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self.crear_operacion(
            nombre_metodo="mediana",
            categoria=CategoriaOperacion.FILTRACION,
            tipo_aplicacion=tipo,
            canal=canal,
            nombre=nombre,
            params={"tamano": tamano},
        )

    def crear_operacion_bilateral(
        self,
        sigma_color: float = 0.1,
        sigma_spatial: float = 2.0,
        tipo: TipoAplicacion = PorCorteEspaciotemporal(),
        canal: Optional[int] = 0,
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self.crear_operacion(
            nombre_metodo="bilateral",
            categoria=CategoriaOperacion.FILTRACION,
            tipo_aplicacion=tipo,
            canal=canal,
            nombre=nombre,
            params={"sigma_color": sigma_color, "sigma_spatial": sigma_spatial},
        )

    def crear_operacion_difusion_anisotropica(
        self,
        iteraciones: int = 10,
        kappa: float = 50.0,
        gamma: float = 0.1,
        tipo: TipoAplicacion = PorCorteEspaciotemporal(),
        canal: Optional[int] = 0,
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self.crear_operacion(
            nombre_metodo="difusion_anisotropica",
            categoria=CategoriaOperacion.FILTRACION,
            tipo_aplicacion=tipo,
            canal=canal,
            nombre=nombre,
            params={"iteraciones": iteraciones, "kappa": kappa, "gamma": gamma},
        )

    def crear_operacion_pasa_bajo(
        self,
        radio: int = 30,
        tipo: TipoAplicacion = Global(),
        canal: Optional[int] = 0,
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self.crear_operacion(
            nombre_metodo="fft_pasa_bajo",
            categoria=CategoriaOperacion.FILTRACION,
            tipo_aplicacion=tipo,
            canal=canal,
            nombre=nombre,
            params={"radio": radio},
        )

    def crear_operacion_pasa_alto(
        self,
        radio: int = 30,
        tipo: TipoAplicacion = Global(),
        canal: Optional[int] = 0,
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self.crear_operacion(
            nombre_metodo="fft_pasa_alto",
            categoria=CategoriaOperacion.FILTRACION,
            tipo_aplicacion=tipo,
            canal=canal,
            nombre=nombre,
            params={"radio": radio},
        )

    def crear_operacion_pasa_banda(
        self,
        radio_bajo: int = 10,
        radio_alto: int = 50,
        tipo: TipoAplicacion = Global(),
        canal: Optional[int] = 0,
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self.crear_operacion(
            nombre_metodo="fft_pasa_banda",
            categoria=CategoriaOperacion.FILTRACION,
            tipo_aplicacion=tipo,
            canal=canal,
            nombre=nombre,
            params={"radio_bajo": radio_bajo, "radio_alto": radio_alto},
        )

    def crear_operacion_bandstop(
        self,
        frecuencia_central: float = 0.25,
        ancho_banda: float = 0.05,
        tipo: TipoAplicacion = Global(),
        canal: Optional[int] = 0,
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self.crear_operacion(
            nombre_metodo="fft_bandstop",
            categoria=CategoriaOperacion.FILTRACION,
            tipo_aplicacion=tipo,
            canal=canal,
            nombre=nombre,
            params={"frecuencia_central": frecuencia_central, "ancho_banda": ancho_banda},
        )

    def crear_operacion_notch(
        self,
        frecuencias_eliminar: list[tuple[float, float]],
        tipo: TipoAplicacion = Global(),
        canal: Optional[int] = 0,
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self.crear_operacion(
            nombre_metodo="filtrado_notch",
            categoria=CategoriaOperacion.FILTRACION,
            tipo_aplicacion=tipo,
            canal=canal,
            nombre=nombre,
            params={"frecuencias_eliminar": frecuencias_eliminar},
        )

    def crear_operacion_diferencia_gaussiana(
        self,
        sigma1: float = 1.0,
        sigma2: float = 2.0,
        tipo: TipoAplicacion = PorCorteEspaciotemporal(),
        canal: Optional[int] = 0,
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self.crear_operacion(
            nombre_metodo="diferencia_gaussiana",
            categoria=CategoriaOperacion.FILTRACION,
            tipo_aplicacion=tipo,
            canal=canal,
            nombre=nombre,
            params={"sigma1": sigma1, "sigma2": sigma2},
        )

    def crear_operacion_laplaciano(
        self,
        tipo_lap: str = "discreto",
        tipo: TipoAplicacion = PorCorteEspaciotemporal(),
        canal: Optional[int] = 0,
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self.crear_operacion(
            nombre_metodo="diferencia_laplaciana",
            categoria=CategoriaOperacion.FILTRACION,
            tipo_aplicacion=tipo,
            canal=canal,
            nombre=nombre,
            params={"tipo": tipo_lap},
        )

    def crear_operacion_wavelet(
        self,
        wavelet: str = "db4",
        niveles: int = 3,
        umbral: float = 0.1,
        tipo: TipoAplicacion = Global(),
        canal: Optional[int] = 0,
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self.crear_operacion(
            nombre_metodo="wavelet_transform",
            categoria=CategoriaOperacion.FILTRACION,
            tipo_aplicacion=tipo,
            canal=canal,
            nombre=nombre,
            params={"wavelet": wavelet, "niveles": niveles, "umbral": umbral},
        )

    def crear_operacion_piramide_laplaciana(
        self,
        niveles: int = 3,
        tipo: TipoAplicacion = Global(),
        canal: Optional[int] = 0,
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self.crear_operacion(
            nombre_metodo="piramide_laplaciana",
            categoria=CategoriaOperacion.FILTRACION,
            tipo_aplicacion=tipo,
            canal=canal,
            nombre=nombre,
            params={"niveles": niveles},
        )

    def crear_operacion_nl_means(
        self,
        h: float = 0.1,
        patch: int = 7,
        busqueda: int = 21,
        tipo: TipoAplicacion = PorCorteEspaciotemporal(),
        canal: Optional[int] = 0,
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self.crear_operacion(
            nombre_metodo="non_local_means",
            categoria=CategoriaOperacion.FILTRACION,
            tipo_aplicacion=tipo,
            canal=canal,
            nombre=nombre,
            params={"h": h, "patch": patch, "busqueda": busqueda},
        )

    def crear_operacion_bm3d(
        self,
        sigma: float = 25.0,
        tipo: TipoAplicacion = PorCorteEspaciotemporal(),
        canal: Optional[int] = 0,
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self.crear_operacion(
            nombre_metodo="block_matching_3d",
            categoria=CategoriaOperacion.FILTRACION,
            tipo_aplicacion=tipo,
            canal=canal,
            nombre=nombre,
            params={"sigma": sigma},
        )

    # =========================================================
    # USO IMPERATIVO (scripting / tests / debug)
    # =========================================================

    def aplicar(
        self,
        data: BioImagenData,
        metodo: MetodoFiltro,
        tipo: TipoAplicacion,
        canal: int = 0,
    ) -> Resultado[BioImagenData, ErrorBioImagen]:
        """Aplica directamente sobre data. Para scripts y tests."""
        op = self._crear(metodo, tipo, canal)
        return op(data)                    # op ya captura canal — 1 arg

    def aplicar_multicanal(
        self,
        data: BioImagenData,
        metodo: MetodoFiltro,
        tipo: TipoAplicacion,
    ) -> Resultado[BioImagenData, ErrorBioImagen]:
        op = self._crear_multicanal(metodo, tipo)
        return op(data)

    def __repr__(self):
        nombre = getattr(self._ultimo_metodo, "nombre", "None")
        return f"<Controlador_Filtrador ultimo={nombre}>"