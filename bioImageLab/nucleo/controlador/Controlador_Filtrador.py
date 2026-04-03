# === filtrador/Controlador_Filtrador.py ===
from __future__ import annotations

from typing import Union, Callable, Optional, Dict, Any

# Core
from .Controlador_Base import ControladorBase, crear_operador, crear_operador_multicanal

# Sistema
from .Resultado_Either import Resultado
from .Controlador_BioImagen import BioImagenData, ErrorBioImagen
from ..gestorLab.Categoria_Operacion import CategoriaOperacion
from ..gestorLab.Operacion import Operacion, TipoSalida

# Métodos
from ..filtrador.locales.Filtros_Locales import (
    CajaBlur, Gaussiano, Bilateral, Mediana, DifusionAnisotropica
)
from ..filtrador.espectrales.Filtros_Ffts import (
    FFTPasaBajo, FFTPasaAlto, FFTPasaBanda, FFTBandStop, FiltradoNotch
)
from ..filtrador.multiescala.Filtros_Multiescala import (
    DiferenciaLaplaciana, DiferenciaGaussiana, WaveletTransform, PiramideLaplaciana
)
from ..filtrador.noLocales.Filtros_NoLocales import (
    NonLocalMeans, BlockMatching3D
)

# Estrategias
from .Estrategias_Aplicacion import TipoAplicacion, Global, PorCorteEspaciotemporal


# ==================== TIPADO ====================

MetodoFiltro = Union[
    CajaBlur, Gaussiano, Bilateral, Mediana, DifusionAnisotropica,
    FFTPasaBajo, FFTPasaAlto, FFTPasaBanda, FFTBandStop, FiltradoNotch,
    DiferenciaLaplaciana, DiferenciaGaussiana, WaveletTransform, PiramideLaplaciana,
    NonLocalMeans, BlockMatching3D
]


# ==================== CONTROLADOR ====================

class Controlador_Filtrador(ControladorBase):
    """
    Controlador de filtrado basado en ControladorBase.
    SOLO define factories y metadata.
    """

    def __init__(self):
        super().__init__(etapa="filtracion")
        self._ultimo_metodo: Optional[MetodoFiltro] = None

    # ===== CORE WRAPPERS =====

    def _crear(
        self,
        metodo: MetodoFiltro,
        tipo: TipoAplicacion,
    ):
        self._ultimo_metodo = metodo
        return crear_operador(
            metodo=metodo,
            tipo=tipo,
            etapa=self.etapa
        )

    def _crear_multicanal(
        self,
        metodo: MetodoFiltro,
        tipo: TipoAplicacion,
    ):
        self._ultimo_metodo = metodo
        return crear_operador_multicanal(
            metodo=metodo,
            tipo=tipo,
            etapa=self.etapa
        )

    # ===== FACTORIES =====

    def crear_gaussiano(
        self,
        sigma: float = 1.0,
        tipo: TipoAplicacion = Global(),
        canal: int = 0
    ):
        op = self._crear(Gaussiano(sigma=sigma), tipo)
        return lambda data: op(data, canal)

    def crear_mediana(
        self,
        tamano: int = 3,
        tipo: TipoAplicacion = Global(),
        canal: int = 0
    ):
        op = self._crear(Mediana(tamano=tamano), tipo)
        return lambda data: op(data, canal)

    def crear_bilateral(
        self,
        sigma_color: float = 0.1,
        sigma_spatial: float = 2.0,
        tipo: TipoAplicacion = PorCorteEspaciotemporal(),
        canal: int = 0
    ):
        op = self._crear(Bilateral(sigma_color, sigma_spatial), tipo)
        return lambda data: op(data, canal)

    def crear_difusion_anisotropica(
        self,
        iteraciones: int = 10,
        kappa: float = 50.0,
        gamma: float = 0.1,
        tipo: TipoAplicacion = PorCorteEspaciotemporal(),
        canal: int = 0
    ):
        op = self._crear(DifusionAnisotropica(iteraciones, kappa, gamma), tipo)
        return lambda data: op(data, canal)

    def crear_pasa_bajo(
        self,
        radio: int = 30,
        tipo: TipoAplicacion = Global(),
        canal: int = 0
    ):
        op = self._crear(FFTPasaBajo(radio), tipo)
        return lambda data: op(data, canal)

    def crear_pasa_alto(
        self,
        radio: int = 30,
        tipo: TipoAplicacion = Global(),
        canal: int = 0
    ):
        op = self._crear(FFTPasaAlto(radio), tipo)
        return lambda data: op(data, canal)

    def crear_pasa_banda(
        self,
        radio_bajo: int = 10,
        radio_alto: int = 50,
        tipo: TipoAplicacion = Global(),
        canal: int = 0
    ):
        op = self._crear(FFTPasaBanda(radio_bajo=radio_bajo, 
                                    radio_alto=radio_alto),
                                    tipo)
        return lambda data: op(data, canal)

    def crear_bandstop(
        self,
        frecuencia_central: float = 0.25,
        ancho_banda: float = 0.05,
        tipo: TipoAplicacion = Global(),
        canal: int = 0
    ):
        op = self._crear(FFTBandStop(frecuencia_central=frecuencia_central,
                                    ancho_banda=ancho_banda),
                                    tipo)
        return lambda data: op(data, canal)

    def crear_notch(
        self,
        recuencias_eliminar: list[tuple[float, float]],
        tipo: TipoAplicacion = Global(),
        canal: int = 0
    ):
        op = self._crear(FiltradoNotch(frecuencias_eliminar=frecuencias_eliminar), tipo)
        return lambda data: op(data, canal)

    def crear_diferencia_gaussiana(
        self,
        sigma1: float = 1.0,
        sigma2: float = 2.0,
        tipo: TipoAplicacion = PorCorteEspaciotemporal(),
        canal: int = 0
    ):
        op = self._crear(DiferenciaGaussiana(sigma1, sigma2), tipo)
        return lambda data: op(data, canal)

    def crear_laplaciano(
        self,
        tipo_lap: str = "discreto",
        tipo: TipoAplicacion = PorCorteEspaciotemporal(),
        canal: int = 0
    ):
        op = self._crear(DiferenciaLaplaciana(tipo=tipo_lap), tipo)
        return lambda data: op(data, canal)

    def crear_wavelet(
        self,
        wavelet: str = "db4",
        niveles: int = 3,
        umbral: float = 0.1,
        tipo: TipoAplicacion = Global(),
        canal: int = 0
    ):
        op = self._crear(WaveletTransform(wavelet, niveles, umbral), tipo)
        return lambda data: op(data, canal)

    def crear_piramide_laplaciana(
        self,
        niveles: int = 3,
        tipo: TipoAplicacion = Global(),
        canal: int = 0
    ):
        op = self._crear(PiramideLaplaciana(niveles=niveles), tipo)
        return lambda data: op(data, canal)    

    def crear_nl_means(
        self,
        h: float = 0.1,
        patch: int = 7,
        busqueda: int = 21,
        tipo: TipoAplicacion = PorCorteEspaciotemporal(),
        canal: int = 0
    ):
        op = self._crear(NonLocalMeans(h, patch, busqueda), tipo)
        return lambda data: op(data, canal)

    def crear_bm3d(
        self,
        sigma: float = 25.0,
        tipo: TipoAplicacion = PorCorteEspaciotemporal(),
        canal: int = 0
    ):
        op = self._crear(BlockMatching3D(sigma), tipo)
        return lambda data: op(data, canal)

    # ===== IMPERATIVO =====

    def aplicar(
        self,
        data: BioImagenData,
        metodo: MetodoFiltro,
        tipo: TipoAplicacion,
        canal: int = 0
    ) -> Resultado[BioImagenData, ErrorBioImagen]:
        op = self._crear(metodo, tipo)
        return op(data, canal)

    def aplicar_multicanal(
        self,
        data: BioImagenData,
        metodo: MetodoFiltro,
        tipo: TipoAplicacion
    ):
        op = self._crear_multicanal(metodo, tipo)
        return op(data)

    def __repr__(self):
        nombre = getattr(self._ultimo_metodo, "nombre", "None")
        return f"<Controlador_Filtrador ultimo={nombre}>"