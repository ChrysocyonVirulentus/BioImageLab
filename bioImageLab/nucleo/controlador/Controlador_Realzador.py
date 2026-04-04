# === realzador/Controlador_Realzador.py ===
from __future__ import annotations

import numpy as np
from typing import Union, Optional, Tuple

# Core
from .Controlador_Base import Controlador_Base

# Sistema
from .Resultado_Either import Resultado
from .Controlador_BioImagen import BioImagenData, ErrorBioImagen
from ..gestorLab.Categoria_Operacion import CategoriaOperacion
from ..gestorLab.Operacion import Operacion, TipoSalida

# Estrategias — sin override, idéntico al filtrador
from .Estrategias_Aplicacion import (
    TipoAplicacion,
    Global,
    PorCorteZ,
    PorTimepoint,
    PorCorteEspaciotemporal,
    PorVolumen3D,
    ConReferencia,
)

# Métodos
from ..realzador.contraste.Realzadores_Constraste import (
    CLAHE, Gamma, Logaritmico, Retinex, EcualizacionHistograma,
)
from ..realzador.convolucion.Realzadores_Convolucion import (
    KernelPersonalizado, PSFSimulacion, KernelSeparable,
    ConvolucionFrecuencia, CorreccionBordes,
)
from ..realzador.deconvolucion.Realzadores_Deconvolucion import (
    Wiener, RichardsonLucy, BlindDeconvolucion, Tikhonov,
)
from ..realzador.morfologicos.Realzadores_Morfologicos import (
    Apertura, Cierre, TopHat, BottomHat,
    GradienteMorfologico, ReconstruccionMorfologica,
)
from ..realzador.afilacion.Realzadores_Afilacion import (
    AfilacionLaplaciana, FiltroHighBoost, MascaraEnfoque,
    AfilacionGradiente, AfilacionWavelet,
)
from ..realzador.estructura.Realzadores_Estructurales import (
    Hessiano, Frangi, Sato, TensorEstructural,
)
from ..realzador.gradientes.Realzadores_Gradientes import (
    Laplaciano, LaplacianoCero, Canny, Sobel, Scharr, Prewitt, Roberts,
)


MetodoRealce = Union[
    CLAHE, Gamma, Logaritmico, Retinex, EcualizacionHistograma,
    KernelPersonalizado, PSFSimulacion, KernelSeparable,
    ConvolucionFrecuencia, CorreccionBordes,
    Wiener, RichardsonLucy, BlindDeconvolucion, Tikhonov,
    Apertura, Cierre, TopHat, BottomHat,
    GradienteMorfologico, ReconstruccionMorfologica,
    AfilacionLaplaciana, FiltroHighBoost, MascaraEnfoque,
    AfilacionGradiente, AfilacionWavelet,
    Hessiano, Frangi, Sato, TensorEstructural,
    Laplaciano, LaplacianoCero, Canny, Sobel, Scharr, Prewitt, Roberts,
]


class Controlador_Realzador(Controlador_Base):
    """
    Controlador de realce.

    Idéntico en estructura al Controlador_Filtrador.
    Sin hooks sobreescritos: todos los métodos operan sobre
    slices 2D [Y,X], igual que los filtros.
    Sin Realce_* propios: usa TipoAplicacion de Estrategias_Aplicacion.
    """

    def __init__(self):
        # dominio="realzado" debe coincidir con @registrar_en("realzado")
        super().__init__(etapa="procesamiento", dominio="realzado")
        self._ultimo_metodo: Optional[MetodoRealce] = None

    # =========================================================
    # HELPER INTERNO
    # =========================================================

    def _crear(
        self,
        metodo: MetodoRealce,
        tipo: TipoAplicacion,
        canal: int = 0,
    ):
        self._ultimo_metodo = metodo
        return self.crear_operador(metodo=metodo, tipo_aplicacion=tipo, canal=canal)

    def _crear_multicanal(self, metodo: MetodoRealce, tipo: TipoAplicacion):
        self._ultimo_metodo = metodo
        return self.crear_operador_multicanal(metodo=metodo, tipo_aplicacion=tipo)

    # =========================================================
    # FACTORIES YAML / CLI — devuelven Operacion
    # =========================================================

    # ── CONTRASTE ────────────────────────────────────────────

    def crear_operacion_clahe(
        self,
        clip_limit: float = 2.0,
        grid_size: Tuple[int, int] = (8, 8),
        tipo: TipoAplicacion = PorCorteEspaciotemporal(),
        canal: Optional[int] = 0,
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self.crear_operacion(
            nombre_metodo="clahe",
            categoria=CategoriaOperacion.REALCE,
            tipo_aplicacion=tipo,
            canal=canal,
            nombre=nombre,
            params={"clip_limit": clip_limit, "grid_size": grid_size},
        )

    def crear_operacion_gamma(
        self,
        gamma: float = 1.0,
        tipo: TipoAplicacion = Global(),
        canal: Optional[int] = 0,
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self.crear_operacion(
            nombre_metodo="gamma",
            categoria=CategoriaOperacion.REALCE,
            tipo_aplicacion=tipo,
            canal=canal,
            nombre=nombre,
            params={"gamma": gamma},
        )

    def crear_operacion_logaritmico(
        self,
        escala: float = 1.0,
        tipo: TipoAplicacion = Global(),
        canal: Optional[int] = 0,
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self.crear_operacion(
            nombre_metodo="logaritmico",
            categoria=CategoriaOperacion.REALCE,
            tipo_aplicacion=tipo,
            canal=canal,
            nombre=nombre,
            params={"escala": escala},
        )

    def crear_operacion_retinex(
        self,
        sigma: float = 15.0,
        tipo_retinex: str = "msrcr",
        tipo: TipoAplicacion = PorCorteEspaciotemporal(),
        canal: Optional[int] = 0,
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self.crear_operacion(
            nombre_metodo="retinex",
            categoria=CategoriaOperacion.REALCE,
            tipo_aplicacion=tipo,
            canal=canal,
            nombre=nombre,
            params={"sigma": sigma, "tipo": tipo_retinex},
        )

    def crear_operacion_ecualizacion_histograma(
        self,
        metodo_ecuacion: str = "global",
        tipo: TipoAplicacion = Global(),
        canal: Optional[int] = 0,
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self.crear_operacion(
            nombre_metodo="ecualizacion_histograma",
            categoria=CategoriaOperacion.REALCE,
            tipo_aplicacion=tipo,
            canal=canal,
            nombre=nombre,
            params={"metodo": metodo_ecuacion},
        )

    # ── CONVOLUCIÓN ───────────────────────────────────────────

    def crear_operacion_kernel_personalizado(
        self,
        kernel: np.ndarray,
        tipo: TipoAplicacion = PorCorteEspaciotemporal(),
        canal: Optional[int] = 0,
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self.crear_operacion(
            nombre_metodo="kernel_personalizado",
            categoria=CategoriaOperacion.REALCE,
            tipo_aplicacion=tipo,
            canal=canal,
            nombre=nombre,
            params={"kernel": kernel},
        )

    def crear_operacion_psf_simulacion(
        self,
        tipo_microscopio: str = "confocal",
        na: float = 1.4,
        lambda_nm: float = 550.0,
        tipo: TipoAplicacion = Global(),
        canal: Optional[int] = 0,
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self.crear_operacion(
            nombre_metodo="psf_simulacion",
            categoria=CategoriaOperacion.REALCE,
            tipo_aplicacion=tipo,
            canal=canal,
            nombre=nombre,
            params={"tipo_microscopio": tipo_microscopio, "na": na, "lambda_nm": lambda_nm},
        )

    def crear_operacion_kernel_separable(
        self,
        kernel_x: np.ndarray,
        kernel_y: np.ndarray,
        tipo: TipoAplicacion = PorCorteEspaciotemporal(),
        canal: Optional[int] = 0,
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self.crear_operacion(
            nombre_metodo="kernel_separable",
            categoria=CategoriaOperacion.REALCE,
            tipo_aplicacion=tipo,
            canal=canal,
            nombre=nombre,
            params={"kernel_x": kernel_x, "kernel_y": kernel_y},
        )

    def crear_operacion_convolucion_frecuencia(
        self,
        kernel: np.ndarray,
        tipo: TipoAplicacion = Global(),
        canal: Optional[int] = 0,
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self.crear_operacion(
            nombre_metodo="convolucion_frecuencia",
            categoria=CategoriaOperacion.REALCE,
            tipo_aplicacion=tipo,
            canal=canal,
            nombre=nombre,
            params={"kernel": kernel},
        )

    def crear_operacion_correccion_bordes(
        self,
        metodo: str = "espejo",
        tipo: TipoAplicacion = PorCorteEspaciotemporal(),
        canal: Optional[int] = 0,
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self.crear_operacion(
            nombre_metodo="correccion_bordes",
            categoria=CategoriaOperacion.REALCE,
            tipo_aplicacion=tipo,
            canal=canal,
            nombre=nombre,
            params={"metodo": metodo},
        )

    # ── DECONVOLUCIÓN ─────────────────────────────────────────

    def crear_operacion_wiener(
        self,
        psf: np.ndarray,
        k: float = 0.01,
        balance: float = 0.5,
        iteraciones: int = 1,
        tipo: TipoAplicacion = Global(),
        canal: Optional[int] = 0,
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self.crear_operacion(
            nombre_metodo="wiener",
            categoria=CategoriaOperacion.REALCE,
            tipo_aplicacion=tipo,
            canal=canal,
            nombre=nombre,
            params={"psf": psf, "k": k, "balance": balance, "iteraciones": iteraciones},
        )

    def crear_operacion_richardson_lucy(
        self,
        psf: np.ndarray,
        iteraciones: int = 10,
        tipo: TipoAplicacion = Global(),
        canal: Optional[int] = 0,
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self.crear_operacion(
            nombre_metodo="richardson_lucy",
            categoria=CategoriaOperacion.REALCE,
            tipo_aplicacion=tipo,
            canal=canal,
            nombre=nombre,
            params={"psf": psf, "iteraciones": iteraciones},
        )

    def crear_operacion_blind_deconvolucion(
        self,
        tamano_psf_estimado: Tuple[int, int] = (15, 15),
        iteraciones: int = 10,
        tipo: TipoAplicacion = Global(),
        canal: Optional[int] = 0,
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self.crear_operacion(
            nombre_metodo="blind_deconvolucion",
            categoria=CategoriaOperacion.REALCE,
            tipo_aplicacion=tipo,
            canal=canal,
            nombre=nombre,
            params={"tamano_psf_estimado": tamano_psf_estimado, "iteraciones": iteraciones},
        )

    def crear_operacion_tikhonov(
        self,
        lambda_reg: float = 0.01,
        orden: int = 1,
        tipo: TipoAplicacion = Global(),
        canal: Optional[int] = 0,
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self.crear_operacion(
            nombre_metodo="tikhonov",
            categoria=CategoriaOperacion.REALCE,
            tipo_aplicacion=tipo,
            canal=canal,
            nombre=nombre,
            params={"lambda_reg": lambda_reg, "orden": orden},
        )

    # ── MORFOLÓGICOS ──────────────────────────────────────────

    def crear_operacion_apertura(
        self,
        elemento_estructurante: np.ndarray = None,
        iteraciones: int = 1,
        tipo: TipoAplicacion = PorCorteEspaciotemporal(),
        canal: Optional[int] = 0,
        nombre: Optional[str] = None,
    ) -> Operacion:
        if elemento_estructurante is None:
            elemento_estructurante = np.ones((3, 3))
        return self.crear_operacion(
            nombre_metodo="apertura",
            categoria=CategoriaOperacion.REALCE,
            tipo_aplicacion=tipo,
            canal=canal,
            nombre=nombre,
            params={"elemento_estructurante": elemento_estructurante, "iteraciones": iteraciones},
        )

    def crear_operacion_cierre(
        self,
        elemento_estructurante: np.ndarray = None,
        iteraciones: int = 1,
        tipo: TipoAplicacion = PorCorteEspaciotemporal(),
        canal: Optional[int] = 0,
        nombre: Optional[str] = None,
    ) -> Operacion:
        if elemento_estructurante is None:
            elemento_estructurante = np.ones((3, 3))
        return self.crear_operacion(
            nombre_metodo="cierre",
            categoria=CategoriaOperacion.REALCE,
            tipo_aplicacion=tipo,
            canal=canal,
            nombre=nombre,
            params={"elemento_estructurante": elemento_estructurante, "iteraciones": iteraciones},
        )

    def crear_operacion_top_hat(
        self,
        elemento_estructurante: np.ndarray = None,
        tipo: TipoAplicacion = PorCorteEspaciotemporal(),
        canal: Optional[int] = 0,
        nombre: Optional[str] = None,
    ) -> Operacion:
        if elemento_estructurante is None:
            elemento_estructurante = np.ones((15, 15))
        return self.crear_operacion(
            nombre_metodo="top_hat",
            categoria=CategoriaOperacion.REALCE,
            tipo_aplicacion=tipo,
            canal=canal,
            nombre=nombre,
            params={"elemento_estructurante": elemento_estructurante},
        )

    def crear_operacion_bottom_hat(
        self,
        elemento_estructurante: np.ndarray = None,
        tipo: TipoAplicacion = PorCorteEspaciotemporal(),
        canal: Optional[int] = 0,
        nombre: Optional[str] = None,
    ) -> Operacion:
        if elemento_estructurante is None:
            elemento_estructurante = np.ones((15, 15))
        return self.crear_operacion(
            nombre_metodo="bottom_hat",
            categoria=CategoriaOperacion.REALCE,
            tipo_aplicacion=tipo,
            canal=canal,
            nombre=nombre,
            params={"elemento_estructurante": elemento_estructurante},
        )

    def crear_operacion_gradiente_morfologico(
        self,
        elemento_estructurante: np.ndarray = None,
        tipo: TipoAplicacion = PorCorteEspaciotemporal(),
        canal: Optional[int] = 0,
        nombre: Optional[str] = None,
    ) -> Operacion:
        if elemento_estructurante is None:
            elemento_estructurante = np.ones((3, 3))
        return self.crear_operacion(
            nombre_metodo="gradiente_morfologico",
            categoria=CategoriaOperacion.REALCE,
            tipo_aplicacion=tipo,
            canal=canal,
            nombre=nombre,
            params={"elemento_estructurante": elemento_estructurante},
        )

    def crear_operacion_reconstruccion_morfologica(
        self,
        mascara: np.ndarray = None,
        metodo: str = "dilatacion",
        iteraciones: int = -1,
        tipo: TipoAplicacion = PorCorteEspaciotemporal(),
        canal: Optional[int] = 0,
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self.crear_operacion(
            nombre_metodo="reconstruccion_morfologica",
            categoria=CategoriaOperacion.REALCE,
            tipo_aplicacion=tipo,
            canal=canal,
            nombre=nombre,
            params={"mascara": mascara, "metodo": metodo, "iteraciones": iteraciones},
        )

    # ── AFILACIÓN ─────────────────────────────────────────────

    def crear_operacion_afilacion_laplaciana(
        self,
        alpha: float = 1.0,
        tipo_laplaciano: str = "discreto",
        tipo: TipoAplicacion = PorCorteEspaciotemporal(),
        canal: Optional[int] = 0,
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self.crear_operacion(
            nombre_metodo="afilacion_laplaciana",
            categoria=CategoriaOperacion.REALCE,
            tipo_aplicacion=tipo,
            canal=canal,
            nombre=nombre,
            params={"alpha": alpha, "tipo": tipo_laplaciano},
        )

    def crear_operacion_high_boost(
        self,
        A: float = 1.5,
        tipo: TipoAplicacion = PorCorteEspaciotemporal(),
        canal: Optional[int] = 0,
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self.crear_operacion(
            nombre_metodo="filtro_high_boost",
            categoria=CategoriaOperacion.REALCE,
            tipo_aplicacion=tipo,
            canal=canal,
            nombre=nombre,
            params={"A": A},
        )

    def crear_operacion_mascara_enfoque(
        self,
        radio_desenfoque: float = 2.0,
        cantidad: float = 1.0,
        umbral: float = 0.0,
        tipo: TipoAplicacion = PorCorteEspaciotemporal(),
        canal: Optional[int] = 0,
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self.crear_operacion(
            nombre_metodo="mascara_enfoque",
            categoria=CategoriaOperacion.REALCE,
            tipo_aplicacion=tipo,
            canal=canal,
            nombre=nombre,
            params={"radio_desenfoque": radio_desenfoque, "cantidad": cantidad, "umbral": umbral},
        )

    def crear_operacion_afilacion_gradiente(
        self,
        operador: str = "sobel",
        escala: float = 1.0,
        tipo: TipoAplicacion = PorCorteEspaciotemporal(),
        canal: Optional[int] = 0,
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self.crear_operacion(
            nombre_metodo="afilacion_gradiente",
            categoria=CategoriaOperacion.REALCE,
            tipo_aplicacion=tipo,
            canal=canal,
            nombre=nombre,
            params={"operador": operador, "escala": escala},
        )

    def crear_operacion_afilacion_wavelet(
        self,
        wavelet: str = "db4",
        umbral: float = 0.1,
        tipo: TipoAplicacion = Global(),
        canal: Optional[int] = 0,
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self.crear_operacion(
            nombre_metodo="afilacion_wavelet",
            categoria=CategoriaOperacion.REALCE,
            tipo_aplicacion=tipo,
            canal=canal,
            nombre=nombre,
            params={"wavelet": wavelet, "umbral": umbral},
        )

    # ── ESTRUCTURALES ─────────────────────────────────────────

    def crear_operacion_hessiano(
        self,
        sigma: float = 1.0,
        tipo_estructura: str = "blob",
        tipo: TipoAplicacion = PorCorteEspaciotemporal(),
        canal: Optional[int] = 0,
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self.crear_operacion(
            nombre_metodo="hessiano",
            categoria=CategoriaOperacion.REALCE,
            tipo_aplicacion=tipo,
            canal=canal,
            nombre=nombre,
            params={"sigma": sigma, "tipo": tipo_estructura},
        )

    def crear_operacion_frangi(
        self,
        sigmas: Tuple[float, ...] = (1.0, 2.0, 4.0),
        alpha: float = 0.5,
        beta: float = 0.5,
        gamma: float = 15.0,
        tipo: TipoAplicacion = PorCorteEspaciotemporal(),
        canal: Optional[int] = 0,
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self.crear_operacion(
            nombre_metodo="frangi",
            categoria=CategoriaOperacion.REALCE,
            tipo_aplicacion=tipo,
            canal=canal,
            nombre=nombre,
            params={"sigmas": sigmas, "alpha": alpha, "beta": beta, "gamma": gamma},
        )

    def crear_operacion_sato(
        self,
        sigmas: Tuple[float, ...] = (1.0, 2.0, 4.0),
        tipo: TipoAplicacion = PorCorteEspaciotemporal(),
        canal: Optional[int] = 0,
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self.crear_operacion(
            nombre_metodo="sato",
            categoria=CategoriaOperacion.REALCE,
            tipo_aplicacion=tipo,
            canal=canal,
            nombre=nombre,
            params={"sigmas": sigmas},
        )

    def crear_operacion_tensor_estructural(
        self,
        sigma_gradiente: float = 1.0,
        sigma_tension: float = 2.0,
        tipo: TipoAplicacion = PorCorteEspaciotemporal(),
        canal: Optional[int] = 0,
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self.crear_operacion(
            nombre_metodo="tensor_estructural",
            categoria=CategoriaOperacion.REALCE,
            tipo_aplicacion=tipo,
            canal=canal,
            nombre=nombre,
            params={"sigma_gradiente": sigma_gradiente, "sigma_tension": sigma_tension},
        )

    # ── GRADIENTES ────────────────────────────────────────────

    def crear_operacion_laplaciano(
        self,
        tipo_lap: str = "discreto",
        tipo: TipoAplicacion = PorCorteEspaciotemporal(),
        canal: Optional[int] = 0,
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self.crear_operacion(
            nombre_metodo="laplaciano",
            categoria=CategoriaOperacion.REALCE,
            tipo_aplicacion=tipo,
            canal=canal,
            nombre=nombre,
            params={"tipo": tipo_lap},
        )

    def crear_operacion_laplaciano_cero(
        self,
        umbral: float = 0.0,
        tipo: TipoAplicacion = PorCorteEspaciotemporal(),
        canal: Optional[int] = 0,
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self.crear_operacion(
            nombre_metodo="laplaciano_cero",
            categoria=CategoriaOperacion.REALCE,
            tipo_aplicacion=tipo,
            canal=canal,
            nombre=nombre,
            params={"umbral": umbral},
        )

    def crear_operacion_canny(
        self,
        sigma: float = 1.0,
        umbral_bajo: float = 0.1,
        umbral_alto: float = 0.3,
        tipo: TipoAplicacion = PorCorteEspaciotemporal(),
        canal: Optional[int] = 0,
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self.crear_operacion(
            nombre_metodo="canny",
            categoria=CategoriaOperacion.REALCE,
            tipo_aplicacion=tipo,
            canal=canal,
            nombre=nombre,
            params={"sigma": sigma, "umbral_bajo": umbral_bajo, "umbral_alto": umbral_alto},
        )

    def crear_operacion_sobel(
        self,
        dx: int = 1,
        dy: int = 1,
        ksize: int = 3,
        tipo: TipoAplicacion = PorCorteEspaciotemporal(),
        canal: Optional[int] = 0,
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self.crear_operacion(
            nombre_metodo="sobel",
            categoria=CategoriaOperacion.REALCE,
            tipo_aplicacion=tipo,
            canal=canal,
            nombre=nombre,
            params={"dx": dx, "dy": dy, "ksize": ksize},
        )

    def crear_operacion_scharr(
        self,
        dx: int = 1,
        dy: int = 1,
        tipo: TipoAplicacion = PorCorteEspaciotemporal(),
        canal: Optional[int] = 0,
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self.crear_operacion(
            nombre_metodo="scharr",
            categoria=CategoriaOperacion.REALCE,
            tipo_aplicacion=tipo,
            canal=canal,
            nombre=nombre,
            params={"dx": dx, "dy": dy},
        )

    def crear_operacion_prewitt(
        self,
        dx: int = 1,
        dy: int = 1,
        tipo: TipoAplicacion = PorCorteEspaciotemporal(),
        canal: Optional[int] = 0,
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self.crear_operacion(
            nombre_metodo="prewitt",
            categoria=CategoriaOperacion.REALCE,
            tipo_aplicacion=tipo,
            canal=canal,
            nombre=nombre,
            params={"dx": dx, "dy": dy},
        )

    def crear_operacion_roberts(
        self,
        tipo: TipoAplicacion = PorCorteEspaciotemporal(),
        canal: Optional[int] = 0,
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self.crear_operacion(
            nombre_metodo="roberts",
            categoria=CategoriaOperacion.REALCE,
            tipo_aplicacion=tipo,
            canal=canal,
            nombre=nombre,
            params={},
        )

    # =========================================================
    # USO IMPERATIVO
    # =========================================================

    def aplicar(
        self,
        data: BioImagenData,
        metodo: MetodoRealce,
        tipo: TipoAplicacion,
        canal: int = 0,
    ) -> Resultado[BioImagenData, ErrorBioImagen]:
        return self._crear(metodo, tipo, canal)(data)

    def aplicar_multicanal(
        self,
        data: BioImagenData,
        metodo: MetodoRealce,
        tipo: TipoAplicacion,
    ) -> Resultado[BioImagenData, ErrorBioImagen]:
        return self._crear_multicanal(metodo, tipo)(data)

    def __repr__(self):
        nombre = getattr(self._ultimo_metodo, "nombre", "None")
        return f"<Controlador_Realzador ultimo={nombre}>"