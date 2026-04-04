# === segmentador/Controlador_Segmentador.py ===
from __future__ import annotations

import numpy as np
from dataclasses import replace
from typing import Union, Optional, Tuple, List, Any, Dict

# Core
from .Controlador_Base import Controlador_Base

# Sistema
from .Resultado_Either import Resultado
from .Controlador_BioImagen import BioImagenData, ErrorBioImagen
from ..gestorLab.Categoria_Operacion import CategoriaOperacion
from ..gestorLab.Operacion import Operacion, TipoDato

# Estrategias — sin override
from .Estrategias_Aplicacion import (
    TipoAplicacion,
    Global,
    PorCorteZ,
    PorTimepoint,
    PorCorteEspaciotemporal,
)

# Métodos
from ..segmentador.binarizacion.Segmentadores_Binarizacion import (
    Otsu, Global as UmbralGlobal, Adaptativo, Percentil,
    Triangle, Isodata, Minimum, Mean,
)
from ..segmentador.instancial.Segmentadores_Instanciales import (
    Watershed, WatershedMarcado, DistanciaWatershed,
    SplitDistancial, WatershedHibrido, SplitWatershed,
)
from ..segmentador.regional.Segmentadores_Regionales import (
    RegionGrowing, RandomWalk, CorteGrafico,
    SuperpixelSLIC, SuperpixelFelzenszwalb,
    WatershedRegiones, MeanShiftSegmentacion,
)


# Métodos instanciales que actúan como puntos de split en el pipeline
_METODOS_SPLIT = {
    "watershed", "watershed_marcado", "distancia_watershed",
    "split_distancial", "watershed_hibrido", "split_watershed",
}

MetodoSegmentacion = Union[
    Otsu, UmbralGlobal, Adaptativo, Percentil, Triangle, Isodata, Minimum, Mean,
    Watershed, WatershedMarcado, DistanciaWatershed,
    SplitDistancial, WatershedHibrido, SplitWatershed,
    RegionGrowing, RandomWalk, CorteGrafico,
    SuperpixelSLIC, SuperpixelFelzenszwalb, WatershedRegiones, MeanShiftSegmentacion,
]


class Controlador_Segmentador(Controlador_Base):
    """
    Controlador de segmentación.

    Diferencias con Controlador_Filtrador/Realzador:
      Los métodos producen máscaras enteras (uint16), no float64.
      Algunos métodos devuelven shapes con dimensiones extra que hay
      que normalizar (squeeze) antes de reinsertar.

    Hooks sobreescritos (solo dos):
      _postprocesar → squeeze + conversión a uint16 + reinserción
      _validar_salida → valida shape (T,Z,Y,X) tras el squeeze

    _preprocesar NO se sobreescribe:
      el base ya devuelve (T,Z,Y,X) float64 — las estrategias
      se encargan de iterar y pasar [Y,X] 2D al método.
    """

    def __init__(self):
        # dominio="segmentacion" debe coincidir con @registrar_en("segmentacion")
        super().__init__(etapa="segmentacion", dominio="segmentacion")
        self._ultimo_metodo: Optional[MetodoSegmentacion] = None

    # =========================================================
    # HOOKS SOBREESCRITOS
    # =========================================================

    def _postprocesar(
        self,
        data: BioImagenData,
        resultado: Any,
        canal: int,
    ) -> Any:
        """
        Normaliza shape y dtype del resultado antes de reinsertar.

        El squeeze resuelve casos donde el método devuelve [Y,X,1]
        o [1,Y,X] en lugar de [Y,X].

        La conversión a uint16 es necesaria porque:
          - Los métodos de binarización devuelven bool o uint8
          - Los de instancial/watershed devuelven int32 con etiquetas
          - Todos deben unificarse a uint16 para el pipeline posterior

        Si el método devuelve float (algunos umbralizadores avanzados
        producen mapas de probabilidad), se redondea y convierte.
        """
        if not isinstance(resultado, np.ndarray):
            return resultado  # Figure, DataFrame, etc. — sin tocar

        arr = resultado

        # ── Normalizar shape ──────────────────────────────────
        # Caso: [Y, X, 1] → [Y, X]
        if arr.ndim == 3 and arr.shape[2] == 1:
            arr = arr[:, :, 0]
        # Caso: [1, Y, X] → [Y, X]
        elif arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]
        # Caso 1D con tamaño compatible (edge case, pero existe)
        elif arr.ndim == 1:
            T, Z, Y, X = data.datos.shape[0], data.datos.shape[1], \
                         data.datos.shape[3], data.datos.shape[4]
            esperado = Y * X
            if arr.size == esperado:
                arr = arr.reshape(Y, X)
            else:
                raise ValueError(
                    f"Shape 1D incompatible: {arr.size} elementos, "
                    f"esperado {esperado} para ({Y}, {X})"
                )

        # ── Conversión a uint16 ───────────────────────────────
        # float → redondear antes de convertir (evita overflow silencioso)
        if np.issubdtype(arr.dtype, np.floating):
            arr = np.round(arr).astype(np.uint16)
        elif arr.dtype != np.uint16:
            arr = arr.astype(np.uint16)

        # ── Reinsertar canal en copia de datos ────────────────
        nuevos = data.datos.copy()
        nuevos[:, :, canal, :, :] = arr
        return replace(data, datos=nuevos)

    def _validar_salida(self, resultado: Any, canal_data: np.ndarray) -> None:
        """
        Valida shape (T,Z,Y,X) tras el squeeze que hace _postprocesar.
        El base compara shape exacto; aquí el resultado puede ser 2D [Y,X]
        (un solo slice) — ese caso es válido y el _postprocesar lo maneja.
        Solo rechazamos shapes incompatibles.
        """
        if not isinstance(resultado, np.ndarray):
            return

        # Shapes aceptados:
        #   [Y, X]          → slice 2D (caso normal de estrategia 2D)
        #   [T, Z, Y, X]    → volumen completo (estrategia Global)
        #   [T, Y, X]       → PorCorteZ
        #   [Z, Y, X]       → PorTimepoint
        T, Z, Y, X = canal_data.shape
        shapes_validos = {
            (Y, X),
            (T, Z, Y, X),
            (T, Y, X),
            (Z, Y, X),
        }

        # Normalizar dims extra antes de validar (igual que _postprocesar)
        arr = resultado
        if arr.ndim == 3 and arr.shape[2] == 1:
            arr = arr[:, :, 0]
        elif arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]

        if arr.shape not in shapes_validos:
            raise ValueError(
                f"Shape inválido para segmentación: {arr.shape}. "
                f"Aceptados: {shapes_validos}"
            )

    # =========================================================
    # HELPER INTERNO
    # =========================================================

    def _crear(
        self,
        metodo: MetodoSegmentacion,
        tipo: TipoAplicacion,
        canal: int = 0,
    ):
        self._ultimo_metodo = metodo
        return self.crear_operador(metodo=metodo, tipo_aplicacion=tipo, canal=canal)

    def _crear_multicanal(self, metodo: MetodoSegmentacion, tipo: TipoAplicacion):
        self._ultimo_metodo = metodo
        return self.crear_operador_multicanal(metodo=metodo, tipo_aplicacion=tipo)

    def _crear_operacion_seg(
        self,
        nombre_metodo: str,
        tipo: TipoAplicacion,
        canal: Optional[int],
        nombre: Optional[str],
        params: Dict[str, Any],
    ) -> Operacion:
        """
        Wrapper sobre crear_operacion que fija tipo_salida=MASCARA
        y determina es_operacion_split según el nombre del método.
        No pertenece al base porque split y MASCARA son exclusivos
        del dominio de segmentación.
        """
        op = self.crear_operacion(
            nombre_metodo=nombre_metodo,
            categoria=CategoriaOperacion.SEGMENTACION,
            tipo_aplicacion=tipo,
            canal=canal,
            nombre=nombre,
            params=params,
            tipo_salida=TipoDato.MASCARA,
        )
        # Operacion es frozen → usamos replace para agregar el flag
        return replace(op, es_operacion_split=(nombre_metodo in _METODOS_SPLIT))

    # =========================================================
    # FACTORIES YAML / CLI — devuelven Operacion
    # =========================================================

    # ── BINARIZACIÓN ─────────────────────────────────────────

    def crear_operacion_otsu(
        self,
        tipo: TipoAplicacion = PorCorteEspaciotemporal(),
        canal: Optional[int] = 0,
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self._crear_operacion_seg("otsu", tipo, canal, nombre, {})

    def crear_operacion_umbral_global(
        self,
        umbral: float = 128.0,
        tipo: TipoAplicacion = PorCorteEspaciotemporal(),
        canal: Optional[int] = 0,
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self._crear_operacion_seg(
            "umbral_global", tipo, canal, nombre, {"umbral": umbral}
        )

    def crear_operacion_adaptativo(
        self,
        tamano_bloque: int = 11,
        C: float = 2.0,
        metodo: str = "gaussian",
        tipo: TipoAplicacion = PorCorteEspaciotemporal(),
        canal: Optional[int] = 0,
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self._crear_operacion_seg(
            "adaptativo", tipo, canal, nombre,
            {"tamano_bloque": tamano_bloque, "C": C, "metodo": metodo},
        )

    def crear_operacion_percentil(
        self,
        percentil: float = 95.0,
        tipo: TipoAplicacion = PorCorteEspaciotemporal(),
        canal: Optional[int] = 0,
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self._crear_operacion_seg(
            "percentil", tipo, canal, nombre, {"percentil": percentil}
        )

    def crear_operacion_triangle(
        self,
        nbins: int = 256,
        tipo: TipoAplicacion = PorCorteEspaciotemporal(),
        canal: Optional[int] = 0,
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self._crear_operacion_seg(
            "triangle", tipo, canal, nombre, {"nbins": nbins}
        )

    def crear_operacion_isodata(
        self,
        nbins: int = 256,
        tipo: TipoAplicacion = PorCorteEspaciotemporal(),
        canal: Optional[int] = 0,
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self._crear_operacion_seg(
            "isodata", tipo, canal, nombre, {"nbins": nbins}
        )

    def crear_operacion_minimum(
        self,
        nbins: int = 256,
        tipo: TipoAplicacion = PorCorteEspaciotemporal(),
        canal: Optional[int] = 0,
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self._crear_operacion_seg(
            "minimum", tipo, canal, nombre, {"nbins": nbins}
        )

    def crear_operacion_mean(
        self,
        tipo: TipoAplicacion = PorCorteEspaciotemporal(),
        canal: Optional[int] = 0,
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self._crear_operacion_seg("mean", tipo, canal, nombre, {})

    # ── INSTANCIALES ──────────────────────────────────────────

    def crear_operacion_watershed(
        self,
        compactness: float = 0.0,
        tipo: TipoAplicacion = PorCorteEspaciotemporal(),
        canal: Optional[int] = 0,
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self._crear_operacion_seg(
            "watershed", tipo, canal, nombre, {"compactness": compactness}
        )

    def crear_operacion_watershed_marcado(
        self,
        marcadores: np.ndarray = None,
        tipo: TipoAplicacion = PorCorteEspaciotemporal(),
        canal: Optional[int] = 0,
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self._crear_operacion_seg(
            "watershed_marcado", tipo, canal, nombre,
            {"marcadores": marcadores},
        )

    def crear_operacion_distancia_watershed(
        self,
        umbral_distancia: float = 0.5,
        tipo: TipoAplicacion = PorCorteEspaciotemporal(),
        canal: Optional[int] = 0,
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self._crear_operacion_seg(
            "distancia_watershed", tipo, canal, nombre,
            {"umbral_distancia": umbral_distancia},
        )

    def crear_operacion_split_distancial(
        self,
        min_distancia: float = 10.0,
        max_objetos: int = 1000,
        tipo: TipoAplicacion = PorCorteEspaciotemporal(),
        canal: Optional[int] = 0,
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self._crear_operacion_seg(
            "split_distancial", tipo, canal, nombre,
            {"min_distancia": min_distancia, "max_objetos": max_objetos},
        )

    def crear_operacion_watershed_hibrido(
        self,
        umbral_binarizacion: float = 0.5,
        compactness: float = 1.0,
        tipo: TipoAplicacion = PorCorteEspaciotemporal(),
        canal: Optional[int] = 0,
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self._crear_operacion_seg(
            "watershed_hibrido", tipo, canal, nombre,
            {"umbral_binarizacion": umbral_binarizacion, "compactness": compactness},
        )

    def crear_operacion_split_watershed(
        self,
        metodo_split: str = "distancia",
        tipo: TipoAplicacion = PorCorteEspaciotemporal(),
        canal: Optional[int] = 0,
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self._crear_operacion_seg(
            "split_watershed", tipo, canal, nombre,
            {"metodo_split": metodo_split},
        )

    # ── REGIONALES ────────────────────────────────────────────

    def crear_operacion_region_growing(
        self,
        semillas: List[Tuple[int, int]] = None,
        criterio: str = "intensidad",
        tolerancia: float = 10.0,
        tipo: TipoAplicacion = PorCorteEspaciotemporal(),
        canal: Optional[int] = 0,
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self._crear_operacion_seg(
            "region_growing", tipo, canal, nombre,
            {"semillas": semillas or [], "criterio": criterio, "tolerancia": tolerancia},
        )

    def crear_operacion_random_walk(
        self,
        semillas: np.ndarray = None,
        beta: float = 130.0,
        modo: str = "bf",
        tipo: TipoAplicacion = PorCorteEspaciotemporal(),
        canal: Optional[int] = 0,
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self._crear_operacion_seg(
            "random_walk", tipo, canal, nombre,
            {"semillas": semillas, "beta": beta, "modo": modo},
        )

    def crear_operacion_corte_grafico(
        self,
        tipo: TipoAplicacion = PorCorteEspaciotemporal(),
        canal: Optional[int] = 0,
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self._crear_operacion_seg(
            "corte_grafico", tipo, canal, nombre, {}
        )

    def crear_operacion_slic(
        self,
        n_segmentos: int = 100,
        compactness: float = 10.0,
        sigma: float = 1.0,
        tipo: TipoAplicacion = Global(),
        canal: Optional[int] = 0,
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self._crear_operacion_seg(
            "superpixel_slic", tipo, canal, nombre,
            {"n_segmentos": n_segmentos, "compactness": compactness, "sigma": sigma},
        )

    def crear_operacion_felzenszwalb(
        self,
        escala: float = 1.0,
        sigma: float = 0.8,
        min_size: int = 20,
        tipo: TipoAplicacion = Global(),
        canal: Optional[int] = 0,
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self._crear_operacion_seg(
            "superpixel_felzenszwalb", tipo, canal, nombre,
            {"escala": escala, "sigma": sigma, "min_size": min_size},
        )

    def crear_operacion_watershed_regiones(
        self,
        tipo: TipoAplicacion = PorCorteEspaciotemporal(),
        canal: Optional[int] = 0,
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self._crear_operacion_seg(
            "watershed_regiones", tipo, canal, nombre, {}
        )

    def crear_operacion_mean_shift(
        self,
        ancho_banda_espacial: float = 15.0,
        ancho_banda_rango: float = 15.0,
        min_densidad: int = 50,
        tipo: TipoAplicacion = Global(),
        canal: Optional[int] = 0,
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self._crear_operacion_seg(
            "mean_shift_segmentacion", tipo, canal, nombre,
            {
                "ancho_banda_espacial": ancho_banda_espacial,
                "ancho_banda_rango": ancho_banda_rango,
                "min_densidad": min_densidad,
            },
        )

    # =========================================================
    # USO IMPERATIVO
    # =========================================================

    def aplicar(
        self,
        data: BioImagenData,
        metodo: MetodoSegmentacion,
        tipo: TipoAplicacion,
        canal: int = 0,
    ) -> Resultado[BioImagenData, ErrorBioImagen]:
        return self._crear(metodo, tipo, canal)(data)

    def aplicar_multicanal(
        self,
        data: BioImagenData,
        metodo: MetodoSegmentacion,
        tipo: TipoAplicacion,
    ) -> Resultado[BioImagenData, ErrorBioImagen]:
        return self._crear_multicanal(metodo, tipo)(data)

    def __repr__(self):
        nombre = getattr(self._ultimo_metodo, "nombre", "None")
        return f"<Controlador_Segmentador ultimo={nombre}>"