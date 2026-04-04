# === transformador/Controlador_Transformador.py ===
from __future__ import annotations

import numpy as np
from typing import Union, Optional

# Core
from .Controlador_Base import Controlador_Base

# Sistema
from .Resultado_Either import Resultado
from .Controlador_BioImagen import BioImagenData, ErrorBioImagen
from ..gestorLab.Categoria_Operacion import CategoriaOperacion
from ..gestorLab.Operacion import Operacion, TipoSalida

# Estrategias
from .Estrategias_Aplicacion import (
    TipoAplicacion,
    Global,
    PorCorteZ,
    PorTimepoint,
    PorCorteEspaciotemporal,
    PorVolumen3D,
)

# Métodos
from ..transformador.geometricos.Transformadores_Geometricos import (
    TransformacionDistancia,
    Esqueletizacion,
    EjeMedial,
    Deformar,
    Redimensionar,
    Rotacion,
    Remuestreo,
)
from ..transformador.espectrales.Transformadores_Espectrales import (
    Fourier,
    Wavelet,
    Gabor,
)
from ..transformador.integrales.Transformadores_Proyectivos import (
    Radon,
    IntegralDeLinea,
    Hough,
    TransformadaDistanciaGeodesica,
    TransformadaHilbert,
    Abel,
)


MetodoGeometrico = Union[
    TransformacionDistancia, Esqueletizacion, EjeMedial,
    Deformar, Redimensionar, Rotacion, Remuestreo,
]

MetodoEspectral = Union[
    Fourier, Wavelet, Gabor,
]

MetodoProyectivo = Union[
    Radon, IntegralDeLinea, Hough,
    TransformadaDistanciaGeodesica, TransformadaHilbert, Abel,
]

MetodoTransformador = Union[
    MetodoGeometrico, MetodoEspectral, MetodoProyectivo,
]

# Proyectivos que cambian el shape de salida — _validar_salida los excluye
_PROYECTIVOS_SHAPE_LIBRE = {
    "radon", "integral_de_linea", "hough",
    "transformada_hilbert", "abel",
}


class Controlador_Transformador(Controlador_Base):
    """
    Controlador de transformaciones geométricas, espectrales y proyectivas.

    Tres submódulos con comportamientos distintos:

      Geométricos (Rotacion, Redimensionar, Deformar, ...):
        - [Y,X] → [Y,X]  mismo shape
        - Sin hooks sobreescritos

      Espectrales (Fourier, Wavelet, Gabor):
        - [Y,X] → [Y,X]  mismo shape (módulo, magnitud, o respuesta)
        - Sin hooks sobreescritos
        - Fourier devuelve float64 (módulo), no complejo

      Proyectivos/Integrales (Radon, Hough, Abel, ...):
        - [Y,X] → shape distinto (sinograma, acumulador, proyección)
        - _validar_salida sobreescrito para permitir shape libre
        - TipoSalida.FEATURES en lugar de IMAGEN

    Hook sobreescrito (solo uno):
      _validar_salida → omite validación de shape para proyectivos
    """

    def __init__(self):
        # dominio="transformacion" debe coincidir con @registrar_en("transformacion")
        super().__init__(etapa="procesamiento", dominio="transformacion")
        self._ultimo_metodo: Optional[MetodoTransformador] = None

    # =========================================================
    # HOOK SOBREESCRITO
    # =========================================================

    def _validar_salida(self, resultado, canal_data: np.ndarray) -> None:
        """
        Los proyectivos producen shapes distintos al input (sinograma,
        acumulador de Hough, etc.) — no validamos shape para ellos.
        Para geométricos y espectrales se mantiene la validación del base.
        """
        if not isinstance(resultado, np.ndarray):
            return

        nombre = getattr(self._ultimo_metodo, "nombre", "")
        if nombre in _PROYECTIVOS_SHAPE_LIBRE:
            return  # shape libre — el método es responsable de su output

        # Geométricos y espectrales: mismo shape que el input
        if resultado.shape != canal_data.shape:
            raise ValueError(
                f"Shape inválido en '{nombre}': "
                f"esperado {canal_data.shape}, obtenido {resultado.shape}"
            )

    # =========================================================
    # HELPER INTERNO
    # =========================================================

    def _crear(
        self,
        metodo: MetodoTransformador,
        tipo: TipoAplicacion,
        canal: int = 0,
    ):
        self._ultimo_metodo = metodo
        return self.crear_operador(metodo=metodo, tipo_aplicacion=tipo, canal=canal)

    def _crear_multicanal(
        self,
        metodo: MetodoTransformador,
        tipo: TipoAplicacion,
    ):
        self._ultimo_metodo = metodo
        return self.crear_operador_multicanal(metodo=metodo, tipo_aplicacion=tipo)

    def _tipo_salida(self, nombre_metodo: str) -> TipoSalida:
        """
        Proyectivos producen features/sinogramas, no imágenes.
        El resto conserva TipoSalida.IMAGEN.
        """
        return (
            TipoSalida.FEATURES
            if nombre_metodo in _PROYECTIVOS_SHAPE_LIBRE
            else TipoSalida.IMAGEN
        )

    # =========================================================
    # FACTORIES YAML / CLI — devuelven Operacion
    # =========================================================

    # ── GEOMÉTRICOS ───────────────────────────────────────────

    def crear_operacion_rotacion(
        self,
        # TODO: completar parámetros (angulo, orden_interpolacion, ...)
        tipo: TipoAplicacion = PorCorteEspaciotemporal(),
        canal: Optional[int] = 0,
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self.crear_operacion(
            nombre_metodo="rotacion",
            categoria=CategoriaOperacion.TRANSFORMACION,
            tipo_aplicacion=tipo,
            canal=canal,
            nombre=nombre,
            params={},  # TODO
            tipo_salida=TipoSalida.IMAGEN,
        )

    def crear_operacion_redimensionar(
        self,
        # TODO: completar parámetros (factor, shape_objetivo, ...)
        tipo: TipoAplicacion = PorCorteEspaciotemporal(),
        canal: Optional[int] = 0,
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self.crear_operacion(
            nombre_metodo="redimensionar",
            categoria=CategoriaOperacion.TRANSFORMACION,
            tipo_aplicacion=tipo,
            canal=canal,
            nombre=nombre,
            params={},  # TODO
            tipo_salida=TipoSalida.IMAGEN,
        )

    def crear_operacion_remuestreo(
        self,
        # TODO: completar parámetros (factor_z, factor_xy, ...)
        tipo: TipoAplicacion = PorVolumen3D(),
        canal: Optional[int] = 0,
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self.crear_operacion(
            nombre_metodo="remuestreo",
            categoria=CategoriaOperacion.TRANSFORMACION,
            tipo_aplicacion=tipo,
            canal=canal,
            nombre=nombre,
            params={},  # TODO
            tipo_salida=TipoSalida.IMAGEN,
        )

    def crear_operacion_deformar(
        self,
        # TODO: completar parámetros (campo_deformacion, modo, ...)
        tipo: TipoAplicacion = PorCorteEspaciotemporal(),
        canal: Optional[int] = 0,
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self.crear_operacion(
            nombre_metodo="deformar",
            categoria=CategoriaOperacion.TRANSFORMACION,
            tipo_aplicacion=tipo,
            canal=canal,
            nombre=nombre,
            params={},  # TODO
            tipo_salida=TipoSalida.IMAGEN,
        )

    def crear_operacion_transformacion_distancia(
        self,
        # TODO: completar parámetros (metrica, ...)
        tipo: TipoAplicacion = PorCorteEspaciotemporal(),
        canal: Optional[int] = 0,
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self.crear_operacion(
            nombre_metodo="transformacion_distancia",
            categoria=CategoriaOperacion.TRANSFORMACION,
            tipo_aplicacion=tipo,
            canal=canal,
            nombre=nombre,
            params={},  # TODO
            tipo_salida=TipoSalida.IMAGEN,
        )

    def crear_operacion_esqueletizacion(
        self,
        # TODO: completar parámetros (metodo, ...)
        tipo: TipoAplicacion = PorCorteEspaciotemporal(),
        canal: Optional[int] = 0,
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self.crear_operacion(
            nombre_metodo="esqueletizacion",
            categoria=CategoriaOperacion.TRANSFORMACION,
            tipo_aplicacion=tipo,
            canal=canal,
            nombre=nombre,
            params={},  # TODO
            tipo_salida=TipoSalida.IMAGEN,
        )

    def crear_operacion_eje_medial(
        self,
        # TODO: completar parámetros
        tipo: TipoAplicacion = PorCorteEspaciotemporal(),
        canal: Optional[int] = 0,
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self.crear_operacion(
            nombre_metodo="eje_medial",
            categoria=CategoriaOperacion.TRANSFORMACION,
            tipo_aplicacion=tipo,
            canal=canal,
            nombre=nombre,
            params={},  # TODO
            tipo_salida=TipoSalida.IMAGEN,
        )

    # ── ESPECTRALES ───────────────────────────────────────────

    def crear_operacion_fourier(
        self,
        # TODO: completar parámetros (shift, modulo, ...)
        tipo: TipoAplicacion = PorCorteEspaciotemporal(),
        canal: Optional[int] = 0,
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self.crear_operacion(
            nombre_metodo="fourier",
            categoria=CategoriaOperacion.TRANSFORMACION,
            tipo_aplicacion=tipo,
            canal=canal,
            nombre=nombre,
            params={},  # TODO
            tipo_salida=TipoSalida.IMAGEN,
        )

    def crear_operacion_wavelet(
        self,
        # TODO: completar parámetros (wavelet, nivel, ...)
        tipo: TipoAplicacion = PorCorteEspaciotemporal(),
        canal: Optional[int] = 0,
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self.crear_operacion(
            nombre_metodo="wavelet",
            categoria=CategoriaOperacion.TRANSFORMACION,
            tipo_aplicacion=tipo,
            canal=canal,
            nombre=nombre,
            params={},  # TODO
            tipo_salida=TipoSalida.IMAGEN,
        )

    def crear_operacion_gabor(
        self,
        # TODO: completar parámetros (frecuencia, theta, sigma_x, sigma_y, ...)
        tipo: TipoAplicacion = PorCorteEspaciotemporal(),
        canal: Optional[int] = 0,
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self.crear_operacion(
            nombre_metodo="gabor",
            categoria=CategoriaOperacion.TRANSFORMACION,
            tipo_aplicacion=tipo,
            canal=canal,
            nombre=nombre,
            params={},  # TODO
            tipo_salida=TipoSalida.IMAGEN,
        )

    # ── PROYECTIVOS / INTEGRALES ──────────────────────────────
    # Estos devuelven TipoSalida.FEATURES — shape distinto al input

    def crear_operacion_radon(
        self,
        # TODO: completar parámetros (angulos, ...)
        tipo: TipoAplicacion = PorCorteEspaciotemporal(),
        canal: Optional[int] = 0,
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self.crear_operacion(
            nombre_metodo="radon",
            categoria=CategoriaOperacion.TRANSFORMACION,
            tipo_aplicacion=tipo,
            canal=canal,
            nombre=nombre,
            params={},  # TODO
            tipo_salida=TipoSalida.FEATURES,  # sinograma [n_angulos, proyeccion]
        )

    def crear_operacion_integral_de_linea(
        self,
        # TODO: completar parámetros (angulos, metodo, ...)
        tipo: TipoAplicacion = PorCorteEspaciotemporal(),
        canal: Optional[int] = 0,
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self.crear_operacion(
            nombre_metodo="integral_de_linea",
            categoria=CategoriaOperacion.TRANSFORMACION,
            tipo_aplicacion=tipo,
            canal=canal,
            nombre=nombre,
            params={},  # TODO
            tipo_salida=TipoSalida.FEATURES,
        )

    def crear_operacion_hough(
        self,
        # TODO: completar parámetros (tipo_primitiva, ...)
        tipo: TipoAplicacion = PorCorteEspaciotemporal(),
        canal: Optional[int] = 0,
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self.crear_operacion(
            nombre_metodo="hough",
            categoria=CategoriaOperacion.TRANSFORMACION,
            tipo_aplicacion=tipo,
            canal=canal,
            nombre=nombre,
            params={},  # TODO
            tipo_salida=TipoSalida.FEATURES,  # acumulador de votos
        )

    def crear_operacion_distancia_geodesica(
        self,
        # TODO: completar parámetros (semillas, metrica, ...)
        tipo: TipoAplicacion = PorCorteEspaciotemporal(),
        canal: Optional[int] = 0,
        nombre: Optional[str] = None,
    ) -> Operacion:
        # Geodésica sí preserva shape → IMAGEN
        return self.crear_operacion(
            nombre_metodo="transformada_distancia_geodesica",
            categoria=CategoriaOperacion.TRANSFORMACION,
            tipo_aplicacion=tipo,
            canal=canal,
            nombre=nombre,
            params={},  # TODO
            tipo_salida=TipoSalida.IMAGEN,
        )

    def crear_operacion_hilbert(
        self,
        # TODO: completar parámetros (eje, ...)
        tipo: TipoAplicacion = PorCorteEspaciotemporal(),
        canal: Optional[int] = 0,
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self.crear_operacion(
            nombre_metodo="transformada_hilbert",
            categoria=CategoriaOperacion.TRANSFORMACION,
            tipo_aplicacion=tipo,
            canal=canal,
            nombre=nombre,
            params={},  # TODO
            tipo_salida=TipoSalida.FEATURES,
        )

    def crear_operacion_abel(
        self,
        # TODO: completar parámetros (metodo, eje_simetria, ...)
        tipo: TipoAplicacion = PorCorteEspaciotemporal(),
        canal: Optional[int] = 0,
        nombre: Optional[str] = None,
    ) -> Operacion:
        return self.crear_operacion(
            nombre_metodo="abel",
            categoria=CategoriaOperacion.TRANSFORMACION,
            tipo_aplicacion=tipo,
            canal=canal,
            nombre=nombre,
            params={},  # TODO
            tipo_salida=TipoSalida.FEATURES,
        )

    # =========================================================
    # USO IMPERATIVO
    # =========================================================

    def aplicar(
        self,
        data: BioImagenData,
        metodo: MetodoTransformador,
        tipo: TipoAplicacion,
        canal: int = 0,
    ) -> Resultado[BioImagenData, ErrorBioImagen]:
        return self._crear(metodo, tipo, canal)(data)

    def aplicar_multicanal(
        self,
        data: BioImagenData,
        metodo: MetodoTransformador,
        tipo: TipoAplicacion,
    ) -> Resultado[BioImagenData, ErrorBioImagen]:
        return self._crear_multicanal(metodo, tipo)(data)

    def __repr__(self):
        nombre = getattr(self._ultimo_metodo, "nombre", "None")
        return f"<Controlador_Transformador ultimo={nombre}>"