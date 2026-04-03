# Estrategias de la aplicacion de los metodos : Si por corte o por slices.

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Callable, Union, Protocol, runtime_checkable


# ==================== PROTOCOLS (POLIMORFISMO ESTRUCTURAL) ====================

@runtime_checkable
class MetodoSimple(Protocol):
    """
    Método que opera SOLO sobre el dato.
    
    Ej:
        metodo(data: np.ndarray) -> np.ndarray
    """
    def __call__(self, data: np.ndarray) -> np.ndarray: ...


@runtime_checkable
class MetodoConReferencia(Protocol):
    """
    Método que requiere referencia externa.
    
    Ej:
        metodo(referencia: np.ndarray, data: np.ndarray) -> np.ndarray
    """
    def __call__(self, referencia: np.ndarray, data: np.ndarray) -> np.ndarray: ...


"""
POLIMORFISMO (IMPORTANTE):

Acá NO usamos herencia ni flags.

El polimorfismo ocurre en runtime vía:

    isinstance(metodo, MetodoConReferencia)

Esto es: Polimorfismo estructural (duck typing tipado)

Es decir:
- Si el método "parece" uno con referencia → se usa como tal
- Si no → se trata como método simple

SIN necesidad de modificar las clases existentes.
"""


# ==================== TIPOS ALGEBRAICOS DE APLICACIÓN ====================

@dataclass(frozen=True)
class Global:
    """
    Aplica el método sobre TODO el volumen [T, Z, Y, X].
    
    - El método recibe el volumen completo.
    - Puede usar contexto global (ej: normalización, estadísticas globales).
    """
    def estrategia(self):
        return aplicar_global


@dataclass(frozen=True)
class PorCorteZ:
    """
    Fija Z y procesa cada plano considerando todos los T de ese Z.
    
    El método recibe un bloque [T, Y, X].
    """
    def estrategia(self):
        return aplicar_por_corte_z


@dataclass(frozen=True)
class PorTimepoint:
    """
    Fija T y procesa cada volumen considerando todos los Z de ese T.
    
    El método recibe un bloque [Z, Y, X].
    """
    def estrategia(self):
        return aplicar_por_timepoint


@dataclass(frozen=True)
class PorCorteEspaciotemporal:
    """
    Aplica el método a cada slice independiente [Y, X].
    
    - (t, z) completamente desacoplados.
    - Máxima granularidad.
    """
    def estrategia(self):
        return aplicar_por_corte_espaciotemporal


@dataclass(frozen=True)
class PorVolumen3D:
    """
    Fija T y aplica el método sobre volumen completo [Z, Y, X].
    
    - Útil para operaciones 3D reales (ej: deconvolución volumétrica).
    """
    def estrategia(self):
        return aplicar_por_volumen_3d


@dataclass(frozen=True)
class ConReferencia:
    """
    El método recibe (referencia, dato).
    
    - La referencia es GLOBAL (volumen completo).
    - Cada slice se procesa con contexto global.
    
    Ej:
        metodo(referencia_global, slice)
    """
    def estrategia(self):
        return aplicar_con_referencia


TipoAplicacion = Union[
    Global,
    PorCorteZ,
    PorTimepoint,
    PorCorteEspaciotemporal,
    PorVolumen3D,
    ConReferencia
]


# ==================== ESTRATEGIAS DE APLICACIÓN ====================

def aplicar_global(
    canal_data: np.ndarray,
    metodo: Callable
) -> np.ndarray:
    """
    canal_data: [T, Z, Y, X]

    El método decide cómo operar sobre TODO el volumen.
    
    Comportamiento:
    - MetodoSimple  → metodo(canal_data)
    - MetodoRef     → metodo(canal_data, canal_data)

    Debe devolver mismo shape.
    """
    if isinstance(metodo, MetodoConReferencia):
        resultado = metodo(canal_data, canal_data)
    else:
        resultado = metodo(canal_data)

    if resultado.shape != canal_data.shape:
        raise ValueError(
            f"[Global] esperado {canal_data.shape}, "
            f"pero devolvió {resultado.shape}"
        )

    return resultado


def aplicar_por_corte_z(
    canal_data: np.ndarray,
    metodo: Callable
) -> np.ndarray:
    """
    canal_data: [T, Z, Y, X]

    Para cada Z:
        - toma bloque [T, Y, X]
        - aplica metodo
        - reconstruye

    Polimorfismo:
        - MetodoSimple → metodo(bloque)
        - MetodoRef    → metodo(bloque, bloque)
    """
    T, Z, Y, X = canal_data.shape
    resultado = np.zeros_like(canal_data, dtype=np.float64)

    for z in range(Z):
        bloque = canal_data[:, z, :, :]

        if isinstance(metodo, MetodoConReferencia):
            procesado = metodo(bloque, bloque)
        else:
            procesado = metodo(bloque)

        if procesado.shape != bloque.shape:
            raise ValueError(
                f"[PorCorteZ] esperado {bloque.shape}, "
                f"pero devolvió {procesado.shape}"
            )

        resultado[:, z, :, :] = procesado

    return resultado


def aplicar_por_timepoint(
    canal_data: np.ndarray,
    metodo: Callable
) -> np.ndarray:
    """
    canal_data: [T, Z, Y, X]

    Para cada T:
        - toma bloque [Z, Y, X]
        - aplica metodo
        - reconstruye
    """
    T, Z, Y, X = canal_data.shape
    resultado = np.zeros_like(canal_data, dtype=np.float64)

    for t in range(T):
        bloque = canal_data[t, :, :, :]

        if isinstance(metodo, MetodoConReferencia):
            procesado = metodo(bloque, bloque)
        else:
            procesado = metodo(bloque)

        if procesado.shape != bloque.shape:
            raise ValueError(
                f"[PorTimepoint] esperado {bloque.shape}, "
                f"pero devolvió {procesado.shape}"
            )

        resultado[t, :, :, :] = procesado

    return resultado


def aplicar_por_corte_espaciotemporal(
    canal_data: np.ndarray,
    metodo: Callable
) -> np.ndarray:
    """
    canal_data: [T, Z, Y, X]

    Para cada (t, z):
        - slice [Y, X]
        - aplica metodo
    """
    T, Z, Y, X = canal_data.shape
    resultado = np.zeros_like(canal_data, dtype=np.float64)

    for t in range(T):
        for z in range(Z):
            slice_2d = canal_data[t, z]

            if isinstance(metodo, MetodoConReferencia):
                procesado = metodo(slice_2d, slice_2d)
            else:
                procesado = metodo(slice_2d)

            if procesado.shape != slice_2d.shape:
                raise ValueError(
                    f"[PorCorteEspaciotemporal] esperado {slice_2d.shape}, "
                    f"pero devolvió {procesado.shape}"
                )

            resultado[t, z] = procesado

    return resultado


def aplicar_por_volumen_3d(
    canal_data: np.ndarray,
    metodo: Callable
) -> np.ndarray:
    """
    canal_data: [T, Z, Y, X]

    Para cada T:
        - volumen [Z, Y, X]
        - aplica metodo
    """
    T, Z, Y, X = canal_data.shape
    resultado = np.zeros_like(canal_data, dtype=np.float64)

    for t in range(T):
        volumen = canal_data[t]

        if isinstance(metodo, MetodoConReferencia):
            procesado = metodo(volumen, volumen)
        else:
            procesado = metodo(volumen)

        if procesado.shape != volumen.shape:
            raise ValueError(
                f"[PorVolumen3D] esperado {volumen.shape}, "
                f"pero devolvió {procesado.shape}"
            )

        resultado[t] = procesado

    return resultado


def aplicar_con_referencia(
    canal_data: np.ndarray,
    metodo: Callable
) -> np.ndarray:
    """
    canal_data: [T, Z, Y, X]

    ✔ referencia GLOBAL
    ✔ cada slice usa contexto completo

    SOLO válido para MetodoConReferencia
    """
    if not isinstance(metodo, MetodoConReferencia):
        raise TypeError("Este método no soporta referencia externa")

    T, Z, Y, X = canal_data.shape
    resultado = np.zeros_like(canal_data, dtype=np.float64)

    referencia = canal_data  # GLOBAL

    for t in range(T):
        for z in range(Z):
            resultado[t, z] = metodo(referencia, canal_data[t, z])

    return resultado