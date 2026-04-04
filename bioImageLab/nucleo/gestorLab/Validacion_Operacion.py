# === gestorLab/Validaciones_Operaciones.py ===

from typing import Dict, Set, Tuple, Optional
from .Categoria_Operacion import CategoriaOperacion, TipoDato


# =========================================================
# MAPEOS SEMÁNTICOS
# =========================================================

TIPO_ENTRADA: Dict[CategoriaOperacion, Set[TipoDato]] = {
    CategoriaOperacion.PREPROCESAMIENTO: {TipoDato.IMAGEN},
    CategoriaOperacion.FILTRACION:       {TipoDato.IMAGEN},
    CategoriaOperacion.REALZADOR:        {TipoDato.IMAGEN},
    CategoriaOperacion.TRANSFORMADOR:    {TipoDato.IMAGEN},

    CategoriaOperacion.SEGMENTADOR:      {TipoDato.IMAGEN},

    CategoriaOperacion.CUANTIFICADOR:    {TipoDato.IMAGEN, TipoDato.MASCARA},

    CategoriaOperacion.MODELADOR:        {TipoDato.TABLA, TipoDato.FEATURES},

    CategoriaOperacion.ANALIZADOR: {
        TipoDato.IMAGEN,
        TipoDato.TABLA,
        TipoDato.FEATURES
    },
}


TIPO_SALIDA: Dict[CategoriaOperacion, TipoDato] = {
    CategoriaOperacion.PREPROCESAMIENTO: TipoDato.IMAGEN,
    CategoriaOperacion.FILTRACION:       TipoDato.IMAGEN,
    CategoriaOperacion.REALZADOR:        TipoDato.IMAGEN,
    CategoriaOperacion.TRANSFORMADOR:    TipoDato.IMAGEN,

    CategoriaOperacion.SEGMENTADOR:      TipoDato.MASCARA,

    CategoriaOperacion.CUANTIFICADOR:    TipoDato.TABLA,

    CategoriaOperacion.MODELADOR:        TipoDato.MODELO,

    CategoriaOperacion.ANALIZADOR:       TipoDato.VISUALIZACION,
}


# =========================================================
# ADAPTADORES
# =========================================================

ADAPTADORES: Dict[Tuple[TipoDato, TipoDato], str] = {
    (TipoDato.IMAGEN, TipoDato.TABLA): "cuantificador_default",
    (TipoDato.MASCARA, TipoDato.TABLA): "cuantificador_mask",
    (TipoDato.MODELO, TipoDato.TABLA): "modelo_to_tabla",
}


# =========================================================
# API
# =========================================================

def tipo_entrada(cat: CategoriaOperacion) -> Set[TipoDato]:
    return TIPO_ENTRADA[cat]


def tipo_salida(cat: CategoriaOperacion) -> TipoDato:
    return TIPO_SALIDA[cat]


def es_compatible(cat1: CategoriaOperacion, cat2: CategoriaOperacion) -> bool:
    return tipo_salida(cat1) in tipo_entrada(cat2)


def requiere_adaptador(cat1: CategoriaOperacion, cat2: CategoriaOperacion) -> bool:
    return not es_compatible(cat1, cat2)


def obtener_adaptador(
    cat1: CategoriaOperacion,
    cat2: CategoriaOperacion
) -> Optional[str]:

    salida = tipo_salida(cat1)

    for entrada in tipo_entrada(cat2):
        key = (salida, entrada)
        if key in ADAPTADORES:
            return ADAPTADORES[key]

    return None