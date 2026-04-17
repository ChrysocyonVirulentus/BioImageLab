# === gestorLab/Constructor_Flujo_Trabajo.py ===

from __future__ import annotations

from typing import Dict, Any, Optional

from .Flujo_Trabajo import FlujoTrabajo, GrafoPipeline, NodoPipeline, AristaOperacion
from .Categoria_Operacion import CategoriaOperacion, TipoDato
from .Validacion_Operacion import es_compatible, obtener_adaptador, tipo_salida as tipo_salida_default
from .Registro_Controladores import CONTROLADORES

# Importar estrategias para el mapeo
from ..controlador.Estrategias_Aplicacion import (
    Global,
    PorCorteZ,
    PorTimepoint,
    PorCorteEspaciotemporal,
    PorVolumen3D,
    ConReferencia,
    TipoAplicacion,
)


# =========================================================
# HELPERS INTERNOS
# =========================================================

def _op_identidad(nombre: str, categoria: CategoriaOperacion, tipo_dato: TipoDato) -> Operacion:
    """Operación pass-through para aristas estructurales del merge."""
    return Operacion(
        nombre             = nombre,
        categoria          = categoria,
        instancia_callable = lambda x: Ok(x),
        tipo_dato_salida   = tipo_dato,
    )

# ── Merge combiners ─────────────────────────────────────────────

def _combinar_imagen_mascara(datos: List[Any]) -> BioImagenData:
    """
    Combina imagen[0] + mascara[1] en una BioImagenData unificada.
    La máscara se guarda en metadata["mascara_datos"] para que
    el Controlador_Cuantificador la extraiga en _preprocesar.
    """
    imagen  = datos[0]   # BioImagenData (imagen filtrada)
    mascara = datos[1]   # BioImagenData (segmentación)
    return replace(
        imagen,
        metadata={
            **imagen.metadata,
            "mascara_datos": mascara.datos,     # shape [T, Z, C, Y, X] uint16
            "mascara_dims":  mascara.dims,
        }
    )


def _combinar_concatenar(datos: List[Any]) -> Any:
    """Devuelve lista tal cual — para merges personalizados."""
    return datos


_FUSION_FNS: Dict[str, Callable] = {
    "imagen_mascara": _combinar_imagen_mascara,
    "concatenar":     _combinar_concatenar,
}

class ConstructorFlujoTrabajo:

    """
    Construye FlujoTrabajo desde un dict de configuración (YAML/JSON).

    Soporta:
      - Etapas lineales
      - anchor: en cualquier operación → guarda el nodo como checkpoint
      - dividir o split: bifurcación paralela que arranca desde un checkpoint
      - fusion o merge: combina dos ramas en una sola BioImagenData
    """

    def __init__(self):
        self.grafo       = GrafoPipeline()
        self._checkpoints: Dict[str, str] = {}   # nombre → nodo_id

    # =========================================================
    # API PRINCIPAL
    # =========================================================

    def construir(self, config: Dict[str, Any]) -> FlujoTrabajo:
        self.grafo        = GrafoPipeline()
        self._checkpoints = {}
        nombre            = config.get("nombre_pipeline", "Pipeline")

        nodo_actual = "input"
        self.grafo.agregar_nodo(NodoPipeline(id=nodo_actual, tipo_dato=TipoDato.IMAGEN))
        self._checkpoints["input"] = nodo_actual

        for etapa_dict in config.get("etapas", []):
            nodo_actual = self._procesar_etapa(etapa_dict, nodo_actual)

        flujo        = FlujoTrabajo(self.grafo)
        flujo.nombre = nombre
        return flujo

    # =========================================================
    # PROCESAMIENTO DE ETAPAS
    # =========================================================

    def _procesar_etapa(self, etapa_cfg: Dict, nodo_entrada: str) -> str:
        """Despacha cada bloque del YAML al handler correspondiente."""
        for nombre_bloque, contenido in etapa_cfg.items():

            if nombre_bloque == "split":
                # Split no cambia nodo_actual de la rama principal
                self._procesar_split(contenido, nodo_entrada)

            elif nombre_bloque == "merge":
                nodo_entrada = self._procesar_merge(contenido)

            else:
                # Etapa normal: lista de operaciones
                categoria = self._mapear_categoria(nombre_bloque)
                for op_cfg in contenido:
                    nodo_entrada = self._crear_y_conectar_operacion(
                        op_cfg, categoria, nodo_entrada
                    )
                # Auto-checkpoint por nombre de etapa (último nodo del bloque)
                self._checkpoints[nombre_bloque] = nodo_entrada

        return nodo_entrada

    # =========================================================
    # SPLIT — bifurcación paralela
    # =========================================================

    def _procesar_split(self, split_cfg: Dict, nodo_actual: str) -> None:
        """
        Crea una rama paralela que parte de un checkpoint.
        NO actualiza nodo_actual del pipeline principal.
        Al final guarda el último nodo de la rama como checkpoint[nombre].
        """
        nombre      = split_cfg["nombre"]
        desde       = split_cfg.get("desde", nodo_actual)
        nodo_inicio = self._checkpoints.get(desde, desde)

        nodo_branch = nodo_inicio
        for etapa in split_cfg.get("etapas", []):
            nodo_branch = self._procesar_etapa(etapa, nodo_branch)

        self._checkpoints[nombre] = nodo_branch

    # =========================================================
    # MERGE — fusión de ramas
    # =========================================================

    def _procesar_merge(self, merge_cfg: Dict) -> str:
        """
        Crea un nodo de merge que:
          1. Recibe inputs de DOS checkpoints vía aristas identidad
          2. Aplica merge_fn cuando ambos inputs están disponibles
          3. Devuelve una BioImagenData unificada al pipeline principal
        """
        nombre     = merge_cfg.get("nombre", "merge")
        estrategia = merge_cfg.get("estrategia", "imagen_mascara")

        if estrategia not in _MERGE_FNS:
            raise ValueError(
                f"Estrategia de merge desconocida: '{estrategia}'. "
                f"Válidas: {list(_MERGE_FNS.keys())}"
            )

        nodo_img_id  = self._resolver_checkpoint(merge_cfg["imagen"])
        nodo_mask_id = self._resolver_checkpoint(merge_cfg["mascara"])

        nodo_merge_id = f"merge_{nombre}"
        merge_fn      = _MERGE_FNS[estrategia]

        self.grafo.agregar_nodo(NodoPipeline(
            id        = nodo_merge_id,
            tipo_dato = TipoDato.IMAGEN,
            es_merge  = True,
            merge_fn  = merge_fn,
        ))

        # Arista identidad desde imagen → merge
        self.grafo.agregar_arista(AristaOperacion(
            origen    = nodo_img_id,
            destino   = nodo_merge_id,
            operacion = _op_identidad(
                f"pass_imagen_{nombre}",
                CategoriaOperacion.PREPROCESAMIENTO,
                TipoDato.IMAGEN,
            )
        ))

        # Arista identidad desde máscara → merge
        self.grafo.agregar_arista(AristaOperacion(
            origen    = nodo_mask_id,
            destino   = nodo_merge_id,
            operacion = _op_identidad(
                f"pass_mascara_{nombre}",
                CategoriaOperacion.SEGMENTADOR,
                TipoDato.MASCARA,
            )
        ))

        self._checkpoints[nombre] = nodo_merge_id
        return nodo_merge_id

    # =========================================================
    # CREAR OPERACION
    # =========================================================

    def _crear_y_conectar_operacion(
        self,
        op_cfg:     Dict,
        categoria:  CategoriaOperacion,
        nodo_entrada: str,
    ) -> str:
        nombre_metodo   = op_cfg["metodo"]
        params          = op_cfg.get("params", {})
        dominio         = op_cfg.get("dominio") or self._inferir_dominio(categoria)
        canal           = op_cfg.get("canal", None)
        tipo_aplicacion = self._mapear_tipo_aplicacion(
            op_cfg.get("tipo_aplicacion"), categoria
        )

        controlador = CONTROLADORES[dominio]
        operacion   = controlador.crear_operacion(
            nombre_metodo   = nombre_metodo,
            categoria       = categoria,
            params          = params,
            canal           = canal,
            tipo_aplicacion = tipo_aplicacion,
        )

        self._validar_conexion(nodo_entrada, operacion)

        nodo_salida = f"{nodo_entrada}_{operacion.nombre}"
        self.grafo.agregar_nodo(NodoPipeline(
            id        = nodo_salida,
            tipo_dato = operacion.tipo_dato_salida,
        ))
        self.grafo.agregar_arista(AristaOperacion(
            origen    = nodo_entrada,
            destino   = nodo_salida,
            operacion = operacion,
        ))

        # Anchor explícito en la operación
        if "anchor" in op_cfg:
            self._checkpoints[op_cfg["anchor"]] = nodo_salida

        return nodo_salida

    # =========================================================
    # VALIDACIÓN
    # =========================================================

    def _validar_conexion(self, nodo_entrada: str, operacion):
        if nodo_entrada not in self.grafo.nodos:
            return
        entrantes_prev = self.grafo.entrantes(nodo_entrada)
        if not entrantes_prev:
            return
        cat_prev  = entrantes_prev[-1].operacion.categoria
        cat_nueva = operacion.categoria

        if not cat_prev.puede_preceder_a(cat_nueva):
            raise ValueError(f"Orden inválido: {cat_prev.name} → {cat_nueva.name}")

        if not es_compatible(cat_prev, cat_nueva):
            adaptador = obtener_adaptador(cat_prev, cat_nueva)
            if adaptador is None:
                raise ValueError(
                    f"Incompatibilidad sin adaptador: {cat_prev.name} → {cat_nueva.name}"
                )
            print(f"[WARN] Adaptador requerido: {adaptador}")

    # =========================================================
    # HELPERS
    # =========================================================

    def _resolver_checkpoint(self, nombre: str) -> str:
        """Resuelve nombre de checkpoint a nodo_id, o lo usa directo si es nodo_id."""
        if nombre in self._checkpoints:
            return self._checkpoints[nombre]
        if nombre in self.grafo.nodos:
            return nombre
        raise KeyError(
            f"Checkpoint '{nombre}' no encontrado. "
            f"Disponibles: {list(self._checkpoints.keys())}"
        )

    def _mapear_tipo_aplicacion(
        self,
        nombre:    Optional[str],
        categoria: CategoriaOperacion,
    ) -> Optional[TipoAplicacion]:
        sin_estrategia = {
            CategoriaOperacion.CUANTIFICADOR,
            CategoriaOperacion.MODELADOR,
            CategoriaOperacion.ANALIZADOR,
        }
        if categoria in sin_estrategia:
            return None
        if nombre is not None:
            mapping = {
                "global":                    Global(),
                "por_corte_z":               PorCorteZ(),
                "por_timepoint":             PorTimepoint(),
                "por_corte_espaciotemporal": PorCorteEspaciotemporal(),
                "por_volumen_3d":            PorVolumen3D(),
                "con_referencia":            ConReferencia(),
            }
            if nombre not in mapping:
                raise ValueError(
                    f"tipo_aplicacion desconocido: '{nombre}'. "
                    f"Válidos: {list(mapping.keys())}"
                )
            return mapping[nombre]
        return PorCorteEspaciotemporal()

    def _mapear_categoria(self, nombre: str) -> CategoriaOperacion:
        mapping = {
            "preprocesamiento": CategoriaOperacion.PREPROCESAMIENTO,
            "filtracion":       CategoriaOperacion.FILTRACION,
            "realzado":         CategoriaOperacion.REALZADOR,
            "transformacion":   CategoriaOperacion.TRANSFORMADOR,
            "segmentacion":     CategoriaOperacion.SEGMENTADOR,
            "cuantificacion":   CategoriaOperacion.CUANTIFICADOR,
            "modelado":         CategoriaOperacion.MODELADOR,
            "analisis":         CategoriaOperacion.ANALIZADOR,
        }
        if nombre not in mapping:
            raise ValueError(f"Categoría desconocida en YAML: '{nombre}'")
        return mapping[nombre]

    def _inferir_dominio(self, categoria: CategoriaOperacion) -> str:
        mapping = {
            CategoriaOperacion.PREPROCESAMIENTO: "normalizacion",
            CategoriaOperacion.FILTRACION:       "filtracion",
            CategoriaOperacion.REALZADOR:        "realzado",
            CategoriaOperacion.TRANSFORMADOR:    "transformacion",
            CategoriaOperacion.SEGMENTADOR:      "segmentacion",
            CategoriaOperacion.CUANTIFICADOR:    "cuantificacion",
            CategoriaOperacion.MODELADOR:        "modelado",
            CategoriaOperacion.ANALIZADOR:       "analisis",
        }
        return mapping[categoria]