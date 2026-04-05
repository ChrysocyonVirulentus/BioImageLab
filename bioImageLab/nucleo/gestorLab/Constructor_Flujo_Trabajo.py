# === gestorLab/Constructor_Flujo_Trabajo.py ===

from __future__ import annotations

from typing import Dict, Any, Optional

from .Flujo_Trabajo import FlujoTrabajo, GrafoPipeline, NodoPipeline, AristaOperacion
from .Categoria_Operacion import CategoriaOperacion, TipoDato
from .Validacion_Operacion import es_compatible, obtener_adaptador, tipo_salida as tipo_salida_default
from .Registro_Controladores import CONTROLADORES


class ConstructorFlujoTrabajo:

    def __init__(self):
        self.grafo = GrafoPipeline()

    # =========================================================
    # API PRINCIPAL
    # =========================================================

    def construir(self, config: Dict[str, Any]) -> FlujoTrabajo:
        self.grafo = GrafoPipeline()
        nombre     = config.get("nombre_pipeline", "Pipeline")

        nodo_actual = "input"
        self.grafo.agregar_nodo(NodoPipeline(id=nodo_actual, tipo_dato=TipoDato.IMAGEN))

        for etapa in config.get("etapas", []):
            nodo_actual = self._procesar_etapa(etapa, nodo_actual)

        flujo        = FlujoTrabajo(self.grafo)
        flujo.nombre = nombre
        return flujo

    # =========================================================
    # PROCESAMIENTO DE ETAPAS
    # =========================================================

    def _procesar_etapa(self, etapa_cfg: Dict, nodo_entrada: str) -> str:
        for nombre_etapa, contenido in etapa_cfg.items():
            categoria   = self._mapear_categoria(nombre_etapa)
            for op_cfg in contenido:
                nodo_entrada = self._crear_y_conectar_operacion(
                    op_cfg, categoria, nodo_entrada
                )
        return nodo_entrada

    # =========================================================
    # CREAR OPERACION
    # =========================================================

    def _crear_y_conectar_operacion(
        self,
        op_cfg: Dict,
        categoria: CategoriaOperacion,
        nodo_entrada: str
    ) -> str:
        nombre_metodo = op_cfg["metodo"]
        params        = op_cfg.get("params", {})
        dominio       = op_cfg.get("dominio") or self._inferir_dominio(categoria)
        canal         = op_cfg.get("canal", None)

        controlador = CONTROLADORES[dominio]

        # Controladores que no usan tipo_aplicacion ni canal (Modelador, Analizador)
        # tienen crear_operacion con la misma firma base — canal=None se ignora internamente
        operacion = controlador.crear_operacion(
            nombre_metodo = nombre_metodo,
            categoria     = categoria,
            params        = params,
            canal         = canal,
        )

        self._validar_conexion(nodo_entrada, operacion)

        nodo_salida = f"{nodo_entrada}_{operacion.nombre}"

        self.grafo.agregar_nodo(NodoPipeline(
            id        = nodo_salida,
            tipo_dato = operacion.tipo_dato_salida
        ))

        self.grafo.agregar_arista(AristaOperacion(
            origen    = nodo_entrada,
            destino   = nodo_salida,
            operacion = operacion
        ))

        return nodo_salida

    # =========================================================
    # VALIDACIÓN
    # =========================================================

    def _validar_conexion(self, nodo_entrada: str, operacion):
        if nodo_entrada not in self.grafo.nodos:
            return

        nodo = self.grafo.nodos[nodo_entrada]

        # CORREGIDO: NodoPipeline tiene tipo_dato, no categoria
        # La validación de orden se hace por tipo_dato del nodo de entrada
        # vs tipo_dato esperado por la nueva operación
        tipo_nodo = nodo.tipo_dato

        # Verificar compatibilidad de tipo de dato
        # (la validación de orden de categorías ocurre en Operacion._validar_semantica)
        cat_nueva = operacion.categoria

        entrantes_prev = self.grafo.entrantes(nodo_entrada)
        if not entrantes_prev:
            return  # nodo raíz — sin validación previa

        # Obtener categoría previa desde la arista entrante
        cat_prev = entrantes_prev[-1].operacion.categoria

        if not cat_prev.puede_preceder_a(cat_nueva):
            raise ValueError(
                f"Orden inválido: {cat_prev.name} → {cat_nueva.name}"
            )

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