# === gestorLab/Constructor_Flujo_Trabajo.py ===

from __future__ import annotations

from typing import Dict, Any
from .Flujo_Trabajo import FlujoTrabajo
from .Categoria_Operacion import CategoriaOperacion
from .Validacion_Operacion import es_compatible, obtener_adaptador
from .Registro_Controladores import CONTROLADORES


class ConstructorFlujoTrabajo:

    def __init__(self):
        self.pipeline = None

    # =========================================================
    # API PRINCIPAL
    # =========================================================

    def construir(self, config: Dict[str, Any]) -> FlujoTrabajo:

        nombre = config.get("nombre_pipeline", "Pipeline")
        self.pipeline = FlujoTrabajo(nombre)

        nodo_actual = "input"

        # INPUT (solo nodo estructural)
        self.pipeline.set_input(nodo_actual)

        # Guardar metadata opcional
        self.pipeline.metadata = {
            "input_config": config.get("input", {})
        }

        # ETAPAS
        etapas = config.get("etapas", [])

        for etapa in etapas:
            nodo_actual = self._procesar_etapa(etapa, nodo_actual)

        return self.pipeline

    # =========================================================
    # PROCESAMIENTO DE ETAPAS
    # =========================================================

    def _procesar_etapa(self, etapa_cfg: Dict, nodo_entrada: str) -> str:

        for nombre_etapa, contenido in etapa_cfg.items():

            categoria = self._mapear_categoria(nombre_etapa)

            for op_cfg in contenido:
                nodo_entrada = self._crear_y_conectar_operacion(
                    op_cfg,
                    categoria,
                    nodo_entrada
                )

        return nodo_entrada

    # =========================================================
    # CREAR OPERACION DESDE CONTROLADOR
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

        controlador = CONTROLADORES[dominio]

        operacion = controlador.crear_operacion(
            nombre_metodo = nombre_metodo,
            categoria     = categoria,
            params        = params,
            canal         = op_cfg.get("canal"),
        )

        # VALIDACIÓN SEMÁNTICA
        self._validar_conexion(nodo_entrada, operacion)

        nodo_salida = f"{nodo_entrada}_{operacion.nombre}"

        self.pipeline.agregar_operacion(
            nodo_entrada,
            nodo_salida,
            operacion
        )

        return nodo_salida

    # =========================================================
    # VALIDACIÓN
    # =========================================================

    def _validar_conexion(self, nodo_entrada: str, operacion):

        if nodo_entrada not in self.pipeline.nodos:
            return

        nodo = self.pipeline.nodos[nodo_entrada]

        if nodo.categoria is None:
            return

        cat_prev = nodo.categoria
        cat_next = operacion.categoria

        # ORDEN
        if not cat_prev.puede_preceder_a(cat_next):
            raise ValueError(
                f"Orden inválido: {cat_prev} → {cat_next}"
            )

        # TIPOS
        if not es_compatible(cat_prev, cat_next):

            adaptador = obtener_adaptador(cat_prev, cat_next)

            if adaptador is None:
                raise ValueError(
                    f"Incompatibilidad sin adaptador: "
                    f"{cat_prev} → {cat_next}"
                )
            else:
                print(f"[WARN] Adaptador requerido: {adaptador}")

    # =========================================================
    # HELPERS
    # =========================================================

    def _mapear_categoria(self, nombre: str) -> CategoriaOperacion:

        mapping = {
            "preprocesamiento": CategoriaOperacion.PREPROCESAMIENTO,
            "filtracion": CategoriaOperacion.FILTRACION,
            "realzado": CategoriaOperacion.REALZADOR,
            "transformacion": CategoriaOperacion.TRANSFORMADOR,
            "segmentacion": CategoriaOperacion.SEGMENTADOR,
            "cuantificacion": CategoriaOperacion.CUANTIFICADOR,
            "modelado": CategoriaOperacion.MODELADOR,
            "analisis": CategoriaOperacion.ANALIZADOR,
        }

        return mapping[nombre]

    def _inferir_dominio(self, categoria: CategoriaOperacion) -> str:
        return categoria.name.lower()