# gestorLab/Constructor_Flujo_Trabajo.py

from __future__ import annotations

import yaml
from typing import List, Callable

from .Operacion import Operacion
from .Registro_Controladores import obtener_controlador
from ..controlador.Resultado_Either import Resultado, Ok
from .Categoria_Operacion import CategoriaOperacion
from Log import guardar_log

class Constructor_Flujo_Trabajo:

    def __init__(self):
        self._operaciones: List[Operacion] = []

    # =========================================================
    # BUILD DESDE YAML
    # =========================================================

    def desde_yaml(self, path: str) -> Callable:
        with open(path, "r") as f:
            config = yaml.safe_load(f)

        pipeline_cfg = config.get("pipeline", [])

        self._operaciones = [
            self._crear_operacion_desde_config(op_cfg)
            for op_cfg in pipeline_cfg
        ]

        return self._construir_pipeline()

    # =========================================================
    # CREACIÓN DE OPERACIONES
    # =========================================================

    def _crear_operacion_desde_config(self, cfg: dict) -> Operacion:
        dominio = cfg["dominio"]
        metodo = cfg["metodo"]
        params = cfg.get("params", {})
        canal  = cfg.get("canal", None)
        nombre = cfg.get("nombre", None)

        controlador = obtener_controlador(dominio)

        return controlador.crear_operacion(
            nombre_metodo=metodo,
            categoria=self._inferir_categoria(dominio),
            canal=canal,
            nombre=nombre,
            params=params,
        )

    def _inferir_categoria(self, dominio: str):        

        mapa = {
            "filtrado": CategoriaOperacion.FILTRACION,
            "normalizacion": CategoriaOperacion.PREPROCESAMIENTO,
            "realzado": CategoriaOperacion.REALZADOR,
            "transformacion": CategoriaOperacion.TRANSFORMADOR,
            "segmentacion": CategoriaOperacion.SEGMENTACION,
            "analisis": CategoriaOperacion.ANALISIS,
        }

        return mapa.get(dominio, CategoriaOperacion.OTROS)

    # =========================================================
    # CONSTRUCCIÓN DEL PIPELINE
    # =========================================================

    def _construir_pipeline(self) -> Callable:

        def pipeline(data):

            resultado: Resultado = Ok(data)

            for op in self._operaciones:
                resultado = resultado.bind(op.ejecutar)

                if resultado.es_err():
                    break

            return resultado

        return pipeline

    # =========================================================
    # LOGGING
    # =========================================================

    def guardar_log(self, resultado: Resultado, path: str = "pipeline.log"):
        from ..logging.Log import guardar_log

        guardar_log(resultado, path)

    # =========================================================

    def __repr__(self):
        ops = " → ".join(op.nombre for op in self._operaciones)
        return f"<Pipeline {ops}>"