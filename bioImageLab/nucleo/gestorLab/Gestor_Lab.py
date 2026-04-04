# === gestorLab/Gestor_Flujo_Trabajo.py ===

from __future__ import annotations

from typing import Dict
from pathlib import Path

from .Flujo_Trabajo import FlujoTrabajo
from .Constructor_Flujo_Trabajo import ConstructorFlujoTrabajo

from ..controlador.Resultado_Either import Resultado, Ok, Err
from ..controlador.Controlador_BioImagen import ControladorBioImagen


class GestorFlujoTrabajo:

    def __init__(self):
        self._pipelines: Dict[str, FlujoTrabajo] = {}
        self._constructor = ConstructorFlujoTrabajo()

    # =========================================================
    # REGISTRO
    # =========================================================

    def registrar_desde_config(self, config: dict) -> FlujoTrabajo:

        pipeline = self._constructor.construir(config)
        self._pipelines[pipeline.nombre] = pipeline

        return pipeline

    def obtener(self, nombre: str) -> FlujoTrabajo:
        return self._pipelines[nombre]

    def listar(self):
        return list(self._pipelines.keys())

    # =========================================================
    # EJECUCIÓN (ALTO NIVEL)
    # =========================================================

    def ejecutar_desde_ruta(
        self,
        nombre_pipeline: str,
        ruta_imagen: str | Path
    ) -> Resultado:

        # 1. Crear controlador (entrypoint real del sistema)
        ctrl_bio = ControladorBioImagen(ruta_imagen)

        # 2. Cargar imagen
        resultado_carga = ctrl_bio.cargar_ImagenResultado()

        if resultado_carga.es_err():
            return resultado_carga

        data = resultado_carga.unwrap()

        # 3. Ejecutar pipeline
        return self._ejecutar_pipeline(nombre_pipeline, data)

    # =========================================================
    # EJECUCIÓN (CORE)
    # =========================================================

    def _ejecutar_pipeline(
        self,
        nombre_pipeline: str,
        data
    ) -> Resultado:

        pipeline = self._pipelines[nombre_pipeline]

        nodo_actual = pipeline.nodo_input
        resultado = Ok(data)

        visitados = set()

        while True:

            if nodo_actual in visitados:
                raise RuntimeError("Loop detectado en pipeline")

            visitados.add(nodo_actual)

            salidas = pipeline.obtener_salidas(nodo_actual)

            if not salidas:
                break

            # (por ahora lineal)
            arista = salidas[0]
            operacion = arista.operacion

            resultado = resultado.bind(operacion.ejecutar)

            if resultado.es_err():
                return resultado

            nodo_actual = arista.destino

        return resultado