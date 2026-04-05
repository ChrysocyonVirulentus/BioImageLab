# === gestorLab/Gestor_Lab.py ===

from __future__ import annotations

import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union

from .Flujo_Trabajo import FlujoTrabajo
from .Constructor_Flujo_Trabajo import ConstructorFlujoTrabajo
from .Validar_Flujo_Trabajo import validar_pipeline

from ..controlador.Resultado_Either import Resultado, Ok, Err
from ..controlador.Controlador_BioImagen import (
    ControladorBioImagen,
    BioImagenData,
    ErrorBioImagen,
    ModoImagen,
)


class GestorLab:
    """
    Punto de entrada único del sistema.

    Responsabilidades:
      - Registrar y almacenar FlujoTrabajo por nombre
      - Construir flujos desde config (YAML / JSON / dict)
      - Validar el pipeline antes de ejecutar
      - Ejecutar con el input correcto según el tipo de entrada

    Tipos de entrada soportados:
      - Ruta de imagen  → BioImagenData  (pipelines imagen)
      - BioImagenData   → directo        (cuando ya está cargada)
      - pd.DataFrame    → directo        (pipelines Modelador/Analizador)
      - dict            → directo        (cualquier otro input)
    """

    def __init__(self):
        self._flujos: Dict[str, FlujoTrabajo] = {}

    # =========================================================
    # REGISTRO MANUAL
    # =========================================================

    def registrar(self, flujo: FlujoTrabajo) -> None:
        if not flujo.nombre:
            raise ValueError("FlujoTrabajo debe tener nombre antes de registrar")
        self._flujos[flujo.nombre] = flujo

    def obtener(self, nombre: str) -> FlujoTrabajo:
        if nombre not in self._flujos:
            raise KeyError(f"Pipeline '{nombre}' no registrado")
        return self._flujos[nombre]

    def listar(self) -> list[str]:
        return list(self._flujos.keys())

    # =========================================================
    # REGISTRO DESDE CONFIG
    # =========================================================

    def registrar_desde_config(self, config: Dict[str, Any]) -> FlujoTrabajo:
        """Construye y registra un pipeline desde un dict de configuración."""
        flujo = ConstructorFlujoTrabajo().construir(config)
        self.registrar(flujo)
        return flujo

    def registrar_desde_yaml(self, ruta: Union[str, Path]) -> FlujoTrabajo:
        """Construye y registra un pipeline desde un archivo YAML."""
        ruta = Path(ruta)
        if not ruta.exists():
            raise FileNotFoundError(f"Archivo YAML no encontrado: {ruta}")
        with open(ruta, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        return self.registrar_desde_config(config)

    def registrar_desde_json(self, ruta: Union[str, Path]) -> FlujoTrabajo:
        """Construye y registra un pipeline desde un archivo JSON."""
        ruta = Path(ruta)
        if not ruta.exists():
            raise FileNotFoundError(f"Archivo JSON no encontrado: {ruta}")
        with open(ruta, "r", encoding="utf-8") as f:
            config = json.load(f)
        return self.registrar_desde_config(config)

    # =========================================================
    # EJECUCIÓN — entrada imagen (flujo principal)
    # =========================================================

    def ejecutar_desde_ruta(
        self,
        nombre: str,
        ruta_imagen: Union[str, Path],
        modo: ModoImagen = ModoImagen.AUTO,
        validar: bool = True,
        debug: bool = False,
    ) -> Resultado[Dict[str, Any], Any]:
        """
        Carga la imagen desde disco y ejecuta el pipeline.
        Flujo principal para pipelines de bioimagen.
        """
        flujo = self._obtener_flujo(nombre)

        # Validación pre-ejecución
        if validar:
            resultado_val = self._validar(flujo, debug)
            if resultado_val.es_err():
                return resultado_val

        # Carga
        if debug:
            print(f"[GestorLab] Cargando imagen: {ruta_imagen}")

        ctrl  = ControladorBioImagen(ruta_imagen)
        carga = ctrl.cargar_ImagenResultado(modo)

        if carga.es_err():
            return carga

        data = carga.unwrap()

        if debug:
            print(f"[GestorLab] Imagen cargada: {data.dims.shape} canales={data.canales}")

        return self._ejecutar(flujo, data, debug)

    def ejecutar_desde_data(
        self,
        nombre: str,
        data: BioImagenData,
        validar: bool = True,
        debug: bool = False,
    ) -> Resultado[Dict[str, Any], Any]:
        """
        Ejecuta el pipeline con BioImagenData ya cargada.
        Útil para re-ejecutar sin releer el disco.
        """
        flujo = self._obtener_flujo(nombre)

        if validar:
            resultado_val = self._validar(flujo, debug)
            if resultado_val.es_err():
                return resultado_val

        return self._ejecutar(flujo, data, debug)

    # =========================================================
    # EJECUCIÓN — entrada tabular (Modelador / Analizador)
    # =========================================================

    def ejecutar_desde_dataframe(
        self,
        nombre: str,
        df,                          # pd.DataFrame — sin importar pandas en este nivel
        validar: bool = True,
        debug: bool = False,
    ) -> Resultado[Dict[str, Any], Any]:
        """
        Ejecuta el pipeline con un DataFrame como entrada.
        Flujo principal para pipelines Modelador/Analizador.
        """
        flujo = self._obtener_flujo(nombre)

        if validar:
            resultado_val = self._validar(flujo, debug)
            if resultado_val.es_err():
                return resultado_val

        return self._ejecutar(flujo, df, debug)

    # =========================================================
    # CORE INTERNO
    # =========================================================

    def _obtener_flujo(self, nombre: str) -> FlujoTrabajo:
        if nombre not in self._flujos:
            raise KeyError(
                f"Pipeline '{nombre}' no registrado. "
                f"Disponibles: {self.listar()}"
            )
        return self._flujos[nombre]

    def _validar(
        self,
        flujo: FlujoTrabajo,
        debug: bool,
    ) -> Resultado[bool, Any]:
        if debug:
            print(f"[GestorLab] Validando pipeline '{flujo.nombre}'...")

        resultado = validar_pipeline(flujo.grafo)

        if resultado.es_err():
            if debug:
                print(f"[GestorLab] Validación fallida: {resultado.error.mensaje}")
        elif debug:
            print(f"[GestorLab] Validación OK")

        return resultado

    def _ejecutar(
        self,
        flujo: FlujoTrabajo,
        data: Any,
        debug: bool,
    ) -> Resultado[Dict[str, Any], Any]:
        flujo.reset_datos()

        if debug:
            print(f"[GestorLab] Ejecutando '{flujo.nombre}'...")

        resultado = flujo.ejecutar(data)

        if debug:
            if resultado.es_ok():
                salida = resultado.unwrap()
                nodos  = list(salida.keys())
                print(f"[GestorLab] OK — nodos finales: {nodos}")
            else:
                print(f"[GestorLab] ERR — {resultado.error.mensaje}")

        return resultado

    # =========================================================
    # DEBUG / VISUALIZACIÓN
    # =========================================================

    def mostrar_grafo(self, nombre: str) -> None:
        flujo = self._obtener_flujo(nombre)
        grafo = flujo.grafo

        print(f"\n{'='*50}")
        print(f"  PIPELINE: {nombre}")
        print(f"{'='*50}")

        print(f"\n[NODOS] ({len(grafo.nodos)})")
        for nodo in grafo.nodos.values():
            prefijo = "→ " if grafo.nodos_iniciales().__contains__(nodo) else \
                      "✓ " if nodo in grafo.nodos_finales() else "  "
            print(f"  {prefijo}{nodo.id} ({nodo.tipo_dato.name})")

        print(f"\n[ARISTAS] ({len(grafo.aristas)})")
        for arista in grafo.aristas:
            print(f"  {arista.origen}")
            print(f"    └─[{arista.operacion.nombre}]→ {arista.destino}")

        ok, errores = flujo.validar_pipeline()
        print(f"\n[VALIDACIÓN] {'OK' if ok else f'ERRORES ({len(errores)})'}")
        for e in errores:
            print(f"  ✗ {e}")

        print()

    def mostrar_orden_ejecucion(self, nombre: str) -> None:
        flujo = self._obtener_flujo(nombre)
        try:
            orden = flujo.grafo.orden_topologico()
            print(f"\n[Orden topológico — {nombre}]")
            for i, nodo_id in enumerate(orden):
                nodo = flujo.grafo.nodos[nodo_id]
                print(f"  {i+1}. {nodo_id} ({nodo.tipo_dato.name})")
        except ValueError as e:
            print(f"[ERROR] {e}")

    def __repr__(self) -> str:
        return f"<GestorLab flujos={list(self._flujos.keys())}>"