# === gestorLab/Gestor_Lab.py ===

from __future__ import annotations

import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union

from .Flujo_Trabajo import FlujoTrabajo
from .Constructor_Flujo_Trabajo import ConstructorFlujoTrabajo
from .Validar_Flujo_Trabajo import validar_pipeline
from .Log import RecolectorLog

from ..controlador.Resultado_Either import Resultado, Ok, Err, LogEvento, NivelLog
from ..controlador.Controlador_BioImagen import (
    ControladorBioImagen,
    BioImagenData,
    ErrorBioImagen,
    ModoImagen,
)


class GestorLab:
    """
    Punto de entrada único del sistema.

    El resultado de ejecutar ahora es:
        Ok((salida_dict, logs))   → éxito con lista de LogEvento
        Err(error)                → fallo, el error lleva su _log interno

    Si el YAML o JSON tiene 'ruta_log', los logs se guardan automáticamente.

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
        self._rutas_log: Dict[str, Optional[Path]] = {}   # por pipeline

    # =========================================================
    # REGISTRO MANUAL
    # =========================================================

    def registrar(self, flujo: FlujoTrabajo, ruta_log: Optional[Path] = None) -> None:
        if not flujo.nombre:
            raise ValueError("FlujoTrabajo debe tener nombre antes de registrar")
        self._flujos[flujo.nombre]    = flujo
        self._rutas_log[flujo.nombre] = ruta_log

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
        flujo    = ConstructorFlujoTrabajo().construir(config)
        ruta_log = Path(config["ruta_log"]) if "ruta_log" in config else None
        self.registrar(flujo, ruta_log)
        return flujo

    def registrar_desde_yaml(self, ruta: Union[str, Path]) -> FlujoTrabajo:
        """Construye y registra un pipeline desde un archivo YAML."""
        ruta = Path(ruta)
        if not ruta.exists():
            raise FileNotFoundError(f"YAML no encontrado: {ruta}")
        with open(ruta, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        return self.registrar_desde_config(config)

    def registrar_desde_json(self, ruta: Union[str, Path]) -> FlujoTrabajo:
        """Construye y registra un pipeline desde un archivo JSON."""
        ruta = Path(ruta)
        if not ruta.exists():
            raise FileNotFoundError(f"JSON no encontrado: {ruta}")
        with open(ruta, "r", encoding="utf-8") as f:
            config = json.load(f)
        return self.registrar_desde_config(config)

    # =========================================================
    # EJECUCIÓN — entrada imagen (flujo principal)
    # =========================================================

    def ejecutar_desde_ruta(
        self,
        nombre:       str,
        ruta_imagen:  Union[str, Path],
        modo:         ModoImagen = ModoImagen.AUTO,
        validar:      bool = True,
        debug:        bool = False,
    )  -> Resultado[Tuple[Dict[str, Any], List[LogEvento]], Any]:

        flujo = self._obtener_flujo(nombre)
        recolector = RecolectorLog(nombre)

        if validar:
            val = self._validar(flujo, debug, recolector)
            if val.es_err():
                self._guardar_si_corresponde(nombre, recolector)
                return val

        if debug:
            print(f"[GestorLab] Cargando imagen: {ruta_imagen}")

        ctrl  = ControladorBioImagen(ruta_imagen)
        carga = ctrl.cargar_ImagenResultado(modo)

        if carga.es_err():
            recolector.agregar_manual(
                etapa="carga", mensaje=str(carga.error.mensaje), nivel=NivelLog.ERROR
            )
            self._guardar_si_corresponde(nombre, recolector)
            return carga

        data = carga.unwrap()
        recolector.agregar_manual(
            etapa="carga",
            mensaje=f"Imagen cargada: shape={data.dims.shape} canales={data.canales}",
        )

        if debug:
            print(f"[GestorLab] Imagen cargada: {data.dims.shape}")

        return self._ejecutar(flujo, data, debug, nombre, recolector)


    def ejecutar_desde_data(
        self,
        nombre: str,
        data: BioImagenData,
        validar: bool = True,
        debug: bool = False,
    ) -> Resultado[Tuple[Dict[str, Any], List[LogEvento]], Any]:
        """
        Ejecuta el pipeline con BioImagenData ya cargada.
        Útil para re-ejecutar sin releer el disco.
        """
        flujo      = self._obtener_flujo(nombre)
        recolector = RecolectorLog(nombre)

        if validar:
            val = self._validar(flujo, debug, recolector)
            if val.es_err():
                self._guardar_si_corresponde(nombre, recolector)
                return val

        return self._ejecutar(flujo, data, debug, nombre, recolector)

    # =========================================================
    # EJECUCIÓN — entrada tabular (Modelador / Analizador)
    # =========================================================

    def ejecutar_desde_dataframe(
        self,
        nombre: str,
        df,                          # pd.DataFrame — sin importar pandas en este nivel
        validar: bool = True,
        debug:   bool = False,
    ) -> Resultado[Tuple[Dict[str, Any], List[LogEvento]], Any]:
        """
        Ejecuta el pipeline con un DataFrame como entrada.
        Flujo principal para pipelines Modelador/Analizador.
        """
        flujo      = self._obtener_flujo(nombre)
        recolector = RecolectorLog(nombre)

        if validar:
            val = self._validar(flujo, debug, recolector)
            if val.es_err():
                self._guardar_si_corresponde(nombre, recolector)
                return val

        return self._ejecutar(flujo, df, debug, nombre, recolector)

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
        flujo:     FlujoTrabajo,
        debug:     bool,
        recolector: RecolectorLog,
    ) -> Resultado[bool, Any]:

        if debug:
            print(f"[GestorLab] Validando '{flujo.nombre}'...")

        resultado = validar_pipeline(flujo.grafo)

        if resultado.es_err():
            recolector.agregar_manual(
                etapa   = "validacion",
                mensaje = resultado.error.mensaje,
                nivel   = NivelLog.ERROR,
            )
            if debug:
                print(f"[GestorLab] ✗ {resultado.error.mensaje}")
            return resultado

        # Cosechar warnings de validación
        for w in resultado.unwrap():
            recolector.agregar_manual(
                etapa   = w.etapa,
                mensaje = w.mensaje,
                nivel   = NivelLog.WARN,
            )
            if debug:
                print(f"[GestorLab] ⚠ {w.mensaje}")

        if debug:
            n_warn = len(resultado.unwrap())
            print(f"[GestorLab] ✓ Validación OK ({n_warn} warnings)")

        return Ok(True)

    def _ejecutar(
        self,
        flujo:      FlujoTrabajo,
        data:       Any,
        debug:      bool,
        nombre:     str,
        recolector: RecolectorLog,
    ) -> Resultado[Tuple[Dict[str, Any], List[LogEvento]], Any]:

        flujo.reset_datos()

        if debug:
            print(f"[GestorLab] Ejecutando '{flujo.nombre}'...")

        resultado = flujo.ejecutar(data)

        if resultado.es_ok():
            salida, logs_pipeline = resultado.unwrap()

            # Cosechar todos los logs del pipeline
            recolector.cosechar_varios(
                [Ok(None, tuple(logs_pipeline))]  # envolver para que cosechar los encuentre
            )
            # Forma directa más simple:
            for ev in logs_pipeline:
                recolector._eventos.append(ev)

            recolector.agregar_manual(
                etapa   = "pipeline",
                mensaje = f"Ejecución completada. Nodos finales: {list(salida.keys())}",
            )

            if debug:
                print(f"[GestorLab] ✓ OK — nodos: {list(salida.keys())}")

            self._guardar_si_corresponde(nombre, recolector)
            return Ok((salida, recolector.eventos))

        else:
            error = resultado.error

            # Cosechar logs que viajan en el Err
            recolector.cosechar(resultado)

            recolector.agregar_manual(
                etapa   = "pipeline",
                mensaje = f"Ejecución fallida: {getattr(error, 'mensaje', str(error))}",
                nivel   = NivelLog.ERROR,
            )

            if debug:
                print(f"[GestorLab] ✗ ERR — {getattr(error, 'mensaje', str(error))}")

            self._guardar_si_corresponde(nombre, recolector)
            return resultado

    def _guardar_si_corresponde(
        self, nombre: str, recolector: RecolectorLog
    ) -> None:
        ruta = self._rutas_log.get(nombre)
        if ruta is None:
            return
        formato = "json" if ruta.suffix == ".json" else "txt"
        recolector.guardar(ruta, formato=formato)

    # =========================================================
    # DEBUG / VISUALIZACIÓN
    # =========================================================

    def mostrar_grafo(self, nombre: str) -> None:
        flujo = self._obtener_flujo(nombre)
        grafo = flujo.grafo

        print(f"\n{'='*60}")
        print(f"  PIPELINE : {nombre}")
        print(f"{'='*60}")

        print(f"\n[NODOS] ({len(grafo.nodos)})")
        iniciales = {n.id for n in grafo.nodos_iniciales()}
        finales   = {n.id for n in grafo.nodos_finales()}
        for nodo in grafo.nodos.values():
            marca = "→" if nodo.id in iniciales else ("✓" if nodo.id in finales else " ")
            print(f"  {marca} {nodo.id} ({nodo.tipo_dato.name})")

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
            for i, nid in enumerate(orden):
                nodo = flujo.grafo.nodos[nid]
                print(f"  {i+1}. {nid} ({nodo.tipo_dato.name})")
        except ValueError as e:
            print(f"[ERROR] {e}")

    def __repr__(self) -> str:
        return f"<GestorLab flujos={self.listar()}>"