# === gestorLab/Gestor_Lab.py ===

from __future__ import annotations

import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple

from .Flujo_Trabajo import FlujoTrabajo
from .Constructor_Flujo_Trabajo import ConstructorFlujoTrabajo
from .Validar_Flujo_Trabajo import validar_pipeline
from .Log import RecolectorLog
from .QC_Visualizacion import visualizar_pasos

from ..controlador.Resultado_Either import Resultado, Ok, Err, LogEvento, NivelLog
from ..controlador.Controlador_BioImagen import (
    ControladorBioImagen, BioImagenData, ErrorBioImagen, ModoImagen,
)


class GestorLab:
    """
    Punto de entrada único del sistema.

    El resultado de ejecutar ahora es:
        Ok((salida_dict, logs))   → éxito con lista de LogEvento
        Err(error)                → fallo, el error lleva su _log interno

    Si el YAML o JSON tiene 'ruta_log', los logs se guardan automáticamente.


    Se organiza como un DAG : Directed Acyclic Graph o Grafo Acíclico Dirigido

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
        self._flujos:    Dict[str, FlujoTrabajo]    = {}
        self._rutas_log: Dict[str, Optional[Path]]  = {}
        self._modo_qc:   Dict[str, bool]            = {}
        self._rutas_qc:  Dict[str, Optional[Path]]  = {}
        self._canal_qc:  Dict[str, int]             = {}

    # ── Registro ──────────────────────────────────────────────

    def registrar(
        self,
        flujo:    FlujoTrabajo,
        ruta_log: Optional[Path] = None,
        modo_qc:  bool           = False,
        ruta_qc:  Optional[Path] = None,
        canal_qc: int            = 0,
    ) -> None:
        if not flujo.nombre:
            raise ValueError("FlujoTrabajo debe tener nombre antes de registrar")
        self._flujos[flujo.nombre]    = flujo
        self._rutas_log[flujo.nombre] = ruta_log
        self._modo_qc[flujo.nombre]   = modo_qc
        self._rutas_qc[flujo.nombre]  = ruta_qc
        self._canal_qc[flujo.nombre]  = canal_qc

    def obtener(self, nombre: str) -> FlujoTrabajo:
        if nombre not in self._flujos:
            raise KeyError(f"Pipeline '{nombre}' no registrado")
        return self._flujos[nombre]

    def listar(self) -> List[str]:
        return list(self._flujos.keys())

    # ── Registro desde config ─────────────────────────────────

    def registrar_desde_config(self, config: Dict[str, Any]) -> FlujoTrabajo:
        flujo = ConstructorFlujoTrabajo().construir(config)
        self.registrar(
            flujo    = flujo,
            ruta_log = Path(config["ruta_log"]) if "ruta_log" in config else None,
            modo_qc  = config.get("modo_qc", False),
            ruta_qc  = Path(config["ruta_qc"]) if "ruta_qc" in config else None,
            canal_qc = config.get("canal_qc", 0),
        )
        return flujo

    def registrar_desde_yaml(self, ruta: Union[str, Path]) -> FlujoTrabajo:
        ruta = Path(ruta)
        if not ruta.exists():
            raise FileNotFoundError(f"YAML no encontrado: {ruta}")
        with open(ruta, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        return self.registrar_desde_config(config)

    def registrar_desde_json(self, ruta: Union[str, Path]) -> FlujoTrabajo:
        ruta = Path(ruta)
        if not ruta.exists():
            raise FileNotFoundError(f"JSON no encontrado: {ruta}")
        with open(ruta, "r", encoding="utf-8") as f:
            config = json.load(f)
        return self.registrar_desde_config(config)

    # =========================================================
    # EJECUCIÓN — entrada imagen (flujo principal)
    # =========================================================

    # ── Ejecución imagen ──────────────────────────────────────

    def ejecutar_desde_ruta(
        self,
        nombre:      str,
        ruta_imagen: Union[str, Path],
        modo:        ModoImagen = ModoImagen.AUTO,
        validar:     bool = True,
        debug:       bool = False,
    ) -> Resultado[Tuple[Dict[str, Any], List[LogEvento]], Any]:

        flujo      = self._obtener_flujo(nombre)
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
            recolector.agregar_manual("carga", str(carga.error.mensaje), NivelLog.ERROR)
            self._guardar_si_corresponde(nombre, recolector)
            return carga

        data = carga.unwrap()
        recolector.agregar_manual(
            "carga",
            f"Imagen cargada: shape={data.dims.shape} canales={data.canales}",
        )
        if debug:
            print(f"[GestorLab] Imagen cargada: {data.dims.shape}")

        return self._ejecutar(flujo, data, debug, nombre, recolector)

    def ejecutar_desde_data(
        self,
        nombre:  str,
        data:    BioImagenData,
        validar: bool = True,
        debug:   bool = False,
    ) -> Resultado[Tuple[Dict[str, Any], List[LogEvento]], Any]:

        flujo      = self._obtener_flujo(nombre)
        recolector = RecolectorLog(nombre)

        if validar:
            val = self._validar(flujo, debug, recolector)
            if val.es_err():
                self._guardar_si_corresponde(nombre, recolector)
                return val

        return self._ejecutar(flujo, data, debug, nombre, recolector)

    def ejecutar_desde_dataframe(
        self,
        nombre:  str,
        df,
        validar: bool = True,
        debug:   bool = False,
    ) -> Resultado[Tuple[Dict[str, Any], List[LogEvento]], Any]:

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
                f"Pipeline '{nombre}' no registrado. Disponibles: {self.listar()}"
            )
        return self._flujos[nombre]

    def _validar(
        self,
        flujo:      FlujoTrabajo,
        debug:      bool,
        recolector: Optional[RecolectorLog] = None,
    ) -> Resultado[bool, Any]:

        if recolector is None:
            recolector = RecolectorLog(flujo.nombre)

        if debug:
            print(f"[GestorLab] Validando '{flujo.nombre}'...")

        resultado = validar_pipeline(flujo.grafo)

        if resultado.es_err():
            recolector.agregar_manual("validacion", resultado.error.mensaje, NivelLog.ERROR)
            if debug:
                print(f"[GestorLab] ✗ {resultado.error.mensaje}")
            return resultado

        warnings = resultado.unwrap()
        for w in warnings:
            recolector.agregar_manual(w.etapa, w.mensaje, NivelLog.WARN)
            if debug:
                print(f"[GestorLab] ⚠ {w.mensaje}")

        if debug:
            print(f"[GestorLab] ✓ Validación OK ({len(warnings)} warnings)")

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

            for ev in logs_pipeline:
                recolector._eventos.append(ev)

            recolector.agregar_manual(
                "pipeline",
                f"Ejecución completada. Nodos finales: {list(salida.keys())}",
            )

            if debug:
                print(f"[GestorLab] ✓ OK — nodos: {list(salida.keys())}")

            self._guardar_si_corresponde(nombre, recolector)

            # ── QC mode ──────────────────────────────────────
            if self._modo_qc.get(nombre, False):
                self._generar_qc(flujo, nombre, debug)

            return Ok((salida, recolector.eventos))

        else:
            error = resultado.error
            recolector.cosechar(resultado)
            recolector.agregar_manual(
                "pipeline",
                f"Ejecución fallida: {getattr(error, 'mensaje', str(error))}",
                NivelLog.ERROR,
            )
            if debug:
                print(f"[GestorLab] ✗ ERR — {getattr(error, 'mensaje', str(error))}")

            self._guardar_si_corresponde(nombre, recolector)
            return resultado

    def _generar_qc(self, flujo: FlujoTrabajo, nombre: str, debug: bool) -> None:
        """Genera el plot QC paso a paso tras una ejecución exitosa."""
        ruta_qc  = self._rutas_qc.get(nombre)
        canal_qc = self._canal_qc.get(nombre, 0)

        if debug:
            print(f"[GestorLab] Generando QC (canal={canal_qc})...")

        try:
            visualizar_pasos(
                grafo   = flujo.grafo,
                canal   = canal_qc,
                ruta    = ruta_qc,
                titulo  = f"QC — {nombre}",
            )
        except Exception as e:
            print(f"[GestorLab] ⚠ Error generando QC: {e}")

    def _guardar_si_corresponde(self, nombre: str, recolector: RecolectorLog) -> None:
        ruta = self._rutas_log.get(nombre)
        if ruta is None:
            return
        formato = "json" if Path(ruta).suffix == ".json" else "txt"
        recolector.guardar(Path(ruta), formato=formato)

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
            merge = " [MERGE]" if nodo.es_merge else ""
            print(f"  {marca} {nodo.id} ({nodo.tipo_dato.name}){merge}")

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
                nodo  = flujo.grafo.nodos[nid]
                merge = " ← MERGE" if nodo.es_merge else ""
                print(f"  {i+1}. {nid} ({nodo.tipo_dato.name}){merge}")
        except ValueError as e:
            print(f"[ERROR] {e}")

    def __repr__(self) -> str:
        return f"<GestorLab flujos={self.listar()}>"