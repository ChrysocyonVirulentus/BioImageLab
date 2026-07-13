# === gestorLab/Gestor_Lab.py ===
"""
Punto de entrada único del sistema BioImageLab.

Modos de operación
──────────────────
  normal  : ejecuta siempre — los errores van al log, nunca rompen el flujo
  estricto: aborta si la validación detecta errores duros (opt-in)
  debug   : genera QC visual paso a paso usando QC_Visualizacion
  batch   : ejecuta el mismo pipeline sobre varios archivos (ver ejecutar_batch)

Patrón Result
─────────────
  Ok((salida, logs))  → ejecución completada (puede tener errores en el log)
  Err(error)          → solo en fallos catastróficos (imagen ilegible, etc.)
  El log SIEMPRE se escribe si ruta_log está definida.
"""
from __future__ import annotations

import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple

from .Flujo_Trabajo import FlujoTrabajo
from .Constructor_Flujo_Trabajo import ConstructorFlujoTrabajo
from .Validar_Flujo_Trabajo import validar_pipeline, DiagnosticoPipeline
from .Log import RecolectorLog

from ..controlador.Resultado_Either import Resultado, Ok, Err, LogEvento, NivelLog
from ..controlador.Controlador_BioImagen import (
    ControladorBioImagen, BioImagenData, ErrorBioImagen, ModoImagen,
)


class GestorLab:

    def __init__(self):
        self._flujos:    Dict[str, FlujoTrabajo]   = {}
        self._rutas_log: Dict[str, Optional[Path]] = {}
        self._modo_qc:   Dict[str, bool]           = {}
        self._rutas_qc:  Dict[str, Optional[Path]] = {}
        self._canal_qc:  Dict[str, int]            = {}
        self._modo_estricto: Dict[str, bool]       = {}

    # =========================================================
    # REGISTRO
    # =========================================================

    def registrar(
        self,
        flujo:         FlujoTrabajo,
        ruta_log:      Optional[Path] = None,
        modo_qc:       bool           = False,
        ruta_qc:       Optional[Path] = None,
        canal_qc:      int            = 0,
        modo_estricto: bool           = False,
    ) -> None:
        if not flujo.nombre:
            raise ValueError("FlujoTrabajo debe tener nombre antes de registrar")
        self._flujos[flujo.nombre]        = flujo
        self._rutas_log[flujo.nombre]     = ruta_log
        self._modo_qc[flujo.nombre]       = modo_qc
        self._rutas_qc[flujo.nombre]      = ruta_qc
        self._canal_qc[flujo.nombre]      = canal_qc
        self._modo_estricto[flujo.nombre] = modo_estricto

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
            flujo         = flujo,
            ruta_log      = Path(config["ruta_log"]) if "ruta_log" in config else None,
            modo_qc       = config.get("modo_qc", False),
            ruta_qc       = Path(config["ruta_qc"]) if "ruta_qc" in config else None,
            canal_qc      = config.get("canal_qc", 0),
            modo_estricto = config.get("modo_estricto", False),
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
    # EJECUCIÓN INDIVIDUAL
    # =========================================================

    def ejecutar_desde_ruta(
        self,
        nombre:      str,
        ruta_imagen: Union[str, Path],
        modo:        ModoImagen = ModoImagen.AUTO,
        debug:       bool = False,
    ) -> Resultado[Tuple[Dict[str, Any], List[LogEvento]], Any]:

        flujo      = self._obtener_flujo(nombre)
        recolector = RecolectorLog(nombre)

        # Validación — siempre se ejecuta, NUNCA bloquea
        self._validar_y_loguear(flujo, debug, recolector)

        if debug:
            print(f"[GestorLab] Cargando imagen: {ruta_imagen}")

        ctrl  = ControladorBioImagen(ruta_imagen)
        carga = ctrl.cargar_ImagenResultado(modo)

        if carga.es_err():
            # Fallo catastrófico: imagen ilegible
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
        debug:   bool = False,
    ) -> Resultado[Tuple[Dict[str, Any], List[LogEvento]], Any]:
        flujo      = self._obtener_flujo(nombre)
        recolector = RecolectorLog(nombre)
        self._validar_y_loguear(flujo, debug, recolector)
        return self._ejecutar(flujo, data, debug, nombre, recolector)

    def ejecutar_desde_dataframe(
        self,
        nombre: str,
        df,
        debug:  bool = False,
    ) -> Resultado[Tuple[Dict[str, Any], List[LogEvento]], Any]:
        flujo      = self._obtener_flujo(nombre)
        recolector = RecolectorLog(nombre)
        self._validar_y_loguear(flujo, debug, recolector)
        return self._ejecutar(flujo, df, debug, nombre, recolector)

    # =========================================================
    # MODO BATCH
    # =========================================================

    def ejecutar_batch(
        self,
        nombre:        str,
        rutas_imagenes: List[Union[str, Path]],
        modo:          ModoImagen = ModoImagen.AUTO,
        debug:         bool = False,
        ruta_log_batch: Optional[Path] = None,
    ) -> Dict[str, Resultado]:
        """
        Ejecuta el mismo pipeline sobre una lista de rutas de imagen.

        Retorna dict {ruta_str: Resultado} — nunca lanza excepciones.
        Los fallos individuales quedan como Err en el dict; los éxitos como Ok.

        Si ruta_log_batch está definida, escribe un resumen TSV con:
          archivo | estado | n_errores_log | n_warnings_log | nodos_finales
        """
        resultados: Dict[str, Resultado] = {}
        filas_resumen = []

        for ruta in rutas_imagenes:
            ruta_str = str(ruta)
            if debug:
                print(f"\n[Batch] ── {ruta_str}")
            try:
                res = self.ejecutar_desde_ruta(nombre, ruta, modo=modo, debug=debug)
            except Exception as e:
                res = Err(ErrorBioImagen(
                    etapa="batch",
                    mensaje=f"Excepción no capturada: {e}",
                    causa=e,
                ))

            resultados[ruta_str] = res

            if res.es_ok():
                salida, logs = res.unwrap()
                n_err  = sum(1 for l in logs if l.nivel == NivelLog.ERROR)
                n_warn = sum(1 for l in logs if l.nivel == NivelLog.WARN)
                nodos  = list(salida.keys())
                filas_resumen.append({
                    "archivo": ruta_str,
                    "estado": "ok",
                    "n_errores_log": n_err,
                    "n_warnings_log": n_warn,
                    "nodos_finales": ";".join(nodos),
                })
            else:
                err = res.error
                filas_resumen.append({
                    "archivo": ruta_str,
                    "estado": "err",
                    "n_errores_log": 1,
                    "n_warnings_log": 0,
                    "nodos_finales": getattr(err, "mensaje", str(err))[:80],
                })

        # Escribir resumen TSV
        if ruta_log_batch:
            ruta_log_batch = Path(ruta_log_batch)
            ruta_log_batch.parent.mkdir(parents=True, exist_ok=True)
            cabecera = "archivo\testado\tn_errores_log\tn_warnings_log\tnodos_finales\n"
            lineas   = [
                f"{f['archivo']}\t{f['estado']}\t{f['n_errores_log']}\t"
                f"{f['n_warnings_log']}\t{f['nodos_finales']}\n"
                for f in filas_resumen
            ]
            ruta_log_batch.write_text(cabecera + "".join(lineas), encoding="utf-8")
            if debug:
                print(f"\n[Batch] Resumen guardado en: {ruta_log_batch}")

        return resultados

    def ejecutar_batch_desde_archivo(
        self,
        nombre:        str,
        ruta_lista:    Union[str, Path],
        directorio:    Optional[Union[str, Path]] = None,
        modo:          ModoImagen = ModoImagen.AUTO,
        debug:         bool = False,
        ruta_log_batch: Optional[Path] = None,
    ) -> Dict[str, Resultado]:
        """
        Lee una lista de nombres de archivo (uno por línea) y ejecuta batch.

        ruta_lista : archivo .txt con nombres de archivo (uno por línea)
        directorio : directorio base donde viven los archivos
                     (si None, los nombres deben ser rutas absolutas)
        """
        ruta_lista = Path(ruta_lista)
        if not ruta_lista.exists():
            raise FileNotFoundError(f"Lista de archivos no encontrada: {ruta_lista}")

        nombres = [
            l.strip() for l in ruta_lista.read_text(encoding="utf-8").splitlines()
            if l.strip() and not l.startswith("#")
        ]

        if directorio:
            base = Path(directorio)
            rutas = [base / n for n in nombres]
        else:
            rutas = [Path(n) for n in nombres]

        return self.ejecutar_batch(
            nombre=nombre,
            rutas_imagenes=rutas,
            modo=modo,
            debug=debug,
            ruta_log_batch=ruta_log_batch,
        )

    # =========================================================
    # CORE INTERNO
    # =========================================================

    def _obtener_flujo(self, nombre: str) -> FlujoTrabajo:
        if nombre not in self._flujos:
            raise KeyError(
                f"Pipeline '{nombre}' no registrado. Disponibles: {self.listar()}"
            )
        return self._flujos[nombre]

    def _validar_y_loguear(
        self,
        flujo:      FlujoTrabajo,
        debug:      bool,
        recolector: RecolectorLog,
    ) -> DiagnosticoPipeline:
        """
        Ejecuta la validación y vuelca todos los eventos al recolector.
        NUNCA bloquea — siempre retorna el diagnóstico.
        """
        if debug:
            print(f"[GestorLab] Validando '{flujo.nombre}'...")

        resultado = validar_pipeline(flujo.grafo)
        diag: DiagnosticoPipeline = resultado.unwrap()  # Ok siempre

        for ev in diag.eventos:
            recolector._eventos.append(ev)
            if debug:
                icono = {"info": "✓", "warn": "⚠", "error": "✗"}.get(ev.nivel.value, "·")
                print(f"  {icono} [{ev.nivel.value.upper()}] {ev.etapa}: {ev.mensaje}")

        if debug:
            print(f"[GestorLab] {diag.resumen()}")

        # En modo estricto, el llamador puede inspeccionar el diagnóstico
        # para decidir si aborta — pero eso es responsabilidad del llamador,
        # no del gestor.
        return diag

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

            # ── QC modo debug ────────────────────────────────
            if self._modo_qc.get(nombre, False):
                self._generar_qc(flujo, nombre, debug)

            return Ok((salida, recolector.eventos))

        else:
            # Err del pipeline — lo registramos pero SEGUIMOS: retornamos
            # el Err envuelto con el log completo para que el llamador
            # pueda inspeccionar qué pasó
            error = resultado.error
            recolector.cosechar(resultado)
            recolector.agregar_manual(
                "pipeline",
                f"Ejecución con fallo: {getattr(error, 'mensaje', str(error))}",
                NivelLog.ERROR,
            )
            if debug:
                print(f"[GestorLab] ✗ ERR — {getattr(error, 'mensaje', str(error))}")

            self._guardar_si_corresponde(nombre, recolector)
            # Retornar el Err original — el log ya está guardado
            return resultado

    def _generar_qc(self, flujo: FlujoTrabajo, nombre: str, debug: bool) -> None:
        """
        Genera el plot QC paso a paso usando el nuevo QC_Visualizacion.
        Construye la secuencia [(op_nombre, BioImagenData)] desde los nodos.
        """
        from ..controlador.Controlador_BioImagen import BioImagenData as _BID
        from ..analizador.qc.QC_Visualizacion import visualizar_pasos, ConfigQC

        ruta_qc  = self._rutas_qc.get(nombre)
        canal_qc = self._canal_qc.get(nombre, 0)

        if debug:
            print(f"[GestorLab] Generando QC (canal={canal_qc})...")

        try:
            orden    = flujo.grafo.orden_topologico()
            secuencia = []
            for nodo_id in orden:
                nodo = flujo.grafo.nodos[nodo_id]
                if not nodo.data:
                    continue
                ultimo = nodo.data[-1]
                if not isinstance(ultimo, _BID):
                    continue
                entrantes = flujo.grafo.entrantes(nodo_id)
                op_nombre = (
                    entrantes[-1].operacion.nombre if entrantes else "input"
                )
                secuencia.append((op_nombre, ultimo))

            if not secuencia:
                if debug:
                    print("[GestorLab] QC: no hay nodos con BioImagenData")
                return

            cfg = ConfigQC(canal=canal_qc)
            res = visualizar_pasos(
                secuencia = secuencia,
                config    = cfg,
                ruta      = Path(ruta_qc) if ruta_qc else None,
                titulo    = f"QC — {nombre}",
            )
            if res.es_err() and debug:
                print(f"[GestorLab] ⚠ QC error: {res.error.mensaje}")

        except Exception as e:
            if debug:
                print(f"[GestorLab] ⚠ Error generando QC: {e}")

    def _guardar_si_corresponde(self, nombre: str, recolector: RecolectorLog) -> None:
        ruta = self._rutas_log.get(nombre)
        if ruta is None:
            return
        formato = "json" if Path(ruta).suffix == ".json" else "txt"
        try:
            recolector.guardar(Path(ruta), formato=formato)
        except Exception as e:
            print(f"[GestorLab] ⚠ No se pudo guardar log: {e}")

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
        print(f"\n[VALIDACIÓN INTRÍNSECA] {'OK' if ok else f'ERRORES ({len(errores)})'}")
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

    def mostrar_diagnostico(self, nombre: str) -> None:
        """Imprime el diagnóstico completo de validación sin ejecutar."""
        flujo = self._obtener_flujo(nombre)
        res   = validar_pipeline(flujo.grafo)
        diag  = res.unwrap()
        print(f"\n{diag.resumen()}")
        for ev in diag.eventos:
            icono = {"info": "✓", "warn": "⚠", "error": "✗"}.get(ev.nivel.value, "·")
            print(f"  {icono} [{ev.etapa}] {ev.mensaje}")

    def __repr__(self) -> str:
        return f"<GestorLab flujos={self.listar()}>"
