# === gestorLab/Log.py ===

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, UTC
from pathlib import Path
from typing import List, Optional, Union, Any

from ..controlador.Resultado_Either import Resultado, Ok, Err, LogEvento, NivelLog


# =========================================================
# RECOLECTOR DE LOGS
# =========================================================

class RecolectorLog:
    """
    Cosecha LogEventos de cajas Resultado a medida que el pipeline ejecuta.

    Uso:
        recolector = RecolectorLog("mi_pipeline")
        recolector.cosechar(resultado)        # extrae _log de la caja
        recolector.agregar_manual(evento)     # evento construido a mano
        recolector.guardar(Path("out.log"))   # escribe a disco
    """

    def __init__(self, nombre_pipeline: str):
        self._nombre    = nombre_pipeline
        self._eventos:  List[LogEvento] = []
        self._inicio:   datetime = datetime.now(UTC) # Ya no se usa datetime.utcnow()

    # ── cosecha ──────────────────────────────────────────────

    def cosechar(self, resultado: Resultado) -> None:
        """Extrae el _log de cualquier caja Ok o Err."""
        log = getattr(resultado, "_log", ())
        for evento in log:
            if isinstance(evento, LogEvento):
                self._eventos.append(evento)

    def cosechar_varios(self, resultados: List[Resultado]) -> None:
        for r in resultados:
            self.cosechar(r)

    def agregar_manual(
        self,
        etapa:   str,
        mensaje: str,
        nivel:   NivelLog = NivelLog.INFO,
        metadata: dict    = None,
    ) -> None:
        self._eventos.append(LogEvento(
            etapa    = etapa,
            mensaje  = mensaje,
            nivel    = nivel,
            metadata = metadata or {},
        ))

    # ── acceso ───────────────────────────────────────────────

    @property
    def eventos(self) -> List[LogEvento]:
        return list(self._eventos)

    def tiene_errores(self) -> bool:
        return any(e.nivel == NivelLog.ERROR for e in self._eventos)

    def tiene_warnings(self) -> bool:
        return any(e.nivel == NivelLog.WARN for e in self._eventos)

    def filtrar(self, nivel: NivelLog) -> List[LogEvento]:
        return [e for e in self._eventos if e.nivel == nivel]

    # ── escritura ────────────────────────────────────────────

    def guardar(
        self,
        ruta:   Path,
        formato: str = "txt",   # "txt" | "json"
    ) -> None:
        ruta = Path(ruta)
        ruta.parent.mkdir(parents=True, exist_ok=True)

        if formato == "json":
            self._guardar_json(ruta)
        else:
            self._guardar_txt(ruta)

    def _guardar_txt(self, ruta: Path) -> None:
        lineas = [
            f"{'='*60}",
            f"  PIPELINE : {self._nombre}",
            f"  INICIO   : {self._inicio.isoformat()}",
            f"  FIN      : {datetime.now(UTC).isoformat()}", # ya no se usa datetime.utcnow()
            f"  EVENTOS  : {len(self._eventos)}",
            f"  ERRORES  : {len(self.filtrar(NivelLog.ERROR))}",
            f"  WARNINGS : {len(self.filtrar(NivelLog.WARN))}",
            f"{'='*60}",
            "",
        ]

        for ev in self._eventos:
            icono = _icono(ev.nivel)
            lineas.append(f"[{ev.timestamp}] {icono} [{ev.nivel.value.upper()}]")
            lineas.append(f"  etapa  : {ev.etapa}")
            lineas.append(f"  mensaje: {ev.mensaje}")
            if ev.metadata:
                for k, v in ev.metadata.items():
                    lineas.append(f"  {k:8}: {v}")
            lineas.append("")

        ruta.write_text("\n".join(lineas), encoding="utf-8")

    def _guardar_json(self, ruta: Path) -> None:
        payload = {
            "pipeline":  self._nombre,
            "inicio":    self._inicio.isoformat(),
            "fin":       datetime.now(UTC).isoformat(), # ya no se usa datetime.utcnow()
            "resumen": {
                "total":    len(self._eventos),
                "errores":  len(self.filtrar(NivelLog.ERROR)),
                "warnings": len(self.filtrar(NivelLog.WARN)),
                "infos":    len(self.filtrar(NivelLog.INFO)),
            },
            "eventos": [_evento_a_dict(e) for e in self._eventos],
        }
        ruta.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )

    def __repr__(self) -> str:
        return (
            f"<RecolectorLog '{self._nombre}' "
            f"eventos={len(self._eventos)} "
            f"errores={len(self.filtrar(NivelLog.ERROR))}>"
        )


# =========================================================
# HELPERS INTERNOS
# =========================================================

def _icono(nivel: NivelLog) -> str:
    return {
        NivelLog.INFO:  "✓",
        NivelLog.WARN:  "⚠",
        NivelLog.ERROR: "✗",
    }.get(nivel, "·")


def _evento_a_dict(evento: LogEvento) -> dict:
    return {
        "timestamp": evento.timestamp,
        "nivel":     evento.nivel.value,
        "etapa":     evento.etapa,
        "mensaje":   evento.mensaje,
        "metadata":  evento.metadata or {},
    }