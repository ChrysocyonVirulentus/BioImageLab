from __future__ import annotations # Para manejar el tipado limpio
from dataclasses import dataclass, field
from typing import Generic, Dict, TypeVar, Callable, Union, Final, Generator, Any, Iterable
from enum import Enum
from datetime import datetime
from pathlib import Path
import numpy as np

# Patrón Result Refactorizado para que se comporte funcionalmente como Either.

T = TypeVar("T") # Type (Tipo exitoso), dato "bueno"
U = TypeVar("U") # Updated (Nuevo Tipo) , luego de una transformación
E = TypeVar("E") # Error (Excepcion de error)
F = TypeVar("F") # Failed-Transformation (Nuevo tipo de error su se decide transformar el error anterior)

@dataclass(frozen=True)
class ErrorPipeline:
    """Error universal del pipeline — todas las capas pueden convertirse a este."""
    etapa:   str
    mensaje: str
    metodo:  str            = ""
    ruta:    Path | None    = None
    causa:   Exception | None = None

    def con_contexto(self, nueva_etapa: str) -> ErrorPipeline:
        from dataclasses import replace
        return replace(self, etapa=f"{nueva_etapa} -> {self.etapa}")

# Para los logs de error o exito:
class NivelLog(Enum):
    INFO = "info"
    WARN  = "warn"
    ERROR = "error"

@dataclass(frozen=True)
class LogEvento:
    etapa:     str
    mensaje:   str
    nivel:     NivelLog
    metadata:  Dict[str, Any]  = field(default_factory=dict)
    timestamp: str             = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> dict:
        return {"etapa": self.etapa, "mensaje": self.mensaje, "metadata": self.metadata}

# La idea de esta arquitectura es que quede a la Haskell de: Resultado = Either + Writer, o sea un  WriterT [Log] (Either Error) a
class Resultado(Generic[T, E]): # Caja con un exito tipo T o un error tipo E
    """
        Either Result para pipeline de bioimágenes, con el comportamiento de encadenamiento de Monada
    """
    
    def es_ok(self) -> bool:
        raise NotImplementedError

    def es_err(self) -> bool:
        raise NotImplementedError

    def map(self, f: Callable[[T], U]) -> Resultado[U, E]:
        raise NotImplementedError

    def bind(self, f: Callable[[T], Resultado[U, E]]) -> Resultado[U, E]:
        raise NotImplementedError
        
    def map_err(self, f: Callable[[E], F]) -> Resultado[T, F]:
        raise NotImplementedError

    def unwrap(self) -> T: # Si da OK, se abre la caja y se saca el dato
        raise NotImplementedError

    def unwrap_or(self, default: T) -> T: # Si hay un error, me da el default.
        raise NotImplementedError

    def unwrap_or_else(self, f: Callable[[E], T]) -> T: # Para valor de rescate
        raise NotImplementedError
        
    # Para los Pipelines
    
    def pipe(self, *functions: Callable[[T], U]) -> Resultado[U, E]:
        """
            Aplica secuencia de funciones puras
        """
        result: Resultado[Any, E] = self
        for f in functions:
            if result.es_err():
                return result  # type: ignore
            result = result.map(f)
        return result  # type: ignore

    def log(self, evento: LogEvento) -> Resultado[T, E]:
        if isinstance(self, Ok):
            return Ok(self._value, self._log + (evento,))
        else:
            return Err(self._error, self._log + (evento,))
    
    def tap(self, f: Callable[[T], None]) -> Resultado[T, E]: # Para hacer logging.
        """
            Efecto secundario sin alterar el flujo (logging, debug)
        """
        if self.es_ok():
            f(self.unwrap())
        return self


@dataclass(frozen=True)
class Ok(Resultado[T, E]):
    _value: T  # privado por convención
    _log:   tuple = ()  # tuple[LogEvento, ...]

    @property
    def value(self) -> T:
        return self._value

    def es_ok(self) -> bool:
        return True

    def es_err(self) -> bool:
        return False

    def map(self, f: Callable[[T], U]) -> Resultado[U, E]:# Transforma si es OK
        try:
            return Ok(f(self._value), self._log)
        except Exception as e:
            # Nota: Aquí convertir 'e' al tipo 'E' esperado
            return Err(e, self._log)  # type: ignore

    def bind(self, f: Callable[[T], Resultado[U, E]]) -> Resultado[U, E]: # Transfforma, pero si viene de Result (aplanar la estructura). Encadenado si hay fallo
        resultado = f(self._value)

        if isinstance(resultado, Ok):
            return Ok(resultado._value, self._log + resultado._log)
        else:
            return Err(resultado._error, self._log + resultado._log)

    def map_err(self, f: Callable[[E], F]) -> Resultado[T, F]:
        return Ok(self._value, self._log)  # Ignora transformación de error

    def unwrap(self) -> T:
        return self._value

    def unwrap_or(self, default: T) -> T:
        return self._value

    def unwrap_or_else(self, f: Callable[[E], T]) -> T:
        return self._value
    
    def agregar_log(self, mensaje: str, etapa: str = "", nivel: NivelLog = NivelLog.INFO) -> "Ok[T, E]":
        # CORREGIDO: 'etapa' era undefined; ahora es parámetro con default
        evento = LogEvento(etapa=etapa, mensaje=mensaje, nivel=nivel)
        return Ok(self._value, self._log + (evento,))


@dataclass(frozen=True)
class Err(Resultado[T, E]):
    _error: E
    _log:   tuple = ()  # tuple[LogEvento, ...]

    @property
    def error(self) -> E:
        return self._error

    def es_ok(self) -> bool:
        return False

    def es_err(self) -> bool:
        return True

    def map(self, f: Callable[[T], U]) -> Resultado[U, E]:
        return Err(self._error, self._log)  # Propaga error sin ejecutar f

    def bind(self, f: Callable[[T], Resultado[U, E]]) -> Resultado[U, E]:
        return Err(self._error, self._log)

    def map_err(self, f: Callable[[E], F]) -> Resultado[T, F]:
        try:
            return Err(f(self._error), self._log)
        except Exception as e:
            return Err(e, self._log)  # type: ignore

    def unwrap(self) -> T:
        if isinstance(self._error, Exception):
            raise self._error
        raise ValueError(f"Cannot unwrap Err: {self._error}")

    def unwrap_or(self, default: T) -> T:
        return default

    def unwrap_or_else(self, f: Callable[[E], T]) -> T:
        return f(self._error)

    def agregar_log(self, mensaje: str, etapa: str = "", nivel: NivelLog = NivelLog.ERROR) -> "Err[T, E]":
        evento = LogEvento(etapa=etapa, mensaje=mensaje, nivel=nivel)
        return Err(self._error, self._log + (evento,))
        
# Simular Do-Notation de Haskell
def result_do(func: Callable[..., Generator[Resultado[Any, E], Any, Resultado[T, E]]]) -> Callable[..., Resultado[T, E]]:
    def wrapper(*args, **kwargs) -> Resultado[T, E]:
        gen = func(*args, **kwargs)
        try:
            # Obtener el primer Resultado
            proximo_resultado = next(gen)
            while True:
                if proximo_resultado.es_err():
                    return proximo_resultado  # Cortocircuito (Short-circuit)
                
                # "Desempaquetar" y enviamos de vuelta al yield
                proximo_resultado = gen.send(proximo_resultado.unwrap())
        except StopIteration as e:
            # El 'return' de la función generadora es el resultado final
            return e.value
    return wrapper