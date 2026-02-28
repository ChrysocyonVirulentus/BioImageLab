from __future__ import annotations # Para manejar el tipado limpio
from dataclasses import dataclass
from typing import Generic, TypeVar, Callable, Union, Final, Generator, Any, Iterable
from pathlib import Path
import numpy as np

# Patrón Result Refactorizado para que se comporte funcionalmente como Either.

T = TypeVar("T") # Type (Tipo exitoso), dato "bueno"
U = TypeVar("U") # Updated (Nuevo Tipo) , luego de una transformación
E = TypeVar("E") # Error (Excepcion de error)
F = TypeVar("F") # Failed-Transformation (Nuevo tipo de error su se decide transformar el error anterior)

@dataclass(frozen=True)
class ErrorPipeline:
    etapa: str  # "lectura", "preprocesamiento", "procesamiento"
    mensaje: str
    ruta: Path | None = None
    causa: Exception | None = None
    
    def con_contexto(self, nueva_etapa: str) -> ErrorPipeline:
        """Añade contexto al error (patrón de error wrapping)"""
        return ErrorPipeline(
            etapa=f"{nueva_etapa} -> {self.etapa}",
            mensaje=self.mensaje,
            ruta=self.ruta,
            causa=self.causa
        )

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
    
    def tap(self, f: Callable[[T], None]) -> Resultado[T, E]: # Para hacer logging.
        """
            Efecto secundario sin alterar el flujo (logging, debug)
        """
        if self.es_ok():
            f(self.unwrap())  # type: ignore
        return self  # type: ignore


@dataclass(frozen=True)
class Ok(Resultado[T, E]):
    _value: T  # privado por convención

    @property
    def value(self) -> T:
        return self._value

    def es_ok(self) -> bool:
        return True

    def es_err(self) -> bool:
        return False

    def map(self, f: Callable[[T], U]) -> Resultado[U, E]:# Transforma si es OK
        try:
            return Ok(f(self._value))
        except Exception as e:
            # Nota: Aquí convertir 'e' al tipo 'E' esperado
            return Err(e)  # type: ignore

    def bind(self, f: Callable[[T], Resultado[U, E]]) -> Resultado[U, E]: # Transfforma, pero si viene de Result (aplanar la estructura). Encadenado si hay fallo
        return f(self._value)

    def map_err(self, f: Callable[[E], F]) -> Resultado[T, F]:
        return Ok(self._value)  # Ignora transformación de error

    def unwrap(self) -> T:
        return self._value

    def unwrap_or(self, default: T) -> T:
        return self._value

    def unwrap_or_else(self, f: Callable[[E], T]) -> T:
        return self._value


@dataclass(frozen=True)
class Err(Resultado[T, E]):
    _error: E

    @property
    def error(self) -> E:
        return self._error

    def es_ok(self) -> bool:
        return False

    def es_err(self) -> bool:
        return True

    def map(self, f: Callable[[T], U]) -> Resultado[U, E]:
        return Err(self._error)  # Propaga error sin ejecutar f

    def bind(self, f: Callable[[T], Resultado[U, E]]) -> Resultado[U, E]:
        return Err(self._error)  # Propaga error sin ejecutar f

    def map_err(self, f: Callable[[E], F]) -> Resultado[T, F]:
        try:
            return Err(f(self._error))
        except Exception as e:
            return Err(e)  # type: ignore

    def unwrap(self) -> T:
        if isinstance(self._error, Exception):
            raise self._error
        raise ValueError(f"Cannot unwrap Err: {self._error}")

    def unwrap_or(self, default: T) -> T:
        return default

    def unwrap_or_else(self, f: Callable[[E], T]) -> T:
        return f(self._error)

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