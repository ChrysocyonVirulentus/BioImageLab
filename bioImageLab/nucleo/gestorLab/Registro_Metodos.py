from typing import Dict, Type, Any


class RegistroMetodos:
    """
    Registry central de métodos del sistema.

    Soporta múltiples dominios:
    - filtrado
    - segmentacion
    - realce
    - analisis
    - etc.
    """

    def __init__(self):
        self._registro: Dict[str, Dict[str, Type[Any]]] = {}

    # REGISTRO

    def registrar(self, dominio: str, clase: Type[Any]):
        nombre = getattr(clase, "nombre", clase.__name__.lower())

        if dominio not in self._registro:
            self._registro[dominio] = {}

        if nombre in self._registro[dominio]:
            raise ValueError(
                f"Metodo '{nombre}' ya registrado en dominio '{dominio}'"
            )

        self._registro[dominio][nombre] = clase

    # GET

    def obtener(self, dominio: str, nombre: str) -> Type[Any]:
        if dominio not in self._registro:
            raise ValueError(f"Dominio '{dominio}' no existe")

        if nombre not in self._registro[dominio]:
            raise ValueError(
                f"Metodo '{nombre}' no registrado en dominio '{dominio}'"
            )

        return self._registro[dominio][nombre]

    def listar(self, dominio: str):
        return list(self._registro.get(dominio, {}).keys())


# Singleton global
registro_metodos = RegistroMetodos()


# DECORADORES

def registrar_en(dominio: str):
    """
    Decorador:

    @registrar_en("filtrado")
    class Gaussiano: ...
    """
    def wrapper(cls):
        registro_metodos.registrar(dominio, cls)
        return cls

    return wrapper