import yaml
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Type

from ResultEither import Resultado, Ok, Err, result_do, ErrorBioImagen
from ControladorBioImagen import ControladorBioImagen, ModoImagen

class FlujoProcesamiento:
    COMPONENTES_DISPONIBLES: List[Type] = [
        # Aquí van tus clases: Normalizador, FiltroEspectral, etc.
    ]

    def __init__(self, controlador: ControladorBioImagen, debug: bool = False):
        self.controlador = controlador
        self.debug = debug
        # Diccionario para inspección: { "0_normalizador": array, "1_fft": array }
        self.procesados: Dict[str, np.ndarray] = {}
        self.log_ejecucion: List[str] = []

    def _log(self, mensaje: str):
        """Helper interno para logging."""
        self.log_ejecucion.append(mensaje)
        if self.debug:
            print(f"[PIPELINE LOG] {mensaje}")

    @result_do
    def ejecutar_pipeline_yaml(self, ruta_yaml: Path) -> Resultado[np.ndarray, ErrorBioImagen]:
        self.log_ejecucion = [] # Reset log
        self._log(f"Iniciando pipeline desde {ruta_yaml}")

        # 1. Leer Config
        config = yield self._leer_config(ruta_yaml)
        
        # 2. Cargar imagen (vía yield)
        data = yield self.controlador.cargar_ImagenResultado(ModoImagen.AUTO)
        imagen_actual = data.datos
        
        self._log("Imagen cargada exitosamente.")

        # 3. Iteración dinámica con inspección
        for i, paso in enumerate(config.get('pipeline', [])):
            id_op = paso.get('operacion')
            params = paso.get('parametros', {})
            
            clase_proceso = self._buscar_clase_por_nombre(id_op)
            if not clase_proceso:
                return Err(ErrorBioImagen("configuracion", f"Operación no registrada: {id_op}"))

            # Instanciación y Ejecución
            try:
                instancia = clase_proceso(**params)
                
                # Ejecutamos y usamos .tap para loguear éxito sin salir del do-notation
                resultado_paso = instancia(imagen_actual)
                
                # Si falla, el yield cortará aquí. 
                # Si tiene éxito, actualizamos la imagen.
                imagen_actual = yield resultado_paso.tap(
                    lambda _: self._log(f"Etapa {i} ({id_op}): Completada OK.")
                )
                
                # METODO DE DEBUGGING: Almacenamiento temporal para verificación real
                if self.debug:
                    clave = f"{i}_{id_op}"
                    # Guardamos una copia para que modificaciones posteriores no afecten el histórico
                    self.procesados[clave] = imagen_actual.copy()
                    self._log(f"Snapshot guardado en 'procesados['{clave}']'")

            except Exception as e:
                return Err(ErrorBioImagen("procesamiento", f"Fallo en {id_op}: {str(e)}"))

        self._log("Pipeline finalizado con éxito.")
        return Ok(imagen_actual)

    def _buscar_clase_por_nombre(self, nombre: str) -> Optional[Type]:
        return next((cls for cls in self.COMPONENTES_DISPONIBLES if getattr(cls, 'nombre', None) == nombre), None)

    def _leer_config(self, ruta: Path) -> Resultado[dict, ErrorBioImagen]:
        try:
            with open(ruta, 'r') as f:
                return Ok(yaml.safe_load(f))
        except Exception as e:
            return Err(ErrorBioImagen("lectura", f"YAML Inválido: {e}", ruta))

    def obtener_log(self) -> str:
        """Devuelve el historial de ejecución como string."""
        return "\n".join(self.log_ejecucion)