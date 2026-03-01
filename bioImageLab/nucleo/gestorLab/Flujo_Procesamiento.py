# === nucleo/pipeline/FlujoProcesamiento.py ===
import yaml
from pathlib import Path
from typing import Dict, List, Any, Type, Optional, Callable
from dataclasses import dataclass

from nucleo.controlador.Resultado_Either import Resultado, Ok, Err, result_do, ErrorBioImagen
from nucleo.controlador.Controlador_BioImagen import ControladorBioImagen, ModoImagen, BioImagenData

from .Categoria_Operacion import CategoriaOperacion
from .Operacion import Operacion
from .Pipeline import Pipeline
from .Pipeline_Builder import PipelineBuilder


@dataclass(frozen=True)
class RegistroWrapper:
    """
    Registra un "wrapper" (ej: Normalizador, Controlador_Filtrador) que sabe
    cómo instanciar componentes individuales desde parámetros YAML.
    """
    nombre: str                      # Nombre en YAML (ej: "fft_pasabajo")
    categoria: CategoriaOperacion    # Categoría semántica
    clase_wrapper: Type              # Clase del wrapper (ej: Controlador_Filtrador)
    metodo_factory: str             # Método del wrapper que crea el callable (ej: "crear_filtro")
    
    def crear_operacion(
        self, 
        params: Dict[str, Any], 
        canal: Optional[int] = None
    ) -> Resultado[Operacion, ErrorBioImagen]:
        """
        Instancia el wrapper, le pide que cree el componente, 
        y envuelve todo en una Operacion lista para el pipeline.
        """
        try:
            # Instanciar wrapper (ej: Controlador_Filtrador())
            # Nota: Algunos wrappers pueden necesitar parámetros de inicialización
            wrapper = self.clase_wrapper()
            
            # Llamar método factory para obtener el callable (ej: FFTPasabajo instanciado)
            metodo = getattr(wrapper, self.metodo_factory)
            instancia_callable = metodo(**params)  # Esto retorna el objeto callable (ej: FFTPasabajo(radio=30))
            
            # Verificar que es realmente callable
            if not callable(instancia_callable):
                return Err(ErrorBioImagen(
                    etapa="registro",
                    mensaje=f"{self.clase_wrapper.__name__}.{self.metodo_factory} no retornó un callable",
                ))
            
            # Crear operación adaptada
            return Ok(Operacion(
                nombre=self.nombre,
                categoria=self.categoria,
                instancia_callable=instancia_callable,
                canal_objetivo=canal,
                parametros_originales=params,
                descripcion=f"Creado via {self.clase_wrapper.__name__}"
            ))
            
        except Exception as e:
            return Err(ErrorBioImagen(
                etapa="registro",
                mensaje=f"Error creando {self.nombre}: {str(e)}",
                causa=e
            ))


class CatalogoWrappers:
    """Catálogo de wrappers disponibles para construir operaciones desde YAML"""
    
    def __init__(self):
        self._registros: Dict[str, RegistroWrapper] = {}
    
    def registrar(
        self,
        nombre_yaml: str,
        categoria: CategoriaOperacion,
        clase_wrapper: Type,
        metodo_factory: str = "__call__"  # Por defecto, el wrapper es directamente callable
    ):
        """
        Registra un wrapper.
        
        Ejemplo:
            catalogo.registrar(
                "fft_pasabajo", 
                CategoriaOperacion.FILTRACION,
                Controlador_Filtrador,
                "crear_pasabajo"  # Método que retorna FFTPasabajo configurado
            )
        """
        self._registros[nombre_yaml] = RegistroWrapper(
            nombre=nombre_yaml,
            categoria=categoria,
            clase_wrapper=clase_wrapper,
            metodo_factory=metodo_factory
        )
        return self
    
    def obtener(self, nombre: str) -> Optional[RegistroWrapper]:
        return self._registros.get(nombre)
    
    def nombres_disponibles(self) -> List[str]:
        return list(self._registros.keys())


class FlujoProcesamiento:
    """
    Orquestador que:
    1. Lee YAML con especificación de operaciones
    2. Usa CatalogoWrappers para instanciar componentes via sus wrappers
    3. Construye Pipeline validado
    4. Ejecuta con do-notation y logging
    """
    
    def __init__(
        self,
        controlador: ControladorBioImagen,
        catalogo: Optional[CatalogoWrappers] = None,
        debug: bool = False
    ):
        self.controlador = controlador
        self.catalogo = catalogo or CatalogoWrappers()
        self.debug = debug
        self.procesados: Dict[str, BioImagenData] = {}
        self.log_ejecucion: List[str] = []
        self._pipeline: Optional[Pipeline] = None

    def _log(self, mensaje: str):
        self.log_ejecucion.append(mensaje)
        if self.debug:
            print(f"[PIPELINE] {mensaje}")

    @result_do
    def ejecutar_pipeline_yaml(self, ruta_yaml: Path) -> Resultado[BioImagenData, ErrorBioImagen]:
        """
        Ejecuta pipeline completo desde YAML usando do-notation.
        """
        self.log_ejecucion = []
        self.procesados = {}
        self._log(f"Iniciando: {ruta_yaml}")

        # 1. Leer YAML
        config = yield self._leer_config(ruta_yaml)
        pasos_yaml = config.get('pipeline', [])
        self._log(f"{len(pasos_yaml)} pasos configurados")

        # 2. Construir operaciones desde wrappers
        operaciones: List[Operacion] = []
        
        for i, paso in enumerate(pasos_yaml):
            id_op = paso.get('operacion')
            params = paso.get('parametros', {})
            canal = paso.get('canal')  # Opcional: canal específico
            
            registro = self.catalogo.obtener(id_op)
            if not registro:
                disponibles = ", ".join(self.catalogo.nombres_disponibles())
                return Err(ErrorBioImagen(
                    etapa="configuracion",
                    mensaje=f"Operación '{id_op}' no registrada. Disponibles: {disponibles}"
                ))
            
            # Crear operación via wrapper
            resultado_op = registro.crear_operacion(params, canal)
            if resultado_op.es_err():
                return resultado_op  # Propaga error de instanciación
            
            operaciones.append(resultado_op.unwrap())
            self._log(f"[{i}] {registro.categoria.name}::{id_op}")

        # 3. Construir pipeline (valida orden semántico)
        try:
            builder = PipelineBuilder()
            for op in operaciones:
                builder.agregar(op)
            self._pipeline = builder.construir()
            self._log(f"Pipeline validado: {len(self._pipeline)} etapas")
        except ValueError as e:
            return Err(ErrorBioImagen(etapa="validacion", mensaje=str(e)))

        # 4. Cargar imagen
        data = yield self.controlador.cargar_ImagenResultado(ModoImagen.AUTO)
        self._log(f"Imagen: {data.dims}")

        # 5. Ejecutar con inspección paso a paso (para debug)
        if self.debug:
            # Modo debug: snapshots después de cada operación
            resultado, snapshots = self._pipeline.ejecutar_con_snapshots(data)
            for i, nombre, snap in snapshots:
                if "despues_" in nombre:
                    clave = f"{i:02d}_{nombre.replace('despues_', '')}"
                    self.procesados[clave] = snap
            data_final = yield resultado
        else:
            # Modo normal: ejecución directa
            data_final = yield self._pipeline.ejecutar(
                data,
                callback_progreso=lambda i, op, d: self._log(f"  ✓ {op.nombre}")
            )

        self._log("Completado exitosamente")
        return Ok(data_final)

    def _leer_config(self, ruta: Path) -> Resultado[dict, ErrorBioImagen]:
        try:
            with open(ruta, 'r') as f:
                return Ok(yaml.safe_load(f))
        except Exception as e:
            return Err(ErrorBioImagen("lectura", f"YAML inválido: {e}", ruta))

    def obtener_snapshot(self, clave: str) -> Optional[BioImagenData]:
        return self.procesados.get(clave)

    def obtener_log(self) -> str:
        return "\n".join(self.log_ejecucion)