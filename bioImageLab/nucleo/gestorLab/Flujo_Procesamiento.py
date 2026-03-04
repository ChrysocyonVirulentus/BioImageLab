# === gestorLab/Flujo_Procesamiento.py (integrado con GestorRamas) ===
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional

from ..controlador.Resultado_Either import Resultado, Ok, Err, result_do
from ..controlador.Controlador_BioImagen import ControladorBioImagen, ModoImagen, BioImagenData, ErrorBioImagen

from .Categoria_Operacion import CategoriaOperacion
from .Operacion import Operacion, TipoSalida
from .Pipeline import Pipeline
from .Pipeline_Builder import PipelineBuilder
from .Gestor_Ramas import GestorRamas, TipoRama, Rama

class FlujoProcesamiento:
    """
    Interfaz unificada que puede ejecutar:
    - Pipelines lineales simples (sin splits)
    - Pipelines con ramas (con splits/merges) via GestorRamas
    """
    
    def __init__(
        self,
        controlador: ControladorBioImagen,
        catalogo: Optional['CatalogoWrappers'] = None,  # Forward ref
        debug: bool = False,
        modo_estricto: bool = True,
        max_workers: int = 4
    ):
        self.controlador = controlador
        self.catalogo = catalogo
        self.debug = debug
        self.modo_estricto = modo_estricto
        self.max_workers = max_workers
        self.procesados: Dict[str, BioImagenData] = {}
        self.log_ejecucion: List[str] = []
        self._ultimo_gestor: Optional[GestorRamas] = None

    def _log(self, mensaje: str):
        self.log_ejecucion.append(mensaje)
        if self.debug:
            print(f"[FLUJO] {mensaje}")

    @result_do
    def ejecutar_pipeline_yaml(self, ruta_yaml: Path) -> Resultado[BioImagenData, ErrorBioImagen]:
        """
        Ejecuta pipeline desde YAML, detectando automáticamente si necesita ramas.
        """
        self.log_ejecucion = []
        self.procesados = {}
        self._log(f"Iniciando: {ruta_yaml}")

        # 1. Leer YAML
        config = yield self._leer_config(ruta_yaml)
        
        # 2. Detectar si YAML define ramas explícitas o pipeline lineal
        if 'ramas' in config:
            # Modo ramas explícitas
            return self._ejecutar_modo_ramas_explicitas(config)
        else:
            # Modo pipeline lineal (detectar splits implícitos)
            return self._ejecutar_modo_lineal_o_auto_split(config)

    def _ejecutar_modo_lineal_o_auto_split(
        self, 
        config: Dict[str, Any]
    ) -> Resultado[BioImagenData, ErrorBioImagen]:
        """Ejecuta pipeline lineal, convirtiendo a ramas si hay splits detectados"""
        from ResultEither import result_do
        
        @result_do
        def _ejecutar():
            pasos = config.get('pipeline', [])
            
            # Construir operaciones
            operaciones = []
            for i, paso in enumerate(pasos):
                op_result = self._crear_operacion_desde_paso(paso, i)
                if op_result.es_err():
                    return op_result
                operaciones.append(op_result.unwrap())
            
            # Crear pipeline y detectar splits
            pipeline = Pipeline("main", tuple(operaciones))
            
            if pipeline.puntos_split:
                self._log(f"Detectados {len(pipeline.puntos_split)} splits automáticos")
                gestor = GestorRamas(self.max_workers)
                gestor.construir_desde_pipeline_unitario(pipeline)
                self._ultimo_gestor = gestor
                
                # Cargar imagen
                data = yield self.controlador.cargar_ImagenResultado(ModoImagen.AUTO)
                
                # Ejecutar ramas
                resultados = yield gestor.ejecutar(data, paralelo=True)
                
                # Retornar resultado de rama principal o merge
                if "base" in resultados:
                    return resultados["base"]
                elif "principal" in resultados:
                    return resultados["principal"]
                else:
                    # Retornar primer resultado exitoso
                    for res in resultados.values():
                        if isinstance(res, Resultado) and res.es_ok():
                            return res
                            
                return Err(ErrorBioImagen(
                    etapa="orquestacion",
                    mensaje="Ninguna rama produjo resultado válido"
                ))
            else:
                # Pipeline simple sin splits
                builder = PipelineBuilder(modo_estricto=self.modo_estricto)
                for op in operaciones:
                    try:
                        builder.agregar(op)
                    except ValueError as e:
                        return Err(ErrorBioImagen("validacion", str(e)))
                
                pipeline_final = builder.construir()
                data = yield self.controlador.cargar_ImagenResultado(ModoImagen.AUTO)
                
                if self.debug:
                    resultado, snapshots = pipeline_final.ejecutar_con_snapshots(data)
                    for i, nombre, snap in snapshots:
                        if "despues_" in nombre:
                            clave = f"{i:02d}_{nombre.replace('despues_', '')}"
                            self.procesados[clave] = snap
                    return resultado
                else:
                    return pipeline_final.ejecutar(data)
        
        return _ejecutar()

    def _ejecutar_modo_ramas_explicitas(
        self,
        config: Dict[str, Any]
    ) -> Resultado[BioImagenData, ErrorBioImagen]:
        """Ejecuta cuando YAML define ramas explícitamente"""
        from ResultEither import result_do
        
        @result_do
        def _ejecutar():
            gestor = GestorRamas(self.max_workers)
            
            # Definir cada rama desde YAML
            for id_rama, def_rama in config['ramas'].items():
                ops = []
                for i, paso in enumerate(def_rama.get('operaciones', [])):
                    op_result = self._crear_operacion_desde_paso(paso, i)
                    if op_result.es_err():
                        return op_result
                    ops.append(op_result.unwrap())
                
                gestor.definir_rama(
                    id_rama=id_rama,
                    nombre=def_rama.get('nombre', id_rama),
                    tipo=TipoRama[def_rama.get('tipo', 'LINEAL')],
                    operaciones=tuple(ops),
                    requiere=set(def_rama.get('requiere', [])),
                    exporta_para=set(def_rama.get('exporta_para', []))
                )
            
            self._ultimo_gestor = gestor
            
            # Validar y ejecutar
            valido, errores = gestor.validar_grafo()
            if not valido:
                return Err(ErrorBioImagen(
                    etapa="configuracion",
                    mensaje=f"Grafo de ramas inválido: {'; '.join(errores)}"
                ))
            
            data = yield self.controlador.cargar_ImagenResultado(ModoImagen.AUTO)
            resultados = yield gestor.ejecutar(data, paralelo=True)
            
            # Retornar resultado de rama final especificada o default
            rama_final = config.get('rama_salida', 'principal')
            if rama_final in resultados:
                return resultados[rama_final]
            
            # Fallback: primer resultado válido
            for res in resultados.values():
                if isinstance(res, Resultado) and res.es_ok():
                    return res
                    
            return Err(ErrorBioImagen(
                etapa="orquestacion",
                mensaje="No se pudo determinar resultado final"
            ))
        
        return _ejecutar()

    def _crear_operacion_desde_paso(
        self,
        paso: Dict[str, Any],
        indice: int
    ) -> Resultado[Operacion, ErrorBioImagen]:
        """Factory para crear Operacion desde config YAML"""
        id_op = paso.get('operacion')
        params = paso.get('parametros', {})
        canal = paso.get('canal')
        
        if not self.catalogo:
            return Err(ErrorBioImagen(
                "configuracion",
                f"No hay catálogo registrado para crear '{id_op}'"
            ))
        
        registro = self.catalogo.obtener(id_op)
        if not registro:
            return Err(ErrorBioImagen(
                "configuracion",
                f"Operación '{id_op}' no existe en catálogo"
            ))
        
        return registro.crear_operacion(params, canal)

    def _leer_config(self, ruta: Path) -> Resultado[Dict, ErrorBioImagen]:
        try:
            import yaml
            with open(ruta, 'r') as f:
                return Ok(yaml.safe_load(f))
        except Exception as e:
            return Err(ErrorBioImagen("lectura", f"YAML inválido: {e}", ruta))

    def obtener_gestor_ramas(self) -> Optional[GestorRamas]:
        return self._ultimo_gestor
    
    def obtener_resultado_rama(self, id_rama: str) -> Optional[Resultado[Any, ErrorBioImagen]]:
        if self._ultimo_gestor:
            return self._ultimo_gestor.obtener_resultado_rama(id_rama)
        return None

    def obtener_log(self) -> str:
        return "\n".join(self.log_ejecucion)