# === gestorLab/Gestor_Ramas.py ===
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any, Callable
from enum import Enum, auto
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy

from ..controlador.Resultado_Either import Resultado, Ok, Err
from ..controlador.Controlador_BioImagen import BioImagenData, ErrorBioImagen

from .Categoria_Operacion import CategoriaOperacion, RequisitoRama
from .Operacion import Operacion, TipoSalida
from .Pipeline import Pipeline, ContextoRama, EstadoRama

class TipoRama(Enum):
    LINEAL = "lineal"           # Sin bifurcación
    DIVERGENTE = "divergente"   # Genera múltiples salidas (split)
    CONVERGENTE = "convergente" # Requiere múltiples entradas (merge)
    PARALELA = "paralela"       # Ejecuta en paralelo con otras

@dataclass
class Rama:
    """
    Una rama de procesamiento con su propio pipeline parcial.
    """
    id: str  # "A", "B", "segmentacion", "cuantificacion", etc.
    nombre: str  # Descripción legible
    tipo: TipoRama
    pipeline: Pipeline  # Sub-conjunto de operaciones para esta rama
    requiere_de: Set[str] = field(default_factory=set)  # IDs de ramas de las que depende
    exporta_para: Set[str] = field(default_factory=set)  # IDs de ramas que dependen de esta
    
    # Estado de ejecución
    contexto: Optional[ContextoRama] = None
    resultado: Optional[Resultado[Any, ErrorBioImagen]] = None
    
    def es_hoja(self) -> bool:
        """True si no tiene ramas dependientes (es última)"""
        return not self.exporta_para
    
    def es_raiz(self) -> bool:
        """True si no depende de otras ramas (es primera)"""
        return not self.requiere_de

class GestorRamas:
    """
    Orquesta múltiples ramas de procesamiento, manejando splits y merges.
    
    Caso de uso típico:
    1. Rama Principal: Imagen → Normalizar → Filtrar → Realzar → [SPLIT]
    2. Rama A (Procesada): continúa con Transformar → Cuantificar intensidad
    3. Rama B (Segmentada): Segmentar → [espera Rama A] → Cuantificar morfometría
    4. Merge: Combinar métricas de A y B en tabla final
    """
    
    def __init__(self, max_workers: int = 4):
        self.ramas: Dict[str, Rama] = {}
        self.max_workers = max_workers
        self.log: List[str] = []
    
    def _log(self, mensaje: str):
        self.log.append(mensaje)
        print(f"[GESTOR_RAMAS] {mensaje}")
    
    def definir_rama(
        self,
        id_rama: str,
        nombre: str,
        tipo: TipoRama,
        operaciones: Tuple[Operacion, ...],
        requiere: Optional[Set[str]] = None,
        exporta_para: Optional[Set[str]] = None
    ) -> 'GestorRamas':
        """
        Define una nueva rama en el grafo de procesamiento.
        """
        self.ramas[id_rama] = Rama(
            id=id_rama,
            nombre=nombre,
            tipo=tipo,
            pipeline=Pipeline(id_rama, operaciones),
            requiere_de=requiere or set(),
            exporta_para=exporta_para or set()
        )
        return self
    
    def construir_desde_pipeline_unitario(
        self, 
        pipeline: Pipeline,
        estrategia_split: Optional[Callable[[Operacion], bool]] = None
    ) -> 'GestorRamas':
        """
        Analiza un pipeline lineal y detecta automáticamente puntos de split.
        Crea ramas implícitas basadas en categorías.
        """
        if not pipeline.puntos_split:
            # No hay splits, una sola rama
            self.definir_rama("principal", "Procesamiento lineal", TipoRama.LINEAL, pipeline.operaciones)
            return self
        
        # Hay splits: crear ramas
        ops = pipeline.operaciones
        idx_split = pipeline.puntos_split[0]  # Primer split (ej: después de SEGMENTADOR)
        
        # Rama base: desde inicio hasta split (inclusive)
        rama_base_ops = ops[:idx_split+1]
        self.definir_rama(
            "base", 
            "Preparación común", 
            TipoRama.LINEAL, 
            tuple(rama_base_ops),
            exporta_para={"procesada", "segmentada"}
        )
        
        # Analizar operaciones post-split para inferir ramas
        ops_post_split = ops[idx_split+1:]
        
        # Rama procesada: operaciones que NO son segmentación adicional
        ops_procesada = tuple(
            op for op in ops_post_split 
            if op.categoria not in {CategoriaOperacion.SEGMENTADOR}
        )
        if ops_procesada:
            self.definir_rama(
                "procesada",
                "Imagen procesada (intensidad)",
                TipoRama.DIVERGENTE,
                ops_procesada,
                requiere={"base"},
                exporta_para={"cuantificacion"}
            )
        
        # Rama segmentada: operaciones de segmentación y post-segmentación
        ops_segmentada = tuple(
            op for op in ops_post_split 
            if op.categoria in {CategoriaOperacion.SEGMENTADOR, CategoriaOperacion.CUANTIFICADOR}
        )
        if ops_segmentada:
            self.definir_rama(
                "segmentada",
                "Máscara segmentada (morfometría)",
                TipoRama.DIVERGENTE,
                ops_segmentada,
                requiere={"base"},
                exporta_para={"cuantificacion"}
            )
        
        return self
    
    def validar_grafo(self) -> Tuple[bool, List[str]]:
        """
        Valida que el grafo de ramas sea válido (sin ciclos, dependencias resolubles).
        """
        errores = []
        
        # Verificar que todas las dependencias existen
        for id_rama, rama in self.ramas.items():
            for dep in rama.requiere_de:
                if dep not in self.ramas:
                    errores.append(f"Rama '{id_rama}' requiere '{dep}' que no existe")
        
        # Detectar ciclos (simplificado)
        visitados = set()
        en_proceso = set()
        
        def tiene_ciclo(id_rama: str) -> bool:
            if id_rama in en_proceso:
                return True
            if id_rama in visitados:
                return False
            
            en_proceso.add(id_rama)
            for dep in self.ramas[id_rama].requiere_de:
                if tiene_ciclo(dep):
                    return True
            en_proceso.remove(id_rama)
            visitados.add(id_rama)
            return False
        
        for id_rama in self.ramas:
            if tiene_ciclo(id_rama):
                errores.append(f"Ciclo detectado involucrando a '{id_rama}'")
        
        return (not errores, errores)
    
    def ejecutar(
        self,
        data_inicial: BioImagenData,
        paralelo: bool = True
    ) -> Resultado[Dict[str, Any], ErrorBioImagen]:
        """
        Ejecuta todas las ramas resolviendo dependencias.
        """
        valido, errores = self.validar_grafo()
        if not valido:
            return Err(ErrorBioImagen(
                etapa="orquestacion",
                mensaje=f"Grafo inválido: {'; '.join(errores)}"
            ))
        
        self._log(f"Iniciando ejecución de {len(self.ramas)} ramas")
        
        # Topología: orden de ejecución (Kahn's algorithm simplificado)
        grados_entrada = {
            id_r: len(r.requiere_de) 
            for id_r, r in self.ramas.items()
        }
        
        resultados: Dict[str, Any] = {}
        ramas_completadas: Set[str] = set()
        
        while len(ramas_completadas) < len(self.ramas):
            # Encontrar ramas listas (grado de entrada 0 y no completadas)
            listas = [
                id_r for id_r, grado in grados_entrada.items() 
                if grado == 0 and id_r not in ramas_completadas
            ]
            
            if not listas:
                return Err(ErrorBioImagen(
                    etapa="orquestacion",
                    mensaje="Deadlock: ramas con dependencias no resolubles"
                ))
            
            # Ejecutar ramas listas (en paralelo si es posible)
            if paralelo and len(listas) > 1:
                self._ejecutar_paralelo(listas, data_inicial, resultados, ramas_completadas, grados_entrada)
            else:
                for id_r in listas:
                    self._ejecutar_rama(id_r, data_inicial, resultados, ramas_completadas, grados_entrada)
        
        return Ok(resultados)
    
    def _ejecutar_rama(
        self,
        id_rama: str,
        data_inicial: BioImagenData,
        resultados: Dict[str, Any],
        completadas: Set[str],
        grados: Dict[str, int]
    ):
        """Ejecuta una rama individual"""
        rama = self.ramas[id_rama]
        self._log(f"Ejecutando rama '{id_rama}': {rama.nombre}")
        
        # Preparar input: mezclar data_inicial con resultados de dependencias
        data_input = self._preparar_input_rama(rama, data_inicial, resultados)
        
        # Ejecutar pipeline de la rama
        resultado = self._ejecutar_pipeline_rama(rama, data_input)
        
        # Almacenar resultado
        resultados[id_rama] = resultado
        if resultado.es_ok():
            rama.resultado = resultado
            completadas.add(id_rama)
            
            # Reducir grado de ramas dependientes
            for id_dependiente in rama.exporta_para:
                if id_dependiente in grados:
                    grados[id_dependiente] -= 1
                    self._log(f"  Rama '{id_dependiente}': dependencia satisfecha ({grados[id_dependiente]} restantes)")
        else:
            self._log(f"  ✗ Rama '{id_rama}' falló: {resultado.error.mensaje}")
    
    def _ejecutar_paralelo(
        self,
        ids_ramas: List[str],
        data_inicial: BioImagenData,
        resultados: Dict[str, Any],
        completadas: Set[str],
        grados: Dict[str, int]
    ):
        """Ejecuta múltiples ramas en paralelo cuando no tienen dependencias entre sí"""
        self._log(f"Ejecutando en paralelo: {ids_ramas}")
        
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Preparar futures
            futures = {}
            for id_r in ids_ramas:
                rama = self.ramas[id_r]
                data_input = self._preparar_input_rama(rama, data_inicial, resultados)
                future = executor.submit(self._ejecutar_pipeline_rama, rama, data_input)
                futures[future] = id_r
            
            # Recolectar resultados
            for future in as_completed(futures):
                id_r = futures[future]
                try:
                    resultado = future.result()
                    resultados[id_r] = resultado
                    
                    if resultado.es_ok():
                        completadas.add(id_r)
                        self.ramas[id_r].resultado = resultado
                        
                        # Actualizar grados de dependientes
                        for id_dep in self.ramas[id_r].exporta_para:
                            if id_dep in grados:
                                grados[id_dep] -= 1
                    else:
                        self._log(f"  ✗ Rama '{id_r}' falló en paralelo")
                        
                except Exception as e:
                    resultados[id_r] = Err(ErrorBioImagen(
                        etapa="ejecucion_paralela",
                        mensaje=f"Excepción en rama '{id_r}': {str(e)}",
                        causa=e
                    ))
    
    def _preparar_input_rama(
        self,
        rama: Rama,
        data_inicial: BioImagenData,
        resultados_previos: Dict[str, Any]
    ) -> BioImagenData:
        """
        Prepara el input para una rama, combinando data inicial con resultados de dependencias.
        Caso especial: CUANTIFICADOR necesita imagen + máscara de ramas diferentes.
        """
        if not rama.requiere_de:
            return data_inicial
        
        # Caso merge: cuantificación necesita múltiples inputs
        if rama.tipo == TipoRama.CONVERGENTE or rama.pipeline.operaciones[0].categoria == CategoriaOperacion.CUANTIFICADOR:
            # Buscar imagen procesada y máscara en resultados previos
            imagen_procesada = None
            mascara_segmentada = None
            
            for dep_id in rama.requiere_de:
                if dep_id not in resultados_previos:
                    continue
                    
                res = resultados_previos[dep_id]
                if not res.es_ok():
                    continue
                
                data_dep = res.unwrap()
                
                # Detectar tipo por operaciones de la rama dependencia
                rama_dep = self.ramas[dep_id]
                ultima_cat = rama_dep.pipeline.operaciones[-1].categoria if rama_dep.pipeline.operaciones else None
                
                if ultima_cat == CategoriaOperacion.SEGMENTADOR:
                    mascara_segmentada = data_dep
                else:
                    imagen_procesada = data_dep
            
            # Combinar en un solo BioImagenData con metadatos especiales
            if imagen_procesada and mascara_segmentada:
                from dataclasses import replace
                # Guardar máscara en metadatos para que cuantificador la use
                return replace(
                    imagen_procesada,
                    metadata={
                        **imagen_procesada.__dict__.get('metadata', {}),
                        'mascara_segmentada': mascara_segmentada.datos,
                        'rama_merge': True
                    }
                )
            
            return imagen_procesada or data_inicial
        
        # Caso simple: tomar resultado de única dependencia
        dep_id = list(rama.requiere_de)[0]
        if dep_id in resultados_previos:
            res = resultados_previos[dep_id]
            if res.es_ok():
                return res.unwrap()
        
        return data_inicial
    
    def _ejecutar_pipeline_rama(
        self,
        rama: Rama,
        data_input: BioImagenData
    ) -> Resultado[Any, ErrorBioImagen]:
        """Ejecuta secuencialmente las operaciones de una rama"""
        resultado: Resultado[BioImagenData, ErrorBioImagen] = Ok(data_input)
        
        for op in rama.pipeline.operaciones:
            if resultado.es_err():
                break
            
            resultado = resultado.bind(op.ejecutar)
        
        return resultado
    
    def obtener_resultado_rama(self, id_rama: str) -> Optional[Resultado[Any, ErrorBioImagen]]:
        return self.ramas.get(id_rama, Rama("", "", TipoRama.LINEAL, Pipeline("", ()))).resultado
    
    def obtener_log_completo(self) -> str:
        return "\n".join(self.log)