# === nucleo/pipeline/PipelineBuilder.py ===
from typing import List, Optional, Dict, Any, Callable
from .CategoriaOperacion import Categoria_Operacion
from .Operacion import Operacion
from .Pipeline import Pipeline

class PipelineBuilder:
    """
    Construye pipelines validando orden semántico de categorías.
    No crea instancias de componentes, solo recibe Operaciones ya adaptadas.
    """
    
    def __init__(self):
        self._operaciones: List[Operacion] = []
        self._ultima_categoria: Optional[CategoriaOperacion] = None
        self._errores: List[str] = []
    
    def _validar_orden(self, nueva_op: Operacion) -> bool:
        """Valida que la operación respete el orden canónico"""
        if self._ultima_categoria is None:
            return True
        
        if nueva_op.categoria == self._ultima_categoria:
            return True  # Misma categoría siempre permitida
        
        if not self._ultima_categoria.puede_preceder_a(nueva_op.categoria):
            self._errores.append(
                f"Orden inválido: {self._ultima_categoria.name} → {nueva_op.categoria.name} "
                f"en operación '{nueva_op.nombre}'. "
                f"El orden correcto es: PREPROCESAMIENTO → FILTRACION → REALZADOR → ..."
            )
            return False
        
        return True
    
    def agregar(self, operacion: Operacion) -> 'PipelineBuilder':
        """Agrega operación validando orden"""
        if not self._validar_orden(operacion):
            raise ValueError(self._errores[-1])
        
        self._operaciones.append(operacion)
        self._ultima_categoria = operacion.categoria
        return self
    
    def agregar_desde_wrapper(
        self,
        nombre: str,
        categoria: CategoriaOperacion,
        instancia_callable: Callable[[np.ndarray], np.ndarray],
        canal: Optional[int] = None,
        parametros: Optional[Dict[str, Any]] = None,
        descripcion: str = ""
    ) -> 'PipelineBuilder':
        """
        Método conveniente para agregar desde wrappers como Normalizador o Controlador_Filtrador.
        
        El wrapper ya instanció el componente (ej: FFTPasabajo(radio=30)),
        aquí solo lo envolvemos en Operacion.
        """
        operacion = Operacion(
            nombre=nombre,
            categoria=categoria,
            instancia_callable=instancia_callable,
            canal_objetivo=canal,
            parametros_originales=parametros or {},
            descripcion=descripcion
        )
        return self.agregar(operacion)
    
    def construir(self) -> Pipeline:
        if not self._operaciones:
            raise ValueError("Pipeline sin operaciones")
        return Pipeline(tuple(self._operaciones))
    
    def obtener_errores(self) -> List[str]:
        return self._errores.copy()
    
    def __len__(self) -> int:
        return len(self._operaciones)