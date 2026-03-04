# === nucleo/pipeline/PipelineBuilder.py ===
from typing import List, Optional, Dict, Any, Callable, Set
from .Categoria_Operacion import CategoriaOperacion
from .Operacion import Operacion
from .Pipeline import Pipeline

class ValidacionPipeline:
    """Resultado de validación con detalles"""
    def __init__(self):
        self.errores: List[str] = []
        self.advertencias: List[str] = []
        self.categorias_presentes: Set[CategoriaOperacion] = set()
        self.es_valido: bool = False
    
    def agregar_error(self, msg: str):
        self.errores.append(msg)
        self.es_valido = False
    
    def agregar_advertencia(self, msg: str):
        self.advertencias.append(msg)
    
    def resumen(self) -> str:
        lines = []
        if self.errores:
            lines.append("❌ ERRORES (pipeline inválido):")
            lines.extend(f"  - {e}" for e in self.errores)
        if self.advertencias:
            lines.append("⚠️  ADVERTENCIAS:")
            lines.extend(f"  - {a}" for a in self.advertencias)
        if not self.errores and not self.advertencias:
            lines.append("✅ Pipeline válido y óptimo")
        return "\n".join(lines)


class PipelineBuilder:
    """
    Builder con validación estricta de dependencias semánticas.
    No solo valida orden, sino que todas las dependencias requeridas estén presentes.
    """
    
    def __init__(self, modo_estricto: bool = True):
        self._operaciones: List[Operacion] = []
        self._ultima_categoria: Optional[CategoriaOperacion] = None
        self._categorias_presentes: Set[CategoriaOperacion] = set()
        self._validacion = ValidacionPipeline()
        self._modo_estricto = modo_estricto  # Si False, permite pipelines incompletos (para desarrollo)
    
    def _validar_orden(self, nueva_op: Operacion) -> bool:
        """Valida orden canónico (necesario pero no suficiente)"""
        if self._ultima_categoria is None:
            return True
        
        if nueva_op.categoria == self._ultima_categoria:
            return True
        
        if not self._ultima_categoria.puede_preceder_a(nueva_op.categoria):
            self._validacion.agregar_error(
                f"Orden incorrecto: {self._ultima_categoria.name} → {nueva_op.categoria.name}. "
                f"No se puede ir 'hacia atrás' en el pipeline."
            )
            return False
        
        return True
    
    def _validar_dependencias_nueva_op(self, nueva_op: Operacion) -> bool:
        """
        Valida que la nueva operación tenga todas sus dependencias satisfechas
        POR LAS OPERACIONES ANTERIORES (no futuras).
        """
        es_valido, mensajes = nueva_op.categoria.validar_dependencias(self._categorias_presentes)
        
        for msg in mensajes:
            if "REQUIERE" in msg:
                self._validacion.agregar_error(msg)
            else:
                self._validacion.agregar_advertencia(msg)
        
        return es_valido or not self._modo_estricto
    
    def agregar(self, operacion: Operacion) -> 'PipelineBuilder':
        """
        Agrega operación validando:
        1. Orden canónico
        2. Dependencias estrictas de la operación
        """
        # Validar orden
        if not self._validar_orden(operacion):
            raise ValueError(f"Orden inválido: {self._validacion.errores[-1]}")
        
        # Validar dependencias de la nueva operación
        if not self._validar_dependencias_nueva_op(operacion):
            raise ValueError(f"Dependencias insatisfechas: {self._validacion.errores[-1]}")
        
        # Agregar
        self._operaciones.append(operacion)
        self._categorias_presentes.add(operacion.categoria)
        self._ultima_categoria = operacion.categoria
        
        return self
    
    def agregar_con_auto_completado(self, operacion: Operacion) -> 'PipelineBuilder':
        """
        Versión que inserta automáticamente dependencias faltantes con valores por defecto.
        Útil para pipelines declarativos donde el usuario no especifica todo.
        """
        categoria = operacion.categoria
        faltantes = categoria.dependencias_estrictas - self._categorias_presentes
        
        # Insertar dependencias faltantes con operaciones por defecto
        for dep in sorted(faltantes, key=lambda c: c.orden_ejecucion()):
            op_default = self._crear_operacion_default(dep)
            self._validacion.agregar_advertencia(
                f"Auto-insertado: {dep.name} (requerido por {categoria.name})"
            )
            self.agregar(op_default)  # Recursivo, valida sus propias deps
        
        # Ahora sí agregar la operación original
        return self.agregar(operacion)
    
    def _crear_operacion_default(self, categoria: CategoriaOperacion) -> Operacion:
        """Crea operación mínima por defecto para una categoría"""
        defaults = {
            CategoriaOperacion.PREPROCESAMIENTO: ("normalizacion_basica", lambda x: x / x.max()),
            CategoriaOperacion.FILTRACION: ("filtro_identidad", lambda x: x),
            CategoriaOperacion.REALZADOR: ("realzador_identidad", lambda x: x),
        }
        
        nombre, fn = defaults.get(categoria, (f"default_{categoria.name.lower()}", lambda x: x))
        
        return Operacion(
            nombre=nombre,
            categoria=categoria,
            instancia_callable=fn,
            es_default=True  # Flag para saber que fue auto-insertada
        )
    
    def validar_pipeline_completo(self) -> ValidacionPipeline:
        """
        Validación final: verifica que el pipeline completo tenga sentido científico.
        Por ejemplo, si termina en CUANTIFICADOR pero nunca hubo SEGMENTADOR.
        """
        validacion = ValidacionPipeline()
        validacion.categorias_presentes = self._categorias_presentes.copy()
        
        # Verificar que si hay CUANTIFICADOR, hubo SEGMENTADOR
        if CategoriaOperacion.CUANTIFICADOR in self._categorias_presentes:
            if CategoriaOperacion.SEGMENTADOR not in self._categorias_presentes:
                validacion.agregar_error(
                    "Pipeline con CUANTIFICADOR pero sin SEGMENTADOR. "
                    "No hay objetos segmentados para cuantificar."
                )
        
        # Verificar que si hay MODELADOR, hubo CUANTIFICADOR
        if CategoriaOperacion.MODELADOR in self._categorias_presentes:
            if CategoriaOperacion.CUANTIFICADOR not in self._categorias_presentes:
                validacion.agregar_error(
                    "Pipeline con MODELADOR pero sin CUANTIFICADOR. "
                    "No hay features extraídos para modelar."
                )
        
        # Verificar gaps peligrosos
        ordenes = sorted([c.orden_ejecucion() for c in self._categorias_presentes])
        for i in range(len(ordenes) - 1):
            if ordenes[i+1] - ordenes[i] > 1:
                cat_antes = list(CategoriaOperacion)[ordenes[i] - 1]
                cat_despues = list(CategoriaOperacion)[ordenes[i+1] - 1]
                validacion.agregar_advertencia(
                    f"Salto de {cat_antes.name} → {cat_despues.name}. "
                    f"Etapas intermedias omitidas pueden afectar calidad."
                )
        
        validacion.es_valido = not validacion.errores
        return validacion
    
    def construir(self, forzar: bool = False) -> Pipeline:
        """
        Construye pipeline si es válido.
        Si forzar=True, ignora advertencias (pero no errores).
        """
        if not self._operaciones:
            raise ValueError("Pipeline vacío")
        
        validacion_final = self.validar_pipeline_completo()
        
        if validacion_final.errores:
            raise ValueError(f"Pipeline inválido:\n{validacion_final.resumen()}")
        
        if validacion_final.advertencias and not forzar:
            raise ValueError(
                f"Pipeline con advertencias (usar forzar=True para ignorar):\n"
                f"{validacion_final.resumen()}"
            )
        
        # CORRECCIÓN: Crear Pipeline solo con nombre y operaciones
        # No pasar validacion_final como argumento
        return Pipeline(
            nombre="pipeline_construido",
            operaciones=tuple(self._operaciones)
        )
    
    def construir_desde_operaciones(self, operaciones: List[Operacion]) -> Pipeline:
        """Reconstruye desde lista (usado por FlujoProcesamiento)"""
        builder = PipelineBuilder(modo_estricto=self._modo_estricto)
        for op in operaciones:
            builder.agregar(op)
        return builder.construir()
    
    def obtener_resumen(self) -> str:
        """Resumen del estado actual del builder"""
        ops = " → ".join(f"{op.categoria.name}::{op.nombre}" for op in self._operaciones)
        return f"Builder[{len(self._operaciones)} ops: {ops}]"
    
    def __len__(self) -> int:
        return len(self._operaciones)