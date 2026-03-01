# === nucleo/pipeline/Pipeline.py ===
from dataclasses import dataclass
from typing import Tuple, List, Callable, Optional
from nucleo.controlador.Resultado_Either import Resultado, Ok, Err, ErrorBioImagen
from nucleo.controlador.BioImagenData import BioImagenData
from .Operacion import Operacion

@dataclass(frozen=True)
class Pipeline:
    """
    Secuencia inmutable de operaciones adaptadas.
    Cada operación contiene su instancia callable lista para ejecutar.
    """
    operaciones: Tuple[Operacion, ...]
    
    def ejecutar(
        self, 
        data: BioImagenData,
        callback_progreso: Optional[Callable[[int, Operacion, BioImagenData], None]] = None
    ) -> Resultado[BioImagenData, ErrorBioImagen]:
        """
        Ejecuta pipeline secuencial. Fail-fast en primer error.
        """
        resultado: Resultado[BioImagenData, ErrorBioImagen] = Ok(data)
        
        for i, op in enumerate(self.operaciones):
            if resultado.es_err():
                break
            
            # Ejecutar operación (ya contiene su instancia callable)
            resultado = resultado.bind(op.ejecutar)
            
            if callback_progreso and resultado.es_ok():
                callback_progreso(i, op, resultado.unwrap())
        
        return resultado
    
    def ejecutar_con_snapshots(
        self,
        data: BioImagenData
    ) -> Tuple[Resultado[BioImagenData, ErrorBioImagen], List[Tuple[int, str, BioImagenData]]]:
        """
        Ejecuta guardando copia después de cada operación (para debug).
        """
        snapshots: List[Tuple[int, str, BioImagenData]] = []
        resultado: Resultado[BioImagenData, ErrorBioImagen] = Ok(data)
        
        for i, op in enumerate(self.operaciones):
            if resultado.es_err():
                break
            
            # Snapshot antes
            if resultado.es_ok():
                snapshots.append((i, f"antes_{op.nombre}", resultado.unwrap()))
            
            # Ejecutar
            resultado = resultado.bind(op.ejecutar)
            
            # Snapshot después
            if resultado.es_ok():
                snapshots.append((i, f"despues_{op.nombre}", resultado.unwrap()))
        
        return resultado, snapshots
    
    def __or__(self, otro: 'Pipeline') -> 'Pipeline':
        """Concatenación de pipelines"""
        return Pipeline(self.operaciones + otro.operaciones)
    
    def __len__(self) -> int:
        return len(self.operaciones)
    
    def __repr__(self) -> str:
        ops = " → ".join(str(op) for op in self.operaciones)
        return f"Pipeline[{len(self)}: {ops}]"