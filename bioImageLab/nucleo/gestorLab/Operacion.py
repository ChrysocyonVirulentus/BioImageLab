# === nucleo/pipeline/Operacion.py ===
from dataclasses import dataclass, field
from typing import Callable, Dict, Any, Optional
from nucleo.controlador.Resultado_Either import Resultado, ErrorBioImagen, BioImagenData
from .Categoria_Operacion import CategoriaOperacion

@dataclass(frozen=True)
class Operacion:
    """
    Adaptador que envuelve un componente individual (ej: FFTPasabajo)
    para integrarlo en el pipeline funcional.
    
    El componente original trabaja con np.ndarray, esta clase
    adapta a BioImagenData -> Resultado[BioImagenData, Error].
    """
    nombre: str                      # Nombre del componente (ej: "fft_pasabajo")
    categoria: CategoriaOperacion    # Categoría semántica
    instancia_callable: Callable[[np.ndarray], np.ndarray]  # El objeto callable (ej: FFTPasabajo(radio=30))
    canal_objetivo: Optional[int] = None  # Si aplica a canal específico, None = todos
    
    # Metadatos para debugging y reproducibilidad
    parametros_originales: Dict[str, Any] = field(default_factory=dict)
    descripcion: str = ""
    
    def ejecutar(self, data: BioImagenData) -> Resultado[BioImagenData, ErrorBioImagen]:
        """
        Ejecuta el componente adaptado sobre BioImagenData.
        Maneja la lógica de canales y preservación de metadatos.
        """
        from dataclasses import replace
        
        try:
            T, Z, C, Y, X = data.dims.shape
            
            # Determinar rango de canales a procesar
            canales_procesar = [self.canal_objetivo] if self.canal_objetivo is not None else range(C)
            
            # Validar canal específico si se proporcionó
            if self.canal_objetivo is not None and not (0 <= self.canal_objetivo < C):
                return Err(ErrorBioImagen(
                    etapa="operacion",
                    mensaje=f"Operación {self.nombre}: canal {self.canal_objetivo} fuera de rango [0, {C-1}]",
                    ruta=data.ruta_origen
                ))
            
            # Crear copia para inmutabilidad
            nuevos_datos = data.datos.copy().astype(np.float64)
            
            # Aplicar operación canal por canal
            for c in canales_procesar:
                for t in range(T):
                    for z in range(Z):
                        corte_2d = data.datos[t, z, c, :, :]
                        
                        # Ejecutar el callable del componente (ej: FFTPasabajo.__call__)
                        resultado_corte = self.instancia_callable(corte_2d)
                        
                        # Validar que no cambió shape (salvo que sea intencional)
                        if resultado_corte.shape != (Y, X):
                            return Err(ErrorBioImagen(
                                etapa="operacion",
                                mensaje=f"{self.nombre} cambió shape de corte: {(Y,X)} -> {resultado_corte.shape}",
                                ruta=data.ruta_origen
                            ))
                        
                        nuevos_datos[t, z, c, :, :] = resultado_corte
            
            # Reconstruir BioImagenData con nuevos datos
            return Ok(replace(data, datos=nuevos_datos))
            
        except Exception as e:
            return Err(ErrorBioImagen(
                etapa="operacion",
                mensaje=f"Error en {self.nombre}: {str(e)}",
                ruta=data.ruta_origen,
                causa=e
            ))
    
    def __repr__(self) -> str:
        canal_str = f"[C={self.canal_objetivo}]" if self.canal_objetivo is not None else "[C=all]"
        return f"{self.categoria.name}::{self.nombre}{canal_str}"