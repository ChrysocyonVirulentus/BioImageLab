from __future__ import annotations
import numpy as np
import cv2
from dataclasses import dataclass, replace
from pathlib import Path
from typing import List, Optional, Tuple, Callable, Any, Iterator
from bioio import BioImage
import bioio_bioformats
from enum import Enum

from Resultado_Either import Resultado, Ok, Err # Para manejar Funcionalmente los Errores.

# Tipos inmutables útiles

@dataclass(frozen=True)
class Dimensiones:
    T: int  # Tiempo
    Z: int  # Z-stack
    C: int  # Canales
    Y: int  # Altura
    X: int  # Ancho
    
    @property
    def shape(self) -> Tuple[int, int, int, int, int]:
        return (self.T, self.Z, self.C, self.Y, self.X)
    
    def total_cortes(self) -> int:
        return self.T * self.Z * self.C

@dataclass(frozen=True)
class BioImagenData:
    """
        Estructura unificada para cualquier tipo de imagen (estándar o bioimagen)
    """
    datos: np.ndarray  # Shape [T, Z, C, Y, X]
    dims: Dimensiones
    canales: Tuple[str, ...]
    ruta_origen: Path
    es_bioformato: bool = False  # Metadata para saber el origen
    
    def __post_init__(self):
        # Validación inmutable: verificar consistencia
        assert self.datos.shape == self.dims.shape, \
            f"Shape {self.datos.shape} no coincide con dims {self.dims.shape}"
        assert len(self.canales) == self.dims.C, \
            f"Canales {len(self.canales)} no coincide con C={self.dims.C}"

@dataclass(frozen=True)
class ErrorBioImagen:
    etapa: str  # "lectura", "procesamiento", "indexacion"
    mensaje: str
    ruta: Optional[Path] = None
    causa: Optional[Exception] = None
    
    def con_contexto(self, nueva_etapa: str) -> ErrorBioImagen:
        """
            Añade contexto al pipeline de error
        """
        return replace(self, etapa=f"{nueva_etapa} -> {self.etapa}")

@dataclass(frozen=True)
class ModoImagen(Enum):
    AUTO = "auto"
    RGB = "rgb"
    GRIS = "gris"

# Funciones puras

def clasificar_extension(ruta: Path) -> bool:
    """True si es bioformato, False si es estándar"""
    formatos_bio = {".ids", ".ics", ".tiff", ".tif"}
    return ruta.suffix.lower() in formatos_bio


def leer_bioformato(ruta: Path) -> Resultado[BioImage, ErrorBioImagen]:
    """
        Función pura para leer bioformatos
    """
    try:
        img = BioImage(ruta, reader=bioio_bioformats.Reader)
        return Ok(img)
    except FileNotFoundError as e:
        return Err(ErrorBioImagen(
            etapa="lectura",
            mensaje=f"Archivo bioformato no encontrado: {ruta}",
            ruta=ruta,
            causa=e
        ))
    except Exception as e:
        return Err(ErrorBioImagen(
            etapa="lectura",
            mensaje=f"Error BioFormats: {str(e)}",
            ruta=ruta,
            causa=e
        ))


def leer_estandar(ruta: Path) -> Resultado[np.ndarray, ErrorBioImagen]:
    """
        Función pura para leer imágenes estándar con OpenCV
    """
    try:
        img_raw = cv2.imread(str(ruta), cv2.IMREAD_UNCHANGED)
        if img_raw is None:
            return Err(ErrorBioImagen(
                etapa="lectura",
                mensaje=f"OpenCV no pudo leer: {ruta}",
                ruta=ruta
            ))
        # Pasaje a 16-bit
        if img_raw.dtype == np.uint8:
            img_raw = img_raw.astype(np.uint16) * 257 # (65535 // 255)
        return Ok(img_raw)
    except Exception as e:
        return Err(ErrorBioImagen(
            etapa="lectura",
            mensaje=f"Error OpenCV: {str(e)}",
            ruta=ruta,
            causa=e
        ))

def procesar_bioformato(
    bioimg: BioImage, 
    modo: ModoImagen
) -> Resultado[BioImagenData, ErrorBioImagen]:
    """
        Procesa BioImage a estructura unificada.

        Argumentos:
            bioimg: BioImage
            ruta: string o path

        Retorna:
            Callable: Retorna un Resultado con el array 2D.
        
        Complejidad:
            O(1)
    """
    try:
        img_data = bioimg.get_image_data("TZCYX")
        channel_names = tuple(bioimg.channel_names)
        
        match modo:
            case ModoImagen.AUTO:
                datos = img_data
                canales = channel_names
                
            case ModoImagen.RGB:
                if img_data.shape[2] != 3:
                    return Err(ErrorBioImagen(
                        etapa="procesamiento",
                        mensaje=f"Modo RGB requiere 3 canales, tiene {img_data.shape[2]}"
                    ))
                datos = img_data
                canales = ("Rojo", "Verde", "Azul")
                
            case ModoImagen.GRIS:
                datos = np.mean(img_data, axis=2, keepdims=True)
                canales = ("Gris",)
        
        T, Z, C, Y, X = datos.shape
        return Ok(BioImagenData(
            datos=datos,
            dims=Dimensiones(T, Z, C, Y, X),
            canales=canales,
            ruta_origen=Path(bioimg.path),
            es_bioformato=True
        ))
        
    except Exception as e:
        return Err(ErrorBioImagen(
            etapa="procesamiento",
            mensaje=f"Error procesando bioformato: {str(e)}",
            causa=e
        ))

def procesar_estandar(
    img_raw: np.ndarray, 
    modo: ModoImagen,
    ruta: Path
) -> Resultado[BioImagenData, ErrorBioImagen]:
    """
        Procesa imagen estándar a estructura 5D unificada.

        Argumentos:
            Array: np.ndarray
            modo: ModoImagen
            ruta: string o path

        Retorna:
            Callable: Retorna un Resultado con el array 2D.
        
        Complejidad:
            O(1)
    """
    try:
        match modo:
            case ModoImagen.AUTO:
                if img_raw.ndim == 3 and img_raw.shape[2] == 3:
                    # RGB: separar canales y crear dims 5D
                    canales_rgb = [img_raw[:, :, i] for i in range(3)]
                    datos = np.stack(canales_rgb, axis=0)[np.newaxis, np.newaxis, :, :, :]
                    canales = ("Rojo", "Verde", "Azul")
                else:
                    # Grayscale: añadir dims
                    datos = img_raw[np.newaxis, np.newaxis, np.newaxis, :, :]
                    canales = ("Gris",)
                    
            case ModoImagen.RGB:
                if not (img_raw.ndim == 3 and img_raw.shape[2] == 3):
                    return Err(ErrorBioImagen(
                        etapa="procesamiento",
                        mensaje="Modo RGB requiere imagen color (3 canales)"
                    ))
                canales_rgb = [img_raw[:, :, i] for i in range(3)]
                datos = np.stack(canales_rgb, axis=0)[np.newaxis, np.newaxis, :, :, :]
                canales = ("Rojo", "Verde", "Azul")
                
            case ModoImagen.GRIS:
                if img_raw.ndim == 3 and img_raw.shape[2] == 3:
                    img_raw = cv2.cvtColor(img_raw, cv2.COLOR_BGR2GRAY)
                datos = img_raw[np.newaxis, np.newaxis, np.newaxis, :, :]
                canales = ("Gris",)
        
        T, Z, C, Y, X = datos.shape
        return Ok(BioImagenData(
            datos=datos,
            dims=Dimensiones(T, Z, C, Y, X),
            canales=canales,
            ruta_origen=ruta,
            es_bioformato=False
        ))
        
    except Exception as e:
        return Err(ErrorBioImagen(
            etapa="procesamiento",
            mensaje=f"Error procesando imagen estándar: {str(e)}",
            causa=e
        ))


def cargar_imagen(ruta: Path, modo: ModoImagen = ModoImagen.AUTO) -> Resultado[BioImagenData, ErrorBioImagen]:
    """
        Función completo de carga: detecta tipo y procesa
        
        Argumentos:
            ruta: String o Path
            modo: ModoImagen

        Retorna:
            Callable: Retorna un Resultado con el array 2D.
        
        Complejidad:
            O(1)
    """
    if clasificar_extension(ruta):
        # Mini-Pipeline bioformato: leer -> procesar
        return (
            leer_bioformato(ruta)
            .bind(lambda bioimg: procesar_bioformato(bioimg, modo))
        )
    else:
        # Mini-Pipeline estándar: leer -> procesar
        return (
            leer_estandar(ruta)
            .bind(lambda img_raw: procesar_estandar(img_raw, modo, ruta))
        )

# Funciones de transformación y de soporte

def extraer_corte(
    t: int = 0, 
    z: int = 0, 
    c: int = 0
) -> Callable[[BioImagenData], Resultado[np.ndarray, ErrorBioImagen]]:
    """
        Genera una función para extraer un corte 2D específico de un BioImagenData.

        Argumentos:
            t (int): Índice temporal.
            z (int): Índice de profundidad (Z-stack).
            c (int): Índice del canal cromático.

        Retorna:
            Callable: Una función que recibe BioImagenData y retorna un Resultado con el array 2D.

        Complejidad:
            O(1) - Acceso directo por índice y copia de la vista 2D.
    """
    def _extraer(data: BioImagenData) -> Resultado[np.ndarray, ErrorBioImagen]:
        dims = data.dims
        if not (0 <= t < dims.T and 0 <= z < dims.Z and 0 <= c < dims.C):
            return Err(ErrorBioImagen(
                etapa="indexacion",
                mensaje=f"Índices ({t},{z},{c}) fuera de rango {dims.shape[:3]}"
            ))
        return Ok(data.datos[t, z, c].copy())
    return _extraer


def aplicar_a_corte(
    t: int, z: int, c: int,
    operacion: Callable[[np.ndarray], np.ndarray]
) -> Callable[[BioImagenData], Resultado[BioImagenData, ErrorBioImagen]]:
    """
        Aplica una operación de procesamiento a un corte y retorna una nueva estructura.

        Argumentos:
            t (int): Índice temporal.
            z (int): Índice de profundidad.
            c (int): Índice del canal.
            operacion (Callable): Función pura np.ndarray -> np.ndarray.

        Retorna:
            Callable: Función que retorna un nuevo BioImagenData con el corte modificado.

        Complejidad:
            O(N*M) donde N,M son dimensiones Y,X (debido a la copia del array completo).
    """
    def _aplicar(data: BioImagenData) -> Resultado[BioImagenData, ErrorBioImagen]:
        # Extraer
        resultado_corte = extraer_corte(t, z, c)(data)
        if resultado_corte.is_err():
            return resultado_corte.map(lambda _: data)  # Propaga error
        
        corte = resultado_corte.unwrap()
        
        # Aplicar operación
        try:
            corte_procesado = operacion(corte)
            if corte_procesado.shape != corte.shape:
                return Err(ErrorBioImagen(
                    etapa="procesamiento",
                    mensaje=f"La operación alteró las dimensiones Y,X: {corte.shape} -> {corte_procesado.shape}"
                ))
        except Exception as e:
            return Err(ErrorBioImagen(
                etapa="procesamiento",
                mensaje=f"Error en operación: {str(e)}",
                causa=e
            ))
        
        # Crear nuevo array con el corte modificado (inmutable)
        nuevos_datos = data.datos.copy()
        nuevos_datos[t, z, c] = corte_procesado
        
        return Ok(replace(data, datos=nuevos_datos))
    
    return _aplicar


def iterar_cortes(
    canal: Optional[int] = None
) -> Callable[[BioImagenData], Resultado[List[Tuple[int, int, int, np.ndarray]], ErrorBioImagen]]:
    """
        Crea una lista de todos los cortes disponibles para iteración.

        Arguments:
            canal (Optional[int]): Si se especifica, filtra solo por ese canal.

        Retorna:
            Resultado: Una lista de tuplas (C, T, Z, array_2d).

        Complejidad:
            O(T * Z * C) en espacio y tiempo.
    """
    def _iterar(data: BioImagenData) -> Resultado[List[Tuple[int, int, int, np.ndarray]], ErrorBioImagen]:
        try:
            dims = data.dims
            cortes = []
            
            rango_canales = [canal] if canal is not None else range(dims.C)
            
            if canal is not None and not (0 <= canal < dims.C):
                return Err(ErrorBioImagen(
                    etapa="indexacion",
                    mensaje=f"Canal {canal} fuera de rango [0, {dims.C-1}]"
                ))
            
            for c in rango_canales:
                for t in range(dims.T):
                    for z in range(dims.Z):
                        cortes.append((c, t, z, data.datos[t, z, c].copy()))
            
            return Ok(cortes)
        except Exception as e:
            return Err(ErrorBioImagen(
                etapa="iteracion",
                mensaje=f"Error iterando: {str(e)}",
                causa=e
            ))
    
    return _iterar

# Clase Controladora 
class ControladorBioImagen:
    """
    Clase "Handler" para leer y preprocesar imágenes de microscopía en formato .png, .jpg, .tiff y formatos de bioimagen confocal como .ics/.ids.
    Wrapper stateful que mantiene API compatible pero usa funciones puras internamente.
    Permite :
        - Leer y abrir este tipo de archivos.
        - Preprocesarlos a escala de grises y transformarlos en un MultiArray para "handlear" los diferentes tipos de canales, "z-stacking" y "time-lapse" según el tipo de imagen.
        - Permite transformar las imágenes MultiArray 5D en una estructura indexeable para los 
        tratamientos de operaciones atómicas (formato de solo Y, X) y luego reestructurarlo en 5D.
        - Permite manipular la imagen modificada por operaciones externas para comparar con la original.
    Nota :
        - El MultiArray es [T, Z, C, Y, X] donde T es el "timelapse", Z el "Z-stacking" (diferentes planos en el eje Z), C es el Canal de fluorescencia ("Azul", "Rojo", "Verde" y "Campo" que puede
        ser claro u oscuro), y los ejes de pixeles X e Y son las dimensiones de la imagen.
        - Una imagen bidimensional sería (1, 1, 1, Y, X), por ejemplo, de formatos .jpg y .png.
        - Se usa .copy() en getters y el iterador dado que sino permitiria la sobreescritura en algo que deberia
        ser solo lectura.
    """

    def __init__(self, ruta_imagen: str | Path):
        self.ruta_imagen = Path(ruta_imagen)
        self._data: Optional[BioImagenData] = None
        self._procesada: Optional[BioImagenData] = None # La imagen procesada
        self._ultimo_error: Optional[ErrorBioImagen] = None

        # Errores
        self._ultimo_error: Optional[ErrorBioImagen] = None

    def __bool__(self) -> bool:
        return self._data is not None

    @property
    def forma(self) -> Tuple[int, ...]:
        """
            Retorna la forma 5D (T, Z, C, Y, X).
        """
        return self._data.dims.shape if self._data else (0, 0, 0, 0, 0)
    
    @property
    def canales(self) -> Tuple[str, ...]:
        """
            Retorna los nombres de los canales cromáticos.
        """
        return self._data.channel_names if self._data else ()


    def cargar_ImagenResultado(self, modo: ModoImagen = ModoImagen.AUTO) -> Resultado[BioImagenData, ErrorBioImagen]:
        """
            Versión Resultado de cargar.

            Retorna:
                Resultado[BioImagenData, ErrorBioImagen]: Éxito o detalle del error.
        """
        resultado = cargar_imagen(self.ruta_imagen, modo)
        
        if resultado.es_ok():
            self._data = resultado.unwrap()
            self._procesada = None  # Reset
            self._ultimo_error = None
        else:
            self._ultimo_error = resultado.error
        
        return resultado

    def cargar(self, modo: ModoImagen = ModoImagen.AUTO) -> bool:
        """
            Carga la imagen. Retorna True si éxito, False si error.
            Para manejo completo de errores, usar cargar_ImagenResultado()
        """
        return self.cargar_ImagenResultado(modo).es_ok()

    def iterar(self, canal: Optional[int] = None):
        """
            Iterador para poder buscar o iterar para todos los canales por tiempo y por z-stack, 
            obteniendo imágenes bidimensionales para su procesamiento.

            Yields:
                canal, t, z, Tupla (t, z, img_2d) donde img_2d es np.ndarray de forma (Y, X)

            Complejidad:
                O(T*Z*C) iteraciones
        """
        if self._data is None:
            return iter([])
        
        return iter(iterar_cortes(canal)(self._data).unwrap_or([]))

    def __iter__(self):
        return self.iterar()

    def get_corte(self, 
                t: int = 0, 
                z: int = 0, 
                c: int = 0, 
                procesado: bool = False
                ) -> Optional[np.ndarray]:
        """
            Obtiene un corte 2D. Prioriza la imagen procesada si se solicita.
        
            Argumentos:
                t, z, c (int): Índices 5D.
                procesado (bool): Si es True, intenta obtenerlo de la imagen modificada.

            Costo:
                O(1)
        """
        fuente = self._procesada if (procesado and self._procesada) else self._data
        if fuente is None: return None
        return extraer_corte(t, z, c)(fuente).unwrap_or(None)

    def set_corte(self,
                c: int = 0,
                t: int = 0,
                z: int = 0,
                img_2d: Optional[np.ndarray] = None):
        """
            Método setter para guardar una imagen 2D procesada en la estructura tensor 5D.
            
            Complejidad: O(N) donde N es el tamaño del tensor completo por la copia inmutable.
        """
        if self._data is None or img_2d is None: return False
        base = self._procesada if self._procesada else self._data
        
        # Operación constante: ignora el valor anterior y pone el nuevo
        resultado = aplicar_a_corte(t, z, c, lambda _: img_2d)(base)
        
        if resultado.es_ok():
            self._procesada = resultado.unwrap()
            return True
        return False

    def __eq__(self, other) -> bool:
        """
            Metodo para comparar dos controladores por tura y por forma. 
            Retorna : Bool
            Complejida : O(1)
        """

        if not isinstance(other, ControladorBioImagen):
            return False

        return (
            self.ruta_imagen == other.ruta_imagen and
            self.forma == other.forma
        )


    def __len__(self) -> int:
        """
            Metodo para determinar la cantidad de fotos totales existentes.
            Retorna : Int
            Complejida : O(1)
        """

        if self._data is None: return 0
        return self._data.dims.T * self._data.dims.Z

    # Metodos para I/O : Abrir imagenes, liberar memoria y cachear los bioformatos.
    def __enter__(self):
        self.cargar()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb): 
        # O(1)
        self._data = None
        self._procesada = None

    def __repr__(self) -> str:
        """
            Metodo para debugging.
            Retorna : F-String
            Complejidad : O(1)
        """
        if self._data is None:
            return f"<ControladorBioImagen ruta='{self.ruta_imagen.name}' (no cargada)>"
        return f"<ControladorBioImagen ruta='{self.ruta_imagen.name}' forma={self.forma} canales={self.canales}>"