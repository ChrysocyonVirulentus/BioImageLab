import numpy as np
import cv2
from dataclasses import dataclass
from pathlib import Path
from typing import Union, List, Optional, Tuple
from bioio import BioImage
import bioio_bioformats
import matplotlib.pyplot as plt
from enum import Enum

# Tipos inmutables para manejar archivos

@dataclass(frozen=True)
class ImagenEstandar:
    ruta: Path

@dataclass(frozen=True)
class BioImagen:
    ruta: Path

TipoOrigen = Union[ImagenEstandar, BioImagen]

class ControladorBioImagen:
    """
    Clase "Handler" para leer y preprocesar imágenes de microscopía en formato .png, .jpg, .tiff y formatos de bioimagen confocal como .ics/.ids.
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

    def __init__(self, ruta_imagen):
        self.ruta_imagen = Path(ruta_imagen)
        self.configuracion: TipoOrigen = self._clasificar_imagen(self.ruta_imagen)

        # Datos del MultiArray [T, Z, C, Y, X] pre-procesamiento
        self.img: Optional[np.ndarray] = None
        self.canales: List[str] = []
        self.forma: Tuple[int, ...] = ()

        # Versión del MultiArray post-procesamiento
        self.img_procesada: Optional[np.ndarray] = None

    def __bool__(self) -> bool:
        # Metodo para indicar si o no está cargada la imagen.
        return self.img is not None
    
    class ModoImagen(Enum):
        AUTO = "auto"       # Detección automática (gris o RGB para estándar; nativo para bio)
        RGB = "rgb"         # Fuerza separación/preservación en canales RGB (3 canales)
        GRIS = "gris"       # Fuerza conversión a escala de grises (1 canal)


    def _clasificar_imagen(self, ruta: Path) -> TipoOrigen:
        """
            Determina el tipo de origen basado en la extensión (Fábrica).
        """
        formatos_bio = {".ids", ".ics", ".tiff", ".tif"}
        if ruta.suffix.lower() in formatos_bio:
            return BioImagen(ruta)
        return ImagenEstandar(ruta)

    def leer_bioImagen(self, modo: ModoImagen = ModoImagen.AUTO) -> Optional[np.ndarray]:
        """
            Selector de modos para cargar y normalizar a 5D.
            Delega el procesamiento del modo a funciones auxiliares para reducir anidamiento que son
            _procesar_estandar y _procesar_bioimagen.
            Costo : O(1)
            Retorna : np.ndarray
        """
        try:
            match self.configuracion:
                case BioImagen(ruta):
                    img = BioImage(ruta, reader=bioio_bioformats.Reader)
                    img_data = img.get_image_data("TZCYX")
                    self.img, self.canales = self._procesar_bioimagen(img_data, img.channel_names, modo)

                case ImagenEstandar(ruta):
                    img_raw = cv2.imread(str(ruta), cv2.IMREAD_UNCHANGED)
                    if img_raw is None:
                        raise FileNotFoundError(f"Archivo no encontrado: {ruta}")
                    # Normalización a 16-bit
                    img_raw = img_raw.astype(np.uint16) * (65535 // 255) if img_raw.dtype == np.uint8 else img_raw
                    self.img, self.canales = self._procesar_estandar(img_raw, modo)

            if self.img is not None:
                self.forma = self.img.shape
                self.img_procesada = np.zeros_like(self.img)
            return self.img

        except Exception as e:
            print(f"Error al leer la imagen: {e}")
            return None

    def _procesar_bioimagen(self, img_data: np.ndarray, channel_names: List[str], modo: ModoImagen) -> Tuple[np.ndarray, List[str]]:
        """
            Auxiliar para procesar bioimágenes según modo.
            Retorna: (img_procesada, canales)
        """
        match modo:
            case ModoImagen.AUTO:
                return img_data, channel_names

            case ModoImagen.RGB:
                if img_data.shape[2] == 3:
                    return img_data, ["Rojo", "Verde", "Azul"]
                else:
                    raise ValueError("Modo RGB requiere exactamente 3 canales en bioimagen")

            case ModoImagen.GRIS:
                img_gris = np.mean(img_data, axis=2, keepdims=True)
                return img_gris, ["Gris"]

    def _procesar_estandar(self, img_raw: np.ndarray, modo: ModoImagen) -> Tuple[np.ndarray, List[str]]:
        """
            Auxiliar para procesar imágenes estándar según modo.
            Retorna: (img_procesada, canales)
        """
        match modo:
            case ModoImagen.AUTO:
                if img_raw.ndim == 3 and img_raw.shape[2] == 3:
                    canales_rgb = [img_raw[:, :, i] for i in range(3)]
                    img_rgb = np.stack(canales_rgb, axis=0)[np.newaxis, np.newaxis, :, :, :]
                    return img_rgb, ["Rojo", "Verde", "Azul"]
                else:
                    img_gris = img_raw[np.newaxis, np.newaxis, np.newaxis, :, :]
                    return img_gris, ["Gris"]

            case ModoImagen.RGB:
                if img_raw.ndim == 3 and img_raw.shape[2] == 3:
                    canales_rgb = [img_raw[:, :, i] for i in range(3)]
                    img_rgb = np.stack(canales_rgb, axis=0)[np.newaxis, np.newaxis, :, :, :]
                    return img_rgb, ["Rojo", "Verde", "Azul"]
                else:
                    raise ValueError("Modo RGB requiere imagen en color (3 canales)")

            case ModoImagen.GRIS:
                if img_raw.ndim == 3 and img_raw.shape[2] == 3:
                    img_raw = cv2.cvtColor(img_raw, cv2.COLOR_BGR2GRAY)
                img_gris = img_raw[np.newaxis, np.newaxis, np.newaxis, :, :]
                return img_gris, ["Gris"]

    def __iter__(self):
        """
            Iterador para poder buscar o iterar para todos los canales por tiempo y por z-stack, 
            obteniendo imágenes bidimensionales para su procesamiento.

            Yields:
                canal, t, z, Tupla (t, z, img_2d) donde img_2d es np.ndarray de forma (Y, X)

            Complejidad:
                O(T*Z*C) iteraciones
        """

        if self.img is None:
            return iter(())
        T, Z, C, _, _ = self.forma
        for canal in range(C):
            for t in range(T):
                for z in range(Z):
                    yield canal, t, z, self.img[t, z, canal].copy()

    def iterar_cortes(self, canal: int = 0):
        """
            Iterador para poder buscar o iterar en un canal dado por tiempo y por z-stack, 
            obteniendo imágenes bidimensionales para su procesamiento.

            Argumentos:
                canal: Canal a iterar (default: 0)

            Yields:
                Tupla (t, z, img_2d) donde img_2d es np.ndarray de forma (Y, X)

            Complejidad:
                O(T*Z) iteraciones
        """
        if self.img is None:
            raise ValueError("Imagen no cargada")

        T, Z, C, _, _ = self.forma
        
        if not (0 <= canal < C):
            raise IndexError(f"Canal {canal} fuera de rango. Canales disponibles: 0-{C-1}")

        for t in range(T):
            for z in range(Z):
                yield t, z, self.img[t, z, canal].copy()

    def set_corte_procesado(self,
                            canal: int = 0,
                            t: int = 0,
                            z: int = 0,
                            img_2d: Optional[np.ndarray] = None):
        """
            Método setter para guardar una imagen 2D procesada en la estructura tensor 5D.
            
            Argumentos:
                canal: Índice del canal (default: 0)
                t: Índice del timelapse (default: 0)
                z: Índice del z-stack (default: 0)
                img_2d: Objeto np.ndarray a settear (si None, crea array de ceros)

            Complejidad:
                O(1)
        """
        if self.img_procesada is None:
            raise ValueError("img_procesada no inicializada")

        T, Z, C, Y, X = self.forma
        
        if not (0 <= t < T and 0 <= z < Z and 0 <= canal < C):
            raise IndexError(f"Índices fuera de rango. T max: {T-1}, Z max: {Z-1}, C max: {C-1}")

        if img_2d is None:
            img_2d = np.zeros((Y, X), dtype=self.img.dtype)

        assert img_2d.ndim == 2, "img_2d debe ser 2D con forma (Y, X)"
        assert img_2d.shape == (Y, X), f"img_2d debe tener forma ({Y}, {X}), tiene {img_2d.shape}"
        
        self.img_procesada[t, z, canal] = img_2d
    
    def _get_corte(self, 
                img: np.ndarray, 
                canal: int,
                t: int,
                z: int) -> np.ndarray:
        """
            Función interna para realizar cortes en alguna estructura imagen 5D.
            
            Argumentos:
                img: Array 5D [T, Z, C, Y, X]
                canal: Índice del canal
                t: Índice del timelapse
                z: Índice del z-stack

            Retorna:
                Array 2D [Y, X]

            Complejidad:
                O(1) - solo indexación y copia
        """
        T, Z, C, _, _ = img.shape
        
        if not (0 <= t < T and 0 <= z < Z and 0 <= canal < C):
            raise IndexError(f"Índices fuera de rango. T max: {T-1}, Z max: {Z-1}, C max: {C-1}")
        
        return img[t, z, canal].copy()

    def get_corte_original(self,
                        canal: int = 0,
                        t: int = 0,
                        z: int = 0) -> np.ndarray:
        """
            Método getter para obtener un corte de la estructura tensor 5D original.
            
            Argumentos:
                canal: Índice del canal (default: 0)
                t: Índice del timelapse (default: 0)
                z: Índice del z-stack (default: 0)

            Retorna:
                Imagen 2D np.ndarray de forma (Y, X)

            Complejidad:
                O(1)
        """
        if self.img is None:
            raise ValueError("Imagen original no cargada")

        return self._get_corte(self.img, canal, t, z)

    def get_corte_procesado(self,
                            canal: int = 0,
                            t: int = 0,
                            z: int = 0) -> np.ndarray:
        """
            Método getter para obtener un corte de la estructura tensor 5D procesada.
            
            Argumentos:
                canal: Índice del canal (default: 0)
                t: Índice del timelapse (default: 0)
                z: Índice del z-stack (default: 0)

            Retorna:
                Imagen 2D np.ndarray de forma (Y, X)

            Complejidad:
                O(1)
        """
        if self.img_procesada is None:
            raise ValueError("No se ha hecho ninguna operación de procesamiento")

        return self._get_corte(self.img_procesada, canal, t, z)


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

        if self.img is None:
            return 0
        T, Z, *_ = self.forma

        return T * Z

    # Metodos para I/O externo : Abrir imagenes, liberar memoria y cachear los bioformatos.
    def __enter__(self):
        self.leer_bioImagen()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.img = None
        self.img_procesada = None

    def __repr__(self) -> str:
        """
            Metodo para debugging.
            Retorna : String
            Complejidad : O(1)
        """
        if self.img is None:
            return f"<ControladorBioImagen ruta='{self.ruta_imagen}' (no cargada)>"

        T, Z, C, Y, X = self.forma

        return (
            f"<ControladorBioImagen "
            f"ruta='{self.ruta_imagen.name}', "
            f"forma=(T={T}, Z={Z}, C={C}, Y={Y}, X={X}), "
            f"canales={self.canales}>"
        )

    # Metodos para Control de Calidad (CC):
    
    def cc_validar_estado(self) -> bool:
        """
            Método CC para validar el estado interno de la clase.
            Retorna True si todo está OK; False si hay inconsistencias.
            Útil para debugging: llama después de operaciones críticas.
        """
        errores = []
        
        # Check 1: Atributos básicos
        if not isinstance(self.ruta_imagen, Path):
            errores.append("ruta_imagen no es un Path válido")
        if not isinstance(self.canales, list):
            errores.append("canales no es una lista")
        
        # Check 2: Consistencia si img está cargada
        if self.img is not None:
            if self.forma != self.img.shape:
                errores.append(f"forma ({self.forma}) no coincide con img.shape ({self.img.shape})")
            if len(self.canales) != self.img.shape[2]:
                errores.append(f"len(canales) ({len(self.canales)}) no coincide con C ({self.img.shape[2]})")
            if self.img_procesada is not None and self.img_procesada.shape != self.img.shape:
                errores.append("img_procesada.shape no coincide con img.shape")
        
        # Check 3: Tipos de datos
        if self.img is not None and not isinstance(self.img, np.ndarray):
            errores.append("img no es un np.ndarray")
        
        if errores:
            logging.error(f"Errores de QC en ControladorBioImagen: {errores}")
            return False
        logging.info("QC de estado: OK")
        return True

    def cc_dump_estado(self, archivo: Optional[str] = None) -> str:
        """
            Método debugging para dump del estado interno.
            Retorna string con info; opcionalmente guarda en archivo.
            Útil para inspeccionar en debugging.
        """
        estado = f"""
        Estado de ControladorBioImagen:
        - Ruta: {self.ruta_imagen}
        - Configuración: {self.configuracion}
        - Cargada: {self.__bool__()}
        - Forma: {self.forma}
        - Canales: {self.canales}
        - Img dtype: {self.img.dtype if self.img is not None else 'None'}
        - Img_procesada inicializada: {self.img_procesada is not None}
        - Repr: {self.__repr__()}
        """
        if archivo:
            with open(archivo, 'w') as f:
                f.write(estado)
            logging.info(f"Estado dumpado a {archivo}")
        return estado.strip()

    def cc_log_operacion(self, operacion: str, detalles: dict = None):
        """
            Método para logging de operaciones clave.
            Llama en métodos como leer_bioImagen() para tracing.
        """
        mensaje = f"Operación: {operacion}"
        if detalles:
            mensaje += f" - Detalles: {detalles}"
        logging.debug(mensaje)