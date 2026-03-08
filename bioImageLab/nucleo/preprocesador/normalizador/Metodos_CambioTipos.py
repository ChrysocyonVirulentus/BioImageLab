import numpy as np

class MetodoCambioTipo:
    """
    Clase base para conversión de tipos de imagen.
    """
    nombre = "base_cambio_tipo"

    def __call__(self, img: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class ToUint8(MetodoCambioTipo):
    """
    Convierte cualquier imagen a uint8 (0-255).
    Para float, escala automáticamente.
    """
    nombre = "to_uint8"

    def __call__(self, img: np.ndarray) -> np.ndarray:
        if np.issubdtype(img.dtype, np.floating):
            # Escala a 0-255
            img_min, img_max = img.min(), img.max()
            if img_max > img_min:
                img_scaled = (img - img_min) / (img_max - img_min) * 255
            else:
                img_scaled = np.zeros_like(img)
            return img_scaled.astype(np.uint8)
        elif np.issubdtype(img.dtype, np.integer):
            # Si es uint16, escala a uint8
            if img.dtype == np.uint16:
                return (img / 256).astype(np.uint8)
            elif img.dtype == np.uint8:
                return img
            else:
                # Otros enteros
                info = np.iinfo(img.dtype)
                return (img.astype(np.float64) / info.max * 255).astype(np.uint8)
        else:
            raise TypeError(f"Tipo de imagen no soportado: {img.dtype}")


class ToUint16(MetodoCambioTipo):
    """
    Convierte cualquier imagen a uint16 (0-65535).
    Para float, escala automáticamente.
    """
    nombre = "to_uint16"

    def __call__(self, img: np.ndarray) -> np.ndarray:
        if np.issubdtype(img.dtype, np.floating):
            img_min, img_max = img.min(), img.max()
            if img_max > img_min:
                img_scaled = (img - img_min) / (img_max - img_min) * 65535
            else:
                img_scaled = np.zeros_like(img)
            return img_scaled.astype(np.uint16)
        elif np.issubdtype(img.dtype, np.integer):
            if img.dtype == np.uint8:
                return (img.astype(np.uint16) * 256)
            elif img.dtype == np.uint16:
                return img
            else:
                info = np.iinfo(img.dtype)
                return (img.astype(np.float64) / info.max * 65535).astype(np.uint16)
        else:
            raise TypeError(f"Tipo de imagen no soportado: {img.dtype}")


class ToFloat32(MetodoCambioTipo):
    """
    Convierte cualquier imagen a float32 (0.0-1.0 si era uint8/16).
    """
    nombre = "to_float32"

    def __call__(self, img: np.ndarray) -> np.ndarray:
        if np.issubdtype(img.dtype, np.floating):
            return img.astype(np.float32)
        elif np.issubdtype(img.dtype, np.integer):
            if img.dtype == np.uint8:
                return (img.astype(np.float32) / 255.0)
            elif img.dtype == np.uint16:
                return (img.astype(np.float32) / 65535.0)
            else:
                info = np.iinfo(img.dtype)
                return (img.astype(np.float32) / info.max)
        else:
            raise TypeError(f"Tipo de imagen no soportado: {img.dtype}")


class ToFloat64(MetodoCambioTipo):
    """
    Convierte cualquier imagen a float64 (0.0-1.0 si era uint8/16).
    """
    nombre = "to_float64"

    def __call__(self, img: np.ndarray) -> np.ndarray:
        if np.issubdtype(img.dtype, np.floating):
            return img.astype(np.float64)
        elif np.issubdtype(img.dtype, np.integer):
            if img.dtype == np.uint8:
                return (img.astype(np.float64) / 255.0)
            elif img.dtype == np.uint16:
                return (img.astype(np.float64) / 65535.0)
            else:
                info = np.iinfo(img.dtype)
                return (img.astype(np.float64) / info.max)
        else:
            raise TypeError(f"Tipo de imagen no soportado: {img.dtype}")