import cv2
import numpy as np
import warnings

# Filtro gaussiano

class Gaussiano:
  nombre = "gaussiano"

  """
    Funcion atomica de filtrado gaussiano que permite un suavizado espacial general de los pixeles,
    eliminando ruido electronico de fondo. 
    Problema : Produce difuminado de los bordes, como bordes de celulas o estructuras.
  """

  def __init__(self, 
              sigma: float,
              mascara : tuple[int, int] = (0,0)
              ):
    """
      Filtrado por funcion gaussiana de sigma dado por parametro y una matriz
      mascara del tamano del blur del filtro.

      Argumentos:
        -Sigma : El parametro sigma de desviacion estandar (la fuerza de desenfoque)
        -mascara : La mascara del filtro gaussiano. Deben ser valores impares
      Retorna:
        -MultiArray (imagen) post-filtrado
    """
    self.sigma = sigma
    self.mascara = self._chequear_mascara(mascara)

  def _chequear_mascara(self, mascara: tuple[int, int]) -> tuple[int, int]:
        """
          Verifica que los valores de la máscara sean impares. Esto evita crasheo en runtime.
          Si son pares, les suma 1 para corregirlos automáticamente.
        """
        ancho, alto = mascara
        
        # Si se pasa (0,0), esto el OpenCV lo acepta. Entonces, solo calcula según el sigma
        if ancho == 0 and alto == 0:
            return (0, 0)
            
        # Asegurar que sean impares (OpenCV requiere ksize impar)
        nuevo_ancho = ancho if ancho % 2 != 0 else ancho + 1
        nuevo_alto = alto if alto % 2 != 0 else alto + 1

        if nuevo_ancho != ancho or nuevo_alto != alto:
          warnings.warn(
                f"Nota: Máscara corregida de ({ancho}, {alto})"
                "a ({nuevo_ancho}, {nuevo_alto}) para ser impar.",
                RuntimeWarning
            )
            
        return (nuevo_ancho, nuevo_alto)

  def __call__(self, img: np.ndarray) -> np.ndarray:
    return cv2.GaussianBlur(img, self.mascara, self.sigma)