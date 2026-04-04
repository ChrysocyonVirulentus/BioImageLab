from ..controlador.Controlador_Normalizador import Controlador_Normalizador
from ..controlador.Controlador_Filtrador import Controlador_Filtrador
from ..controlador.Controlador_Realzador import Controlador_Realzador
from ..controlador.Controlador_Segmentador import Controlador_Segmentador
from ..controlador.Controlador_Transformador import Controlador_Transformador
from ..controlador.Controlador_Cuantificador import Controlador_Cuantificador
from ..controlador.Controlador_Modelador import Controlador_Modelador
from ..controlador.Controlador_Analizador import Controlador_Analizador

CONTROLADORES = {
    "normalizacion": Controlador_Normalizador(),
    "filtracion": Controlador_Filtrador(),
    "realzado": Controlador_Realzador(),
    "segmentacion": Controlador_Segmentador(),
    "transformacion": Controlador_Transformador(),
    "cuantificacion": Controlador_Cuantificador(),
    "modelado": Controlador_Modelador(),
    "analisis": Controlador_Analizador(),
}