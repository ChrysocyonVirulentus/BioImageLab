from flujoProcesamiento import FlujoProcesamiento

class GestorRamas:
    def __init__(self, img):
        self.ramas = {}

    def nueva_rama(self, nombre):
        self.ramas[nombre] = FlujoProcesamiento(img)

    def ejecutar(self, rama, nombre_op, operacion):
        return self.ramas[rama].ejecutar(nombre_op, operacion)