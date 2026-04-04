#gestorLab/Log.py

def guardar_log(resultado, path="pipeline.log"):
    with open(path, "w") as f:
        for evento in resultado.logs:  # depende de tu Writer
            f.write(str(evento) + "\n")