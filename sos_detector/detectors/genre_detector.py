from ultralytics import YOLO

# Carrega o modelo YOLO (treinado para detectar pessoas)
model = YOLO("yolov8n.pt")

def detectar_pessoa_genero(frame):
    resultados = model(frame)
    for r in resultados[0].boxes.data:
        classe = int(r[-1])
        if classe == 0:  # classe 0 = pessoa
            return True
    return False
