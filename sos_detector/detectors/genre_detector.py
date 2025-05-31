from ultralytics import YOLO
import cv2
from deepface import DeepFace

model_yolo_pessoas = YOLO("yolov8n.pt")

def analisar_genero_deepface(imagem_pessoa_recortada_rgb):
   
    try:
        analises = DeepFace.analyze(
            img_path=imagem_pessoa_recortada_rgb,
            actions=['gender'],
            enforce_detection=False,
            silent=True
        )
        
        if analises and len(analises) > 0:
            primeira_analise = analises[0]
            genero_dominante = primeira_analise.get('dominant_gender')
            
            if genero_dominante == "Woman":
                return "Woman"
            elif genero_dominante == "Man":
                return "Man"
            else:
                return "Unknown"
        else:
            return "Not detected"
    except ValueError:
        return "Not detected" 
    except Exception:
        return "Error"


def detectar_mulher_deepface(frame):

    resultados_yolo = model_yolo_pessoas(frame, classes=[0], verbose=False)
    
    for dados_caixa in resultados_yolo[0].boxes.data.cpu().numpy():
      
        x1_coord, y1_coord, x2_coord, y2_coord, confianca_det, id_classe = dados_caixa
       
        if int(id_classe) == 0:
            print(f"INFO YOLO: Pessoa detectada com confiança {confianca_det:.2f}")

            x1_int, y1_int, x2_int, y2_int = int(x1_coord), int(y1_coord), int(x2_coord), int(y2_coord)
            
            frame_altura, frame_largura = frame.shape[:2]
            x1_recorte = max(0, x1_int)
            y1_recorte = max(0, y1_int)
            x2_recorte = min(frame_largura, x2_int)
            y2_recorte = min(frame_altura, y2_int)

            if x1_recorte >= x2_recorte or y1_recorte >= y2_recorte:
                continue

            imagem_pessoa_roi = frame[y1_recorte:y2_recorte, x1_recorte:x2_recorte]

            if imagem_pessoa_roi.size == 0:
                continue
            
            genero_identificado = analisar_genero_deepface(imagem_pessoa_roi)
            print(f"INFO DeepFace: Gênero da pessoa no ROI: {genero_identificado}") 

            if genero_identificado == "Woman":
                print("SUCESSO: Mulher identificada pela DeepFace!") 
                return True 
    return False