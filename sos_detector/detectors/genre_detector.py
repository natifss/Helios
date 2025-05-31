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
            
            print(f"DEBUG: Resultado da análise: {primeira_analise}")

            if genero_dominante == "Woman":
                return "Woman"
            elif genero_dominante == "Man":
                return "Man"
            else:
                print(f"AVISO DeepFace: Gênero dominante não foi 'Mulher' ou 'Homem': {genero_dominante}")
                return "Unknown"
        else:
            print("AVISO: Nenhuma face ou gênero detectado no ROI pela DeepFace.")
            return "Not detected" 

    except ValueError as ve:
        print(f"AVISO: ValueError durante a análise - {ve}")
        return "Not detected"
    except Exception as e:
        print(f"ERRO: Exceção inesperada durante a análise de gênero: {e}")
        return "Error"


def detectar_mulher_deepface(frame):
    
    resultados_yolo = model_yolo_pessoas(frame, classes=[0], verbose=False)
    
    for r_yolo in resultados_yolo[0].boxes.data.cpu().numpy():
        r_yolo = [x1, y1, x2, y2, confiança, id_da_classe]
        x1, y1, x2, y2, confianca, classe_id = r_yolo
        
        if int(classe_id) == 0:
            print(f"INFO YOLO: Pessoa detectada com confiança {confianca:.2f}")

            x1_int, y1_int, x2_int, y2_int = int(x1), int(y1), int(x2), int(y2)
            
            frame_altura, frame_largura = frame.shape[:2]
            x1_recorte = max(0, x1_int)
            y1_recorte = max(0, y1_int)
            x2_recorte = min(frame_largura, x2_int)
            y2_recorte = min(frame_altura, y2_int)

            if x1_recorte >= x2_recorte or y1_recorte >= y2_recorte:
                print("AVISO: Caixa de recorte da pessoa inválida (área zero). Pulando.")
                continue

            imagem_pessoa_roi = frame[y1_recorte:y2_recorte, x1_recorte:x2_recorte]

            if imagem_pessoa_roi.size == 0:
                print("AVISO: Imagem recortada da pessoa (ROI) está vazia. Pulando.")
                continue
           
            genero_identificado = analisar_genero_deepface(imagem_pessoa_roi)
            print(f"INFO DeepFace: Gênero da pessoa no ROI: {genero_identificado}")

            if genero_identificado == "Woman":
                print("SUCESSO: Mulher identificada pela DeepFace!")
                return True
            else:
                print(f"INFO: Pessoa não classificada como 'Mulher', ({genero_identificado}). Continuando busca...")
                return False