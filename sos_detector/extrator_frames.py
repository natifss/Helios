import cv2
import os

pasta_videos = "videos_para_anotacao"  
pasta_frames_saida = "frames_para_anotacao" 

intervalo_frames_para_salvar = 15 

if not os.path.exists(pasta_frames_saida):
    os.makedirs(pasta_frames_saida)

lista_videos = [f for f in os.listdir(pasta_videos) if f.endswith(('.mp4', '.avi', '.mov'))]

for nome_video in lista_videos:
    caminho_video = os.path.join(pasta_videos, nome_video)
    nome_base_video = os.path.splitext(nome_video)[0]

    pasta_video_especifico = os.path.join(pasta_frames_saida, nome_base_video)
    if not os.path.exists(pasta_video_especifico):
        os.makedirs(pasta_video_especifico)

    cap = cv2.VideoCapture(caminho_video)
    if not cap.isOpened():
        print(f"Erro ao abrir o vídeo: {caminho_video}")
        continue

    print(f"Processando vídeo: {nome_video}...")
    contador_frame_video = 0
    contador_frames_salvos = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break 

        if contador_frame_video % intervalo_frames_para_salvar == 0:
            nome_frame_salvo = os.path.join(pasta_video_especifico, f"{nome_base_video}_frame_{contador_frames_salvos:04d}.jpg")
            cv2.imwrite(nome_frame_salvo, frame)
            contador_frames_salvos += 1

        contador_frame_video += 1

    cap.release()
    print(f"Vídeo {nome_video} processado. {contador_frames_salvos} frames salvos em '{pasta_video_especifico}'.")

print("Extração de frames concluída para todos os vídeos.")