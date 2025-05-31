import cv2
import datetime
import os
from banco import salvar_alerta
from detectors.genre_detector import detectar_mulher_deepface
from detectors.sos_detector import detectar_sinal_sos

PASTA_GRAVACOES = "gravacoes"
FRAMES_PARA_GRAVAR_VIDEO_SOS = 100 
FPS_DA_GRAVACAO = 20.0

os.makedirs(PASTA_GRAVACOES, exist_ok=True) 

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERRO CRÍTICO: Não foi possível abrir a câmera. Verifique a conexão ou permissões.")
    exit()

gravando_video_sos = False
objeto_video_writer = None 
caminho_do_video_sos = ""
contador_frames_gravados_sos = 0 

print("INFO: Sistema de monitoramento SOS com detecção de gênero (DeepFace) iniciado.")
print("INFO: Pressione 'q' para sair.")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("AVISO: Não foi possível capturar frame da câmera. Encerrando o loop.")
            break

        mulher_detectada = detectar_mulher_deepface(frame)  

        gesto_sos_detectado = detectar_sinal_sos(frame)

        if mulher_detectada and gesto_sos_detectado:
            if not gravando_video_sos:
                gravando_video_sos = True
                timestamp_atual = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
     
                caminho_do_video_sos = os.path.join(PASTA_GRAVACOES, f"sos_mulher_detectada_{timestamp_atual}.mp4")
                
                altura_frame, largura_frame = frame.shape[:2]
                try:
                    objeto_video_writer = cv2.VideoWriter(
                        caminho_do_video_sos,
                        cv2.VideoWriter_fourcc(*'mp4v'), 
                        FPS_DA_GRAVACAO,
                        (largura_frame, altura_frame)
                    )
                    contador_frames_gravados_sos = 0 
                    print(f"INFO: ⚠️ Mulher e Gesto SOS detectados! Iniciando gravação: {caminho_do_video_sos}")
                except Exception as e_writer:
                    print(f"ERRO: Falha ao iniciar o cv2.VideoWriter: {e_writer}")
                    gravando_video_sos = False 
                    objeto_video_writer = None
        
        if gravando_video_sos and objeto_video_writer is not None:
            objeto_video_writer.write(frame)
            contador_frames_gravados_sos += 1 

            if contador_frames_gravados_sos >= FRAMES_PARA_GRAVAR_VIDEO_SOS:
                print(f"INFO: Gravação de SOS ({contador_frames_gravados_sos} frames) concluída: {caminho_do_video_sos}")
                objeto_video_writer.release() 
                objeto_video_writer = None    
                gravando_video_sos = False
                
                print(f"INFO: Tentando salvar o alerta do vídeo '{caminho_do_video_sos}' no banco de dados...")
                try:
                    salvar_alerta(caminho_do_video_sos) 
                    print(f"SUCESSO: 📥 Alerta referente ao vídeo '{caminho_do_video_sos}' salvo no banco!")
                except Exception as e_salvar:
                    print(f"ERRO: Falha ao salvar alerta para '{caminho_do_video_sos}' no banco: {e_salvar}")
                
                contador_frames_gravados_sos = 0 

        cv2.imshow("Monitoramento SOS com DeepFace", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("INFO: Tecla 'q' pressionada, encerrando o loop principal.")
            break
finally:
    print("INFO: Finalizando o sistema de monitoramento...")
    if gravando_video_sos and objeto_video_writer is not None: 
        print(f"INFO: Encerrando gravação de SOS em andamento de '{caminho_do_video_sos}'...")
        objeto_video_writer.release()
        print(f"INFO: Vídeo SOS '{caminho_do_video_sos}' salvo parcialmente com {contador_frames_gravados_sos} frames.")
       
    if os.path.exists(caminho_do_video_sos) and os.path.getsize(caminho_do_video_sos) > 0:
        try:
            print(f"INFO: Tentando salvar alerta parcial para '{caminho_do_video_sos}'...")
            salvar_alerta(caminho_do_video_sos)
            print(f"SUCESSO: Alerta parcial para '{caminho_do_video_sos}' salvo.")
        except Exception as e_partial_save:
            print(f"ERRO: Falha ao salvar alerta parcial para '{caminho_do_video_sos}': {e_partial_save}")
            
    if cap.isOpened():
        cap.release() 
    cv2.destroyAllWindows() 
    print("INFO: Câmera liberada e janelas destruídas. Sistema finalizado.")