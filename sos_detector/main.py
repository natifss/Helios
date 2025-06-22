import cv2
import datetime
import time
import os
from banco import salvar_alerta
from detectors.genre_detector import detectar_mulher_deepface
from detectors.sos_detector import detectar_sinal_sos

PASTA_GRAVACOES = "gravacoes"
FRAMES_PARA_GRAVAR_VIDEO_SOS = 60
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
sos_evento_em_tratamento = False

print("INFO: Sistema de monitoramento SOS com detecção de gênero (DeepFace) iniciado.")
print("INFO: Pressione 'q' para sair.")

fps_frame_count = 0
fps_display_interval = 1
fps_time_last_display = time.time()
fps_accumulated_value = 0.0

frame_count_debug = 0

try:
    while True:
        ret, frame_original = cap.read()
        if not ret:
            print("AVISO: Não foi possível capturar frame da câmera. Encerrando o loop.")
            break

        frame_para_processar_e_exibir = frame_original.copy()
        
        frame_count_debug += 1

        fps_frame_count += 1
        current_time = time.time()

        if (current_time - fps_time_last_display) >= fps_display_interval:
            if (current_time - fps_time_last_display) > 0:
                 fps_actual_calculated = fps_frame_count / (current_time - fps_time_last_display)
                 fps_accumulated_value = fps_actual_calculated
            else:
                 fps_accumulated_value = 0

            fps_frame_count = 0
            fps_time_last_display = current_time
            print(f"INFO: FPS Atual: {fps_accumulated_value:.2f}")

        mulher_detectada = detectar_mulher_deepface(frame_para_processar_e_exibir)
        gesto_sos_detectado_agora = detectar_sinal_sos(frame_para_processar_e_exibir)

        cv2.putText(frame_para_processar_e_exibir, f"FPS: {fps_accumulated_value:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        if mulher_detectada and gesto_sos_detectado_agora:
            if not gravando_video_sos and not sos_evento_em_tratamento:
                timestamp_atual = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                caminho_do_video_sos = os.path.join(PASTA_GRAVACOES, f"sos_mulher_detectada_{timestamp_atual}.mp4")
                altura_frame, largura_frame = frame_original.shape[:2]
                try:
                    objeto_video_writer = cv2.VideoWriter(
                        caminho_do_video_sos,
                        cv2.VideoWriter_fourcc(*'mp4v'),
                        FPS_DA_GRAVACAO,
                        (largura_frame, altura_frame)
                    )
                    if objeto_video_writer.isOpened():
                        gravando_video_sos = True
                        sos_evento_em_tratamento = True
                        contador_frames_gravados_sos = 0
                        print(f"INFO: ⚠️ Mulher e Gesto SOS detectados! Iniciando gravação: {caminho_do_video_sos}")
                    else:
                        print(f"DEBUG ERRO: cv2.VideoWriter NÃO foi aberto após a criação. Gravação não iniciada.")
                        objeto_video_writer = None
                except Exception as e_writer:
                    print(f"ERRO CRÍTICO: Falha ao iniciar o cv2.VideoWriter: {e_writer}")
                    gravando_video_sos = False
                    objeto_video_writer = None
                    sos_evento_em_tratamento = False
        
        if gravando_video_sos and objeto_video_writer is not None:
            if not objeto_video_writer.isOpened():
                print(f"DEBUG ERRO: VideoWriter NÃO ESTÁ ABERTO durante tentativa de escrita! Interrompendo esta gravação.")
                objeto_video_writer.release()
                objeto_video_writer = None
                gravando_video_sos = False
            else:
                objeto_video_writer.write(frame_para_processar_e_exibir)
                contador_frames_gravados_sos += 1

            if contador_frames_gravados_sos >= FRAMES_PARA_GRAVAR_VIDEO_SOS:
                print(f"INFO: Gravação de SOS ({contador_frames_gravados_sos} frames) CONCLUÍDA: {caminho_do_video_sos} (Frame: {frame_count_debug})")
                if objeto_video_writer.isOpened():
                    objeto_video_writer.release()
                objeto_video_writer = None
                gravando_video_sos = False

                print(f"INFO: Tentando salvar o alerta do vídeo '{caminho_do_video_sos}' no banco de dados...")
                try:
                    salvar_alerta(caminho_do_video_sos)
                    print(f"SUCESSO: 📥 Alerta referente ao vídeo '{caminho_do_video_sos}' salvo no banco!")
                except Exception as e_salvar:
                    print(f"ERRO: Falha ao salvar alerta para '{caminho_do_video_sos}' no banco: {e_salvar}")
            
        elif gravando_video_sos and objeto_video_writer is None:
            print(f"DEBUG AVISO: gravando_video_sos é True, mas objeto_video_writer é None. Resetando gravando_video_sos.")
            gravando_video_sos = False

        if not gravando_video_sos and not gesto_sos_detectado_agora:
            if sos_evento_em_tratamento:
                print(f"DEBUG (Frame {frame_count_debug}): Sistema PRONTO para novo evento SOS. (Não está gravando E gesto SOS não detectado). Resetando sos_evento_em_tratamento para False.")
            sos_evento_em_tratamento = False
        
        cv2.imshow("Monitoramento SOS com DeepFace", frame_para_processar_e_exibir)
        key_press = cv2.waitKey(1) & 0xFF

        if key_press == ord('q'):
            print("INFO: Tecla 'q' pressionada, encerrando o loop principal.")
            break

finally:
    print("INFO: Finalizando o sistema de monitoramento...")
    if gravando_video_sos and objeto_video_writer is not None:
        if objeto_video_writer.isOpened():
            print(f"INFO: Encerrando gravação de SOS em andamento de '{caminho_do_video_sos}' (no finally)...")
            objeto_video_writer.release()
            print(f"INFO: Vídeo SOS '{caminho_do_video_sos}' salvo parcialmente com {contador_frames_gravados_sos} frames.")
        else:
            print(f"INFO: objeto_video_writer não estava aberto no bloco finally, gravação '{caminho_do_video_sos}' pode não ter sido salva corretamente.")

        if os.path.exists(caminho_do_video_sos) and os.path.getsize(caminho_do_video_sos) > 0:
            try:
                print(f"INFO: Tentando salvar alerta parcial para '{caminho_do_video_sos}' (no finally)...")
                salvar_alerta(caminho_do_video_sos)
                print(f"SUCESSO: Alerta parcial para '{caminho_do_video_sos}' salvo (no finally).")
            except Exception as e_partial_save:
                print(f"ERRO: Falha ao salvar alerta parcial para '{caminho_do_video_sos}' (no finally): {e_partial_save}")
                
    if cap.isOpened():
        cap.release()
    cv2.destroyAllWindows()
    print("INFO: Câmera liberada e janelas destruídas. Sistema finalizado.")