import cv2
import datetime
import os
from banco import salvar_alerta
from detectors.genre_detector import detectar_pessoa_genero
from detectors.sos_detector import detectar_sinal_sos

os.makedirs("gravacoes", exist_ok=True)

cap = cv2.VideoCapture(0)
gravando = False
video_writer = None
caminho_video = ""

print("Sistema de monitoramento iniciado. Pressione 'q' para sair.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    pessoa_detectada = detectar_pessoa_genero(frame)
    gesto_detectado = detectar_sinal_sos(frame)

    if pessoa_detectada and gesto_detectado:
        if not gravando:
            gravando = True
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            caminho_video = f"gravacoes/sos_{timestamp}.mp4"
            video_writer = cv2.VideoWriter(
                caminho_video, cv2.VideoWriter_fourcc(*'mp4v'), 20,
                (frame.shape[1], frame.shape[0])
            )
            print("⚠️ Gesto SOS detectado. Gravando vídeo...")

    if gravando:
        video_writer.write(frame)

        # Simplesmente grava 60 frames (~3 segundos)
        if video_writer.get(cv2.CAP_PROP_FRAME_COUNT) >= 60:
            video_writer.release()
            gravando = False
            salvar_alerta(caminho_video)
            print("📥 Alerta salvo no banco com sucesso!")

    cv2.imshow("Monitoramento SOS", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
