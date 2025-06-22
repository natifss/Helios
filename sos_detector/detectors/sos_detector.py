import cv2
import mediapipe as mp
# import numpy as np # numpy não estava sendo usado, pode ser omitido

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2)
mp_draw = mp.solutions.drawing_utils

def detectar_sinal_sos(frame):
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    sinal_sos_detectado = False
    frame_altura, frame_largura = frame.shape[:2]

    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            pontos = [(int(lm.x * frame_largura), int(lm.y * frame_altura)) for lm in handLms.landmark]

            x_list = [p[0] for p in pontos]
            y_list = [p[1] for p in pontos]
            x_min, x_max = min(x_list), max(x_list)
            y_min, y_max = min(y_list), max(y_list)

            thumb_tip = handLms.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = handLms.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middle_tip = handLms.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            ring_tip = handLms.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
            pinky_tip = handLms.landmark[mp_hands.HandLandmark.PINKY_TIP]

            index_base = handLms.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
            middle_base = handLms.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
            ring_base = handLms.landmark[mp_hands.HandLandmark.RING_FINGER_MCP]
            pinky_base = handLms.landmark[mp_hands.HandLandmark.PINKY_MCP]

            dedao_dentro = thumb_tip.y > middle_base.y
            dedos_fechados = (
                index_tip.y > index_base.y and
                middle_tip.y > middle_base.y and
                ring_tip.y > ring_base.y and
                pinky_tip.y > pinky_base.y
            )

            if dedao_dentro and dedos_fechados:
                legenda = "Sinal de SOS detectado"
                cor = (0, 0, 255)
                sinal_sos_detectado = True
            else:
                legenda = "Mao"
                cor = (200, 200, 200)

            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), cor, 2)
            cv2.putText(frame, legenda, (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, cor, 2)
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

    return sinal_sos_detectado