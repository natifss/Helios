import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2)
mp_draw = mp.solutions.drawing_utils

def detectar_sinal_sos(frame):
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            thumb_tip = handLms.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = handLms.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

            # Se o dedão estiver abaixo do indicador, consideramos como gesto (simplificado)
            if thumb_tip.y > index_tip.y:
                return True
    return False
