from ultralytics import YOLO
import cv2
import numpy as np
from deepface import DeepFace

model = YOLO("yolov8n.pt") 

video_path = 0 
cap = cv2.VideoCapture(video_path)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0]) 
            conf = box.conf[0].item() 
            cls = int(box.cls[0]) 
            if cls == 0:
                face = frame[y1:y2, x1:x2]
                if face.shape[0] > 0 and face.shape[1] > 0:
                    try:
                        analysis = DeepFace.analyze(face, actions=["gender"], enforce_detection=False)
                        gender = analysis[0]["dominant_gender"]
                        if gender == "Woman":
                            label = f"Mulher ({conf:.2f})"
                            color = (0, 0, 255) 
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    except Exception as e:
                        print("Erro na análise de gênero:", e)

    cv2.imshow("YOLOv8 - Detecção de Mulheres", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    cap.release()
    cv2.destroyAllWindows()