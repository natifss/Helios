from ultralytics import YOLO
import cv2


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
                label = f"Pessoa ({conf:.2f})"
                color = (0, 255, 0) # Verde
            else:
                label = f"Outro ({conf:.2f})"
                color = (255, 0, 0) # Azul
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
    color, 2)
            
        cv2.imshow("YOLOv8 Detecção", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    cap.release()
    cv2.destroyAllWindows()
