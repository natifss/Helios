import cv2
import time

cap = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
timestamp = time.strftime("%Y%m%d-%H%M%S")
out = cv2.VideoWriter(f'video_teste_{timestamp}.mp4', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

print("Gravando vídeo de teste... Pressione 's' para parar.")
start_time = time.time()
duration = 10 

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret:
        out.write(frame)
        cv2.imshow('Gravando Teste', frame)
        if time.time() - start_time > duration or (cv2.waitKey(1) & 0xFF == ord('s')):
            break
    else:
        break
cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Vídeo de teste salvo como video_teste_{timestamp}.mp4")