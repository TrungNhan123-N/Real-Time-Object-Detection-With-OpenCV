from ultralytics import YOLO
import cv2

# Load mô hình YOLOv8 pre-trained (mặc định là YOLOv8n - phiên bản nhẹ)
model = YOLO("yolov8s.pt")  # Hoặc "yolov8s.pt" nếu muốn mô hình mạnh hơn

# Khởi tạo webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Dự đoán vật thể trên từng frame
    results = model(frame)

    # Vẽ bounding box lên hình ảnh
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Lấy tọa độ hộp giới hạn
            conf = box.conf[0]  # Độ tin cậy của dự đoán
            cls = int(box.cls[0])  # ID của lớp vật thể

            # Lấy tên lớp vật thể từ model.names
            label = f"{model.names[cls]}: {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Hiển thị khung hình
    cv2.imshow("YOLOv8 Object Detection", frame)

    # Nhấn 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()
