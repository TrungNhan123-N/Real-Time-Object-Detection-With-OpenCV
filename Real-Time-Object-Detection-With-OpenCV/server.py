from flask import Flask, render_template, Response, request
import cv2
import torch
from ultralytics import YOLO
from imutils.video import VideoStream
import time

app = Flask(__name__)

# Load model YOLOv8
model = YOLO("yolov8s.pt")  # Sử dụng phiên bản YOLOv8 nhỏ (s) cho tốc độ nhanh

vs = None  # Luồng video
is_running = False  # Trạng thái camera

def detect_objects():
    global vs, is_running
    while is_running:
        frame = vs.read()
        frame = cv2.resize(frame, (640, 480))

        # Dự đoán với YOLOv8
        results = model(frame)

        # Vẽ bounding box
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Toạ độ hộp giới hạn
                conf = box.conf[0]  # Độ tin cậy
                cls = int(box.cls[0])  # ID lớp vật thể

                label = f"{model.names[cls]}: {conf:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Chuyển frame sang định dạng JPEG
        ret, jpeg = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detection')
def detection():
    name = request.args.get('name', 'Người dùng')  # Lấy tên từ query parameter
    return render_template('detection.html', name=name)

@app.route('/video_feed')
def video_feed():
    global vs, is_running
    if not is_running:
        vs = VideoStream(src=0).start()
        time.sleep(2.0)  # Đợi camera khởi động
        is_running = True
    return Response(detect_objects(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop', methods=['POST'])
def stop():
    global vs, is_running
    if is_running:
        is_running = False
        vs.stop()
    return "Stopped", 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)