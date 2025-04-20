from flask import Flask, render_template, Response, request, jsonify, redirect, url_for, session, flash, send_file
from flask_mysqldb import MySQL
import cv2
from ultralytics import YOLO
from imutils.video import VideoStream
import time
import numpy as np
from sort import Sort
import io
from datetime import datetime

app = Flask(__name__)

# Cấu hình MySQL
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'nhan123'
app.config['MYSQL_PASSWORD'] = 'nhan123@'
app.config['MYSQL_DB'] = 'yolo_v8'
app.secret_key = 'your_secret_key'

mysql = MySQL(app)

# Load model YOLOv8
model = YOLO("yolov8s.pt")
tracker = Sort()

# Kiểm tra model.names và lấy danh sách tên lớp
if isinstance(model.names, dict):
    class_names = list(model.names.values())
else:
    class_names = model.names

vs = None
is_running = False
settings = {
    'box_color': (0, 255, 0),  # Màu mặc định cho khung
    'label_color': (0, 255, 0),  # Màu mặc định cho nhãn
    'mode': 'detection'
}
stats = {'detections': {}}

def init_db():
    with app.app_context():
        cur = mysql.connection.cursor()
        cur.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INT AUTO_INCREMENT PRIMARY KEY,
                username VARCHAR(50) UNIQUE NOT NULL,
                email VARCHAR(100) UNIQUE NOT NULL,
                password VARCHAR(255) NOT NULL
            )
        ''')
        cur.execute('''
            CREATE TABLE IF NOT EXISTS saved_files (
                id INT AUTO_INCREMENT PRIMARY KEY,
                user_id INT,
                file_type VARCHAR(10),
                file_path VARCHAR(255),
                file_data LONGBLOB,
                created_at DATETIME,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        ''')
        cur.execute("SHOW COLUMNS FROM saved_files LIKE 'file_data'")
        if not cur.fetchone():
            cur.execute("ALTER TABLE saved_files ADD COLUMN file_data LONGBLOB AFTER file_path")
        mysql.connection.commit()
        cur.close()

def detect_objects():
    global vs, is_running
    while is_running:
        if vs is None:
            print("Video stream is not initialized. Attempting to restart...")
            vs = VideoStream(src=0).start()
            time.sleep(2.0)

        frame = vs.read()
        if frame is None:
            print("Error: Cannot read frame from video stream.")
            time.sleep(0.1)
            continue
        frame = cv2.resize(frame, (640, 480))

        try:
            results = model(frame, task=settings['mode'])
            print(f"YOLOv8 detection successful. Number of results: {len(results)}")
        except Exception as e:
            print(f"Error in YOLOv8 detection: {e}")
            results = []

        detections = []
        for result in results:
            for box in result.boxes:
                conf = box.conf[0].item()
                cls = int(box.cls[0].item())
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detections.append([x1, y1, x2, y2, conf, cls])

                label = model.names[cls]
                stats['detections'][label] = stats['detections'].get(label, 0) + 1

        if settings['mode'] == 'tracking' and detections:
            tracked_objects = tracker.update(np.array([[x1, y1, x2, y2, conf] for x1, y1, x2, y2, conf, cls in detections]))
            for track in tracked_objects:
                x1, y1, x2, y2, track_id = map(int, track)
                label = f"ID {track_id}"
                print(f"Drawing tracking box: {x1}, {y1}, {x2}, {y2} with color {settings['box_color']}")
                cv2.rectangle(frame, (x1, y1), (x2, y2), settings['box_color'], 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, settings['label_color'], 2)
        else:
            for x1, y1, x2, y2, conf, cls in detections:
                label = f"{model.names[cls]}: {conf:.2f}"
                print(f"Drawing detection box: {x1}, {y1}, {x2}, {y2} with color {settings['box_color']}")
                cv2.rectangle(frame, (x1, y1), (x2, y2), settings['box_color'], 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, settings['label_color'], 2)

        ret, jpeg = cv2.imencode('.jpg', frame)
        if ret:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
        else:
            print("Error encoding frame to JPEG")

def save_file_to_db(file_path, file_data, file_type):
    if 'user_id' in session:
        cur = mysql.connection.cursor()
        cur.execute('''
            INSERT INTO saved_files (user_id, file_type, file_path, file_data, created_at)
            VALUES (%s, %s, %s, %s, %s)
        ''', (session['user_id'], file_type, file_path, file_data, datetime.now()))
        mysql.connection.commit()
        cur.close()

@app.route('/')
def index():
    return render_template('index.html', user=session.get('username'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        
        cur = mysql.connection.cursor()
        try:
            cur.execute('INSERT INTO users (username, email, password) VALUES (%s, %s, %s)', 
                        (username, email, password))
            mysql.connection.commit()
            flash('Đăng ký thành công! Vui lòng đăng nhập.', 'success')
            return redirect(url_for('login'))
        except Exception as e:
            flash('Tên người dùng hoặc email đã tồn tại!', 'error')
            return render_template('register.html')
        finally:
            cur.close()
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        
        cur = mysql.connection.cursor()
        cur.execute('SELECT * FROM users WHERE email = %s AND password = %s', (email, password))
        user = cur.fetchone()
        cur.close()
        
        if user:
            session['user_id'] = user[0]
            session['username'] = user[1]
            flash('Đăng nhập thành công!', 'success')
            return redirect(url_for('detection'))
        else:
            flash('Sai email hoặc mật khẩu!', 'error')
            return render_template('login.html')
    return render_template('login.html')

@app.route('/detection')
def detection():
    if 'user_id' not in session:
        flash('Vui lòng đăng nhập trước!', 'error')
        return redirect(url_for('login'))
    username = session.get('username', 'Người dùng')
    return render_template('detection.html', name=username)

@app.route('/video_feed')
def video_feed():
    global vs, is_running
    if not is_running:
        print("Starting video stream...")
        if vs is not None:
            vs.stop()
        vs = VideoStream(src=0).start()
        time.sleep(2.0)
        is_running = True
    return Response(detect_objects(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop', methods=['POST'])
def stop():
    global vs, is_running
    if is_running:
        print("Stopping video stream...")
        is_running = False
        if vs is not None:
            vs.stop()
            vs = None
    return "Stopped", 200

@app.route('/set_mode/<mode>', methods=['POST'])
def set_mode(mode):
    settings['mode'] = mode
    print(f"Mode set to: {mode}")
    return "Mode set", 200

@app.route('/update_settings', methods=['POST'])
def update_settings():
    data = request.json
    settings['box_color'] = tuple(data.get('box_color', [0, 255, 0]))
    settings['label_color'] = tuple(data.get('label_color', [0, 255, 0]))
    print(f"Settings updated: {settings}")
    return "Settings updated", 200

@app.route('/capture_image', methods=['POST'])
def capture_image():
    if 'user_id' not in session:
        return "Vui lòng đăng nhập trước.", 401
    if not is_running or vs is None:
        return "Camera chưa được mở. Vui lòng nhấn nút 'Mở' trước khi chụp ảnh.", 400

    frame = vs.read()
    if frame is None:
        return "Không thể chụp ảnh. Vui lòng thử lại.", 400

    frame = cv2.resize(frame, (640, 480))
    _, buffer = cv2.imencode('.jpg', frame)
    img_data = buffer.tobytes()
    save_file_to_db(None, img_data, 'image')
    return "Ảnh đã được chụp và lưu thành công.", 200

@app.route('/download_file/<int:file_id>', methods=['GET'])
def download_file(file_id):
    if 'user_id' not in session:
        return "Please login first.", 401
    
    cur = mysql.connection.cursor()
    cur.execute('SELECT file_type, file_data, created_at FROM saved_files WHERE id = %s AND user_id = %s', 
                (file_id, session['user_id']))
    file = cur.fetchone()
    cur.close()

    if not file:
        return "File not found.", 404

    file_type, file_data, created_at = file
    timestamp = created_at.strftime('%Y%m%d_%H%M%S')
    
    filename = f"snapshot_{timestamp}.jpg"
    content_type = 'image/jpeg'

    return send_file(
        io.BytesIO(file_data),
        mimetype=content_type,
        as_attachment=True,
        download_name=filename
    )

@app.route('/get_saved_files', methods=['GET'])
def get_saved_files():
    if 'user_id' not in session:
        return jsonify([]), 401
    cur = mysql.connection.cursor()
    cur.execute('SELECT id, file_type, created_at FROM saved_files WHERE user_id = %s', (session['user_id'],))
    files = [{'id': row[0], 'type': row[1], 'created_at': row[2].strftime('%Y-%m-%d %H:%M:%S')} for row in cur.fetchall()]
    cur.close()
    return jsonify(files)

@app.route('/get_file_data/<int:file_id>', methods=['GET'])
def get_file_data(file_id):
    if 'user_id' not in session:
        return jsonify({'error': 'Please login first.'}), 401
    
    cur = mysql.connection.cursor()
    cur.execute('SELECT file_type, file_data FROM saved_files WHERE id = %s AND user_id = %s', 
                (file_id, session['user_id']))
    file = cur.fetchone()
    cur.close()

    if not file:
        return jsonify({'error': 'File not found.'}), 404

    file_type, file_data = file
    import base64
    file_data_b64 = base64.b64encode(file_data).decode('utf-8')
    return jsonify({'type': file_type, 'data': file_data_b64})

@app.route('/get_stats', methods=['GET'])
def get_stats():
    return jsonify(stats)

@app.route('/logout')
def logout():
    global is_running
    if is_running:
        is_running = False
        if vs is not None:
            vs.stop()
            vs = None
    session.pop('user_id', None)
    session.pop('username', None)
    flash('Đăng xuất thành công!', 'success')
    return redirect(url_for('index'))

if __name__ == '__main__':
    init_db()
    app.run(host='0.0.0.0', port=5000, debug=True)