from flask import Flask, render_template, request, send_file
from ultralytics import YOLO
import os
import cv2
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch

app = Flask(__name__)

# Render-la write permission /tmp folder-ku thaan irukkum
UPLOAD_FOLDER = '/tmp/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# YOLO model-a load pannuthu
model = YOLO("yolov8n.pt")

last_result = ""

# Supported formats
IMAGE_EXT = ['jpg', 'jpeg', 'png']
VIDEO_EXT = ['mp4', 'avi', 'mov']
AUDIO_EXT = ['mp3', 'wav', 'aac']
DOC_EXT = ['pdf', 'docx', 'txt']

@app.route('/', methods=['GET', 'POST'])
def index():
    global last_result
    result_text = ""
    media_type = ""
    detected_objects = []

    if request.method == 'POST':
        file = request.files.get('media')
        if file and file.filename != "":
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            file_ext = file.filename.split('.')[-1].lower()

            if file_ext in IMAGE_EXT:
                results = model(filepath)
                for r in results:
                    for box in r.boxes:
                        cls_id = int(box.cls[0])
                        name = model.names[cls_id]
                        detected_objects.append(name)
                result_text = "Detected: " + ", ".join(set(detected_objects)) if detected_objects else "No animal detected"
                media_type = "image"

            elif file_ext in VIDEO_EXT:
                cap = cv2.VideoCapture(filepath)
                while True:
                    ret, frame = cap.read()
                    if not ret: break
                    results = model(frame)
                    for r in results:
                        for box in r.boxes:
                            cls_id = int(box.cls[0])
                            name = model.names[cls_id]
                            detected_objects.append(name)
                cap.release()
                result_text = "Detected in Video: " + ", ".join(set(detected_objects)) if detected_objects else "No animal detected in video"
                media_type = "video"

            elif file_ext in AUDIO_EXT:
                result_text = "Audio file uploaded. Detection not supported."
                media_type = "audio"

            elif file_ext in DOC_EXT:
                result_text = "Document uploaded. Detection not supported."
                media_type = "document"

            if detected_objects:
                health_status = " | Health Status: Appears Normal" if any(x in ["dog", "cat"] for x in detected_objects) else " | Health Status: Needs Manual Check"
                result_text += health_status

            last_result = result_text
            return render_template("index.html", result=last_result, media=filepath, media_type=media_type)

    return render_template("index.html", result="", media=None, media_type=None)

@app.route('/download')
def download():
    # PDF-aiyum /tmp-le save panna thaan permission kidaikum
    file_path = "/tmp/animal_report.pdf"
    doc = SimpleDocTemplate(file_path)
    elements = []
    styles = getSampleStyleSheet()
    elements.append(Paragraph("<b>Animal Health Detection Report</b>", styles["Title"]))
    elements.append(Spacer(1, 0.5 * inch))
    elements.append(Paragraph(last_result if last_result else "No Data", styles["Normal"]))
    doc.build(elements)
    return send_file(file_path, as_attachment=True)

if __name__ == '__main__':
    # Cloud deployment-kku yetha port setting
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
