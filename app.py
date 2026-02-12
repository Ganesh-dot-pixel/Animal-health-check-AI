from flask import Flask, render_template, request, send_file
from ultralytics import YOLO
import os
import cv2
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

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

            # ---------------- IMAGE ----------------
            if file_ext in IMAGE_EXT:
                results = model(filepath)

                for r in results:
                    for box in r.boxes:
                        cls_id = int(box.cls[0])
                        name = model.names[cls_id]
                        detected_objects.append(name)

                if detected_objects:
                    result_text = "Detected: " + ", ".join(set(detected_objects))
                else:
                    result_text = "No animal detected"

                media_type = "image"

            # ---------------- VIDEO ----------------
            elif file_ext in VIDEO_EXT:
                cap = cv2.VideoCapture(filepath)

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    results = model(frame)

                    for r in results:
                        for box in r.boxes:
                            cls_id = int(box.cls[0])
                            name = model.names[cls_id]
                            detected_objects.append(name)

                cap.release()

                if detected_objects:
                    result_text = "Detected in Video: " + ", ".join(set(detected_objects))
                else:
                    result_text = "No animal detected in video"

                media_type = "video"

            # ---------------- AUDIO ----------------
            elif file_ext in AUDIO_EXT:
                result_text = "Audio file uploaded successfully. AI animal detection not supported for audio."
                media_type = "audio"

            # ---------------- DOCUMENT ----------------
            elif file_ext in DOC_EXT:
                result_text = "Document uploaded successfully. Detection not supported for this file type."
                media_type = "document"

            else:
                result_text = "File uploaded successfully. No AI detection available."

            # ---------------- SIMPLE HEALTH LOGIC ----------------
            if detected_objects:
                if "dog" in detected_objects or "cat" in detected_objects:
                    health_status = " | Health Status: Appears Normal"
                else:
                    health_status = " | Health Status: Needs Manual Check"
                result_text += health_status

            last_result = result_text

            return render_template("index.html",
                                   result=last_result,
                                   media=filepath,
                                   media_type=media_type)

    return render_template("index.html",
                           result="",
                           media=None,
                           media_type=None)


# ---------------- PDF REPORT ----------------

@app.route('/download')
def download():
    file_path = "animal_report.pdf"
    doc = SimpleDocTemplate(file_path)
    elements = []

    styles = getSampleStyleSheet()
    elements.append(Paragraph("<b>Animal Health Detection Report</b>", styles["Title"]))
    elements.append(Spacer(1, 0.5 * inch))
    elements.append(Paragraph(last_result if last_result else "No Data", styles["Normal"]))

    doc.build(elements)

    return send_file(file_path, as_attachment=True)


if __name__ == '__main__':
    app.run(debug=True)
