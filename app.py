import cv2.data
from flask import Flask, render_template, request, redirect, url_for
#from flask_migrate import Migrate
#from flask_sqlalchemy import SQLAlchemy
import cv2
import os
from werkzeug.utils import secure_filename
#from config import Config
app = Flask(__name__)
app.config.from_object('config.Config')
#db = SQLAlchemy(app)
#migrate = Migrate(app, db)



UPLOAD_FOLDER = 'upload'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

#بسنجید که آِیا فایل مورد نظر وجود دارد یا نه...add()

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
    
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'svg', 'webp', 'tiff'}

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods = ['GET', 'POST'])

def index():
    if request.method == 'POST':
        # Handle multiple file uploads
        files = request.files.getlist('files')
        accuracy = request.form.get('accuracy', 'medium')
        reference_image = request.files.get('reference_image')
        # Process each uploaded file
        original_images = []
        detected_images = []
        face_labels = []
        for file in files:
            if file and allowed_file(file.filename):
                filename = save_file(file)
                original_images.append(filename)
                # Detect faces in the uploaded image
                detected_filename = detect_faces(filename, accuracy)
                detected_images.append(detected_filename)
                # Handle face labeling
                face_labels.append(request.form.getlist(f'face{i+1}_label' for i in range(len(detected_filename))))
        # Handle face comparison (if reference image is provided)
        if reference_image and allowed_file(reference_image.filename):
            ref_filename = save_file(reference_image)
            comparison_results = compare_faces(original_images[0], ref_filename)
        else:
            comparison_results = None
        return render_template('index.html', original_images=original_images, detected_images=detected_images, face_labels=face_labels, comparison_results=comparison_results)
    return render_template('index.html')

@app.route('/upload', methods=['POST'])

def upload_image():
    if 'file' not in request.files:
        return redirect(url_for('index'), error='No file selected.')
    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('index'), error='No file selected.')
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        try:
            image = cv2.imread(filepath)
            if image is None:
                return redirect(url_for('index'), error='Error reading the image.')
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            for (x, y, w, h) in faces:
                x, y, w, h = int(x), int(y), int(w), int(h)
                cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
            output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'detected_' + filename)
            cv2.imwrite(output_path, image)
            return render_template('index.html', original_image=filename, detected_image='detected_' + filename)
        
        except Exception as e:
            return redirect(url_for('index'), error=f'Error during face detection: {str(e)}')
    else:
        return redirect(url_for('index'), error='Invalid file type. Only images are allowed.')
    #return render_template('index.html')

def save_file(file):
    filename = secure_filename(file.filename)
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    return filename

def detect_faces(filename, accuracy):
    image = cv2.imread(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    # Draw bounding boxes around the detected faces
    for (x, y, w, h) in faces:
        x, y, w, h = int(x), int(y), int(w), int(h)
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
    detected_filename = f'detected_{filename}'
    cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], detected_filename), image)
    return detected_filename

    image = cv2.imread(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    #جعبه های محدود کننده در اطراف چهره های شناسایی شده

    for (x, y ,w, h) in faces:
        x, y ,w, h = int(x), int(y), int(w), int(h)
        cv2.rectangle(image, (x, y), (x + w, y+ h), (255, 0, 0), 2)
    detected_filename = f'detected_{filename}'
    cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], detected_filename), image)
    return detected_filename
def compare_faces(original_image, reference_image):
    
    # منطق مقایسه چهره را در اینجا پیاده کنید
    # می توانید از یک کتابخانه تشخیص چهره مانند Face_Recognition استفاده کنید
    # برای مقایسه چهره ها در تصاویر اصلی و مرجع
    return True
        
if __name__ == '__main__':
    app.run(debug=  True)        
        
    
    
