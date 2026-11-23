
import os
import uuid
import cv2
import numpy as np

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    send_from_directory
)

from werkzeug.utils import secure_filename


# ==========================================
# 1. FLASK CONFIGURATION
# ==========================================

app = Flask(__name__)

app.secret_key = os.environ.get(
    "SECRET_KEY", "change-this-secret-key"
)

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

UPLOAD_FOLDER = os.path.join(BASE_DIR, "upload")

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ==========================================
# 2. OPENCV CONFIGURATION
# ==========================================

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_eye.xml"
)

if face_cascade.empty():
    raise RuntimeError("Unable to load face detection model")


# ==========================================
# 3. ALLOWED IMAGE EXTENSIONS
# ==========================================

ALLOWED_EXTENSIONS = {
    "png",
    "jpg",
    "jpeg",
    "gif",
    "webp",
    "tiff",
    "bmp"
}


def allowed_file(filename):

    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower()
        in ALLOWED_EXTENSIONS
    )


# ==========================================
# 4. DETECTION ACCURACY SETTINGS
# ==========================================

ACCURACY_SETTINGS = {

    "low": {
        "scaleFactor": 1.2,
        "minNeighbors": 3,
        "minSize": (25, 25)
    },

    "medium": {
        "scaleFactor": 1.1,
        "minNeighbors": 5,
        "minSize": (30, 30)
    },

    "high": {
        "scaleFactor": 1.05,
        "minNeighbors": 7,
        "minSize": (40, 40)
    }
}


# ==========================================
# 5. SAVE UPLOADED FILE
# ==========================================

def save_file(file):

    original_name = secure_filename(file.filename)

    extension = original_name.rsplit(".", 1)[1].lower()

    unique_name = f"{uuid.uuid4().hex}.{extension}"

    filepath = os.path.join(
        app.config["UPLOAD_FOLDER"],
        unique_name
    )

    file.save(filepath)

    return unique_name


# ==========================================
# 6. READ IMAGE
# ==========================================

def read_image(filename):

    filepath = os.path.join(
        app.config["UPLOAD_FOLDER"],
        filename
    )

    image = cv2.imread(filepath)

    if image is None:
        raise ValueError("Unable to read image")

    return image


# ==========================================
# 7. FACE DETECTION
# ==========================================

def detect_faces(filename, accuracy="medium"):

    image = read_image(filename)

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    settings = ACCURACY_SETTINGS.get(
        accuracy,
        ACCURACY_SETTINGS["medium"]
    )

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=settings["scaleFactor"],
        minNeighbors=settings["minNeighbors"],
        minSize=settings["minSize"]
    )

    face_count = len(faces)

    # Draw rectangles around detected faces

    for index, (x, y, w, h) in enumerate(faces, start=1):

        cv2.rectangle(
            image,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        cv2.putText(
            image,
            f"Face {index}",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    # Save detected image

    detected_filename = f"detected_{filename}"

    output_path = os.path.join(
        app.config["UPLOAD_FOLDER"],
        detected_filename
    )

    cv2.imwrite(output_path, image)

    return detected_filename, face_count


# ==========================================
# 8. EYE DETECTION
# ==========================================

def detect_eyes(filename):

    image = read_image(filename)

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5
    )

    eye_count = 0

    for (x, y, w, h) in faces:

        roi_gray = gray[y:y+h, x:x+w]

        roi_color = image[y:y+h, x:x+w]

        eyes = eye_cascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.1,
            minNeighbors=5
        )

        for (ex, ey, ew, eh) in eyes:

            cv2.rectangle(
                roi_color,
                (ex, ey),
                (ex + ew, ey + eh),
                (255, 0, 0),
                2
            )

            eye_count += 1

    output_filename = f"eyes_{filename}"

    cv2.imwrite(
        os.path.join(
            app.config["UPLOAD_FOLDER"],
            output_filename
        ),
        image
    )

    return output_filename, eye_count


# ==========================================
# 9. FACE COMPARISON
# ==========================================

def compare_faces(image_one, image_two):

    img1 = read_image(image_one)
    img2 = read_image(image_two)

    gray1 = cv2.cvtColor(
        img1,
        cv2.COLOR_BGR2GRAY
    )

    gray2 = cv2.cvtColor(
        img2,
        cv2.COLOR_BGR2GRAY
    )

    orb = cv2.ORB_create(
        nfeatures=1000
    )

    kp1, des1 = orb.detectAndCompute(
        gray1,
        None
    )

    kp2, des2 = orb.detectAndCompute(
        gray2,
        None
    )

    if des1 is None or des2 is None:

        return {
            "matches": 0,
            "message": "Not enough visual features for comparison"
        }

    matcher = cv2.BFMatcher(
        cv2.NORM_HAMMING,
        crossCheck=True
    )

    matches = matcher.match(
        des1,
        des2
    )

    matches = sorted(
        matches,
        key=lambda x: x.distance
    )

    good_matches = [
        match for match in matches
        if match.distance < 50
    ]

    return {
        "matches": len(good_matches),
        "message": (
            "Visual features compared. "
            "This is not a reliable identity verification."
        )
    }


# ==========================================
# 10. MAIN PAGE
# ==========================================

@app.route("/", methods=["GET", "POST"])
def index():

    if request.method == "POST":

        files = request.files.getlist("files")

        accuracy = request.form.get(
            "accuracy",
            "medium"
        )

        reference_image = request.files.get(
            "reference_image"
        )

        if accuracy not in ACCURACY_SETTINGS:

            accuracy = "medium"

        if not files or not any(f.filename for f in files):

            flash("Please select at least one image.")

            return redirect(url_for("index"))

        original_images = []
        detected_images = []
        face_counts = []
        eye_counts = []

        # Process uploaded images

        for file in files:

            if not file.filename:
                continue

            if not allowed_file(file.filename):

                flash(
                    f"Invalid image format: {file.filename}"
                )

                continue

            try:

                filename = save_file(file)

                original_images.append(filename)

                detected_filename, face_count = detect_faces(
                    filename,
                    accuracy
                )

                detected_images.append(
                    detected_filename
                )

                face_counts.append(face_count)

                eye_filename, eye_count = detect_eyes(
                    filename
                )

                eye_counts.append(eye_count)

            except Exception as error:

                app.logger.exception(
                    "Image processing failed"
                )

                flash(
                    f"Image processing error: {str(error)}"
                )

        # Reference image comparison

        comparison_results = None

        if (
            reference_image
            and reference_image.filename
            and allowed_file(reference_image.filename)
        ):

            if original_images:

                try:

                    ref_filename = save_file(
                        reference_image
                    )

                    comparison_results = compare_faces(
                        original_images[0],
                        ref_filename
                    )

                except Exception as error:

                    flash(
                        f"Comparison error: {str(error)}"
                    )

        return render_template(
            "index.html",

            original_images=original_images,

            detected_images=detected_images,

            face_counts=face_counts,

            eye_counts=eye_counts,

            comparison_results=comparison_results
        )

    return render_template("index.html")


# ==========================================
# 11. IMAGE UPLOAD ROUTE
# ==========================================

@app.route("/upload", methods=["POST"])
def upload_image():

    file = request.files.get("file")

    if not file or not file.filename:

        flash("No file selected.")

        return redirect(url_for("index"))

    if not allowed_file(file.filename):

        flash("Invalid image format.")

        return redirect(url_for("index"))

    try:

        filename = save_file(file)

        detected_filename, face_count = detect_faces(
            filename,
            "medium"
        )

        return render_template(
            "index.html",

            original_images=[filename],

            detected_images=[detected_filename],

            face_counts=[face_count],

            eye_counts=[],

            comparison_results=None
        )

    except Exception as error:

        app.logger.exception(
            "Upload failed"
        )

        flash(
            f"Upload error: {str(error)}"
        )

        return redirect(url_for("index"))


# ==========================================
# 12. DOWNLOAD PROCESSED IMAGE
# ==========================================

@app.route("/download/<filename>")
def download_image(filename):

    filename = os.path.basename(filename)

    return send_from_directory(
        app.config["UPLOAD_FOLDER"],
        filename,
        as_attachment=True
    )


# ==========================================
# 13. VIEW UPLOADED IMAGE
# ==========================================

@app.route("/images/<filename>")
def uploaded_image(filename):

    filename = os.path.basename(filename)

    return send_from_directory(
        app.config["UPLOAD_FOLDER"],
        filename
    )


# ==========================================
# 14. ERROR HANDLING
# ==========================================

@app.errorhandler(413)
def file_too_large(error):

    flash("File is too large. Maximum size is 16 MB.")

    return redirect(url_for("index"))


@app.errorhandler(404)
def page_not_found(error):

    return "404 - Page Not Found", 404


# ==========================================
# 15. RUN APPLICATION
# ==========================================

if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )
