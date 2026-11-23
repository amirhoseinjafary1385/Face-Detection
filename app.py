"""
Face Recognition Application — Flask + OpenCV
"""

import os
import uuid

import cv2
from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    session,
    send_from_directory,
)
from werkzeug.utils import secure_filename

from database.database import Database

# ==========================================
# 1. CONFIGURATION
# ==========================================

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
RESULT_FOLDER = os.path.join(BASE_DIR, "static", "results")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "gif", "webp"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config["SECRET_KEY"] = "change-this-in-production"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["RESULT_FOLDER"] = RESULT_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB

db = Database()

# ==========================================
# 2. CASCADES
# ==========================================

FACE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
EYE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_eye.xml"

face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
eye_cascade = cv2.CascadeClassifier(EYE_CASCADE_PATH)

if face_cascade.empty():
    raise RuntimeError(f"Failed to load face cascade: {FACE_CASCADE_PATH}")
if eye_cascade.empty():
    raise RuntimeError(f"Failed to load eye cascade: {EYE_CASCADE_PATH}")

# ==========================================
# 3. ACCURACY SETTINGS
# ==========================================

ACCURACY_SETTINGS = {
    "high": {"scaleFactor": 1.05, "minNeighbors": 8, "minSize": (60, 60)},
    "medium": {"scaleFactor": 1.10, "minNeighbors": 5, "minSize": (30, 30)},
    "low": {"scaleFactor": 1.30, "minNeighbors": 3, "minSize": (20, 20)},
}

# ==========================================
# 4. HELPERS
# ==========================================


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def save_file(file_storage) -> str:
    """Save an uploaded file with a unique name; return stored filename."""
    original = secure_filename(file_storage.filename)
    unique = f"{uuid.uuid4().hex}_{original}"
    path = os.path.join(app.config["UPLOAD_FOLDER"], unique)
    file_storage.save(path)
    return unique


def read_image(filename: str):
    """Read an image from the upload folder by stored filename."""
    path = os.path.join(app.config["UPLOAD_FOLDER"], os.path.basename(filename))
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {filename}")
    return image


def save_result(image, filename: str) -> str:
    """Write an annotated image to the results folder; return its URL."""
    path = os.path.join(app.config["RESULT_FOLDER"], filename)
    cv2.imwrite(path, image)
    return f"/static/results/{filename}"


# ==========================================
# 5. FACE DETECTION
# ==========================================


def detect_faces(filename: str, accuracy: str = "medium"):
    """Detect faces, save an annotated copy, return (url, count)."""
    settings = ACCURACY_SETTINGS.get(accuracy, ACCURACY_SETTINGS["medium"])

    image = read_image(filename)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=settings["scaleFactor"],
        minNeighbors=settings["minNeighbors"],
        minSize=settings["minSize"],
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    out_name = f"detected_{uuid.uuid4().hex}_{os.path.basename(filename)}"
    url = save_result(image, out_name)
    return url, len(faces)


# ==========================================
# 6. EYE DETECTION
# ==========================================


def detect_eyes(filename: str):
    """Detect eyes inside each detected face, save annotated copy, return (url, count)."""
    image = read_image(filename)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
    )

    eye_count = 0

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = image[y:y + h, x:x + w]

        eyes = eye_cascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.1,
            minNeighbors=5,
        )

        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(
                roi_color,
                (ex, ey),
                (ex + ew, ey + eh),
                (255, 0, 0),
                2,
            )
            eye_count += 1

    out_name = f"eyes_{uuid.uuid4().hex}_{os.path.basename(filename)}"
    url = save_result(image, out_name)
    return url, eye_count


# ==========================================
# 7. FACE COMPARISON
# ==========================================


def compare_faces(image_one: str, image_two: str):
    img1 = read_image(image_one)
    img2 = read_image(image_two)

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(nfeatures=1000)

    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    if des1 is None or des2 is None:
        return {
            "matches": 0,
            "message": "Not enough visual features for comparison",
        }

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des1, des2)
    matches = sorted(matches, key=lambda m: m.distance)

    good_matches = [m for m in matches if m.distance < 50]

    return {
        "matches": len(good_matches),
        "message": (
            "Visual features compared. "
            "This is not a reliable identity verification."
        ),
    }


# ==========================================
# 8. LOGIN / LOGOUT
# ==========================================


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()
        if db.verify_user(username, password):
            session["logged_in"] = True
            session["username"] = username
            flash("Logged in successfully.", "success")
            return redirect(url_for("index"))
        flash("Invalid credentials.", "danger")
    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    return render_template("logout.html")


# ==========================================
# 9. MAIN PAGE
# ==========================================


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method != "POST":
        return render_template("index.html")

    files = request.files.getlist("files")
    accuracy = request.form.get("accuracy", "medium")
    reference_image = request.files.get("reference_image")

    if accuracy not in ACCURACY_SETTINGS:
        accuracy = "medium"

    if not files or not any(f.filename for f in files):
        flash("Please select at least one image.", "warning")
        return redirect(url_for("index"))

    original_images = []
    detected_images = []
    eye_images = []
    face_counts = []
    eye_counts = []

    for file in files:
        if not file.filename:
            continue
        if not allowed_file(file.filename):
            flash(f"Invalid image format: {file.filename}", "danger")
            continue

        try:
            filename = save_file(file)
            original_images.append(filename)

            det_url, face_count = detect_faces(filename, accuracy)
            detected_images.append(det_url)
            face_counts.append(face_count)

            eye_url, eye_count = detect_eyes(filename)
            eye_images.append(eye_url)
            eye_counts.append(eye_count)

        except Exception as exc:
            app.logger.exception("Image processing failed")
            flash(f"Image processing error: {exc}", "danger")

    comparison_results = None
    if (
        reference_image
        and reference_image.filename
        and allowed_file(reference_image.filename)
        and original_images
    ):
        try:
            ref_filename = save_file(reference_image)
            comparison_results = compare_faces(original_images[0], ref_filename)
        except Exception as exc:
            app.logger.exception("Comparison failed")
            flash(f"Comparison error: {exc}", "danger")

    return render_template(
        "index.html",
        original_images=original_images,
        detected_images=detected_images,
        eye_images=eye_images,
        face_counts=face_counts,
        eye_counts=eye_counts,
        comparison_results=comparison_results,
    )


# ==========================================
# 10. SINGLE UPLOAD ROUTE
# ==========================================


@app.route("/upload", methods=["POST"])
def upload_image():
    file = request.files.get("file")
    if not file or not file.filename:
        flash("No file selected.", "warning")
        return redirect(url_for("index"))
    if not allowed_file(file.filename):
        flash("Invalid image format.", "danger")
        return redirect(url_for("index"))

    try:
        filename = save_file(file)
        det_url, face_count = detect_faces(filename, "medium")
        eye_url, eye_count = detect_eyes(filename)

        return render_template(
            "index.html",
            original_images=[filename],
            detected_images=[det_url],
            eye_images=[eye_url],
            face_counts=[face_count],
            eye_counts=[eye_count],
            comparison_results=None,
        )
    except Exception as exc:
        app.logger.exception("Upload failed")
        flash(f"Upload error: {exc}", "danger")
        return redirect(url_for("index"))


# ==========================================
# 11. FILE SERVING
# ==========================================


@app.route("/download/<filename>")
def download_image(filename):
    filename = os.path.basename(filename)
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename, as_attachment=True)


@app.route("/images/<filename>")
def uploaded_image(filename):
    filename = os.path.basename(filename)
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


@app.route("/detect" , methods=["POST"])
def detect():
    return index()


# ==========================================
# 12. ERROR HANDLING
# ==========================================


@app.errorhandler(413)
def file_too_large(error):
    flash("File is too large. Maximum size is 16 MB.", "danger")
    return redirect(url_for("index"))


@app.errorhandler(404)
def page_not_found(error):
    return "404 - Page Not Found", 404


# ==========================================
# 13. RUN
# ==========================================

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)