import os
from flask import (
    Flask, render_template, request, jsonify,
    redirect, url_for, send_from_directory, flash
)
from werkzeug.utils import secure_filename

# ML imports
import cv2
import numpy as np
from PIL import Image
import torch
from torchvision.models import efficientnet_b0
from torchvision import transforms

# Auth
from flask_sqlalchemy import SQLAlchemy
from flask_login import (
    LoginManager, login_user, logout_user, login_required,
    current_user, UserMixin
)

# -----------------------------
# CONFIGURATION
# -----------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
THUMB_FOLDER = os.path.join(UPLOAD_FOLDER, "thumbs")
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model-v3.pt")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(THUMB_FOLDER, exist_ok=True)

ALLOWED_IMAGE_EXT = {"png", "jpg", "jpeg", "webp"}
ALLOWED_VIDEO_EXT = {"mp4", "webm", "mov", "avi", "mkv"}

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["SECRET_KEY"] = "change_this_secret"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///users.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)

# -----------------------------
# USER MODEL
# -----------------------------

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(120))
    email = db.Column(db.String(200), unique=True)
    password_hash = db.Column(db.String(200))

    def set_password(self, pw):
        self.password_hash = pw

    def check_password(self, pw):
        return self.password_hash == pw


with app.app_context():
    db.create_all()

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"

@login_manager.user_loader
def load_user(uid):
    try:
        return User.query.get(int(uid))
    except:
        return None

@app.context_processor
def inject_user():
    return dict(current_user=current_user)




def load_model(path):
    if not os.path.exists(path):
        print("❌ MODEL NOT FOUND:", path)
        return None

    try:
        model = efficientnet_b0(pretrained=False)
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 2)
        model.load_state_dict(torch.load(path, map_location="cpu"))
        model.eval()
        print("✅ Model loaded successfully.")
        return model
    except Exception as e:
        print("❌ Model failed to load:", e)
        return None


effnet_model = load_model(MODEL_PATH)

effnet_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def predict_pil(pil_img):
    if effnet_model is None:
        return 0, 50.0

    tensor = effnet_transform(pil_img).unsqueeze(0)

    with torch.no_grad():
        out = effnet_model(tensor)
        probs = torch.softmax(out, dim=1)[0].numpy()
        pred = int(np.argmax(probs))
        conf = float(probs[pred] * 100)

    return pred, conf


# -----------------------------
# FACE CROP
# -----------------------------

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def safe_crop_face(img):
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)
        if len(faces) > 0:
            x, y, w, h = faces[0]
            crop = img[y:y+h, x:x+w]
            if crop.size > 0:
                return crop
    except:
        pass
    return img


# -----------------------------
# ALLOWED FILE CHECK
# -----------------------------

def allowed_file(filename, allowed_set):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed_set


# -----------------------------
# ROUTES
# -----------------------------

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/image-detect")
@login_required
def image_detect_page():
    return render_template("image.html")

@app.route("/detect")
@login_required
def detect_page():
    return render_template("detect.html")

@app.route("/login", methods=["GET","POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")

        user = User.query.filter_by(email=email).first()
        if user and user.check_password(password):
            login_user(user)
            return redirect(url_for("home"))

        flash("Invalid credentials", "danger")

    return render_template("login.html")

@app.route("/signup", methods=["GET","POST"])
def signup():
    if request.method == "POST":
        username = request.form.get("username")
        email = request.form.get("email")
        password = request.form.get("password")

        if User.query.filter_by(email=email).first():
            flash("Email already registered", "warning")
            return redirect(url_for("signup"))

        user = User(username=username, email=email)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()

        login_user(user)
        return redirect(url_for("home"))

    return render_template("signup.html")

@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("home"))


# -----------------------------
# IMAGE PREDICT API
# -----------------------------

@app.route("/api/image-predict", methods=["POST"])
@login_required
def api_image_predict():
    try:
        file = request.files.get("image")
        if not file:
            return jsonify({"error": "No image uploaded"}), 400

        if not allowed_file(file.filename, ALLOWED_IMAGE_EXT):
            return jsonify({"error": "Invalid file type"}), 400

        filename = secure_filename(file.filename)
        img_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(img_path)

        img = cv2.imread(img_path)
        if img is None:
            return jsonify({"error": "Unable to read image"}), 400

        face = safe_crop_face(img)
        rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        pred, conf = predict_pil(pil_img)
        label = "FAKE" if pred == 1 else "REAL"

        return jsonify({"label": label, "confidence": round(conf, 2)})

    except Exception as e:
        return jsonify({"error": "Server error", "details": str(e)}), 500

    finally:
        try:
            os.remove(img_path)
        except:
            pass


# -----------------------------
# VIDEO PREDICT API
# -----------------------------

@app.route("/api/video-predict", methods=["POST"])
@login_required
def api_video_predict():
    try:
        file = request.files.get("video")
        if not file:
            return jsonify({"error": "No video uploaded"}), 400

        if not allowed_file(file.filename, ALLOWED_VIDEO_EXT):
            return jsonify({"error": "Invalid video type"}), 400

        filename = secure_filename(file.filename)
        video_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(video_path)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return jsonify({"error": "Cannot open video"}), 400

        results = []
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % 10 != 0:
                continue

            face = safe_crop_face(frame)
            rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)

            pred, _ = predict_pil(pil_img)
            results.append(pred)

        cap.release()

        if not results:
            return jsonify({"error": "No frames processed"}), 400

        fake_count = results.count(1)
        real_count = results.count(0)

        label = "FAKE" if fake_count > real_count else "REAL"
        confidence = max(fake_count, real_count) / len(results) * 100

        return jsonify({"label": label, "confidence": round(confidence, 2)})

    except Exception as e:
        return jsonify({"error": "Server error", "details": str(e)}), 500

    finally:
        try:
            os.remove(video_path)
        except:
            pass


# -----------------------------
# RUN SERVER
# -----------------------------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000, debug=True)
