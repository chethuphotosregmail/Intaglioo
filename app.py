from flask import Flask, render_template, request #type:ignore
import cv2, os, tempfile, numpy as np, pywt, joblib #type:ignore
from scipy.stats import skew, kurtosis #type:ignore 
from skimage.feature import local_binary_pattern, hog #type:ignore
from sklearn.metrics.pairwise import cosine_similarity #type:ignore

app = Flask(__name__)

ROI_SIZE = 1080
DEBUG_SAVE = False
GENUINE_REF_DIR = "resolution/genuine"

# ---------------- MODEL LOAD ----------------
if not os.path.exists("model_v5.pkl") or not os.path.exists("scaler_v5.pkl"):
    raise FileNotFoundError("⚠️ model_v5.pkl or scaler_v5.pkl not found!")

rf = joblib.load("model_v5.pkl")
scaler = joblib.load("scaler_v5.pkl")
print("✅ Model_v5 & Scaler_v5 loaded successfully!")


# ---------------- REFERENCE PATTERN ----------------
def compute_reference_hog():
    hogs = []
    for fname in os.listdir(GENUINE_REF_DIR):
        fpath = os.path.join(GENUINE_REF_DIR, fname)
        if not fpath.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.resize(img, (256, 256))
        h = hog(img, orientations=9, pixels_per_cell=(16, 16),
                cells_per_block=(2, 2), visualize=False, block_norm='L2-Hys')
        hogs.append(h)
    if len(hogs) == 0:
        return None
    return np.mean(hogs, axis=0)

print("📘 Generating reference HOG pattern from genuine samples...")
REF_HOG = compute_reference_hog()
if REF_HOG is None:
    print("⚠️ Warning: No genuine reference images found.")
else:
    print("✅ Reference pattern signature ready.")


# ---------------- PREPROCESS ----------------
def detect_note_region(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (7, 7), 0), 60, 160)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img
    contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(contour)
    if w * h < 0.2 * img.shape[0] * img.shape[1]:
        return img
    return img[y:y+h, x:x+w]


def preprocess_image(path, size=(ROI_SIZE, ROI_SIZE)):
    img = cv2.imread(path)
    if img is None:
        raise ValueError("Unreadable image.")
    img = detect_note_region(img)
    img = cv2.resize(img, size)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # normalize lighting
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # mobile lighting correction
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    return img, gray


# ---------------- FEATURE EXTRACTION ----------------
def wavelet_color_features(img, wavelet_name='db2'):
    feats = []
    for channel in cv2.split(img):
        coeffs2 = pywt.dwt2(channel, wavelet_name)
        _, (cH, cV, cD) = coeffs2
        for mat in [cH, cV, cD]:
            hist, _ = np.histogram(mat.flatten(), bins=64, density=True)
            feats.extend([np.var(hist), skew(hist), kurtosis(hist)])
    return feats


def extract_features(img_color, gray):
    feats = wavelet_color_features(img_color)
    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 59), density=True)
    feats.extend(hist.tolist())
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    contrast_std = np.std(cv2.absdiff(gray, blur))
    bright_ratio = np.sum(gray > 220) / gray.size
    bright_mask = cv2.inRange(gray, 230, 255)
    contours, _ = cv2.findContours(bright_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    reflection_clusters = len([c for c in contours if 5 < cv2.contourArea(c) < 200])
    reflection_density = reflection_clusters / (gray.shape[0] * gray.shape[1] / 10000)
    color_std = np.std(cv2.cvtColor(img_color, cv2.COLOR_BGR2LAB)[:, :, 0])
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    edge_mag = np.sqrt(sobelx**2 + sobely**2)
    edge_uniformity = 1 / (np.std(edge_mag) + 1e-5)
    edges = cv2.Canny(gray, 80, 160)
    micro_edge_density = np.sum(edges > 0) / edges.size
    specular_ratio = np.sum(gray > 245) / gray.size
    highpass = gray - cv2.GaussianBlur(gray, (5, 5), 0)
    texture_coarseness = np.std(highpass)
    feats.extend([
        lap_var, contrast_std, bright_ratio, reflection_density,
        color_std, edge_uniformity, micro_edge_density,
        specular_ratio, texture_coarseness
    ])
    return np.array(feats, dtype=np.float32)


# ---------------- PREDICT ----------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        file = request.files.get("file")
        if not file or file.filename == "":
            return render_template("result.html", result="⚠️ No image selected!")

        temp_path = os.path.join(tempfile.gettempdir(), file.filename)
        file.save(temp_path)

        img_color, gray = preprocess_image(temp_path)
        feats = extract_features(img_color, gray).reshape(1, -1)
        feats_scaled = scaler.transform(feats)
        proba = rf.predict_proba(feats_scaled)[0]
        genuine_prob = float(proba[1])

        # --- Pattern and Reflectivity Checks ---
        reflectivity = np.sum(gray > 230) / gray.size
        reflection_strength = np.std(cv2.GaussianBlur(gray, (5, 5), 0))
        test_hog = hog(cv2.resize(gray, (256, 256)), orientations=9,
                       pixels_per_cell=(16, 16), cells_per_block=(2, 2),
                       visualize=False, block_norm='L2-Hys')
        pattern_similarity = (
            cosine_similarity([REF_HOG], [test_hog])[0][0] if REF_HOG is not None else 0.0
        )

        # --- Adjust pattern score based on reflectivity ---
        if reflectivity < 0.001:
            pattern_similarity *= 0.85  # penalize flat Xerox
        elif reflectivity > 0.002:
            pattern_similarity *= 1.05  # boost for real reflective note
            pattern_similarity = min(pattern_similarity, 1.0)

        # --- Final Decision ---
        if genuine_prob >= 0.9 and pattern_similarity > 0.85 and reflectivity > 0.001:
            result = "✅ Genuine Note"
        elif genuine_prob >= 0.8 and pattern_similarity > 0.75 and reflectivity > 0.0008:
            result = "⚠️ Likely Genuine (Low Light)"
        else:
            result = "❌ Fake / Xerox Detected"

        confidence = f"{genuine_prob:.2f} | Pattern: {pattern_similarity:.2f} | Reflectivity: {reflectivity:.5f}"
        os.remove(temp_path)

        return render_template("result.html", result=result, confidence=confidence)

    except Exception as e:
        print("Error:", e)
        return render_template("result.html", result=f"❌ Error: {str(e)}")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
