from flask import Flask, render_template, request
import cv2, os, tempfile, numpy as np, base64
from skimage.feature import hog
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow as tf

app = Flask(__name__)

GENUINE_REF_DIR = "resolution/genuine"
MODEL_PATH = "oneplus_crop.tflite"

# ---------------- MODEL LOAD ----------------
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("⚠️ TFLite model not found!")

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_shape = input_details[0]['shape']
EXPECTED_H, EXPECTED_W = input_shape[1], input_shape[2]
EXPECTED_C = input_shape[3]
print(f"✅ Model loaded. Expected input shape: {input_shape}")


# ---------------- REFERENCE HOG ----------------
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
    print("✅ Reference HOG pattern ready.")


# ---------------- DETECTION & CROPPING ----------------
def detect_and_crop(img):
    """Detect ROI using TFLite model output only when needed."""
    h, w, _ = img.shape
    aspect_ratio = w / h

    # 🔍 If the image is already a cropped region (square-ish), skip detection
    if 0.8 <= aspect_ratio <= 1.2 and min(h, w) < 600:
        print("🟡 Already cropped image detected — skipping model detection.")
        return img

    print("🔍 Running model-based detection...")
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (EXPECTED_W, EXPECTED_H))
    input_data = np.expand_dims(resized.astype(np.float32) / 255.0, axis=0)

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])[0]  # shape: (10647, 7)

    scores = np.maximum(output[:, 5], output[:, 6])
    boxes = output[:, :4]

    conf_mask = scores > 0.4
    boxes, scores = boxes[conf_mask], scores[conf_mask]

    if len(boxes) == 0:
        print("⚠️ No confident detections found — using original image.")
        return img

    boxes_px, areas = [], []
    for b in boxes:
        ymin, xmin, ymax, xmax = b
        x1, y1, x2, y2 = (
            int(xmin * w),
            int(ymin * h),
            int(xmax * w),
            int(ymax * h)
        )
        if (x2 - x1) > 0 and (y2 - y1) > 0:
            boxes_px.append([x1, y1, x2, y2])
            areas.append((x2 - x1) * (y2 - y1))

    if len(boxes_px) == 0:
        print("⚠️ All detections invalid — using original image.")
        return img

    best_idx = int(np.argmax(areas))
    x1, y1, x2, y2 = boxes_px[best_idx]
    conf = float(scores[best_idx])
    cropped = img[y1:y2, x1:x2]

    if cropped.size == 0:
        print("⚠️ Cropped image empty — using full image.")
        return img

    print(f"✅ Cropped region: ({x1},{y1}) → ({x2},{y2}) | Conf: {conf:.2f}")

    debug_img = img.copy()
    cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 3)
    cv2.imwrite("detected_box_debug.png", debug_img)

    return cropped



# ---------------- PREPROCESS ----------------
def preprocess_image(path):
    """Crop using detection, resize, convert to grayscale safely."""
    img = cv2.imread(path)
    if img is None:
        raise ValueError("Unreadable image file.")

    cropped = detect_and_crop(img)
    if cropped is None or cropped.size == 0:
        print("⚠️ Cropping failed — fallback to full image.")
        cropped = img

    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

    return cropped, gray


# ---------------- ROUTES ----------------
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

        cropped_color, gray = preprocess_image(temp_path)

        reflectivity = np.sum(gray > 230) / gray.size
        test_hog = hog(cv2.resize(gray, (256, 256)), orientations=9,
                       pixels_per_cell=(16, 16), cells_per_block=(2, 2),
                       visualize=False, block_norm='L2-Hys')
        pattern_similarity = (
            cosine_similarity([REF_HOG], [test_hog])[0][0]
            if REF_HOG is not None else 0.0
        )

        if pattern_similarity > 0.85 and reflectivity > 0.001:
            result = "✅ Genuine Note"
        elif pattern_similarity > 0.75:
            result = "⚠️ Likely Genuine (Low Light)"
        else:
            result = "❌ Fake / Xerox Detected"

        confidence = f"Pattern: {pattern_similarity:.2f} | Reflectivity: {reflectivity:.5f}"

        _, buffer = cv2.imencode('.png', gray)
        processed_image = base64.b64encode(buffer).decode('utf-8')

        os.remove(temp_path)

        return render_template(
            "result.html",
            result=result,
            confidence=confidence,
            processed_image=processed_image
        )

    except Exception as e:
        print("❌ Error:", e)
        return render_template("result.html", result=f"❌ Error: {str(e)}")


if __name__ == "__main__":
    from waitress import serve
    print("🚀 Starting production server...")
    serve(app, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))


