import os
import cv2 #type:ignore
import numpy as np #type:ignore
import pywt #type:ignore
import joblib #type:ignore 
from scipy.stats import skew, kurtosis #type:ignore
from skimage.feature import local_binary_pattern #type:ignore 
from sklearn.ensemble import RandomForestClassifier #type:ignore
from sklearn.preprocessing import StandardScaler #type:ignore 
from sklearn.model_selection import train_test_split #type:ignore 
from sklearn.metrics import classification_report, accuracy_score #type:ignore

# ---------------- CONFIG ----------------
ROI_SIZE = 1080
DATA_DIR_GENUINE = "resolution/genuine"
DATA_DIR_FAKE = "resolution/fake"
MODEL_NAME = "model_v5.pkl"
SCALER_NAME = "scaler_v5.pkl"


# ---------------- 1. IMAGE PREPROCESSING ----------------
def preprocess_image(path, size=(ROI_SIZE, ROI_SIZE)):
    """Read image, convert to grayscale, equalize lighting, and resize."""
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Unreadable or invalid image: {path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    img = cv2.resize(img, size)
    gray = cv2.resize(gray, size)
    return img, gray


# ---------------- 2. FEATURE EXTRACTORS ----------------
def wavelet_color_features(img, wavelet_name='db2'):
    feats = []
    for channel in cv2.split(img):
        coeffs2 = pywt.dwt2(channel, wavelet_name)
        cA, (cH, cV, cD) = coeffs2
        for mat in [cH, cV, cD]:
            hist, _ = np.histogram(mat.flatten(), bins=64, density=True)
            feats.extend([np.var(hist), skew(hist), kurtosis(hist)])
    return feats


def lbp_texture_features(gray):
    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 59), density=True)
    return hist.tolist()


def extract_features(img_color, gray):
    """Enhanced Xerox-aware features."""
    feats = wavelet_color_features(img_color) + lbp_texture_features(gray)

    # Sharpness, contrast, reflection
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    contrast_map = cv2.absdiff(gray, blur)
    contrast_std = np.std(contrast_map)
    bright_ratio = np.sum(gray > 220) / gray.size

    # Reflection density
    bright_mask = cv2.inRange(gray, 230, 255)
    contours, _ = cv2.findContours(bright_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    reflection_clusters = len([c for c in contours if 5 < cv2.contourArea(c) < 200])
    reflection_density = reflection_clusters / (gray.shape[0] * gray.shape[1] / 10000)

    # Color variance
    color_std = np.std(cv2.cvtColor(img_color, cv2.COLOR_BGR2LAB)[:, :, 0])

    # Edge uniformity
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    edge_mag = np.sqrt(sobelx ** 2 + sobely ** 2)
    edge_uniformity = 1 / (np.std(edge_mag) + 1e-5)

    # Micro-edge density (fine line detection)
    edges = cv2.Canny(gray, 80, 160)
    micro_edge_density = np.sum(edges > 0) / edges.size

    # Specular highlight ratio
    specular_ratio = np.sum(gray > 245) / gray.size

    # Texture coarseness
    highpass = gray - cv2.GaussianBlur(gray, (5, 5), 0)
    texture_coarseness = np.std(highpass)

    feats.extend([
        lap_var, contrast_std, bright_ratio, reflection_density,
        color_std, edge_uniformity, micro_edge_density,
        specular_ratio, texture_coarseness
    ])
    return np.array(feats)


# ---------------- 3. DATASET BUILDING ----------------
def build_dataset():
    X, y = [], []

    print("📂 Loading images from dataset...")

    for label, folder in [(1, DATA_DIR_GENUINE), (0, DATA_DIR_FAKE)]:
        for fname in os.listdir(folder):
            fpath = os.path.join(folder, fname)
            if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            try:
                img_color, gray = preprocess_image(fpath)
                feats = extract_features(img_color, gray)
                X.append(feats)
                y.append(label)
            except Exception as e:
                print(f"⚠️ Skipped {fname}: {e}")

    X, y = np.array(X), np.array(y)
    print(f"✅ Dataset built: {len(X)} samples")
    return X, y


# ---------------- 4. TRAIN MODEL ----------------
def train_model():
    X, y = build_dataset()
    X = np.nan_to_num(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=3,
        min_samples_leaf=2,
        random_state=42,
        class_weight='balanced'
    )
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ Accuracy: {acc:.2f}")

    joblib.dump(rf, MODEL_NAME)
    joblib.dump(scaler, SCALER_NAME)
    print(f"💾 Model saved as {MODEL_NAME} and {SCALER_NAME}")


# ---------------- MAIN ----------------
if __name__ == "__main__":
    train_model()
