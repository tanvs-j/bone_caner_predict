import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import random
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib
import time

# -------------------------------
# 1️⃣ IMAGE PREPROCESSING
# -------------------------------

def enhance_contrast(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    return clahe.apply(gray)

def edge_detection(img):
    edges = cv2.Canny(img, 100, 200)
    return edges

def segment_bone(image_path, k=3):
    img = cv2.imread(image_path)
    if img is None:
        print("⚠️ Image not found:", image_path)
        return None, None, None

    enhanced = enhance_contrast(img)
    edges = edge_detection(enhanced)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pixel_values = img_rgb.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    segmented = centers[labels.flatten()].reshape(img_rgb.shape)

    return enhanced, edges, segmented


# -------------------------------
# 2️⃣ FEATURE EXTRACTION
# -------------------------------
def extract_features(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None
    img = cv2.resize(img, (128, 128))
    mean = np.mean(img, axis=(0, 1))
    std = np.std(img, axis=(0, 1))
    return np.concatenate([mean, std])


# -------------------------------
# 3️⃣ DATA LOADING
# -------------------------------
def load_data(base_dir):
    X, y = [], []
    for label, folder in enumerate(["normal", "cancer"]):
        folder_path = os.path.join(base_dir, folder)
        for file in os.listdir(folder_path):
            img_path = os.path.join(folder_path, file)
            features = extract_features(img_path)
            if features is not None:
                X.append(features)
                y.append(label)
    return np.array(X), np.array(y)


# -------------------------------
# 4️⃣ DOT (BLOB) BASED TUMOR ANALYSIS
# -------------------------------
def analyze_tumor(segmented_img, edges):
    """Detects tumor area based on brightness and structure within bone edges."""
    gray = cv2.cvtColor(segmented_img, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Adaptive threshold — better for variable lighting
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 35, 2)

    # Invert so that bright tumor regions are white
    binary = cv2.bitwise_not(binary)

    # Use bone edges to mask out non-bone (background) regions
    edges_dilated = cv2.dilate(edges, np.ones((5,5), np.uint8), iterations=2)
    bone_mask = cv2.bitwise_not(edges_dilated)
    tumor_mask = cv2.bitwise_and(binary, bone_mask)

    # Morphological cleanup
    kernel = np.ones((5, 5), np.uint8)
    tumor_mask = cv2.morphologyEx(tumor_mask, cv2.MORPH_CLOSE, kernel)
    tumor_mask = cv2.morphologyEx(tumor_mask, cv2.MORPH_OPEN, kernel)

    # Count blobs within bone region
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(tumor_mask, connectivity=8)
    tumor_blobs = num_labels - 1  # subtract background
    tumor_area = np.sum(tumor_mask > 0)

    return tumor_mask, tumor_blobs, tumor_area



def determine_stage(tumor_blobs, tumor_area):
    # Weighted score combining blob count and area
    score = tumor_blobs * 0.5 + (tumor_area / 2000)
    if score < 2:
        return 1, "Low (Stage 1)"
    elif score < 6:
        return 2, "Moderate (Stage 2)"
    else:
        return 3, "High (Stage 3)"


def estimate_lifespan(stage):
    base_years = 10
    if stage == 1:
        return f"{base_years - 1} to {base_years} years (Mild)"
    elif stage == 2:
        return f"{base_years - 4} to {base_years - 1} years (Moderate)"
    else:
        return f"{base_years - 7} to {base_years - 4} years (Severe)"


# -------------------------------
# 5️⃣ MODEL TRAINING (KNN)
# -------------------------------
start_time = time.time()
print("📂 Loading dataset...")
X_train, y_train = load_data("dataset/train")
X_val, y_val = load_data("dataset/valid")
X_test, y_test = load_data("dataset/test")
print(f"✅ Data Loaded: Train={len(X_train)}, Valid={len(X_val)}, Test={len(X_test)}")

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
joblib.dump((knn, scaler), "bone_cancer_dot_model.pkl")

val_pred = knn.predict(X_val)
test_pred = knn.predict(X_test)
print(f"\n🧠 Validation Accuracy: {accuracy_score(y_val, val_pred):.3f}")
print(f"✅ Test Accuracy: {accuracy_score(y_test, test_pred):.3f}")
print("\nClassification Report:\n", classification_report(y_test, test_pred))
print(f"⏱ Training completed in {time.time() - start_time:.2f}s")


# -------------------------------
# 6️⃣ PREDICTION + VISUALIZATION
# -------------------------------
def predict_and_visualize(image_path):
    enhanced, edges, segmented = segment_bone(image_path)
    if segmented is None:
        return

    # ✅ Pass both segmented image and edges
    mask, tumor_blobs, tumor_area = analyze_tumor(segmented, edges)

    highlight = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
    highlight[mask > 0] = [255, 0, 0]

    stage_num, stage_text = determine_stage(tumor_blobs, tumor_area)
    lifespan = estimate_lifespan(stage_num)

    plt.figure(figsize=(16, 6))
    plt.subplot(1, 4, 1)
    plt.imshow(enhanced, cmap='gray')
    plt.title("1️⃣ Contrast Enhanced")
    plt.axis("off")

    plt.subplot(1, 4, 2)
    plt.imshow(edges, cmap='gray')
    plt.title("2️⃣ Edge Detection")
    plt.axis("off")

    plt.subplot(1, 4, 3)
    plt.imshow(segmented)
    plt.title("3️⃣ Segmented Image")
    plt.axis("off")

    plt.subplot(1, 4, 4)
    plt.imshow(highlight)
    plt.title(f"4️⃣ Highlighted Tumor\n{stage_text}\nLifespan: {lifespan}")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    print(f"🩻 Tumor Spots: {tumor_blobs}")
    print(f"📏 Tumor Area: {tumor_area} pixels")
    print(f"🩺 Stage: {stage_text}")
    print(f"⏳ Lifespan: {lifespan}")


# -------------------------------
# 7️⃣ TEST RANDOM IMAGE
# -------------------------------
sample_img = "dataset/test/cancer/" + random.choice(os.listdir("dataset/test/cancer"))
print(f"\n🔍 Testing on: {sample_img}")
predict_and_visualize(sample_img)
