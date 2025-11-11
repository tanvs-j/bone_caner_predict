import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
import random

# -------------------------------
# 1️⃣ K-MEANS SEGMENTATION
# -------------------------------
def segment_bone(image_path, k=3):
    img = cv2.imread(image_path)
    if img is None:
        print("❌ Error: Image not found:", image_path)
        return None, None

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pixel_values = img_rgb.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)

    # K-means clustering
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    centers = np.uint8(centers)
    segmented_image = centers[labels.flatten()].reshape(img_rgb.shape)
    return img_rgb, segmented_image


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
# 3️⃣ LOAD DATA
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
# 4️⃣ STAGE + LIFESPAN ESTIMATION
# -------------------------------
def detect_tumor_area(segmented_img):
    """Detects bright tumor-like regions and returns tumor mask + area."""
    gray = cv2.cvtColor(segmented_img, cv2.COLOR_RGB2GRAY)

    # Normalize and threshold
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)

    # Morphological operations to clean the mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    tumor_area = np.sum(mask > 0)
    return mask, tumor_area


def determine_stage(tumor_area):
    """Decide stage based on tumor area (in pixels)."""
    if tumor_area < 1000:
        return 1, "Low (Stage 1)"
    elif tumor_area < 4000:
        return 2, "Moderate (Stage 2)"
    else:
        return 3, "High (Stage 3)"


def estimate_lifespan(stage):
    """Estimate lifespan based on severity."""
    base_years = 10
    if stage == 1:
        return f"{base_years - 1} to {base_years} years (Good prognosis)"
    elif stage == 2:
        return f"{base_years - 4} to {base_years - 1} years (Moderate)"
    else:
        return f"{base_years - 7} to {base_years - 4} years (Severe)"


# -------------------------------
# 5️⃣ TRAIN MODEL
# -------------------------------
print("📂 Loading data...")
X_train, y_train = load_data("dataset/train")
X_val, y_val = load_data("dataset/valid")
X_test, y_test = load_data("dataset/test")

print(f"✅ Data Loaded: Train={len(X_train)}, Valid={len(X_val)}, Test={len(X_test)}")

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Evaluate model
print("\n📊 Validation Accuracy:", round(accuracy_score(y_val, knn.predict(X_val)), 3))
print("\n✅ Final Test Accuracy:", round(accuracy_score(y_test, knn.predict(X_test)), 3))
print("\nClassification Report:\n", classification_report(y_test, knn.predict(X_test)))


# -------------------------------
# 6️⃣ VISUALIZATION
# -------------------------------
def predict_and_highlight(image_path):
    original, segmented = segment_bone(image_path)
    if segmented is None:
        return

    # Detect tumor area and mask
    mask, tumor_area = detect_tumor_area(segmented)

    # Highlight tumor in red
    highlight = original.copy()
    highlight[mask > 0] = [255, 0, 0]  # Red overlay on tumor region

    # Get stage + lifespan
    stage_num, stage_text = determine_stage(tumor_area)
    lifespan = estimate_lifespan(stage_num)

    # Display results
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(original)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(highlight)
    plt.title(f"{stage_text}\nEstimated Lifespan: {lifespan}")
    plt.axis("off")

    plt.show()
    print(f"🩻 Tumor Area: {tumor_area} pixels → {stage_text}")


# -------------------------------
# 7️⃣ TEST ON RANDOM IMAGE
# -------------------------------
sample_img = "dataset/test/cancer/" + random.choice(os.listdir("dataset/test/cancer"))
print(f"\n🔍 Testing on image: {sample_img}")
predict_and_highlight(sample_img)
