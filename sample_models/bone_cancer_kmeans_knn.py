import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import random
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score

# -------------------------------
# 1️⃣  K-MEANS SEGMENTATION
# -------------------------------
def segment_bone(image_path, k=3):
    img = cv2.imread(image_path)
    if img is None:
        print("❌ Error: Image not found at", image_path)
        return None

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pixel_values = img_rgb.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    centers = np.uint8(centers)
    segmented = centers[labels.flatten()]
    segmented_image = segmented.reshape(img_rgb.shape)
    return segmented_image


# -------------------------------
# 2️⃣  FEATURE EXTRACTION FUNCTION
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
# 3️⃣  DATA LOADING FUNCTION
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
# 4️⃣  STAGE & LIFESPAN ESTIMATION
# -------------------------------
def estimate_stage(segmented_image):
    gray = cv2.cvtColor(segmented_image, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    tumor_area = np.sum(thresh > 200)

    if tumor_area < 500:
        return 1
    elif tumor_area < 2000:
        return 2
    else:
        return 3


def estimate_lifespan(stage):
    base_years = 10
    if stage == 1:
        return f"{base_years - 1} to {base_years} years (Mild)"
    elif stage == 2:
        return f"{base_years - 4} to {base_years - 1} years (Moderate)"
    else:
        return f"{base_years - 7} to {base_years - 4} years (Severe)"


# -------------------------------
# 5️⃣  MAIN EXECUTION
# -------------------------------
print("📂 Loading data...")
X_train, y_train = load_data("dataset/train")
X_val, y_val = load_data("dataset/valid")
X_test, y_test = load_data("dataset/test")

print(f"✅ Data Loaded: Train={len(X_train)}, Valid={len(X_val)}, Test={len(X_test)}")

# Train KNN classifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
print("\n🤖 Model training completed.")

# Evaluate on validation data
y_val_pred = knn.predict(X_val)
print("\n📊 Validation Accuracy:", round(accuracy_score(y_val, y_val_pred), 3))

# Evaluate on test data
y_test_pred = knn.predict(X_test)
print("\n✅ Final Test Results:")
print("Accuracy:", round(accuracy_score(y_test, y_test_pred), 3))
print("\nClassification Report:\n", classification_report(y_test, y_test_pred))


# -------------------------------
# 6️⃣  VISUALIZATION + STAGE PREDICTION
# -------------------------------
def predict_and_visualize(image_path):
    segmented = segment_bone(image_path)
    if segmented is None:
        return

    stage = estimate_stage(segmented)
    lifespan = estimate_lifespan(stage)

    plt.figure(figsize=(7,7))
    plt.imshow(segmented)
    plt.title(f"Predicted Stage: {stage}\nEstimated Lifespan: {lifespan}", fontsize=12)
    plt.axis("off")
    plt.show()


# Pick a sample image from test set
sample_img = "dataset/test/cancer/" + random.choice(os.listdir("dataset/test/cancer"))
predict_and_visualize(sample_img)
