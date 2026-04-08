import os
import cv2
import numpy as np
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

IMG_SIZE = 64


# -----------------------------
# 1. LOAD + PREPROCESS DATA
# -----------------------------
def load_data(data_dir, limit_per_class=None):
    data = []
    labels = []

    for label in ["cats", "dogs"]:
        path = os.path.join(data_dir, label)
        class_label = 0 if label == "cats" else 1

        count = 0

        for img in os.listdir(path):
            if limit_per_class and count >= limit_per_class:
                break

            try:
                img_path = os.path.join(path, img)
                image = cv2.imread(img_path)

                if image is None:
                    continue

                # Resize
                image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

                # Convert to grayscale
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                data.append(image)
                labels.append(class_label)
                count += 1

            except:
                continue

    return np.array(data), np.array(labels)


# -----------------------------
# 2. FEATURE EXTRACTION (HOG)
# -----------------------------
def extract_features(images):
    features = []

    for img in images:
        hog_feature = hog(
            img,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            block_norm='L2-Hys'
        )
        features.append(hog_feature)

    return np.array(features)


# -----------------------------
# 3. MAIN EXECUTION
# -----------------------------
if __name__ == "__main__":

    # ⚡ LIMIT DATA FOR SPEED (VERY IMPORTANT)
    TRAIN_LIMIT = 3000   # per class → total ~4000
    TEST_LIMIT = 500    # per class → total ~1000

    # -----------------------------
    # LOAD TRAIN DATA
    # -----------------------------
    print("Loading training data...")
    X_train, y_train = load_data("train", limit_per_class=TRAIN_LIMIT)

    print("Train shape:", X_train.shape)

    # -----------------------------
    # HOG FEATURES (TRAIN)
    # -----------------------------
    print("Extracting HOG features (train)...")
    X_train_features = extract_features(X_train)

    print("Train features shape:", X_train_features.shape)

    # -----------------------------
    # TRAIN SVM MODEL
    # -----------------------------
    print("Training SVM... (this may take a few seconds)")
    model = SVC(kernel='rbf')
    model.fit(X_train_features, y_train)

    print("Model training completed!")

    # -----------------------------
    # LOAD TEST DATA
    # -----------------------------
    print("Loading test data...")
    X_test, y_test = load_data("test", limit_per_class=TEST_LIMIT)

    print("Test shape:", X_test.shape)

    # -----------------------------
    # HOG FEATURES (TEST)
    # -----------------------------
    print("Extracting HOG features (test)...")
    X_test_features = extract_features(X_test)

    print("Test features shape:", X_test_features.shape)

    # -----------------------------
    # PREDICT + EVALUATE
    # -----------------------------
    print("Predicting...")
    y_pred = model.predict(X_test_features)

    accuracy = accuracy_score(y_test, y_pred)

    print("Final Accuracy:", accuracy)