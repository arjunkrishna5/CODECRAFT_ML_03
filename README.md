# 🐱🐶 Cats vs Dogs Classification using SVM

## 📌 Project Overview

This project implements a **Support Vector Machine (SVM)** to classify images of cats and dogs.
The model uses **HOG (Histogram of Oriented Gradients)** for feature extraction and trains an SVM classifier to distinguish between the two classes.

---

## 🚀 Features

* Image preprocessing (resize + grayscale)
* Feature extraction using HOG
* SVM model training (Linear & RBF kernel tested)
* Model evaluation with accuracy
* Clean and minimal implementation

---

## 📊 Dataset

The dataset contains images of cats and dogs.

Dataset Source:
https://github.com/laxmimerit/dog-cat-full-dataset

⚠️ **Important Note:**
The dataset is **not included in this repository** because of its large size.
Please download it from the link above and organize it as follows:

```text
project/
│
├── train/
│   ├── cats/
│   └── dogs/
│
├── test/
│   ├── cats/
│   └── dogs/
```

---

## 🧠 Approach

1. Data Preprocessing

   * Resize images to 64x64
   * Convert images to grayscale

2. Feature Extraction

   * Apply HOG (Histogram of Oriented Gradients)
   * Convert images into feature vectors

3. Model Training

   * Support Vector Machine (SVM)
   * Tested both Linear and RBF kernels

4. Evaluation

   * Accuracy score on test dataset

---

## ⚙️ Installation

pip install -r requirements.txt

---

## ▶️ Run the Project

python main.py

---

## 📈 Results

| Model        | Accuracy |
| ------------ | -------- |
| SVM (Linear) | ~69%     |
| SVM (RBF)    | ~76%     |

---



## 🛠️ Tech Stack

* Python
* OpenCV
* NumPy
* Scikit-learn
* Scikit-image

---

