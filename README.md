# TumorScan AI | Radiology Assistant

This project is a brain tumor classifier built using a deep learning model (EfficientNetB0 CNN), trained into a custom AI model that predicts brain tumors based on MRI scans.

- Glioma
- Meningioma
- Pituitary
- Normal (No Tumor)

---

## Features

- **Four-Class Detection:** Classifies scans into `glioma`, `meningioma`, `normal`, and `pituitary`.
- **High Performance:** Achieves **97.22% test accuracy** using an optimized deep neural network.
- **Dynamic Learning Rate:** Uses `ReduceLROnPlateau` to improve convergence and stabilize training.
- **Best Model Checkpointing:** Automatically saves the best model based on validation accuracy.
- **User Interface:** Simple web-based UI for uploading MRI scans, getting instant predictions, and generating diagnostic reports.

---

## Model Performance

The model was trained for 50 epochs and showed strong learning ability with high generalization performance.

### Training Highlights
- **Best Validation Accuracy:** 97.28% (reached at Epoch 44)
- **Final Test Accuracy:** 97.22%

### Classification Report

The model shows consistently high precision, recall, and F1 scores across all classes, indicating strong and balanced performance.

```text
              precision    recall  f1-score   support

     glioma       0.96      0.99      0.98      1211
 meningioma       0.98      0.92      0.95      1023
    notumor       0.99      0.98      0.99       530
  pituitary       0.96      1.00      0.98      1052
```

### Confusion Matrix

The high concentration of numbers along the dark blue diagonal proves that the model consistently predicts the correct tumor type with very few mistakes across all classes, proving the 97% accuracy on the unseen test data.

![Confusion Matrix](screenshots/confusion_matrix.png)

---

## User Interface & Results

Here is how the application predicts and classifies brain MRI scans for each category:

---

*The model correctly identifies each type of brain tumor by detecting abnormal tissue patterns in the MRI scan.*

### Glioma Prediction Result
![Glioma Classification Example](screenshots/glioma.png)

### Meningioma Prediction Result
![Meningioma Classification Example](screenshots/meningioma.png)

### Pituitary Prediction Result
![Pituitary Classification Example](screenshots/pituitary.png)

---

*The model correctly classifies scans with no abnormalities as normal MRI scans.*

### Normal (No Tumor) Prediction Result
![Normal Brain Scan Example](screenshots/notumor.png)

---

## Tech Stack 

This project uses deep learning and computer vision to build an end-to-end brain tumor classification system.

### Core Technologies
- **Programming Language:** Python 3.10
- **Deep Learning Framework:** TensorFlow, Keras (EfficientNetB0 CNN)
- **Computer Vision:** OpenCV, Pillow (PIL)
- **Web Interface:** Streamlit
- **Data Processing & Evaluation:** NumPy, Scikit-learn
- **Visualization:** Matplotlib, Seaborn

---

## Setup and Running Instructions

### 1. Clone the Repository

```bash
git clone "https://github.com/Maheen012/TumorScan-AI.git"
```

```bash
cd TumorScan-AI
```

### 2. Prerequisites

Ensure you have Python 3.10 installed on your system.

### 3. Create Virtual Environment

Create a new Python 3.10 virtual environment:

```bash
py -3.10 -m venv venv
```

### 4. Activate Environment

```bash
.\venv\Scripts\Activate
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the Application

This will redirect you to the web app:

```bash
streamlit run app.py
```
---

## ⚠️ Disclaimer

This project is developed for educational and research purposes only. It is not intended to be used as a medical diagnostic tool. The model’s predictions are based on training data and may not always be accurate. It should not replace professional medical advice, diagnosis or treatment.

---

## References

The following datasets were used for training and evaluation of this project:

- Brain Tumor MRI Dataset: https://www.kaggle.com/datasets/mohammadhossein77/brain-tumors-dataset  
- Brain Tumor MRI Dataset (Multi-class): https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset
