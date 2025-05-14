# 🧪 Skin Cancer Detection using CNN & Streamlit

---

An automated, intelligent system for **early detection of skin cancer** using **Convolutional Neural Networks (CNNs)** and deployed through a **Streamlit web app**. The system allows users to upload images of skin lesions and instantly classifies them as **benign** or **malignant**, enabling faster medical decision-making.

---

## 🚀 Features

* 📷 **Image-based skin lesion classification**
* 🧠 **CNN-based deep learning model** with high accuracy
* 🗃️ **Image preprocessing** using Keras’ `ImageDataGenerator`
* 📊 **Performance visualization** through confusion matrix and training plots
* 💾 **Model saving** in HDF5 format for future inference
* 📈 **Classification metrics** (accuracy, precision, recall, F1-score)
* ⚙️ **Real-time prediction** via Streamlit interface

---

## 🛠️ Tech Stack

* **Python 3.x**
* [TensorFlow / Keras](https://www.tensorflow.org/)
* [NumPy](https://numpy.org/)
* [Pandas](https://pandas.pydata.org/)
* [Matplotlib](https://matplotlib.org/)
* [Seaborn](https://seaborn.pydata.org/)
* [Scikit-learn](https://scikit-learn.org/)
* [Streamlit](https://streamlit.io/)

---
## 🗂️ How to Use

### 1. Setup Environment

Install required packages:

```bash
pip install tensorflow numpy pandas matplotlib seaborn scikit-learn streamlit
```

### 2. Prepare Dataset

* Place your dataset zip file (`skin_cancer_detect.zip`) in the working directory.
* Extract it and ensure folders are organized by class names: `benign/`, `malignant/`.

### 3. Train the Model

Run your training script to:

* Load data
* Build & train CNN model
* Save the model as `a1_model.h5`

### 4. Run Streamlit App

```bash
streamlit run app.py
```

### 5. Upload and Predict

* Upload an image
* Get real-time prediction and confidence level

---


## 🤝 Acknowledgements

* [ISIC Archive](https://www.isic-archive.com/) – Skin cancer datasets
* [TensorFlow](https://www.tensorflow.org/) – Deep Learning Framework
* [Streamlit](https://streamlit.io/) – App interface
* [Kaggle](https://www.kaggle.com/datasets/rm1000/skin-cancer-isic-images/data) – Dataset and community resources


