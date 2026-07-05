# Potato Disease Detection Using Deep Learning

A deep learning-based web application that identifies diseases in potato leaves from uploaded images. The project uses a Convolutional Neural Network (CNN) built with TensorFlow/Keras and deployed using Streamlit to classify potato leaves into **Early Blight**, **Late Blight**, or **Healthy**.

---

##  Live Demo

https://sanjana-git12-potato-disease-detection-app-03hkjm.streamlit.app/
---

## Features

- Upload a potato leaf image for disease detection.
- Predicts one of three classes:
  -  Healthy
  -  Early Blight
  -  Late Blight
- Displays prediction confidence.
- Provides disease-specific recommendations.
- Interactive and user-friendly Streamlit interface.

---

## Tech Stack

- Python
- TensorFlow
- Keras
- Streamlit
- NumPy
- Pillow

---

## Project Structure

```
Potato-Disease-Detection/
│
├── app.py                  # Streamlit web application
├── potato_model.keras      # Trained CNN model
├── Potato_Disease_Classification.ipynb
├── requirements.txt
├── README.md
```

---

## 📊 Dataset

This project is trained on the **PlantVillage Potato Leaf Disease Dataset**, containing images of:

- Healthy
- Early Blight
- Late Blight

The dataset is preprocessed and augmented to improve model performance and generalization.

---

## Model Architecture

The CNN model includes:

- Data Augmentation
- Rescaling Layer
- Convolutional Layers
- MaxPooling Layers
- Flatten Layer
- Dense Hidden Layers
- Softmax Output Layer

The model is trained using TensorFlow/Keras for multi-class image classification.

---

##  Installation

### Clone the repository

```bash
git clone https://github.com/sanjana-git12/Potato-Disease-Detection.git
cd Potato-Disease-Detection
```

### Create a virtual environment

```bash
python -m venv .venv
```

### Activate the environment

**Windows**

```bash
.venv\Scripts\activate
```

**Linux / macOS**

```bash
source .venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

---

## Run the Application

```bash
streamlit run app.py
```

Open your browser and visit:

```
http://localhost:8501
```

---

## How It Works

1. Upload a potato leaf image.
2. The image is resized to the model's input size.
3. The trained CNN predicts the disease class.
4. The predicted disease and confidence score are displayed.
5. Recommendations are shown based on the prediction.

---

## Future Improvements

- Support additional crop diseases.
- Mobile-friendly interface.
- Disease severity estimation.
- Real-time detection using a camera.
- Deploy on cloud platforms for public access.

---

## Author

**Sanjana Nimmagadda**

---
